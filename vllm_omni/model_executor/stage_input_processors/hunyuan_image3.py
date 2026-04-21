# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for HunyuanImage3: AR → Diffusion transition.

In IT2I (image editing) mode:
  - Stage 0 (AR) receives (image + edit instruction), generates CoT/latent tokens
  - Stage 1 (DiT) receives the AR output + original image, denoises → edited image

For inter-stage KV cache reuse (text-to-image):
  - Stage 0 (AR) prefills text tokens and sends KV cache to Stage 1
  - expand_cfg_prompts creates a CFG companion request for the negative branch
  - collect_cfg_kv_caches collects the companion KV cache for the DiT

The ar2diffusion function bridges these two stages, following the same
signature pattern as glm_image.ar2diffusion.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def ar2diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Process AR stage outputs to create Diffusion stage inputs.

    Args:
        stage_list: List of stage clients (set by orchestrator).
        engine_input_source: List of source stage IDs (from YAML).
        prompt: Original user prompt (may contain multimodal data).
        requires_multimodal_data: Whether to forward multimodal data.

    Returns:
        List of dicts, each consumable by the HunyuanImage3 diffusion pipeline.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid source stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    ar_outputs = stage_list[source_stage_id].engine_outputs
    diffusion_inputs = []

    # Normalize prompt to list
    if not isinstance(prompt, list):
        prompt = [prompt] if prompt is not None else [{}]

    for i, ar_output in enumerate(ar_outputs):
        output = ar_output.outputs[0]
        generated_token_ids = output.token_ids
        generated_text = getattr(output, "text", "") or ""

        # Get original prompt info
        original_prompt = prompt[i] if i < len(prompt) else {}
        if isinstance(original_prompt, dict):
            pass
        elif hasattr(original_prompt, "_asdict"):
            original_prompt = original_prompt._asdict()
        elif hasattr(original_prompt, "__dict__"):
            original_prompt = vars(original_prompt)
        else:
            original_prompt = {}

        height = original_prompt.get("height", 1024)
        width = original_prompt.get("width", 1024)
        # Prefer clean user_prompt over the full baked AR string so the DiT
        # can reconstruct the sequence with the same template as at training time.
        # Fall back to "prompt" for callers that do not populate "user_prompt".
        text_prompt = original_prompt.get("user_prompt") or original_prompt.get("prompt", "")
        use_system_prompt = original_prompt.get("use_system_prompt")

        logger.info(
            "[ar2diffusion] Request %d: AR generated %d tokens, text length=%d, target size=%dx%d",
            i,
            len(generated_token_ids),
            len(generated_text),
            height,
            width,
        )

        token_tensor = torch.tensor(generated_token_ids, dtype=torch.long)

        diffusion_input: dict[str, Any] = {
            "prompt": text_prompt,
            "height": height,
            "width": width,
            "extra": {
                "ar_token_ids": token_tensor,
                "ar_generated_text": generated_text,
            },
        }

        # Forward use_system_prompt so the DiT can build the same system prefix
        if use_system_prompt is not None:
            diffusion_input["use_system_prompt"] = use_system_prompt

        # Forward multimodal data (original image for IT2I conditioning)
        mm_data = original_prompt.get("multi_modal_data")
        if mm_data:
            prompt_images = mm_data.get("image")
            if prompt_images is None:
                prompt_images = mm_data.get("images")
            if prompt_images is not None:
                diffusion_input["pil_image"] = prompt_images
                diffusion_input["multi_modal_data"] = {"image": prompt_images}

        # Forward multimodal output from AR (if any)
        if hasattr(ar_output, "multimodal_output") and ar_output.multimodal_output:
            mm_output = ar_output.multimodal_output
            if isinstance(mm_output, dict):
                diffusion_input["extra"]["ar_multimodal_output"] = mm_output

        # Forward sampling params
        for key in ["seed", "num_inference_steps", "guidance_scale", "negative_prompt"]:
            if key in original_prompt:
                diffusion_input[key] = original_prompt[key]

        diffusion_inputs.append(diffusion_input)

    return diffusion_inputs


# ---------------------------------------------------------------------------
# CFG prompt expansion & KV cache collection for inter-stage KV reuse
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

CFG_TEXT_SUFFIX = "__cfg_text"


def expand_cfg_prompts(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list:
    """Expand user prompt into companion prompts for HunyuanImage3 CFG.

    Creates a negative/unconditional companion request that the AR stage
    prefills to produce the CFG text KV cache.  The companion request gets
    ``max_tokens=1`` so it stops immediately after prefill – only the text
    KV cache is needed, no decoding.

    For text-to-image (``modalities`` contains ``"image"``):
      - One extra prompt for cfg_text (negative/empty prompt).
      - The negative prompt uses an empty string by default.

    Args:
        prompt: The original user prompt (dict or string).
        sampling_params: Stage-0 sampling params.

    Returns:
        List of ``ExpandedPrompt``.  Empty if no expansion needed.
    """
    from vllm_omni.model_executor.stage_input_processors.bagel import ExpandedPrompt

    if not isinstance(prompt, dict):
        return []

    modalities = prompt.get("modalities", [])
    if "image" not in modalities:
        return []

    # Resolve negative prompt — use a minimal BOS marker when empty so the
    # companion request is never zero-length (vLLM rejects empty prompts).
    neg_prompt = _get_negative_prompt(prompt, sampling_params)
    if not neg_prompt:
        neg_prompt = "<|startoftext|>"

    neg_prompt_dict: dict[str, Any] = {
        "prompt": neg_prompt,
        "modalities": prompt.get("modalities", []),
    }

    # Companion only needs prefill KV – stop after 1 token.
    companion_params = {"max_tokens": 1}

    return [
        ExpandedPrompt(
            prompt=neg_prompt_dict,
            role="cfg_text",
            request_id_suffix=CFG_TEXT_SUFFIX,
            sampling_params_override=companion_params,
        ),
    ]


def collect_cfg_kv_caches(
    request_id: str,
    cfg_request_ids: dict[str, str],
    kv_transfer_manager: Any,
    target_device: Any | None = None,
) -> dict[str, Any]:
    """Collect KV caches for CFG companion requests.

    Called by the diffusion model runner after receiving the primary KV
    cache.  Uses the kv_transfer_manager to fetch companion KV caches by
    their request IDs.

    Args:
        request_id: The original (parent) request ID.
        cfg_request_ids: Mapping of role → companion request ID,
            e.g. ``{"cfg_text": "req_0__cfg_text"}``.
        kv_transfer_manager: The ``OmniKVTransferManager`` instance.
        target_device: Device to move tensors to.

    Returns:
        Dict with keys like ``"cfg_text_past_key_values"``,
        ``"cfg_text_kv_metadata"``, etc.
    """
    result: dict[str, Any] = {}

    for role, companion_rid in cfg_request_ids.items():
        try:
            data, size = kv_transfer_manager.receive_kv_cache_for_request(
                companion_rid,
                target_device,
            )
            if data and "layer_blocks" in data:
                layer_blocks = data["layer_blocks"]
                kv_obj = SimpleNamespace(**layer_blocks)
                result[f"{role}_past_key_values"] = kv_obj
                if "metadata" in data:
                    result[f"{role}_kv_metadata"] = data["metadata"]
                logger.info(
                    "Collected CFG KV cache for role=%s, rid=%s, size=%d bytes",
                    role,
                    companion_rid,
                    size,
                )
            else:
                logger.warning(
                    "Failed to collect CFG KV cache for role=%s, rid=%s",
                    role,
                    companion_rid,
                )
        except Exception as e:
            logger.exception(
                "Error collecting CFG KV cache for role=%s, rid=%s: %s",
                role,
                companion_rid,
                e,
            )

    return result


def _get_negative_prompt(
    prompt: dict[str, Any],
    sampling_params: Any,
) -> str:
    """Resolve the negative prompt from prompt dict or sampling params."""
    neg = prompt.get("negative_prompt")
    if neg:
        return neg

    if hasattr(sampling_params, "extra_args") and sampling_params.extra_args:
        neg = sampling_params.extra_args.get("negative_prompt")
        if neg:
            return neg

    return ""
