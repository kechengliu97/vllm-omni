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

        # When the AR stage is configured with `detokenize: false` (the default
        # for the HunyuanImage-3 IT2I configs, which avoids the cost of streaming
        # text during generation), ``output.text`` is empty even though tokens
        # were produced.  The DiT needs the decoded CoT text to rebuild the
        # joint sequence via ``apply_chat_template``, so decode the tokens here
        # on the fly as a fallback.  Without this fallback the DiT receives an
        # empty ``ar_generated_text`` and silently drops the image conditioning
        # — this is the "text length=0 / model ignores the input image" symptom
        # reported in https://github.com/vllm-project/vllm-omni/pull/2590.
        if not generated_text and generated_token_ids:
            tokenizer = _resolve_ar_tokenizer(stage_list[source_stage_id])
            if tokenizer is not None:
                try:
                    generated_text = tokenizer.decode(list(generated_token_ids), skip_special_tokens=False)
                except Exception as exc:
                    logger.warning(
                        "[ar2diffusion] Failed to decode AR tokens for request %d: %s",
                        i,
                        exc,
                    )

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

        # If the AR stage used a pretrain-format input (KV-reuse path) the
        # trigger tag (e.g. "<think>") was NOT part of the AR prefill and is
        # provided separately via ``original_prompt["trigger_tag"]``.  Concatenate
        # it here (mirroring the official HunyuanImage-3 reference:
        # https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/hunyuan_image_3/modeling_hunyuan_image_3.py#L3355)
        # so the DiT just sees a single ``ar_generated_text`` field without
        # needing to know about the trigger tag.  Guard against double-prepend
        # when the AR model has already emitted the trigger as its first token.
        trigger_tag = original_prompt.get("trigger_tag")
        if trigger_tag and generated_text and not generated_text.startswith(trigger_tag):
            generated_text = trigger_tag + generated_text

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


_AR_TOKENIZER_CACHE: dict[str, Any] = {}


def _resolve_ar_tokenizer(stage_client: Any) -> Any:
    """Best-effort resolution of the AR stage's tokenizer.

    The stage client exposes the resolved ``vllm_config``; we use its
    ``model_config.model`` path to load an ``AutoTokenizer`` lazily and cache
    the result so we don't pay the init cost on every request.
    """
    model_path = None
    vllm_config = getattr(stage_client, "vllm_config", None)
    if vllm_config is not None:
        model_cfg = getattr(vllm_config, "model_config", None)
        if model_cfg is not None:
            model_path = getattr(model_cfg, "tokenizer", None) or getattr(model_cfg, "model", None)
    if model_path is None:
        model_path = getattr(stage_client, "model", None) or getattr(stage_client, "model_name", None)
    if not model_path:
        return None
    if model_path in _AR_TOKENIZER_CACHE:
        return _AR_TOKENIZER_CACHE[model_path]
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        logger.warning("[ar2diffusion] Could not load tokenizer from %r: %s", model_path, exc)
        _AR_TOKENIZER_CACHE[model_path] = None
        return None
    _AR_TOKENIZER_CACHE[model_path] = tokenizer
    return tokenizer


def expand_cfg_prompts(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list:
    """Expand user prompt into companion prompts for HunyuanImage3 CFG.

    Creates a negative/unconditional companion request that the AR stage
    prefills to produce the CFG text KV cache.  The companion request gets
    ``max_tokens=1`` so it stops immediately after prefill – only the text
    KV cache is needed, no decoding.

    The companion must mirror the **structure** of the positive prompt so
    that the positive and negative KV caches have matching layouts
    (same system-prompt section, same image section when present, same
    assistant/trigger prefix).  Otherwise the injected ``L_pos`` and
    ``L_neg`` diverge in both length and per-section semantics, and the
    DiT's CFG becomes mathematically degenerate – the exact failure mode
    we observed as ``L_pos=6833, L_neg=1`` in the logs.

    Official Hunyuan reference behaviour (``tokenization_hunyuan_image_3.py``):

        if uncond_flag and do_uncond_drop:
            text_token = [self.cfg_token_id] * len(text_token)

    i.e. every user-text token is replaced with the ``<cfg>`` token, but
    every other section (system, image, trigger) is preserved.  We
    approximate that here by reusing the positive prompt dict and
    replacing only the user-facing text fields with the ``<cfg>`` token.

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

    CFG_TOKEN = "<cfg>"

    # 1. Build the negative prompt string from the positive prompt string.
    # Substitute the user's text with a single <cfg> token so the companion
    # traverses the exact same sections (BOS, system, image marker,
    # assistant/trigger) as the positive branch.  The small length delta
    # vs the "one <cfg> per user-text token" official behaviour is zero-
    # padded and masked in inject_prompt_kv_cache, which is benign because
    # the padding sits at the end of the user-text section — all the
    # model-critical sections remain aligned.
    pos_prompt_text = prompt.get("prompt")
    user_text = prompt.get("user_prompt")
    explicit_neg = _get_negative_prompt(prompt, sampling_params)

    if explicit_neg:
        neg_prompt = explicit_neg
    elif pos_prompt_text and user_text and user_text in pos_prompt_text:
        # Replace the exact user-text substring with the <cfg> token.
        neg_prompt = pos_prompt_text.replace(user_text, CFG_TOKEN, 1)
    elif pos_prompt_text:
        # Cannot locate user_text; append <cfg> before the assistant
        # marker if present, else fall back to the pos prompt verbatim.
        neg_prompt = pos_prompt_text
    else:
        # Final fallback — at least non-empty.
        neg_prompt = "<|startoftext|>"

    # 2. Build the companion dict, inheriting modalities + multimodal data
    # so the image section is prefilled for the uncond branch too.
    neg_prompt_dict: dict[str, Any] = {
        "prompt": neg_prompt,
        "modalities": prompt.get("modalities", []),
    }
    if "multi_modal_data" in prompt:
        neg_prompt_dict["multi_modal_data"] = prompt["multi_modal_data"]
    if "height" in prompt:
        neg_prompt_dict["height"] = prompt["height"]
    if "width" in prompt:
        neg_prompt_dict["width"] = prompt["width"]
    if "use_system_prompt" in prompt:
        neg_prompt_dict["use_system_prompt"] = prompt["use_system_prompt"]

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
