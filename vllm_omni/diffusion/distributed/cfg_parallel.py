# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Base pipeline class for Diffusion models with shared CFG functionality.
"""

from abc import ABCMeta
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)

# ── debug: one-time capture dirs ──────────────────────────────────────────────
_DBG_PARALLEL_DIR = Path("/home/l30053556/cfg-fix/negative_kwargs/parallel")
_DBG_ORIGINAL_DIR = Path("/home/l30053556/cfg-fix/negative_kwargs/original")
_DBG_PARALLEL_DONE: bool = False
_DBG_ORIGINAL_DONE: bool = False
# ─────────────────────────────────────────────────────────────────────────────


def _dbg_save(
    negative_kwargs: dict[str, Any],
    output: torch.Tensor,
    save_dir: Path,
    branch: str,
    transformer: Any = None,
) -> None:
    """
    One-shot debug helper.

    Saves to save_dir/:
      negative_<key>.pt   – each tensor in negative_kwargs (CPU copy)
      negative_output.pt  – the output of predict_noise (CPU copy)
      meta.txt            – shapes, dtypes, device, model.training flag

    Prints a summary so the output appears in the process log.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"[cfg-debug:{branch}] ========================================"]

    # model state
    if transformer is not None:
        lines.append(f"  transformer.training = {transformer.training}")
        # print all unique devices that model parameters live on
        param_devices = list({str(p.device) for p in transformer.parameters()})
        lines.append(f"  parameter devices    = {param_devices}")

    # save each tensor in negative_kwargs
    for key, val in negative_kwargs.items():
        if isinstance(val, torch.Tensor):
            pt_path = save_dir / f"negative_{key}.pt"
            torch.save(val.detach().cpu(), pt_path)
            lines.append(f"  input  [{key}]: shape={val.shape}, dtype={val.dtype}, device={val.device}")
        else:
            lines.append(f"  input  [{key}]: {type(val).__name__} = {val}")

    # save predict_noise output
    out_path = save_dir / "negative_output.pt"
    torch.save(output.detach().cpu(), out_path)
    lines.append(f"  output : shape={output.shape}, dtype={output.dtype}, device={output.device}")
    lines.append(f"  output : min={output.float().min().item():.6f}  max={output.float().max().item():.6f}  "
                 f"mean={output.float().mean().item():.6f}  std={output.float().std().item():.6f}")
    lines.append(f"[cfg-debug:{branch}] saved to {save_dir}")
    lines.append(f"[cfg-debug:{branch}] ========================================")

    print("\n".join(lines), flush=True)


class CFGParallelMixin(metaclass=ABCMeta):
    """
    Base Mixin class for Diffusion pipelines providing shared CFG methods.

    All pipelines should inherit from this class to reuse
    classifier-free guidance logic.
    """

    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any] | None,
        cfg_normalize: bool = True,
        output_slice: int | None = None,
    ) -> torch.Tensor | None:
        """
        Predict noise with optional classifier-free guidance.

        Args:
            do_true_cfg: Whether to apply CFG
            true_cfg_scale: CFG scale factor
            positive_kwargs: Kwargs for positive/conditional prediction
            negative_kwargs: Kwargs for negative/unconditional prediction
            cfg_normalize: Whether to normalize CFG output (default: True)
            output_slice: If set, slice output to [:, :output_slice] for image editing

        Returns:
            Predicted noise tensor (only valid on rank 0 in CFG parallel mode)
        """
        global _DBG_PARALLEL_DONE, _DBG_ORIGINAL_DONE

        if do_true_cfg:
            # Automatically detect CFG parallel configuration
            cfg_parallel_ready = get_classifier_free_guidance_world_size() > 1

            if cfg_parallel_ready:
                # Enable CFG-parallel: rank0 computes positive, rank1 computes negative.
                cfg_group = get_cfg_group()
                cfg_rank = get_classifier_free_guidance_rank()

                if cfg_rank == 0:
                    local_pred = self.predict_noise(**positive_kwargs)
                else:
                    transformer = getattr(self, "transformer", None)
                    if transformer is not None:
                        transformer._cfg_debug_branch = "parallel"
                    # ── encoder debug logging ────────────────────────────────────────
                    encoder_hs = negative_kwargs.get("encoder_hidden_states", None)
                    if encoder_hs is not None and isinstance(encoder_hs, torch.Tensor):
                        print(
                            f"[cfg-debug:parallel] encoder_hidden_states input: "
                            f"shape={encoder_hs.shape}, dtype={encoder_hs.dtype}, device={encoder_hs.device}, "
                            f"mean={encoder_hs.float().mean().item():.6f}, "
                            f"std={encoder_hs.float().std().item():.6f}, "
                            f"abs_mean={encoder_hs.float().abs().mean().item():.6f}, "
                            f"abs_max={encoder_hs.float().abs().max().item():.6f}",
                            flush=True
                        )
                    # ──────────────────────────────────────────────────────────────────
                    try:
                        local_pred = self.predict_noise(**negative_kwargs)
                    finally:
                        if transformer is not None:
                            transformer._cfg_debug_branch = None
                    if not _DBG_PARALLEL_DONE:
                        _DBG_PARALLEL_DONE = True
                        _dbg_save(
                            negative_kwargs, local_pred,
                            _DBG_PARALLEL_DIR, "parallel",
                            transformer=getattr(self, "transformer", None),
                        )

                # Slice output for image editing pipelines (remove condition latents)
                if output_slice is not None:
                    local_pred = local_pred[:, :output_slice]

                gathered = cfg_group.all_gather(local_pred, separate_tensors=True)

                if cfg_rank == 0:
                    noise_pred = gathered[0]
                    neg_noise_pred = gathered[1]
                    noise_pred = self.combine_cfg_noise(noise_pred, neg_noise_pred, true_cfg_scale, cfg_normalize)
                    return noise_pred
                else:
                    return None
            else:
                # Sequential CFG: compute both positive and negative
                positive_noise_pred = self.predict_noise(**positive_kwargs)
                transformer = getattr(self, "transformer", None)
                if transformer is not None:
                    transformer._cfg_debug_branch = "original"
                # ── encoder debug logging ────────────────────────────────────────
                encoder_hs = negative_kwargs.get("encoder_hidden_states", None)
                if encoder_hs is not None and isinstance(encoder_hs, torch.Tensor):
                    print(
                        f"[cfg-debug:original] encoder_hidden_states input: "
                        f"shape={encoder_hs.shape}, dtype={encoder_hs.dtype}, device={encoder_hs.device}, "
                        f"mean={encoder_hs.float().mean().item():.6f}, "
                        f"std={encoder_hs.float().std().item():.6f}, "
                        f"abs_mean={encoder_hs.float().abs().mean().item():.6f}, "
                        f"abs_max={encoder_hs.float().abs().max().item():.6f}",
                        flush=True
                    )
                # ──────────────────────────────────────────────────────────────────
                try:
                    negative_noise_pred = self.predict_noise(**negative_kwargs)
                finally:
                    if transformer is not None:
                        transformer._cfg_debug_branch = None
                if not _DBG_ORIGINAL_DONE:
                    _DBG_ORIGINAL_DONE = True
                    _dbg_save(
                        negative_kwargs, negative_noise_pred,
                        _DBG_ORIGINAL_DIR, "original",
                        transformer=getattr(self, "transformer", None),
                    )

                # Slice output for image editing pipelines
                if output_slice is not None:
                    positive_noise_pred = positive_noise_pred[:, :output_slice]
                    negative_noise_pred = negative_noise_pred[:, :output_slice]

                noise_pred = self.combine_cfg_noise(
                    positive_noise_pred, negative_noise_pred, true_cfg_scale, cfg_normalize
                )
                return noise_pred
        else:
            # No CFG: only compute positive/conditional prediction
            pred = self.predict_noise(**positive_kwargs)
            if output_slice is not None:
                pred = pred[:, :output_slice]
            return pred

    def cfg_normalize_function(self, noise_pred: torch.Tensor, comb_pred: torch.Tensor) -> torch.Tensor:
        """
        Normalize the combined noise prediction.

        Args:
            noise_pred: positive noise prediction
            comb_pred: combined noise prediction after CFG

        Returns:
            Normalized noise prediction tensor
        """
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        noise_pred = comb_pred * (cond_norm / noise_norm)
        return noise_pred

    def combine_cfg_noise(
        self, noise_pred: torch.Tensor, neg_noise_pred: torch.Tensor, true_cfg_scale: float, cfg_normalize: bool = False
    ) -> torch.Tensor:
        """
        Combine conditional and unconditional noise predictions with CFG.

        Args:
            noise_pred: Conditional noise prediction
            neg_noise_pred: Unconditional noise prediction
            true_cfg_scale: CFG scale factor
            cfg_normalize: Whether to normalize the combined prediction (default: False)

        Returns:
            Combined noise prediction tensor
        """
        comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

        if cfg_normalize:
            noise_pred = self.cfg_normalize_function(noise_pred, comb_pred)
        else:
            noise_pred = comb_pred

        return noise_pred

    def predict_noise(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass through transformer to predict noise.

        Subclasses should override this if they need custom behavior,
        but the default implementation calls self.transformer.
        """
        return self.transformer(*args, **kwargs)[0]

    def diffuse(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Diffusion loop with optional classifier-free guidance.

        Subclasses MUST implement this method to define the complete
        diffusion/denoising loop for their specific model.

        Typical implementation pattern:
        ```python
        def diffuse(self, latents, timesteps, prompt_embeds, negative_embeds, ...):
            for t in timesteps:
                # Prepare kwargs for positive and negative predictions
                positive_kwargs = {...}
                negative_kwargs = {...}

                # Predict noise with automatic CFG handling
                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=True,
                    true_cfg_scale=self.guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                )

                # Step scheduler with automatic CFG sync
                latents = self.scheduler_step_maybe_with_cfg(
                    noise_pred, t, latents, do_true_cfg=True
                )

            return latents
        ```
        """
        raise NotImplementedError("Subclasses must implement diffuse")

    def scheduler_step(self, noise_pred: torch.Tensor, t: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Step the scheduler.

        Args:
            noise_pred: Predicted noise
            t: Current timestep
            latents: Current latents

        Returns:
            Updated latents after scheduler step
        """
        return self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    def scheduler_step_maybe_with_cfg(
        self, noise_pred: torch.Tensor, t: torch.Tensor, latents: torch.Tensor, do_true_cfg: bool
    ) -> torch.Tensor:
        """
        Step the scheduler with (maybe) automatic CFG parallel synchronization.

        In CFG parallel mode, only rank 0 computes the scheduler step,
        then broadcasts the result to other ranks.

        Args:
            noise_pred: Predicted noise (only valid on rank 0 in CFG parallel)
            t: Current timestep
            latents: Current latents
            do_true_cfg: Whether CFG is enabled

        Returns:
            Updated latents (synchronized across all CFG ranks)
        """
        # Automatically detect CFG parallel configuration
        cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1

        if cfg_parallel_ready:
            cfg_group = get_cfg_group()
            cfg_rank = get_classifier_free_guidance_rank()

            # Only rank 0 computes the scheduler step
            if cfg_rank == 0:
                latents = self.scheduler_step(noise_pred, t, latents)

            # Broadcast the updated latents to all ranks
            latents = latents.contiguous()
            cfg_group.broadcast(latents, src=0)
        else:
            # No CFG parallel: directly compute scheduler step
            latents = self.scheduler_step(noise_pred, t, latents)

        return latents
