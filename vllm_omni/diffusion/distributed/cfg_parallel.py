# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Base pipeline class for Diffusion models with shared CFG functionality.
"""

from abc import ABCMeta
import os
from typing import Any

import torch

from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


class CFGParallelMixin(metaclass=ABCMeta):
    """
    Base Mixin class for Diffusion pipelines providing shared CFG methods.

    All pipelines should inherit from this class to reuse
    classifier-free guidance logic.
    """

    _CFG_ORIGINAL_LOG_PATH = "/home/l30053556/cfg-fix/noise_input_original.log"
    _CFG_PARALLEL_LOG_PATH = "/home/l30053556/cfg-fix/noise_input_parallel.log"

    def _next_noise_debug_step(self) -> int:
        step = getattr(self, "_cfg_noise_debug_step", 0) + 1
        self._cfg_noise_debug_step = step
        return step

    def _format_tensor_stats(self, tensor: torch.Tensor) -> str:
        if tensor is None:
            return "None"
        t = tensor.detach().float()
        flat = t.reshape(-1)
        preview_count = min(4, flat.numel())
        preview = ", ".join(f"{float(v):.6g}" for v in flat[:preview_count].cpu())
        return (
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"mean={float(t.mean().item()):.6g}, std={float(t.std(unbiased=False).item()):.6g}, "
            f"min={float(t.min().item()):.6g}, max={float(t.max().item()):.6g}, "
            f"preview=[{preview}]"
        )

    def _extract_input_tensor(self, kwargs: dict[str, Any] | None) -> torch.Tensor | None:
        if not kwargs:
            return None
        if isinstance(kwargs.get("hidden_states"), torch.Tensor):
            return kwargs["hidden_states"]
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                return v
        return None

    def _append_noise_debug_log(self, log_path: str, message: str) -> None:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception:
            # Debug logging must never break generation.
            pass

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
        debug_step = self._next_noise_debug_step()

        if do_true_cfg:
            # Automatically detect CFG parallel configuration
            cfg_parallel_ready = get_classifier_free_guidance_world_size() > 1

            if cfg_parallel_ready:
                # Enable CFG-parallel: rank0 computes positive, rank1 computes negative.
                cfg_group = get_cfg_group()
                cfg_rank = get_classifier_free_guidance_rank()

                self._append_noise_debug_log(
                    self._CFG_PARALLEL_LOG_PATH,
                    (
                        f"step={debug_step}, rank={cfg_rank}, mode=parallel, "
                        f"pos={self._format_tensor_stats(self._extract_input_tensor(positive_kwargs))}, "
                        f"neg={self._format_tensor_stats(self._extract_input_tensor(negative_kwargs))}"
                    ),
                )

                if cfg_rank == 0:
                    local_pred = self.predict_noise(**positive_kwargs)
                else:
                    local_pred = self.predict_noise(**negative_kwargs)

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
                self._append_noise_debug_log(
                    self._CFG_ORIGINAL_LOG_PATH,
                    (
                        f"step={debug_step}, rank=0, mode=sequential, "
                        f"pos={self._format_tensor_stats(self._extract_input_tensor(positive_kwargs))}, "
                        f"neg={self._format_tensor_stats(self._extract_input_tensor(negative_kwargs))}"
                    ),
                )

                positive_noise_pred = self.predict_noise(**positive_kwargs)
                negative_noise_pred = self.predict_noise(**negative_kwargs)

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
