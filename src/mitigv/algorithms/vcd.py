"""VCD — Visual Contrastive Decoding (Leng et al., CVPR 2024).

Training-free hallucination mitigation that contrasts the next-token logits
from the original image against those from a distorted image::

    logits = (1 + alpha) * logits(v) - alpha * logits(v')

optionally followed by an adaptive plausibility constraint (``beta``) that masks
tokens unlikely under the distorted distribution.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any

import torch

from mitigv.backends.hf import HFMitigator, HFMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator
from mitigv.perturbations import build_perturbation

__all__ = ["VCDConfig", "VCD"]


class VCDConfig(HFMitigatorConfig):
    """Hyper-parameters for VCD.

    Attributes:
        alpha: Contrast strength. ``alpha=0`` degenerates to plain decoding.
        beta: Adaptive-plausibility threshold. ``beta=0`` disables the constraint.
        distortion: Name of the registered perturbation applied to the image.
        distortion_kwargs: Keyword arguments forwarded to the perturbation.
    """

    alpha: float = 1.0
    beta: float = 0.1
    distortion: str = "diffusion_noise"
    distortion_kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if self.alpha < 0:
            raise MitigatorConfigError("alpha must be >= 0")
        if self.beta < 0:
            raise MitigatorConfigError("beta must be >= 0")


@register_mitigator("vcd")
class VCD(HFMitigator):
    """Visual Contrastive Decoding.

    At every decoding step it runs the model on both the original and a
    distorted image, contrasts the two logits distributions (weight ``alpha``),
    and optionally applies the adaptive plausibility constraint (``beta``).
    """

    algorithm_name = "vcd"
    config_class = VCDConfig

    # -- input preparation -----------------------------------------------------
    def _prepare_inputs(self, images: Any, prompt: str, cfg: VCDConfig) -> dict[str, Any]:
        inputs = super()._prepare_inputs(images, prompt, cfg)
        self._distorted_inputs = self._distort(inputs, cfg)
        self._distorted_past = None
        return inputs

    def _distort(self, inputs: dict[str, Any], cfg: VCDConfig) -> dict[str, Any]:
        """Return a copy of ``inputs`` with the image tensors perturbed."""
        perturbation = build_perturbation(cfg.distortion, **cfg.distortion_kwargs)
        distorted = dict(inputs)
        for key in self._image_keys(inputs):
            distorted[key] = perturbation(inputs[key])
        return distorted

    def _image_keys(self, inputs: dict[str, Any]) -> list[str]:
        """Identify the image tensors to perturb (override for exotic models)."""
        return [
            k
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor) and (k == "images" or "pixel_values" in k)
        ]

    def _reorder_aux_cache(self, beam_idx: torch.Tensor) -> None:
        """Reorder the distorted-branch cache during beam search."""
        self._distorted_past = self._reorder_cache(self._distorted_past, beam_idx)

    def _expand_inputs_for_beams(self, inputs: dict[str, Any], num_beams: int) -> dict[str, Any]:
        """Expand the main inputs *and* the distorted-branch inputs for beam search."""
        expanded = super()._expand_inputs_for_beams(inputs, num_beams)
        self._distorted_inputs = super()._expand_inputs_for_beams(
            self._distorted_inputs, num_beams
        )
        return expanded

    # -- intervention -----------------------------------------------------------
    def _step_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        inputs: dict[str, Any],
        past_key_values: Any,
        step: int,
        cfg: VCDConfig,
    ) -> tuple[torch.Tensor, Any]:
        logits_v, past = self._forward(input_ids, attention_mask, inputs, past_key_values)
        logits_vp, self._distorted_past = self._forward(
            input_ids, attention_mask, self._distorted_inputs, self._distorted_past
        )
        logits = (1.0 + cfg.alpha) * logits_v - cfg.alpha * logits_vp
        if cfg.beta > 0:
            logits = self._adaptive_plausibility(logits, logits_v, cfg.beta)
        return logits, past

    @staticmethod
    def _adaptive_plausibility(
        logits: torch.Tensor, logits_original: torch.Tensor, beta: float
    ) -> torch.Tensor:
        """Mask tokens whose *original*-branch probability is below ``beta * max``.

        This is the paper's adaptive plausibility constraint, applied in log
        space (equivalent to ``p_original < beta * max(p_original)``):

            cutoff = log(beta) + max(logits_original)

        Tokens whose original logits fall below ``cutoff`` are set to ``-inf``.
        """
        cutoff = torch.log(
            torch.tensor(beta, device=logits.device, dtype=logits.dtype)
        ) + logits_original.max(dim=-1, keepdim=True).values
        return logits.masked_fill(logits_original < cutoff, float("-inf"))
