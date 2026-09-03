"""M3ID — Multi-Modal Mutual Information Decoding (Favero et al., CVPR 2024).

Training-free decoding that counteracts *conditioning dilution*: as more tokens
are generated, the visual prompt's influence fades, so the model increasingly
relies on its language prior and hallucinates. M3ID amplifies the visual
(conditional) branch relative to a no-image (unconditional) branch, with the
amplification weight *growing* with the decoding step::

    gamma_t     = exp(-lambda * t)            # first predicted token uses t = 1
    weight_t    = (1 - gamma_t) / gamma_t     # = exp(lambda * t) - 1
    logits      = l_c + 1[max(l_c) < log(alpha)] * weight_t * (l_c - l_u)

where ``l_c``/``l_u`` are the with-image / no-image logits. The indicator gate
applies the contrast only when the model is *uncertain* (top-1 conditional
probability below ``alpha``), which prevents overcompensation.

Larger ``alpha`` values activate the uncertainty gate more often;
``forgetting_rate=0`` gives a constant weight of 0 (plain decoding).
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

import torch

from mitigv.backends.generic import GenericMitigator, GenericMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator

__all__ = ["M3IDConfig", "M3ID"]


class M3IDConfig(GenericMitigatorConfig):
    """Hyper-parameters for M3ID.

    Attributes:
        alpha: Plausibility threshold (a probability in ``(0, 1]``). The
            contrast is applied only when the conditional branch's top-1
            probability is below ``alpha``. Paper default ``0.3`` (scan ``0.2``,
            ``0.3``, ``0.5``).
        forgetting_rate: Forgetting factor ``lambda`` (``gamma_t = exp(-lambda*t)``).
            Paper default ``0.02`` (scan ``0.001``, ``0.02``, ``0.03``).
    """

    alpha: float = 0.3
    forgetting_rate: float = 0.02

    def validate(self) -> None:
        super().validate()
        if not (0.0 < self.alpha <= 1.0):
            raise MitigatorConfigError("alpha must be in (0, 1]")
        if self.forgetting_rate < 0:
            raise MitigatorConfigError("forgetting_rate must be >= 0")


@register_mitigator("m3id")
class M3ID(GenericMitigator):
    """Multi-Modal Mutual Information Decoding.

    At every decoding step it runs the model with the image and without it, and
    adds ``weight_t * (l_c - l_u)`` to the conditional logits when the model is
    uncertain. Following the paper's one-based indexing, the first token already
    receives weight ``exp(lambda)-1`` and the weight grows thereafter.
    """

    algorithm_name = "m3id"
    config_class = M3IDConfig

    # -- input preparation -----------------------------------------------------
    def _prepare_inputs(
        self, images: Any, prompt: str, cfg: M3IDConfig
    ) -> dict[str, Any]:
        inputs = super()._prepare_inputs(images, prompt, cfg)
        if cfg.forgetting_rate == 0:
            self._uncond_inputs = {}
            self._uncond_attention_mask = None
            self._uncond_past = None
            return inputs
        uncond_prompt = self._remove_image_placeholder(prompt)
        self._uncond_inputs = super()._prepare_inputs(None, uncond_prompt, cfg)
        self._uncond_attention_mask = self._uncond_inputs.get("attention_mask")
        self._uncond_past = None
        return inputs

    @staticmethod
    def _remove_image_placeholder(prompt: str | Sequence[str]) -> str | list[str]:
        """Drop the ``<image>`` placeholder (plus trailing whitespace) from text."""
        if isinstance(prompt, str):
            return re.sub(r"<image>\s*", "", prompt, count=1)
        return [re.sub(r"<image>\s*", "", item, count=1) for item in prompt]

    @staticmethod
    def _step_weight(step: int, forgetting_rate: float) -> float:
        """Return ``exp(lambda * t) - 1`` with the paper's one-based ``t``.

        The backend passes a zero-based ``step``, whereas Algorithm 1 initializes
        ``t = 1`` for the first predicted token. ``expm1`` is more accurate for
        the small forgetting rates normally used by M3ID.
        """
        exponent = forgetting_rate * (step + 1)
        try:
            weight = math.expm1(exponent)
        except OverflowError as exc:
            raise MitigatorConfigError(
                "forgetting_rate is too large for the requested generation step"
            ) from exc
        if not math.isfinite(weight):
            raise MitigatorConfigError(
                "forgetting_rate is too large for the requested generation step"
            )
        return weight

    # -- intervention -----------------------------------------------------------
    def _step_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        inputs: dict[str, Any],
        past_key_values: Any,
        step: int,
        cfg: M3IDConfig,
    ) -> tuple[torch.Tensor, Any]:
        logits_c, past = self._forward(
            input_ids, attention_mask, inputs, past_key_values
        )

        weight = self._step_weight(step, cfg.forgetting_rate)
        if weight == 0.0:
            return logits_c, past

        # Unconditional (no-image) branch: full prompt on step 0, else same token.
        if self._uncond_past is None:
            uncond_ids = self._uncond_inputs["input_ids"]
        else:
            uncond_ids = input_ids
        logits_u, self._uncond_past = self._forward(
            uncond_ids,
            self._uncond_attention_mask,
            self._uncond_inputs,
            self._uncond_past,
        )
        self._grow_uncond_mask()

        # Gate: apply contrast only where the conditional top-1 probability < alpha.
        l_c = torch.log_softmax(logits_c, dim=-1)
        gate = (l_c.max(dim=-1, keepdim=True).values < math.log(cfg.alpha)).to(
            logits_c.dtype
        )

        logits = logits_c + gate * weight * (logits_c - logits_u)
        return logits, past

    def _grow_uncond_mask(self) -> None:
        if self._uncond_attention_mask is None:
            return
        ones = torch.ones(
            (self._uncond_attention_mask.shape[0], 1),
            dtype=self._uncond_attention_mask.dtype,
            device=self._uncond_attention_mask.device,
        )
        self._uncond_attention_mask = torch.cat(
            [self._uncond_attention_mask, ones], dim=-1
        )

    # -- beam search ---------------------------------------------------------
    def _expand_inputs_for_beams(
        self, inputs: dict[str, Any], num_beams: int
    ) -> dict[str, Any]:
        expanded = super()._expand_inputs_for_beams(inputs, num_beams)
        self._uncond_inputs = super()._expand_inputs_for_beams(
            self._uncond_inputs, num_beams
        )
        if self._uncond_attention_mask is not None:
            self._uncond_attention_mask = self._uncond_attention_mask.repeat_interleave(
                num_beams, dim=0
            )
        return expanded

    def _reorder_aux_cache(self, beam_idx: torch.Tensor) -> None:
        self._uncond_past = self._reorder_cache(self._uncond_past, beam_idx)
        if self._uncond_attention_mask is not None:
            self._uncond_attention_mask = self._uncond_attention_mask.index_select(
                0, beam_idx
            )
