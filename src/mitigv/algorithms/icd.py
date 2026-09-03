"""ICD — Instruction Contrastive Decoding (Wang et al., ACL Findings 2024).

Training-free hallucination mitigation that contrasts the next-token logits
from the *standard* instruction against those from a *disturbed* instruction
(a role prefix, e.g. "You are a confused object detector ...", prepended to the
question)::

    logits = logits(std) - lam * logits(disturbed)

optionally followed by an adaptive plausibility constraint (``alpha``) that masks
tokens unlikely under the *standard* distribution.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from mitigv.backends.generic import GenericMitigator, GenericMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator

__all__ = ["ICDConfig", "ICD"]

#: Default disturbance prefix (the paper's negative prefix N2, also used in Fig. 1).
DEFAULT_DISTURBANCE_PREFIX = "You are a confused object detector to provide a fuzzy overview or impression of the image."


class ICDConfig(GenericMitigatorConfig):
    """Hyper-parameters for ICD.

    Attributes:
        lam: Contrast strength λ — penalty on the disturbed distribution
            (analogous to VCD's ``alpha``). ``lam=0`` degenerates to plain decoding.
        alpha: Adaptive-plausibility truncation threshold α (analogous to VCD's
            ``beta``). ``alpha=0`` disables the constraint.
        disturbance_prefix: Role prefix prepended to the instruction to induce
            the disturbance.
    """

    lam: float = 1.0
    alpha: float = 0.1
    disturbance_prefix: str = DEFAULT_DISTURBANCE_PREFIX

    def validate(self) -> None:
        super().validate()
        if self.lam < 0:
            raise MitigatorConfigError("lam must be >= 0")
        if not (0.0 <= self.alpha <= 1.0):
            raise MitigatorConfigError("alpha must be in [0, 1]")
        if not self.disturbance_prefix.strip():
            raise MitigatorConfigError("disturbance_prefix must be non-empty")


@register_mitigator("icd")
class ICD(GenericMitigator):
    """Instruction Contrastive Decoding.

    At every decoding step it runs the model twice — once with the standard
    instruction and once with a disturbed instruction (same image) — and
    subtracts ``lam`` times the disturbed logits from the standard logits.
    """

    algorithm_name = "icd"
    config_class = ICDConfig

    # -- input preparation -----------------------------------------------------
    def _prepare_inputs(
        self, images: Any, prompt: str, cfg: ICDConfig
    ) -> dict[str, Any]:
        inputs = super()._prepare_inputs(images, prompt, cfg)
        if cfg.lam == 0:
            self._disturbed_inputs = {}
            self._disturbed_attention_mask = None
            self._disturbed_past = None
            return inputs
        disturbed_prompt = self._disturb_prompt(prompt, cfg)
        self._disturbed_inputs = super()._prepare_inputs(images, disturbed_prompt, cfg)
        self._disturbed_attention_mask = self._disturbed_inputs.get("attention_mask")
        self._disturbed_past = None
        return inputs

    def _disturb_prompt(
        self, prompt: str | Sequence[str], cfg: ICDConfig
    ) -> str | list[str]:
        """Prepend the disturbance prefix to the instruction (after ``<image>``)."""
        if isinstance(prompt, str):
            return self._disturb_prompt_with_prefix(prompt, cfg.disturbance_prefix)
        return [
            self._disturb_prompt_with_prefix(item, cfg.disturbance_prefix)
            for item in prompt
        ]

    @staticmethod
    def _disturb_prompt_with_prefix(prompt: str, prefix: str) -> str:
        if "<image>\n" in prompt:
            return prompt.replace("<image>\n", f"<image>\n{prefix} ", 1)
        if "<image>" in prompt:
            return prompt.replace("<image>", f"<image> {prefix} ", 1)
        return f"{prefix} {prompt}"

    # -- intervention -----------------------------------------------------------
    def _step_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        inputs: dict[str, Any],
        past_key_values: Any,
        step: int,
        cfg: ICDConfig,
    ) -> tuple[torch.Tensor, Any]:
        logits_std, past = self._forward(
            input_ids, attention_mask, inputs, past_key_values
        )

        # Disturbed branch: full disturbed prompt on its first step, otherwise the
        # same generated token(s) as the standard branch.
        if cfg.lam == 0:
            logits = logits_std
        else:
            if self._disturbed_past is None:
                dst_ids = self._disturbed_inputs["input_ids"]
            else:
                dst_ids = input_ids
            logits_dst, self._disturbed_past = self._forward(
                dst_ids,
                self._disturbed_attention_mask,
                self._disturbed_inputs,
                self._disturbed_past,
            )
            self._grow_disturbed_mask()
            logits = logits_std - cfg.lam * logits_dst
        if cfg.alpha > 0:
            logits = self._adaptive_plausibility(logits, logits_std, cfg.alpha)
        return logits, past

    def _grow_disturbed_mask(self) -> None:
        if self._disturbed_attention_mask is None:
            return
        ones = torch.ones(
            (self._disturbed_attention_mask.shape[0], 1),
            dtype=self._disturbed_attention_mask.dtype,
            device=self._disturbed_attention_mask.device,
        )
        self._disturbed_attention_mask = torch.cat(
            [self._disturbed_attention_mask, ones], dim=-1
        )

    @staticmethod
    def _adaptive_plausibility(
        logits: torch.Tensor, logits_standard: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Mask tokens whose *standard*-branch probability is below ``alpha * max``."""
        cutoff = (
            torch.log(torch.tensor(alpha, device=logits.device, dtype=logits.dtype))
            + logits_standard.max(dim=-1, keepdim=True).values
        )
        return logits.masked_fill(logits_standard < cutoff, float("-inf"))

    # -- beam search ---------------------------------------------------------
    def _expand_inputs_for_beams(
        self, inputs: dict[str, Any], num_beams: int
    ) -> dict[str, Any]:
        expanded = super()._expand_inputs_for_beams(inputs, num_beams)
        self._disturbed_inputs = super()._expand_inputs_for_beams(
            self._disturbed_inputs, num_beams
        )
        if self._disturbed_attention_mask is not None:
            self._disturbed_attention_mask = (
                self._disturbed_attention_mask.repeat_interleave(num_beams, dim=0)
            )
        return expanded

    def _reorder_aux_cache(self, beam_idx: torch.Tensor) -> None:
        self._disturbed_past = self._reorder_cache(self._disturbed_past, beam_idx)
        if self._disturbed_attention_mask is not None:
            self._disturbed_attention_mask = (
                self._disturbed_attention_mask.index_select(0, beam_idx)
            )
