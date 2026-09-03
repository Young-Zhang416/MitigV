"""VISTA — Visual Information Steering (Li & Shi, ICML 2025).

Training-free hallucination mitigation that counteracts the gradual dilution of
visual information in the residual stream. At the start of generation it
extracts a per-layer **Visual Steering Vector (VSV)** as the difference between
the residual streams of the *with-image* and *no-image* prompts, then injects it
back into the residual stream at every layer during decoding::

    V_steer^l = F(X_p)^l[last] - F(X_n)^l[last]      # X_p: with image, X_n: text-only
    h_t^l     = h_t^l + steer_strength * V_steer^l   # l in [0, L)

``F`` forwards a token sequence through the LVLM and returns the residual stream
of the last token per layer. ``steer_strength=0`` degenerates to plain decoding.

(Only the VSV component is implemented here; the paper's second component, SLA —
Self-Logits Augmentation — is logits-level and left as future work.)
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import torch

from mitigv.backends.generic import GenericMitigator, GenericMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator

__all__ = ["VISTAConfig", "VISTA"]


class VISTAConfig(GenericMitigatorConfig):
    """Hyper-parameters for VISTA.

    Attributes:
        steer_strength: Injection strength ``lambda`` of the VSV (paper default
            ``0.01``). ``0`` disables the steering (plain decoding).
    """

    steer_strength: float = 0.01

    def validate(self) -> None:
        super().validate()
        if self.steer_strength < 0:
            raise MitigatorConfigError("steer_strength must be >= 0")


@register_mitigator("vista")
class VISTA(GenericMitigator):
    """Visual Information Steering (VSV component).

    Computes the VSV once per generation from two forward passes (with / without
    image), then injects it into every decoder layer's residual stream during
    decoding. Decoding itself is a single forward per step, so the runtime
    overhead is a one-time 2x forward at the start plus a cheap add per layer.
    """

    algorithm_name = "vista"
    config_class = VISTAConfig

    # -- input preparation -----------------------------------------------------
    def _prepare_inputs(
        self, images: Any, prompt: str, cfg: VISTAConfig
    ) -> dict[str, Any]:
        inputs = super()._prepare_inputs(images, prompt, cfg)
        if cfg.steer_strength == 0:
            self._uncond_inputs = {}
            self._steer_vectors = []
            return inputs
        uncond_prompt = self._remove_image_placeholder(prompt)
        self._uncond_inputs = super()._prepare_inputs(None, uncond_prompt, cfg)
        self._steer_vectors = self._compute_steering_vectors(inputs)
        return inputs

    @staticmethod
    def _remove_image_placeholder(prompt: str | Sequence[str]) -> str | list[str]:
        """Drop the ``<image>`` placeholder (plus trailing whitespace) from text."""
        if isinstance(prompt, str):
            return re.sub(r"<image>\s*", "", prompt, count=1)
        return [re.sub(r"<image>\s*", "", item, count=1) for item in prompt]

    # -- steering vector --------------------------------------------------------
    def _forward_hidden(self, inputs: dict[str, Any]) -> Any:
        """Run one forward pass with ``output_hidden_states=True``."""
        kwargs = {
            key: value
            for key, value in inputs.items()
            if key in ("input_ids", "attention_mask", "pixel_values", "images")
        }
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        return self.model(**kwargs)

    def _compute_steering_vectors(self, inputs: dict[str, Any]) -> list[torch.Tensor]:
        """Compute per-layer VSV ``V_p - V_n`` from the last token's residual."""
        was_training = bool(getattr(self.model, "training", False))
        try:
            with torch.no_grad():
                if hasattr(self.model, "eval"):
                    self.model.eval()
                out_p = self._forward_hidden(inputs)
                out_n = self._forward_hidden(self._uncond_inputs)
                hs_p = out_p.hidden_states
                hs_n = out_n.hidden_states
                if len(hs_p) != len(hs_n):
                    raise RuntimeError(
                        "conditional and text-only branches returned different "
                        "hidden-state depths"
                    )
                vectors = [
                    self._last_hidden(hs_p[layer], inputs.get("attention_mask"))
                    - self._last_hidden(
                        hs_n[layer], self._uncond_inputs.get("attention_mask")
                    )
                    for layer in range(1, len(hs_p))
                ]
        finally:
            if was_training:
                self.model.train()
        return vectors

    @staticmethod
    def _last_hidden(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Select the right-most non-padding hidden state for every batch row."""
        if (
            attention_mask is None
            or attention_mask.ndim != 2
            or hidden_states.shape[1] != attention_mask.shape[1]
        ):
            return hidden_states[:, -1, :]
        valid = attention_mask.to(dtype=torch.bool)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("attention_mask contains an empty input sequence")
        positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
        last_valid = positions.masked_fill(~valid, -1).max(dim=1).values
        batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch, last_valid]

    # -- residual-stream injection ----------------------------------------------
    def _on_generate_start(self, cfg: VISTAConfig) -> None:
        self._steer_hooks: list[Any] = []
        if cfg.steer_strength == 0.0 or not self._steer_vectors:
            return
        vectors = self._steer_vectors
        if cfg.num_beams > 1:
            vectors = [v.repeat_interleave(cfg.num_beams, dim=0) for v in vectors]
        layers = self._language_model_layers()
        strength = cfg.steer_strength
        for index, vector in enumerate(vectors):
            if index >= len(layers):
                break

            def make_hook(vec: torch.Tensor):
                def hook(module: Any, args: Any, output: Any) -> Any:
                    return self._add_to_layer_output(
                        output, strength * vec.unsqueeze(1)
                    )

                return hook

            self._steer_hooks.append(
                layers[index].register_forward_hook(make_hook(vector))
            )

    def _on_generate_end(self) -> None:
        for handle in getattr(self, "_steer_hooks", []):
            handle.remove()
        self._steer_hooks = []
