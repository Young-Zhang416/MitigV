"""AGLA — Assembly of Global and Local Attention (Zhou et al., 2024).

Training-free hallucination mitigation that fuses the *generative global* view
of the original image with the *discriminative local* view of a saliency-cropped
region. A GradCAM-like saliency map is estimated from the LVLM's own attention
to image tokens (no gradients, no matching model), the most salient region is
cropped, and the two views' next-token logits are fused additively::

    logits = logit(global) + alpha * logit(local)

The local view captures prompt-relevant details that the global view neglects,
while the global view keeps the full generative context.
"""

from __future__ import annotations

from typing import Any

import torch

from mitigv.backends.hf import HFMitigator, HFMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator

__all__ = ["AGLAConfig", "AGLA"]


class AGLAConfig(HFMitigatorConfig):
    """Hyper-parameters for AGLA.

    Attributes:
        alpha: Weight of the local (cropped) view's logits in the fusion
            (``alpha=0`` degenerates to plain global decoding).
        crop_ratio: Side length of the cropped region as a fraction of the
            image's shorter edge (e.g. ``0.5`` crops a half-size square).
    """

    alpha: float = 1.0
    crop_ratio: float = 0.5

    def validate(self) -> None:
        super().validate()
        if self.alpha < 0:
            raise MitigatorConfigError("alpha must be >= 0")
        if not (0.0 < self.crop_ratio <= 1.0):
            raise MitigatorConfigError("crop_ratio must be in (0, 1]")


@register_mitigator("agla")
class AGLA(HFMitigator):
    """Assembly of Global and Local Attention.

    At each step it runs the model on the original image and on a saliency-cropped
    region, then adds ``alpha`` times the local logits to the global logits.
    """

    algorithm_name = "agla"
    config_class = AGLAConfig

    # -- input preparation -----------------------------------------------------
    def _prepare_inputs(self, images: Any, prompt: str, cfg: AGLAConfig) -> dict[str, Any]:
        inputs = super()._prepare_inputs(images, prompt, cfg)
        start, end = self._image_token_span(inputs)
        if start >= end:
            self._local_inputs = dict(inputs)
        else:
            saliency = self._compute_saliency(inputs, start, end)
            local_images = self._crop_images(images, saliency, cfg)
            self._local_inputs = super()._prepare_inputs(local_images, prompt, cfg)
        self._local_past = None
        return inputs

    # -- saliency --------------------------------------------------------------
    def _image_token_span(self, inputs: dict[str, Any]) -> tuple[int, int]:
        config = getattr(self.model, "config", None)
        token = getattr(config, "image_token_index", None)
        if token is None:
            return 0, 0
        positions = (inputs["input_ids"] == token).nonzero(as_tuple=False)
        if positions.numel() == 0:
            return 0, 0
        cols = positions[:, 1]
        return int(cols.min().item()), int(cols.max().item()) + 1

    def _compute_saliency(self, inputs: dict[str, Any], start: int, end: int) -> torch.Tensor:
        """Attention-based saliency over image tokens, shape ``(B, n_image_tokens)``."""
        kwargs = {
            key: value
            for key, value in inputs.items()
            if key in ("input_ids", "attention_mask", "pixel_values", "images")
        }
        kwargs["output_attentions"] = True
        kwargs["return_dict"] = True
        self._force_eager_attention()
        try:
            with torch.no_grad():
                out = self.model(**kwargs)
        finally:
            self._restore_attention_implementation()
        attentions = torch.stack(out.attentions)  # (L, B, H, seq, seq)
        attn = attentions.mean(dim=0).mean(dim=1)  # (B, seq, seq)
        saliency = attn[:, -1, start:end]  # (B, n_tokens)
        return saliency / (saliency.sum(dim=-1, keepdim=True) + 1e-8)

    def _crop_images(self, images: Any, saliency: torch.Tensor, cfg: AGLAConfig) -> Any:
        single = not isinstance(images, (list, tuple))
        image_list = [images] if single else list(images)
        crops = [
            self._crop_one(img, saliency[b], cfg)
            for b, img in enumerate(image_list)
        ]
        return crops[0] if single else crops

    def _crop_one(self, image: Any, saliency_1d: torch.Tensor, cfg: AGLAConfig) -> Any:
        grid = int(round(saliency_1d.numel() ** 0.5))
        sal = saliency_1d.view(grid, grid)
        idx = torch.arange(grid, dtype=saliency_1d.dtype, device=saliency_1d.device)
        ys, xs = torch.meshgrid(idx, idx, indexing="ij")
        total = sal.sum() + 1e-8
        cy = float((sal * ys).sum() / total) / grid
        cx = float((sal * xs).sum() / total) / grid
        w, h = image.width, image.height
        side = cfg.crop_ratio * min(w, h)
        cx_px, cy_px = cx * w, cy * h
        left = max(0, int(cx_px - side / 2))
        top = max(0, int(cy_px - side / 2))
        right = min(w, int(cx_px + side / 2))
        bottom = min(h, int(cy_px + side / 2))
        return image.crop((left, top, right, bottom))

    # -- intervention -----------------------------------------------------------
    def _step_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        inputs: dict[str, Any],
        past_key_values: Any,
        step: int,
        cfg: AGLAConfig,
    ) -> tuple[torch.Tensor, Any]:
        logits_g, past = self._forward(input_ids, attention_mask, inputs, past_key_values)
        logits_l, self._local_past = self._forward(
            input_ids, attention_mask, self._local_inputs, self._local_past
        )
        return logits_g + cfg.alpha * logits_l, past

    # -- beam search ---------------------------------------------------------
    def _expand_inputs_for_beams(self, inputs: dict[str, Any], num_beams: int) -> dict[str, Any]:
        expanded = super()._expand_inputs_for_beams(inputs, num_beams)
        self._local_inputs = super()._expand_inputs_for_beams(self._local_inputs, num_beams)
        return expanded

    def _reorder_aux_cache(self, beam_idx: torch.Tensor) -> None:
        self._local_past = self._reorder_cache(self._local_past, beam_idx)
