"""AGLA — Assembly of Global and Local Attention (An et al., 2025).

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

from mitigv.backends.generic import GenericMitigator, GenericMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator

__all__ = ["AGLAConfig", "AGLA"]


class AGLAConfig(GenericMitigatorConfig):
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
class AGLA(GenericMitigator):
    """Assembly of Global and Local Attention.

    At each step it runs the model on the original image and on a saliency-cropped
    region, then adds ``alpha`` times the local logits to the global logits.
    """

    algorithm_name = "agla"
    config_class = AGLAConfig

    # -- input preparation -----------------------------------------------------
    def _prepare_inputs(
        self, images: Any, prompt: str, cfg: AGLAConfig
    ) -> dict[str, Any]:
        inputs = super()._prepare_inputs(images, prompt, cfg)
        if cfg.alpha == 0:
            self._local_inputs = dict(inputs)
            self._local_past = None
            return inputs
        spans = self._image_token_spans(inputs)
        if bool(spans[:, 0].eq(spans[:, 1]).any()):
            raise ValueError("every AGLA batch row must contain an image token")
        saliency = self._compute_saliency_spans(inputs, spans)
        local_images = self._crop_images(images, saliency, cfg)
        self._local_inputs = super()._prepare_inputs(local_images, prompt, cfg)
        self._local_past = None
        return inputs

    # -- saliency --------------------------------------------------------------
    def _image_token_span(self, inputs: dict[str, Any]) -> tuple[int, int]:
        spans = self._image_token_spans(inputs)
        if spans.shape[0] > 1 and not bool(spans.eq(spans[0]).all()):
            raise RuntimeError("batch rows have different image-token spans")
        return int(spans[0, 0]), int(spans[0, 1])

    def _image_token_spans(self, inputs: dict[str, Any]) -> torch.Tensor:
        config = getattr(self.model, "config", None)
        token = getattr(config, "image_token_index", None)
        input_ids = inputs["input_ids"]
        spans = torch.zeros(
            (input_ids.shape[0], 2), dtype=torch.long, device=input_ids.device
        )
        if token is None:
            return spans
        image_seq_length = getattr(config, "image_seq_length", None)
        if not image_seq_length:
            vision = getattr(config, "vision_config", None)
            image_size = getattr(vision, "image_size", None)
            patch_size = getattr(vision, "patch_size", None)
            if image_size and patch_size:
                image_seq_length = (int(image_size) // int(patch_size)) ** 2
        for row in range(input_ids.shape[0]):
            cols = (input_ids[row] == token).nonzero(as_tuple=False).flatten()
            if cols.numel() == 0:
                continue
            if cols.numel() > 1 and not bool(cols.diff().eq(1).all()):
                raise ValueError("AGLA does not support disjoint image-token spans")
            start = int(cols.min())
            end = int(cols.max()) + 1
            # Older processors retain one placeholder which the model expands.
            if cols.numel() == 1 and image_seq_length:
                end = start + int(image_seq_length)
            spans[row] = torch.tensor((start, end), device=input_ids.device)
        return spans

    def _compute_saliency(
        self, inputs: dict[str, Any], start: int, end: int
    ) -> torch.Tensor:
        """Attention-based saliency over image tokens, shape ``(B, n_image_tokens)``."""
        spans = torch.tensor(
            [[start, end]] * inputs["input_ids"].shape[0],
            dtype=torch.long,
            device=inputs["input_ids"].device,
        )
        return self._compute_saliency_spans(inputs, spans)

    def _compute_saliency_spans(
        self, inputs: dict[str, Any], spans: torch.Tensor
    ) -> torch.Tensor:
        """Compute saliency for each row's own image-token interval."""
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
        lengths = spans[:, 1] - spans[:, 0]
        if not bool(lengths.eq(lengths[0]).all()):
            raise RuntimeError("AGLA batch rows must have equal image-token counts")
        saliency = torch.stack(
            [
                attn[row, -1, int(start) : int(end)]
                for row, (start, end) in enumerate(spans.tolist())
            ]
        )
        return saliency / (saliency.sum(dim=-1, keepdim=True) + 1e-8)

    def _crop_images(self, images: Any, saliency: torch.Tensor, cfg: AGLAConfig) -> Any:
        single = not isinstance(images, (list, tuple))
        image_list = [images] if single else list(images)
        if len(image_list) != saliency.shape[0]:
            raise ValueError(
                "number of images must match the processed prompt batch size"
            )
        crops = [
            self._crop_one(img, saliency[b], cfg) for b, img in enumerate(image_list)
        ]
        return crops[0] if single else crops

    def _crop_one(self, image: Any, saliency_1d: torch.Tensor, cfg: AGLAConfig) -> Any:
        grid = int(round(saliency_1d.numel() ** 0.5))
        if grid * grid != saliency_1d.numel():
            raise RuntimeError(
                "AGLA requires a square spatial image-token grid; got "
                f"{saliency_1d.numel()} tokens"
            )
        if (
            not hasattr(image, "width")
            or not hasattr(image, "height")
            or not hasattr(image, "crop")
        ):
            raise TypeError(
                "AGLA cropping requires PIL-like images with width/height/crop"
            )
        sal = saliency_1d.view(grid, grid)
        idx = torch.arange(grid, dtype=saliency_1d.dtype, device=saliency_1d.device)
        ys, xs = torch.meshgrid(idx, idx, indexing="ij")
        total = sal.sum() + 1e-8
        # Grid coordinates denote patch centers, not their top-left corners.
        cy = float((sal * (ys + 0.5)).sum() / total) / grid
        cx = float((sal * (xs + 0.5)).sum() / total) / grid
        w, h = image.width, image.height
        side = max(1, int(round(cfg.crop_ratio * min(w, h))))
        cx_px, cy_px = cx * w, cy * h
        left = min(max(0, int(round(cx_px - side / 2))), w - side)
        top = min(max(0, int(round(cy_px - side / 2))), h - side)
        right = left + side
        bottom = top + side
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
        logits_g, past = self._forward(
            input_ids, attention_mask, inputs, past_key_values
        )
        if cfg.alpha == 0:
            return logits_g, past
        logits_l, self._local_past = self._forward(
            input_ids, attention_mask, self._local_inputs, self._local_past
        )
        return logits_g + cfg.alpha * logits_l, past

    # -- beam search ---------------------------------------------------------
    def _expand_inputs_for_beams(
        self, inputs: dict[str, Any], num_beams: int
    ) -> dict[str, Any]:
        expanded = super()._expand_inputs_for_beams(inputs, num_beams)
        self._local_inputs = super()._expand_inputs_for_beams(
            self._local_inputs, num_beams
        )
        return expanded

    def _reorder_aux_cache(self, beam_idx: torch.Tensor) -> None:
        self._local_past = self._reorder_cache(self._local_past, beam_idx)
