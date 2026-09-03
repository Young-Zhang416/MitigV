"""ONLY — One-Layer Intervention Sufficiently Mitigates Hallucinations (Wan et al., ICCV 2025).

Training-free decoding that intervenes at a *single* decoder layer. At that
layer, attention heads are ranked by their **Text-to-Visual Entropy Ratio
(TVER)** ``Entropy(a^text) / Entropy(a^visual)``; heads below the layer average
are deactivated (their attention weights are zeroed) to produce a *textually
enhanced* logits distribution ``f~``. The original and enhanced logits are then
combined adaptively, guided by the Manhattan distance ``d`` between the two
probability distributions::

    d     = sum_y |p(y) - p~(y)|
    final = f + alpha1 * f~            if d < gamma   (collaborative)
    final = (1 + alpha2) * f - alpha2 * f~   otherwise (contrastive)

``alpha1=3``, ``alpha2=1``, ``gamma=0.2`` (paper defaults).
"""

from __future__ import annotations

from typing import Any

import torch

from mitigv.backends.generic import GenericMitigator, GenericMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator

__all__ = ["ONLYConfig", "ONLY"]


class ONLYConfig(GenericMitigatorConfig):
    """Hyper-parameters for ONLY.

    Attributes:
        layer: Decoder layer index for the intervention (paper: the initial layer).
        alpha1: Collaborative fusion weight (paper default 3).
        alpha2: Contrastive fusion weight (paper default 1).
        gamma: Manhattan-distance threshold between collaborative and contrastive
            (paper default 0.2).
    """

    layer: int = 0
    alpha1: float = 3.0
    alpha2: float = 1.0
    gamma: float = 0.2

    def validate(self) -> None:
        super().validate()
        if self.layer < 0:
            raise MitigatorConfigError("layer must be >= 0")
        if self.alpha1 < 0 or self.alpha2 < 0:
            raise MitigatorConfigError("alpha1/alpha2 must be >= 0")
        if self.gamma < 0:
            raise MitigatorConfigError("gamma must be >= 0")


@register_mitigator("only")
class ONLY(GenericMitigator):
    """One-Layer Intervention (TVER head selection + adaptive fusion).

    The first forward records the selected layer's attention (used to compute
    TVER); the second forward re-runs the model with low-TVER heads deactivated
    at that layer. Both branches keep their own KV cache.
    """

    algorithm_name = "only"
    config_class = ONLYConfig

    # -- input preparation -----------------------------------------------------
    def _prepare_inputs(
        self, images: Any, prompt: str, cfg: ONLYConfig
    ) -> dict[str, Any]:
        inputs = super()._prepare_inputs(images, prompt, cfg)
        self._img_spans = self._image_token_spans(inputs)
        if bool(self._img_spans[:, 0].eq(self._img_spans[:, 1]).any()):
            raise ValueError("ONLY could not locate image tokens in every batch row")
        self._mask_active = False
        self._masked_past = None
        self._patched_layer = None
        self._attn_original = None
        return inputs

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
                raise ValueError("ONLY does not support disjoint image-token spans")
            start = int(cols.min())
            end = int(cols.max()) + 1
            if cols.numel() == 1 and image_seq_length:
                end = start + int(image_seq_length)
            spans[row] = torch.tensor((start, end), device=input_ids.device)
        return spans

    # -- attention patch -------------------------------------------------------
    def _on_generate_start(self, cfg: ONLYConfig) -> None:
        self._force_eager_attention()
        layers = self._language_model_layers()
        if cfg.layer >= len(layers):
            raise MitigatorConfigError(
                f"layer {cfg.layer} out of range for {len(layers)} layers"
            )
        attn = layers[cfg.layer].self_attn
        self._patched_layer = attn
        self._attn_original = attn.forward
        attn.forward = self._make_patched_forward(attn)

    def _on_generate_end(self) -> None:
        if self._patched_layer is not None and self._attn_original is not None:
            self._patched_layer.forward = self._attn_original
        self._patched_layer = None
        self._attn_original = None
        self._restore_attention_implementation()

    def _make_patched_forward(self, attn_module: Any):
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        pai = self

        def forward(
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            attention_mask: torch.Tensor | None = None,
            past_key_values: Any = None,
            cache_position: torch.LongTensor | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn_module.head_dim)
            query_states = (
                attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )
            key_states = (
                attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )
            value_states = (
                attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            if past_key_values is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_values.update(
                    key_states, value_states, attn_module.layer_idx, cache_kwargs
                )

            key_states = repeat_kv(key_states, attn_module.num_key_value_groups)
            value_states = repeat_kv(value_states, attn_module.num_key_value_groups)

            attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3))
                * attn_module.scaling
            )
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            if pai._mask_active and pai._selected_mask is not None:
                selected = pai._selected_mask
                if selected.shape != attn_weights.shape[:2]:
                    raise RuntimeError(
                        "ONLY head mask does not match the current attention batch"
                    )
                attn_weights = attn_weights.masked_fill(selected[:, :, None, None], 0.0)

            pai._last_attn = attn_weights  # (B, H, q, kv) post-softmax

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)
            return attn_output, attn_weights

        return forward

    # -- TVER ------------------------------------------------------------------
    def _compute_selected_mask(
        self, attention: torch.Tensor, cfg: ONLYConfig
    ) -> torch.Tensor:
        """Return a boolean mask over heads: True for *deactivated* heads."""

        def entropy(a: torch.Tensor) -> torch.Tensor:
            # The paper computes entropy inside each modality, so restricted
            # attention mass must be normalized again. Suppress attention-sink
            # outliers as done by the reference implementation first.
            threshold = a.mean(dim=-1, keepdim=True) + a.std(
                dim=-1, keepdim=True, unbiased=False
            )
            filtered = torch.where(a > threshold, torch.zeros_like(a), a)
            probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

        spans = getattr(self, "_img_spans", None)
        if spans is None:
            # Backward-compatible path for subclasses/tests that set the old
            # scalar attributes directly.
            start = getattr(self, "_img_start", 0)
            end = getattr(self, "_img_end", 0)
            spans = torch.tensor(
                [[start, end]] * attention.shape[0], device=attention.device
            )
        if spans.shape[0] != attention.shape[0]:
            raise RuntimeError(
                "ONLY image-token spans do not match the attention batch"
            )

        selected: list[torch.Tensor] = []
        for row, (start_t, end_t) in enumerate(spans.tolist()):
            start = max(0, int(start_t))
            end = min(int(end_t), attention.shape[-1])
            if end <= start:
                selected.append(
                    torch.zeros(
                        attention.shape[1], dtype=torch.bool, device=attention.device
                    )
                )
                continue
            a_v = attention[row, :, -1, start:end]
            text_mask = torch.ones(
                attention.shape[-1], dtype=torch.bool, device=attention.device
            )
            text_mask[start:end] = False
            a_t = attention[row, :, -1, text_mask]
            entropy_text = entropy(a_t)
            entropy_image = entropy(a_v)
            tver = entropy_text / entropy_image.clamp_min(1e-12)
            tver = torch.nan_to_num(tver, nan=0.0, posinf=0.0, neginf=0.0)
            selected.append(tver < tver.mean())
        return torch.stack(selected)

    # -- intervention -----------------------------------------------------------
    def _step_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        inputs: dict[str, Any],
        past_key_values: Any,
        step: int,
        cfg: ONLYConfig,
    ) -> tuple[torch.Tensor, Any]:
        # 1) original forward (records attention at the patched layer)
        self._mask_active = False
        logits_f, past = self._forward(
            input_ids, attention_mask, inputs, past_key_values
        )
        attention = self._last_attn

        # 2) select low-TVER heads, then re-run with them deactivated
        self._selected_mask = self._compute_selected_mask(attention, cfg)
        self._mask_active = True
        if self._masked_past is None:
            masked_ids = inputs["input_ids"]
        else:
            masked_ids = input_ids
        logits_ftilde, self._masked_past = self._forward(
            masked_ids, attention_mask, inputs, self._masked_past
        )
        self._mask_active = False

        # 3) adaptive collaborative / contrastive fusion
        logits = self._fuse(logits_f, logits_ftilde, cfg.alpha1, cfg.alpha2, cfg.gamma)
        return logits, past

    @staticmethod
    def _fuse(
        logits_f: torch.Tensor,
        logits_ftilde: torch.Tensor,
        alpha1: float,
        alpha2: float,
        gamma: float,
    ) -> torch.Tensor:
        """Adaptively fuse original and textually-enhanced logits by Manhattan distance."""
        p = torch.softmax(logits_f, dim=-1)
        pt = torch.softmax(logits_ftilde, dim=-1)
        d = (p - pt).abs().sum(dim=-1, keepdim=True)
        collab = d < gamma
        return torch.where(
            collab,
            logits_f + alpha1 * logits_ftilde,
            (1.0 + alpha2) * logits_f - alpha2 * logits_ftilde,
        )

    # -- beam search ---------------------------------------------------------
    def _expand_inputs_for_beams(
        self, inputs: dict[str, Any], num_beams: int
    ) -> dict[str, Any]:
        expanded = super()._expand_inputs_for_beams(inputs, num_beams)
        self._img_spans = self._img_spans.repeat_interleave(num_beams, dim=0)
        return expanded

    def _reorder_aux_cache(self, beam_idx: torch.Tensor) -> None:
        self._masked_past = self._reorder_cache(self._masked_past, beam_idx)
