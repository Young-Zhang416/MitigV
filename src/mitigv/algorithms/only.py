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

from mitigv.backends.hf import HFMitigator, HFMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator

__all__ = ["ONLYConfig", "ONLY"]


class ONLYConfig(HFMitigatorConfig):
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
class ONLY(HFMitigator):
    """One-Layer Intervention (TVER head selection + adaptive fusion).

    The first forward records the selected layer's attention (used to compute
    TVER); the second forward re-runs the model with low-TVER heads deactivated
    at that layer. Both branches keep their own KV cache.
    """

    algorithm_name = "only"
    config_class = ONLYConfig

    # -- input preparation -----------------------------------------------------
    def _prepare_inputs(self, images: Any, prompt: str, cfg: ONLYConfig) -> dict[str, Any]:
        inputs = super()._prepare_inputs(images, prompt, cfg)
        self._img_start, self._img_end = self._image_token_span(inputs)
        self._mask_active = False
        self._masked_past = None
        self._patched_layer = None
        self._attn_original = None
        return inputs

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

    # -- attention patch -------------------------------------------------------
    def _on_generate_start(self, cfg: ONLYConfig) -> None:
        self._force_eager_attention()
        layers = self._language_model_layers()
        if cfg.layer >= len(layers):
            raise MitigatorConfigError(f"layer {cfg.layer} out of range for {len(layers)} layers")
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
            query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, attn_module.layer_idx, cache_kwargs
                )

            key_states = repeat_kv(key_states, attn_module.num_key_value_groups)
            value_states = repeat_kv(value_states, attn_module.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attn_module.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            if pai._mask_active and pai._selected_mask is not None:
                attn_weights = attn_weights.clone()
                attn_weights[:, pai._selected_mask] = 0.0

            pai._last_attn = attn_weights  # (B, H, q, kv) post-softmax

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)
            return attn_output, attn_weights

        return forward

    # -- TVER ------------------------------------------------------------------
    def _compute_selected_mask(self, attention: torch.Tensor, cfg: ONLYConfig) -> torch.Tensor:
        """Return a boolean mask over heads: True for *deactivated* heads."""
        if self._img_end <= self._img_start:
            return torch.zeros(attention.shape[1], dtype=torch.bool, device=attention.device)
        a_v = attention[:, :, -1, self._img_start : self._img_end]  # (B, H, n_v)
        text_mask = torch.ones(attention.shape[-1], dtype=torch.bool, device=attention.device)
        text_mask[self._img_start : self._img_end] = False
        a_t = attention[:, :, -1, text_mask]  # (B, H, n_t)

        def entropy(a: torch.Tensor) -> torch.Tensor:
            a = a + 1e-12
            return -(a * a.log()).sum(dim=-1)

        tver = entropy(a_t) / entropy(a_v)  # (B, H)
        mean = tver.mean(dim=-1, keepdim=True)
        return (tver < mean)[0]  # (H,) boolean mask of deactivated heads (row 0)

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
        logits_f, past = self._forward(input_ids, attention_mask, inputs, past_key_values)
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
    def _expand_inputs_for_beams(self, inputs: dict[str, Any], num_beams: int) -> dict[str, Any]:
        return super()._expand_inputs_for_beams(inputs, num_beams)

    def _reorder_aux_cache(self, beam_idx: torch.Tensor) -> None:
        self._masked_past = self._reorder_cache(self._masked_past, beam_idx)
