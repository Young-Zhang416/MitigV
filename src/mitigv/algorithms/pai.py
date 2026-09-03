"""PAI — Paying More Attention to Image (Liu et al., ECCV 2024).

Training-free hallucination mitigation with two coordinated interventions:

1. **Attention amplification** — while decoding, the attention weights from the
   newest query token to the image tokens are amplified in log space before the
   softmax::

       A[:, :, -1, img_start:img_end] = |A[...]| * alpha + A[...]

   applied to the language-model layers in ``[start_layer, end_layer)``.

2. **Classifier-free guidance against text** — the next-token logits are
   contrasted with a *text-only* branch (the same prompt with the image removed,
   so no ``pixel_values`` are fed)::

       logits = gamma * (logits(image) - logits(text)) + logits(text)

   optionally followed by an adaptive plausibility constraint (``beta``) that
   masks tokens unlikely under the *with-image* distribution.

The two parts are independent: ``alpha=0`` disables the attention intervention
and ``gamma=1`` disables the text guidance, each degenerating to plain decoding.
"""

from __future__ import annotations

from numbers import Integral
import re
from typing import Any, Sequence

import torch

from mitigv.backends.generic import GenericMitigator, GenericMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator

__all__ = ["PAIConfig", "PAI"]


class PAIConfig(GenericMitigatorConfig):
    """Hyper-parameters for PAI.

    Attributes:
        alpha: Attention-amplification scale on image tokens (``0`` disables
            the attention intervention).
        gamma: Classifier-free guidance scale against the text-only branch
            (``gamma=1`` disables the guidance; ``>1`` penalises the language
            prior / "text inertia").
        beta: Adaptive-plausibility threshold applied to the guided logits
            (``beta=0`` disables the constraint).
        start_layer: First language-model layer (inclusive) whose attention is
            amplified.
        end_layer: First language-model layer (exclusive) NOT amplified.
        num_image_tokens: Number of image tokens a single placeholder expands
            to, or ``None`` to infer from the vision config. Only consulted when
            ``input_ids`` holds a single image placeholder token.
    """

    alpha: float = 0.2
    gamma: float = 1.1
    beta: float = 0.1
    start_layer: int = 2
    end_layer: int = 32
    num_image_tokens: int | None = None

    def validate(self) -> None:
        super().validate()
        if self.alpha < 0:
            raise MitigatorConfigError("alpha must be >= 0")
        if self.gamma < 1.0:
            raise MitigatorConfigError("gamma must be >= 1 (gamma=1 disables guidance)")
        if not (0.0 <= self.beta <= 1.0):
            raise MitigatorConfigError("beta must be in [0, 1]")
        if self.start_layer < 0:
            raise MitigatorConfigError("start_layer must be >= 0")
        if self.end_layer <= self.start_layer:
            raise MitigatorConfigError("end_layer must be > start_layer")
        if self.num_image_tokens is not None and (
            not isinstance(self.num_image_tokens, Integral)
            or isinstance(self.num_image_tokens, bool)
        ):
            raise MitigatorConfigError("num_image_tokens must be an integer or None")
        if self.num_image_tokens is not None and self.num_image_tokens < 1:
            raise MitigatorConfigError("num_image_tokens must be >= 1 when set")


@register_mitigator("pai")
class PAI(GenericMitigator):
    """Paying More Attention to Image.

    Amplifies image-token attention during decoding (weight ``alpha``) and
    subtracts a text-only branch's logits (weight ``gamma``). Unlike VCD/ICD —
    which distort the *image* or the *instruction* — PAI's contrast branch drops
    the image entirely, targeting the language prior ("text inertia").
    """

    algorithm_name = "pai"
    config_class = PAIConfig

    # -- input preparation -----------------------------------------------------
    def _prepare_inputs(
        self, images: Any, prompt: str, cfg: PAIConfig
    ) -> dict[str, Any]:
        inputs = super()._prepare_inputs(images, prompt, cfg)
        if cfg.gamma == 1:
            self._uncond_inputs = {}
            self._uncond_attention_mask = None
        else:
            uncond_prompt = self._remove_image_placeholder(prompt)
            self._uncond_inputs = super()._prepare_inputs(None, uncond_prompt, cfg)
            self._uncond_attention_mask = self._uncond_inputs.get("attention_mask")
        self._uncond_past = None
        self._img_spans = self._locate_image_spans(inputs, cfg)
        if cfg.alpha > 0 and bool(
            self._img_spans[:, 0].eq(self._img_spans[:, 1]).any()
        ):
            raise ValueError(
                "PAI attention intervention could not locate image tokens in every batch row"
            )
        self._attn_active = False
        return inputs

    @staticmethod
    def _remove_image_placeholder(prompt: str | Sequence[str]) -> str | list[str]:
        """Drop the ``<image>`` placeholder (plus trailing whitespace) from text."""
        if isinstance(prompt, str):
            return re.sub(r"<image>\s*", "", prompt, count=1)
        return [re.sub(r"<image>\s*", "", item, count=1) for item in prompt]

    # -- image token span -------------------------------------------------------
    def _image_token_index(self) -> int | None:
        config = getattr(self.model, "config", None)
        return getattr(config, "image_token_index", None)

    def _locate_image_span(
        self, inputs: dict[str, Any], cfg: PAIConfig
    ) -> tuple[int, int]:
        """Return ``(start, end)`` of the image tokens in the attention sequence.

        Modern transformers expand the ``<image>`` placeholder into many explicit
        image tokens in ``input_ids``, so the span is simply the min/max position
        of the ``image_token_index``. When only a *single* placeholder token is
        present (older backends), it is widened by ``num_image_tokens`` because
        the model expands it at the embedding level.
        """
        spans = self._locate_image_spans(inputs, cfg)
        if spans.shape[0] > 1 and not bool(spans.eq(spans[0]).all()):
            raise RuntimeError(
                "batch rows have different image-token spans; use "
                "_locate_image_spans for batched inputs"
            )
        return int(spans[0, 0]), int(spans[0, 1])

    def _locate_image_spans(
        self, inputs: dict[str, Any], cfg: PAIConfig
    ) -> torch.Tensor:
        """Return a ``(batch, 2)`` tensor of per-row image-token spans."""
        input_ids = inputs["input_ids"]
        spans = torch.zeros(
            (input_ids.shape[0], 2), dtype=torch.long, device=input_ids.device
        )
        token = self._image_token_index()
        if token is None:
            return spans
        num = (
            cfg.num_image_tokens
            if cfg.num_image_tokens is not None
            else self._num_image_tokens_from_config()
        )
        for row in range(input_ids.shape[0]):
            cols = (input_ids[row] == token).nonzero(as_tuple=False).flatten()
            if cols.numel() == 0:
                continue
            if cols.numel() > 1 and not bool(cols.diff().eq(1).all()):
                raise ValueError("PAI does not support disjoint image-token spans")
            start = int(cols.min())
            end = int(cols.max()) + 1
            if cols.numel() == 1:
                end = start + num
            spans[row] = torch.tensor((start, end), device=input_ids.device)
        return spans

    def _num_image_tokens_from_config(self) -> int:
        """Infer image tokens as ``(image_size // patch_size) ** 2`` for
        CLIP-style towers; falls back to the LLaVA-1.5 default (576)."""
        config = getattr(self.model, "config", None)
        image_seq_length = getattr(config, "image_seq_length", None)
        if image_seq_length:
            return int(image_seq_length)
        vision = getattr(config, "vision_config", None)
        if vision is not None:
            image_size = getattr(vision, "image_size", None)
            patch_size = getattr(vision, "patch_size", None)
            if image_size and patch_size:
                return (int(image_size) // int(patch_size)) ** 2
        return 576  # LLaVA-1.5 default (336px / 14px patch -> 24x24)

    # -- attention intervention -------------------------------------------------
    def _on_generate_start(self, cfg: PAIConfig) -> None:
        self._attn_originals: list[tuple[Any, Any]] = []
        self._attn_alpha = cfg.alpha
        if cfg.alpha > 0:
            self._patch_attention(cfg)

    def _on_generate_end(self) -> None:
        self._unpatch_attention()

    def _attention_layers(self) -> Any:
        """Return the language model's ``ModuleList`` of decoder layers."""
        try:
            return self._language_model_layers()
        except RuntimeError as exc:
            raise RuntimeError(
                "PAI attention intervention needs a language model with a "
                "``.layers`` self-attention stack (e.g. Llama); set alpha=0 "
                "to disable it."
            ) from exc

    def _patch_attention(self, cfg: PAIConfig) -> None:
        layers = self._attention_layers()
        if cfg.start_layer >= len(layers):
            raise MitigatorConfigError(
                f"start_layer {cfg.start_layer} out of range for "
                f"{len(layers)} decoder layers"
            )
        for index in range(cfg.start_layer, min(cfg.end_layer, len(layers))):
            attn = layers[index].self_attn
            self._attn_originals.append((attn, attn.forward))
            attn.forward = self._make_patched_forward(attn)

    def _unpatch_attention(self) -> None:
        for attn, original in getattr(self, "_attn_originals", []):
            attn.forward = original
        self._attn_originals = []

    def _make_patched_forward(self, attn_module: Any):
        """Return a Llama-style eager attention ``forward`` that amplifies the
        image-token attention of the newest query token before the softmax."""
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

            if pai._attn_active:
                attn_weights = pai._amplify_attention_spans(
                    attn_weights, pai._img_spans, pai._attn_alpha
                )

            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = torch.nn.functional.dropout(
                attn_weights,
                p=0.0 if not attn_module.training else attn_module.attention_dropout,
                training=attn_module.training,
            )
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)
            return attn_output, attn_weights

        return forward

    @staticmethod
    def _amplify_attention(
        attn_weights: torch.Tensor, start: int, end: int, alpha: float
    ) -> torch.Tensor:
        """Amplify the newest query token's attention logits toward image tokens.

        Returns a new tensor; ``attn_weights[:, :, -1, start:end]`` becomes
        ``|w| * alpha + w`` (so ``alpha=0`` is a no-op).
        """
        out = attn_weights.clone()
        span = out[:, :, -1, start:end]
        out[:, :, -1, start:end] = span.abs() * alpha + span
        return out

    @staticmethod
    def _amplify_attention_spans(
        attn_weights: torch.Tensor, spans: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Amplify per-row image spans without touching other batch rows."""
        if spans.shape != (attn_weights.shape[0], 2):
            raise RuntimeError(
                "PAI image-token spans do not match the current attention batch"
            )
        positions = torch.arange(attn_weights.shape[-1], device=attn_weights.device)
        spans = spans.to(attn_weights.device)
        image_mask = (positions >= spans[:, :1]) & (positions < spans[:, 1:])
        out = attn_weights.clone()
        newest = out[:, :, -1, :]
        amplified = newest.abs() * alpha + newest
        out[:, :, -1, :] = torch.where(image_mask[:, None, :], amplified, newest)
        return out

    # -- intervention -----------------------------------------------------------
    def _step_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        inputs: dict[str, Any],
        past_key_values: Any,
        step: int,
        cfg: PAIConfig,
    ) -> tuple[torch.Tensor, Any]:
        # Conditional (with-image) branch — attention amplification is active.
        self._attn_active = True
        try:
            logits_img, past = self._forward(
                input_ids, attention_mask, inputs, past_key_values
            )
        finally:
            self._attn_active = False

        if cfg.gamma == 1.0:
            return logits_img, past

        # Text-only branch: full prompt on its first step, else the same token.
        if self._uncond_past is None:
            uncond_ids = self._uncond_inputs["input_ids"]
        else:
            uncond_ids = input_ids
        logits_txt, self._uncond_past = self._forward(
            uncond_ids,
            self._uncond_attention_mask,
            self._uncond_inputs,
            self._uncond_past,
        )
        self._grow_uncond_mask()

        logits = cfg.gamma * (logits_img - logits_txt) + logits_txt
        if cfg.beta > 0:
            logits = self._adaptive_plausibility(logits, logits_img, cfg.beta)
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

    @staticmethod
    def _adaptive_plausibility(
        logits: torch.Tensor, logits_image: torch.Tensor, beta: float
    ) -> torch.Tensor:
        """Mask tokens whose *with-image* probability is below ``beta * max``."""
        cutoff = (
            torch.log(torch.tensor(beta, device=logits.device, dtype=logits.dtype))
            + logits_image.max(dim=-1, keepdim=True).values
        )
        return logits.masked_fill(logits_image < cutoff, float("-inf"))

    # -- beam search ---------------------------------------------------------
    def _expand_inputs_for_beams(
        self, inputs: dict[str, Any], num_beams: int
    ) -> dict[str, Any]:
        expanded = super()._expand_inputs_for_beams(inputs, num_beams)
        self._img_spans = self._img_spans.repeat_interleave(num_beams, dim=0)
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
