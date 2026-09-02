"""HuggingFace-transformers decoding backend.

``HFMitigator`` sits between the framework-agnostic :class:`BaseMitigator` and
the concrete algorithms (VCD, PAI, ...). It owns a cache-aware autoregressive
decoding loop and exposes small overridable hooks, so an algorithm implements
only its intervention instead of re-writing generation.

The loop targets causal language models / LVLMs that support ``use_cache=True``
(the standard ``transformers`` case). Model families with a different calling
convention can override :meth:`HFMitigator._prepare_inputs` or
:meth:`HFMitigator._forward` without touching the loop itself.
"""

from __future__ import annotations

from typing import Any

import torch

from mitigv.core.base import BaseMitigator, MitigatorConfig, MitigatorConfigError

__all__ = ["HFMitigatorConfig", "HFMitigator"]


class HFMitigatorConfig(MitigatorConfig):
    """Decoding configuration for the HF backend.

    Extends :class:`MitigatorConfig` with HF-specific knobs for sampling and
    beam search.
    """

    top_k: int | None = None
    length_penalty: float = 1.0
    early_stopping: bool = True
    num_return_sequences: int = 1

    def validate(self) -> None:
        super().validate()
        if self.top_k is not None and self.top_k < 1:
            raise MitigatorConfigError("top_k must be >= 1 when set")
        if self.length_penalty < 0:
            raise MitigatorConfigError("length_penalty must be >= 0")
        if self.num_return_sequences < 1:
            raise MitigatorConfigError("num_return_sequences must be >= 1")
        if self.num_return_sequences > self.num_beams:
            raise MitigatorConfigError("num_return_sequences cannot exceed num_beams")


class HFMitigator(BaseMitigator):
    """Base class for mitigators backed by a HuggingFace ``transformers`` model.

    Implements greedy / sampling / beam-search decoding with a per-step hook so
    subclasses can intervene on the logits before the next token is chosen.

    Extension points (all may be overridden):

    * :meth:`_prepare_inputs` — ``(images, prompt, cfg) -> dict`` of tensors.
    * :meth:`_step_logits` — per-step ``(input_ids, attn_mask, inputs, past, step, cfg) -> (logits, past)``.
    * :meth:`_forward` — one raw forward pass of the main model.
    * :meth:`_sample_next` — ``(logits, cfg) -> token ids (B, 1)`` (greedy/sampling).
    * :meth:`_reorder_cache` / :meth:`_reorder_aux_cache` — beam-search cache reordering.
    * :meth:`_decode` — ``generated_ids -> str | list[str]``.
    * :meth:`_eos_token_id` — ``-> int | None``.
    """

    config_class = HFMitigatorConfig

    # -- device / dtype ------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """Infer the model's device from its first parameter (fallback: CPU)."""
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype | None:
        """Infer the model's dtype from its first parameter (``None`` if unknown)."""
        try:
            return next(self.model.parameters()).dtype
        except (StopIteration, AttributeError):
            return None

    # -- template method -----------------------------------------------------
    def generate(self, images: Any, prompt: str, **kwargs: Any) -> str | list[str]:
        """Run generation; ``**kwargs`` override :attr:`config` for this call."""
        self._ensure_ready()
        cfg = self.config.copy(**kwargs)
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        inputs = self._prepare_inputs(images, prompt, cfg)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._on_generate_start(cfg)
        try:
            with torch.no_grad():
                if cfg.num_beams > 1:
                    generated_ids = self._beam_search_loop(inputs, cfg)
                else:
                    generated_ids = self._decode_loop(inputs, cfg)
        finally:
            self._on_generate_end()
        return self._decode(generated_ids)

    # -- input preparation ----------------------------------------------------
    def _prepare_inputs(
        self, images: Any, prompt: str, cfg: HFMitigatorConfig
    ) -> dict[str, Any]:
        """Build the model-input dict from ``images`` + ``prompt``.

        Defaults to ``processor(text=..., images=..., return_tensors="pt")``,
        moves tensors to :attr:`device`, and casts floating-point tensors (e.g.
        ``pixel_values``) to :attr:`dtype` so they match the model's weights.
        Integer tensors (``input_ids``/``attention_mask``) are left untouched.
        Override for model families that need a different calling convention.
        """
        inputs = self.processor(text=prompt, images=images, return_tensors="pt", padding=True)
        prepared: dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
                if value.is_floating_point() and self.dtype is not None:
                    value = value.to(self.dtype)
            prepared[key] = value
        return prepared

    # -- decoding loop ---------------------------------------------------------
    def _decode_loop(self, inputs: dict[str, Any], cfg: HFMitigatorConfig) -> torch.Tensor:
        """Run the autoregressive loop and return the generated token ids (B, T)."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        batch_size = input_ids.shape[0]
        eos = self._eos_token_id()
        past = None
        generated: list[torch.Tensor] = []

        for step in range(cfg.max_new_tokens):
            logits, past = self._step_logits(input_ids, attention_mask, inputs, past, step, cfg)
            next_token = self._sample_next(logits, cfg)
            generated.append(next_token)

            # Prepare inputs for the next step: with a KV cache we only feed the
            # new token and grow the mask by one position.
            input_ids = next_token
            if attention_mask is not None:
                ones = torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, ones], dim=-1)

            if eos is not None and bool((next_token == eos).all()):
                break

        if not generated:
            return torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)
        return torch.cat(generated, dim=-1)

    # -- beam search ----------------------------------------------------------
    def _expand_inputs_for_beams(
        self, inputs: dict[str, Any], num_beams: int
    ) -> dict[str, Any]:
        """Repeat every tensor's batch dim ``num_beams`` times for beam search."""
        return {
            key: (
                value.repeat_interleave(num_beams, dim=0)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in inputs.items()
        }

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.Tensor) -> Any:
        """Reorder a KV cache's batch dim to the new beam order."""
        if past_key_values is None:
            return None
        # modern transformers: the cache object knows how to reorder itself
        # (DynamicCache mutates in place and returns None, so return the object)
        if hasattr(past_key_values, "reorder_cache"):
            result = past_key_values.reorder_cache(beam_idx)
            return past_key_values if result is None else result
        # older transformers: the model owns a `_reorder_cache` helper
        if hasattr(self.model, "_reorder_cache"):
            return self.model._reorder_cache(past_key_values, beam_idx)
        # legacy fallback: tuple of (key, value) layer pairs
        return tuple(
            tuple(t.index_select(0, beam_idx) for t in layer)
            for layer in past_key_values
        )

    def _reorder_aux_cache(self, beam_idx: torch.Tensor) -> None:
        """Hook for subclasses with extra per-beam state (e.g. VCD's distorted
        branch cache). Called by beam search after reordering the main cache."""
        return None

    def _language_model_layers(self) -> Any:
        """Return the language model's decoder-layer ``ModuleList`` (e.g. Llama's).

        Resolves the common wrapper shapes across transformers versions —
        ``model.language_model`` (a ``ForCausalLM`` exposing ``.model.layers``)
        or ``model.model.language_model`` (a base model exposing ``.layers``) —
        and returns the ``.layers`` holder. Raises if none is found.
        """
        lm = getattr(self.model, "language_model", None)
        if lm is None:
            base = getattr(self.model, "model", None)
            lm = getattr(base, "language_model", None)
        if lm is None:
            raise RuntimeError(
                f"{type(self).__name__} requires a language model with a "
                "'.layers' decoder stack (e.g. Llama)."
            )
        holder = getattr(lm, "model", lm)  # ForCausalLM.model -> base model
        if not hasattr(holder, "layers"):
            raise RuntimeError(
                f"{type(self).__name__} requires a language model with a "
                "'.layers' decoder stack (e.g. Llama)."
            )
        return holder.layers

    def _force_eager_attention(self) -> None:
        """Force eager attention so ``output_attentions`` materializes real weights.

        Saves the previous implementation so :meth:`_restore_attention_implementation`
        can revert it. Attention-based algorithms (OPERA/AGLA/ONLY) call this
        before probing attention maps.
        """
        self._saved_attn_impl = None
        layers = self._language_model_layers()
        if len(layers):
            config = getattr(layers[0].self_attn, "config", None)
            if config is not None:
                self._saved_attn_impl = getattr(config, "_attn_implementation", None)
                config._attn_implementation = "eager"

    def _restore_attention_implementation(self) -> None:
        """Restore the attention implementation saved by :meth:`_force_eager_attention`."""
        saved = getattr(self, "_saved_attn_impl", None)
        if saved is not None:
            layers = self._language_model_layers()
            if len(layers):
                config = getattr(layers[0].self_attn, "config", None)
                if config is not None:
                    config._attn_implementation = saved
        self._saved_attn_impl = None

    # -- generation lifecycle hooks -------------------------------------------
    def _on_generate_start(self, cfg: HFMitigatorConfig) -> None:
        """Hook called once per generation, after inputs are prepared and the
        model is set to eval mode, but before the decode loop.

        Subclasses may patch model internals here (e.g. PAI's attention
        intervention). ``cfg`` is the effective config for this call.
        """
        return None

    def _on_generate_end(self) -> None:
        """Hook called once per generation, after the decode loop.

        Always runs (also on error) so subclasses can restore anything patched
        in :meth:`_on_generate_start`.
        """
        return None

    def _beam_search_loop(
        self, inputs: dict[str, Any], cfg: HFMitigatorConfig
    ) -> torch.Tensor:
        """Run greedy beam search and return generated ids ``(B*num_return, T)``."""
        batch_size = inputs["input_ids"].shape[0]
        num_beams = cfg.num_beams
        num_return = cfg.num_return_sequences
        device = self.device
        eos = self._eos_token_id()
        prompt_len = inputs["input_ids"].shape[1]

        beam_inputs = self._expand_inputs_for_beams(inputs, num_beams)
        input_ids = beam_inputs["input_ids"]
        attention_mask = beam_inputs.get("attention_mask")
        full_ids = input_ids  # complete sequences (prompt + generated)

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # (B*N,)

        is_done = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)
        finished: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(batch_size)]

        batch_offset = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
        past = None

        step = 0
        while step < cfg.max_new_tokens:
            if cfg.early_stopping and bool(is_done.all()):
                break

            logits, past = self._step_logits(
                input_ids, attention_mask, beam_inputs, past, step, cfg
            )
            vocab_size = logits.shape[-1]

            scores = logits + beam_scores.unsqueeze(1)  # (B*N, V)
            scores = scores.masked_fill(is_done.unsqueeze(1), -float("inf"))
            scores = scores.view(batch_size, num_beams * vocab_size)
            topk_scores, topk_idx = torch.topk(scores, num_beams, dim=1, largest=True, sorted=True)

            parent = torch.div(topk_idx, vocab_size, rounding_mode="floor")  # (B, N)
            token = topk_idx % vocab_size  # (B, N)
            beam_idx = (batch_offset + parent).view(-1)  # (B*N,) global flat
            next_tokens = token.view(-1, 1)

            full_ids = torch.cat([full_ids[beam_idx], next_tokens], dim=-1)
            if attention_mask is not None:
                ones = torch.ones(
                    (batch_size * num_beams, 1),
                    dtype=attention_mask.dtype,
                    device=device,
                )
                attention_mask = torch.cat([attention_mask[beam_idx], ones], dim=-1)

            past = self._reorder_cache(past, beam_idx)
            self._reorder_aux_cache(beam_idx)

            beam_scores = topk_scores.view(-1)
            is_done = is_done[beam_idx]

            if eos is not None:
                newly_done = next_tokens.squeeze(1).eq(eos) & ~is_done
                if bool(newly_done.any()):
                    cur_len = step + 1
                    norm = beam_scores[newly_done] / (float(cur_len) ** cfg.length_penalty)
                    done_seqs = full_ids[newly_done, prompt_len:]
                    done_flat = torch.nonzero(newly_done, as_tuple=False).squeeze(1)
                    done_batch = torch.div(done_flat, num_beams, rounding_mode="floor")
                    for i in range(done_flat.numel()):
                        finished[int(done_batch[i])].append((norm[i].item(), done_seqs[i]))
                    is_done = is_done | newly_done

            input_ids = next_tokens  # feed only the new tokens next step (cache)
            step += 1

        return self._finalize_beams(
            finished, full_ids, beam_scores, is_done, prompt_len,
            batch_size, num_beams, num_return, cfg, device,
        )

    def _finalize_beams(
        self,
        finished: list[list[tuple[float, torch.Tensor]]],
        full_ids: torch.Tensor,
        beam_scores: torch.Tensor,
        is_done: torch.Tensor,
        prompt_len: int,
        batch_size: int,
        num_beams: int,
        num_return: int,
        cfg: HFMitigatorConfig,
        device: torch.device,
    ) -> torch.Tensor:
        """Select the top ``num_return`` beams per batch and pad to equal length."""
        gen_len = max(full_ids.shape[1] - prompt_len, 1)
        pad = self._pad_token_id() if self._pad_token_id() is not None else 0

        rows: list[torch.Tensor] = []
        for b in range(batch_size):
            candidates = list(finished[b])
            if not cfg.early_stopping or len(candidates) < num_return:
                for i in range(num_beams):
                    flat = b * num_beams + i
                    if not bool(is_done[flat]):
                        score = beam_scores[flat].item() / (float(gen_len) ** cfg.length_penalty)
                        candidates.append((score, full_ids[flat, prompt_len:]))
            candidates.sort(key=lambda item: item[0], reverse=True)
            rows.extend(tokens for _, tokens in candidates[:num_return])

        if not rows:
            return torch.empty((0, 0), dtype=torch.long, device=device)

        max_len = max(r.numel() for r in rows)
        padded = []
        for r in rows:
            if r.numel() < max_len:
                r = torch.cat(
                    [r, torch.full((max_len - r.numel(),), pad, dtype=torch.long, device=device)]
                )
            padded.append(r)
        return torch.stack(padded, dim=0)

    # -- hooks --------------------------------------------------------------
    def _step_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        inputs: dict[str, Any],
        past_key_values: Any,
        step: int,
        cfg: HFMitigatorConfig,
    ) -> tuple[torch.Tensor, Any]:
        """Compute the next-token logits for the current decoding step.

        Default is a single forward pass on the main model. Contrastive
        algorithms (e.g. VCD) override this to run additional forward passes —
        e.g. on a distorted image — and combine the logits before returning.
        ``cfg`` is the *effective* config for this generation call (including
        any per-call overrides), so algorithms can read their hyper-parameters.
        """
        return self._forward(input_ids, attention_mask, inputs, past_key_values)

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        inputs: dict[str, Any],
        past_key_values: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Run one forward pass on the main model.

        Visual inputs (anything except ``input_ids``/``attention_mask``) are
        passed only on the first step, when ``past_key_values`` is ``None``.
        Returns ``(logits (B, V), past_key_values)``.
        """
        model_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "past_key_values": past_key_values,
            "return_dict": True,
        }
        if past_key_values is None:
            for key, value in inputs.items():
                if key not in ("input_ids", "attention_mask"):
                    model_kwargs[key] = value
        outputs = self.model(**model_kwargs)
        return outputs.logits[:, -1, :], outputs.past_key_values

    def _sample_next(self, logits: torch.Tensor, cfg: HFMitigatorConfig) -> torch.Tensor:
        """Turn next-token logits into token ids ``(B, 1)``.

        Greedy (``argmax``) when ``do_sample`` is false; otherwise applies
        temperature / top-k / top-p warpers and multinomial sampling.
        """
        if not cfg.do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        from transformers.generation.logits_process import (
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        scores = logits
        if cfg.temperature != 1.0:
            scores = TemperatureLogitsWarper(temperature=cfg.temperature)(None, scores)
        if cfg.top_k is not None:
            scores = TopKLogitsWarper(top_k=cfg.top_k)(None, scores)
        if cfg.top_p is not None:
            scores = TopPLogitsWarper(top_p=cfg.top_p)(None, scores)
        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # -- tokenizer helpers ---------------------------------------------------
    def _eos_token_id(self) -> int | None:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return getattr(tokenizer, "eos_token_id", None)

    def _pad_token_id(self) -> int | None:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return getattr(tokenizer, "pad_token_id", None)

    def _decode(self, generated_ids: torch.Tensor) -> str | list[str]:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if len(texts) == 1:
            return texts[0]
        return texts
