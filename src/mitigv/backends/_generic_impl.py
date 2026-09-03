"""Generic autoregressive decoding implementation.

The backend sits between the framework-agnostic :class:`BaseMitigator` and
the concrete algorithms (VCD, PAI, ...). It owns a cache-aware autoregressive
decoding loop and exposes small overridable hooks, so an algorithm implements
only its intervention instead of re-writing generation.

The loop targets causal language models / LVLMs that support ``use_cache=True``.
Model families with a different calling convention can override
:meth:`GenericMitigator._prepare_inputs` or :meth:`GenericMitigator._forward`
without touching the loop itself.
"""

from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import torch

from mitigv.core.base import BaseMitigator, MitigatorConfig, MitigatorConfigError
from mitigv.core.interfaces import ModelProtocol, ProcessorProtocol

__all__ = ["GenericMitigatorConfig", "GenericMitigator", "ModelProtocol", "ProcessorProtocol"]


class GenericMitigatorConfig(MitigatorConfig):
    """Portable decoding configuration.

    Extends :class:`MitigatorConfig` with sampling and beam-search knobs.
    """

    top_k: int | None = None
    length_penalty: float = 1.0
    early_stopping: bool = True
    num_return_sequences: int = 1

    def validate(self) -> None:
        super().validate()
        if self.top_k is not None and (
            not isinstance(self.top_k, Integral) or isinstance(self.top_k, bool)
        ):
            raise MitigatorConfigError("top_k must be an integer or None")
        if self.top_k is not None and self.top_k < 1:
            raise MitigatorConfigError("top_k must be >= 1 when set")
        if self.length_penalty < 0:
            raise MitigatorConfigError("length_penalty must be >= 0")
        if self.num_return_sequences < 1:
            raise MitigatorConfigError("num_return_sequences must be >= 1")
        if self.num_return_sequences > self.num_beams:
            raise MitigatorConfigError("num_return_sequences cannot exceed num_beams")


class GenericMitigator(BaseMitigator):
    """Generic autoregressive mitigator backend.

    Despite the historical ``HF`` name, this class only relies on the small
    :class:`~mitigv.core.interfaces.ModelProtocol` and
    :class:`~mitigv.core.interfaces.ProcessorProtocol` contracts.  Any model
    runtime can therefore be used by implementing a callable forward pass and
    a processor that returns a mapping containing ``input_ids`` (plus an
    optional ``attention_mask``) and decodes token ids.  HuggingFace models
    remain fully compatible.

    Extension points (all may be overridden):

    * :meth:`_prepare_inputs` — ``(images, prompt, cfg) -> dict`` of tensors.
    * :meth:`_step_logits` — per-step ``(input_ids, attn_mask, inputs, past, step, cfg) -> (logits, past)``.
    * :meth:`_forward` — one raw forward pass of the main model.
    * :meth:`_sample_next` — ``(logits, cfg) -> token ids (B, 1)`` (greedy/sampling).
    * :meth:`_reorder_cache` / :meth:`_reorder_aux_cache` — beam-search cache reordering.
    * :meth:`_decode` — ``generated_ids -> str | list[str]``.
    * :meth:`_eos_token_id` — ``-> int | None``.
    """

    config_class = GenericMitigatorConfig

    def __init__(self, model: ModelProtocol | Any = None,
                 processor: ProcessorProtocol | Any = None,
                 config: GenericMitigatorConfig | Mapping[str, Any] | None = None,
                 **kwargs: Any) -> None:
        super().__init__(model=model, processor=processor, config=config, **kwargs)
        if model is not None and not callable(model):
            raise TypeError("model must implement the callable ModelProtocol")
        if processor is not None and not (
            callable(processor)
            or hasattr(processor, "prepare_inputs")
            or hasattr(processor, "encode")
        ):
            raise TypeError(
                "processor must implement ProcessorProtocol (__call__, "
                "prepare_inputs, or encode)"
            )

    # -- device / dtype ------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """Infer the model device from ``device`` or its first parameter."""
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            try:
                return torch.device(model_device)
            except (TypeError, RuntimeError):
                pass
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype | None:
        """Infer model dtype from ``dtype`` or its first parameter."""
        model_dtype = getattr(self.model, "dtype", None)
        if isinstance(model_dtype, torch.dtype):
            return model_dtype
        try:
            return next(self.model.parameters()).dtype
        except (StopIteration, AttributeError):
            return None

    # -- template method -----------------------------------------------------
    def generate(self, images: Any, prompt: str, **kwargs: Any) -> str | list[str]:
        """Run generation; ``**kwargs`` override :attr:`config` for this call."""
        self._ensure_ready()
        with self._generation_lock:
            return self._generate_once(images, prompt, **kwargs)

    def _generate_once(
        self, images: Any, prompt: str, **kwargs: Any
    ) -> str | list[str]:
        """Run one serialized generation and restore all process/model state."""
        cfg = self.config.copy(**kwargs)
        was_training = bool(getattr(self.model, "training", False))
        if hasattr(self.model, "eval"):
            self.model.eval()

        # Generation may use randomness both in sampling and in an algorithm's
        # input perturbation.  fork_rng makes a seeded call reproducible without
        # permanently replacing the caller's process-wide RNG state.
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index or torch.cuda.current_device()]
        rng_context = (
            torch.random.fork_rng(devices=cuda_devices)
            if cfg.seed is not None
            else nullcontext()
        )
        try:
            with rng_context:
                if cfg.seed is not None:
                    torch.random.default_generator.manual_seed(cfg.seed)
                    if cuda_devices:
                        with torch.cuda.device(cuda_devices[0]):
                            torch.cuda.manual_seed(cfg.seed)
                inputs = self._prepare_inputs(images, prompt, cfg)
                try:
                    # Keep cleanup inside the try: a lifecycle hook can fail
                    # after partially patching the model.
                    self._on_generate_start(cfg)
                    with torch.inference_mode():
                        if cfg.num_beams > 1:
                            generated_ids = self._beam_search_loop(inputs, cfg)
                        else:
                            generated_ids = self._decode_loop(inputs, cfg)
                finally:
                    self._on_generate_end()
        finally:
            if was_training and hasattr(self.model, "train"):
                self.model.train()
        return self._decode(generated_ids)

    # -- input preparation ----------------------------------------------------
    def _prepare_inputs(
        self, images: Any, prompt: str, cfg: GenericMitigatorConfig
    ) -> dict[str, Any]:
        """Build the model-input dict from ``images`` + ``prompt``.

        Defaults to ``processor(text=..., images=..., return_tensors="pt")``,
        moves tensors to :attr:`device`, and casts floating-point tensors (e.g.
        ``pixel_values``) to :attr:`dtype` so they match the model's weights.
        Integer tensors (``input_ids``/``attention_mask``) are left untouched.
        Override for model families that need a different calling convention.
        """
        inputs = self._processor_inputs(prompt, images)
        if not isinstance(inputs, Mapping):
            raise TypeError(
                "processor must return a mapping of model inputs; got "
                f"{type(inputs).__name__}"
            )
        if "input_ids" not in inputs:
            raise TypeError(
                "processor output must contain 'input_ids' for autoregressive decoding"
            )
        prepared: dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
                if value.is_floating_point() and self.dtype is not None:
                    value = value.to(self.dtype)
            prepared[key] = value
        # A single-example processor often returns a 1-D token vector.  The
        # decoding contract is batched, so normalize that convenient form.
        token_ids = prepared["input_ids"]
        if isinstance(token_ids, torch.Tensor) and token_ids.ndim == 1:
            prepared["input_ids"] = token_ids.unsqueeze(0)
            mask = prepared.get("attention_mask")
            if isinstance(mask, torch.Tensor) and mask.ndim == 1:
                prepared["attention_mask"] = mask.unsqueeze(0)
        return self._left_pad_token_inputs(prepared)

    def _processor_inputs(self, prompt: Any, images: Any) -> Mapping[str, Any]:
        """Call a processor using the portable protocol.

        ``prepare_inputs``/``encode`` are accepted for runtimes that do not
        model HuggingFace's callable processor API.  Their return value still
        has to be a mapping containing ``input_ids``.
        """
        processor = self.processor
        if hasattr(processor, "prepare_inputs"):
            return processor.prepare_inputs(prompt, images)
        if hasattr(processor, "encode") and not callable(processor):
            return processor.encode(prompt, images)
        if not callable(processor):
            raise TypeError("processor does not implement a supported input method")
        try:
            return processor(
                text=prompt, images=images, return_tensors="pt", padding=True
            )
        except TypeError as error:
            if hasattr(processor, "encode"):
                try:
                    return processor.encode(prompt, images)
                except TypeError:
                    pass
            # A lightweight processor may expose ``(prompt, images)`` instead
            # of keyword arguments.  Only fall back when that signature is
            # actually accepted; preserve errors raised by the processor body.
            try:
                return processor(prompt, images)
            except TypeError:
                raise error

    @staticmethod
    def _left_pad_token_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        """Normalize padded token batches to left padding.

        Decoder-only generation appends new tokens at the right edge. With
        right-padded prompts that creates holes between a prompt and its first
        generated token, and attention-based methods inspect a padding query.
        Token-aligned 2-D tensors are shifted together; visual tensors are left
        untouched.
        """
        mask = inputs.get("attention_mask")
        ids = inputs.get("input_ids")
        if (
            not isinstance(mask, torch.Tensor)
            or not isinstance(ids, torch.Tensor)
            or mask.ndim != 2
            or ids.shape != mask.shape
            or mask.shape[0] <= 1
        ):
            return inputs

        valid = mask.to(dtype=torch.bool)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("attention_mask contains an empty input sequence")
        lengths = valid.sum(dim=1)
        canonical = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        canonical = canonical >= (mask.shape[1] - lengths).unsqueeze(1)
        if bool(valid.eq(canonical).all()):
            return inputs

        result = dict(inputs)
        for key, value in inputs.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape != mask.shape
            ):
                continue
            rows = []
            for row in range(mask.shape[0]):
                row_valid = valid[row]
                rows.append(torch.cat([value[row, ~row_valid], value[row, row_valid]]))
            result[key] = torch.stack(rows)
        return result

    # -- decoding loop ---------------------------------------------------------
    def _decode_loop(
        self, inputs: dict[str, Any], cfg: GenericMitigatorConfig
    ) -> torch.Tensor:
        """Run the autoregressive loop and return the generated token ids (B, T)."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        batch_size = input_ids.shape[0]
        eos = self._eos_token_id()
        pad = self._pad_token_id()
        fill_token = pad if pad is not None else eos
        past = None
        generated: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for step in range(cfg.max_new_tokens):
            logits, past = self._step_logits(
                input_ids, attention_mask, inputs, past, step, cfg
            )
            next_token = self._sample_next(logits, cfg)
            if bool(finished.any()) and fill_token is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, fill_token),
                    next_token,
                )
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

            if eos is not None:
                finished |= next_token.squeeze(1).eq(eos)
                if bool(finished.all()):
                    break
            if past is None and step + 1 < cfg.max_new_tokens:
                raise RuntimeError(
                    "model did not return past_key_values with use_cache=True; "
                    "GenericMitigator requires a cache for multi-token generation"
                )

        if not generated:
            return torch.empty(
                (batch_size, 0), dtype=torch.long, device=input_ids.device
            )
        return torch.cat(generated, dim=-1)

    # -- beam search ----------------------------------------------------------
    def _expand_inputs_for_beams(
        self, inputs: dict[str, Any], num_beams: int
    ) -> dict[str, Any]:
        """Repeat batched tensors' leading dimension for beam search."""
        batch_size = inputs["input_ids"].shape[0]
        return {
            key: (
                value.repeat_interleave(num_beams, dim=0)
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
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
        self._attn_impl_saved = False
        layers = self._language_model_layers()
        if len(layers):
            config = getattr(layers[0].self_attn, "config", None)
            if config is not None:
                self._saved_attn_impl = getattr(config, "_attn_implementation", None)
                self._attn_impl_saved = True
                config._attn_implementation = "eager"

    def _restore_attention_implementation(self) -> None:
        """Restore the attention implementation saved by :meth:`_force_eager_attention`.

        Restores even when the original value was ``None`` (the config default).
        """
        if getattr(self, "_attn_impl_saved", False):
            layers = self._language_model_layers()
            if len(layers):
                config = getattr(layers[0].self_attn, "config", None)
                if config is not None:
                    config._attn_implementation = self._saved_attn_impl
        self._saved_attn_impl = None
        self._attn_impl_saved = False

    # -- generation lifecycle hooks -------------------------------------------
    def _on_generate_start(self, cfg: GenericMitigatorConfig) -> None:
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
        self, inputs: dict[str, Any], cfg: GenericMitigatorConfig
    ) -> torch.Tensor:
        """Run beam search (log-probability scores, standard EOS semantics).

        Beam scores accumulate ``log_softmax(logits)``, so different prefixes are
        comparable. A beam that emits EOS is moved to the finished set and is no
        longer carried forward; its slot is refilled from ``2 * num_beams``
        candidates. Returns generated ids ``(B*num_return, T)``.
        """
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

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # (B*N,)

        is_done = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)
        finished: list[list[tuple[float, torch.Tensor]]] = [
            [] for _ in range(batch_size)
        ]

        past = None

        step = 0
        while step < cfg.max_new_tokens:
            # Standard beam-search early stopping waits until a complete beam
            # set exists.  Stopping after only ``num_return`` EOS candidates can
            # discard a still-active hypothesis with a better eventual score.
            if cfg.early_stopping and all(len(f) >= num_beams for f in finished):
                break

            logits, past = self._step_logits(
                input_ids, attention_mask, beam_inputs, past, step, cfg
            )
            self._validate_next_token_logits(logits)
            vocab_size = logits.shape[-1]
            log_probs = torch.log_softmax(logits, dim=-1)

            next_scores = log_probs + beam_scores.unsqueeze(1)  # (B*N, V)
            next_scores = next_scores.masked_fill(is_done.unsqueeze(1), -float("inf"))
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            # Take 2*num_beams candidates so finished beams can be refilled.
            candidate_count = min(2 * num_beams, next_scores.shape[1])
            next_scores, next_tokens = torch.topk(
                next_scores, candidate_count, dim=1, largest=True, sorted=True
            )
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            new_beam_indices: list[int] = []
            new_token_ids: list[int] = []
            new_scores: list[float] = []

            for b in range(batch_size):
                n_selected = 0
                for c in range(candidate_count):
                    if n_selected >= num_beams:
                        break
                    beam_id = int(next_indices[b, c].item())
                    token = int(next_tokens[b, c].item())
                    score = float(next_scores[b, c].item())
                    flat = b * num_beams + beam_id
                    if eos is not None and token == eos:
                        if not bool(is_done[flat]):
                            seq = full_ids[flat, prompt_len:].clone()
                            cur_len = seq.numel() + 1
                            finished[b].append(
                                (score / (float(cur_len) ** cfg.length_penalty), seq)
                            )
                        continue
                    new_beam_indices.append(flat)
                    new_token_ids.append(token)
                    new_scores.append(score)
                    n_selected += 1

            if not new_beam_indices:
                break  # every beam finished
            if past is None and step + 1 < cfg.max_new_tokens:
                raise RuntimeError(
                    "model did not return past_key_values with use_cache=True; "
                    "GenericMitigator requires a cache for multi-token beam search"
                )

            beam_idx = torch.tensor(new_beam_indices, dtype=torch.long, device=device)
            next_token_t = torch.tensor(
                new_token_ids, dtype=torch.long, device=device
            ).unsqueeze(1)
            beam_scores = torch.tensor(new_scores, dtype=torch.float, device=device)

            full_ids = torch.cat([full_ids[beam_idx], next_token_t], dim=-1)
            if attention_mask is not None:
                ones = torch.ones(
                    (len(beam_idx), 1), dtype=attention_mask.dtype, device=device
                )
                attention_mask = torch.cat([attention_mask[beam_idx], ones], dim=-1)

            past = self._reorder_cache(past, beam_idx)
            self._reorder_aux_cache(beam_idx)
            is_done = is_done[beam_idx]  # active beams are never done

            input_ids = next_token_t  # feed only the new tokens next step (cache)
            step += 1

        return self._finalize_beams(
            finished,
            full_ids,
            beam_scores,
            is_done,
            prompt_len,
            batch_size,
            num_beams,
            num_return,
            cfg,
            device,
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
        cfg: GenericMitigatorConfig,
        device: torch.device,
    ) -> torch.Tensor:
        """Select the top ``num_return`` hypotheses per batch and pad to equal length."""
        gen_len = max(full_ids.shape[1] - prompt_len, 1)
        pad = self._pad_token_id() if self._pad_token_id() is not None else 0

        rows: list[torch.Tensor] = []
        for b in range(batch_size):
            candidates = list(finished[b])
            if not cfg.early_stopping or len(candidates) < num_return:
                for i in range(num_beams):
                    flat = b * num_beams + i
                    if not bool(is_done[flat]):
                        score = beam_scores[flat].item() / (
                            float(gen_len) ** cfg.length_penalty
                        )
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
                    [
                        r,
                        torch.full(
                            (max_len - r.numel(),), pad, dtype=torch.long, device=device
                        ),
                    ]
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
        cfg: GenericMitigatorConfig,
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
        if isinstance(outputs, Mapping):
            all_logits = outputs.get("logits")
            returned_past = outputs.get("past_key_values", outputs.get("cache"))
        elif isinstance(outputs, (tuple, list)):
            if not outputs:
                raise TypeError("model output tuple is empty; expected logits")
            all_logits = outputs[0]
            returned_past = outputs[1] if len(outputs) > 1 else None
        else:
            all_logits = getattr(outputs, "logits", None)
            returned_past = getattr(
                outputs, "past_key_values", getattr(outputs, "cache", None)
            )
        if all_logits is None:
            raise TypeError(
                "model output must expose logits as an attribute, mapping key, "
                "or first tuple item"
            )
        next_logits = all_logits[:, -1, :]

        # On the prefill step a processor can right-pad a batch.  In that case
        # ``[:, -1]`` selects a padding position for shorter prompts.  Select
        # each row's right-most valid token whenever logits and mask are aligned.
        if (
            past_key_values is None
            and attention_mask is not None
            and attention_mask.ndim == 2
            and all_logits.shape[1] == attention_mask.shape[1]
        ):
            valid = attention_mask.to(dtype=torch.bool)
            if not bool(valid.any(dim=1).all()):
                raise ValueError("attention_mask contains an empty input sequence")
            positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
            last_valid = positions.masked_fill(~valid, -1).max(dim=1).values
            batch = torch.arange(all_logits.shape[0], device=all_logits.device)
            next_logits = all_logits[batch, last_valid]

        return next_logits, returned_past

    @staticmethod
    def _add_to_layer_output(output: Any, delta: torch.Tensor) -> Any:
        """Add ``delta`` to a decoder layer's hidden state across HF versions.

        Recent Llama decoder layers return a tensor, while older transformers
        releases return a tuple whose first item is the hidden-state tensor.
        Steering algorithms support both without dropping cache/attention data.
        """
        if isinstance(output, torch.Tensor):
            return output + delta
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (output[0] + delta, *output[1:])
        if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            return [output[0] + delta, *output[1:]]
        raise TypeError(
            "decoder layer hook expected a Tensor or a non-empty sequence "
            f"starting with a Tensor, got {type(output).__name__}"
        )

    def _sample_next(
        self, logits: torch.Tensor, cfg: GenericMitigatorConfig
    ) -> torch.Tensor:
        """Turn next-token logits into token ids ``(B, 1)``.

        Greedy (``argmax``) when ``do_sample`` is false; otherwise applies
        temperature / top-k / top-p warpers and multinomial sampling.
        """
        self._validate_next_token_logits(logits)
        if not cfg.do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        scores = logits
        if cfg.temperature != 1.0:
            scores = scores / cfg.temperature
        if cfg.top_k is not None:
            k = min(cfg.top_k, scores.shape[-1])
            threshold = torch.topk(scores, k, dim=-1).values[..., -1, None]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
        if cfg.top_p is not None:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            cumulative = sorted_probs.cumsum(dim=-1)
            remove = cumulative - sorted_probs > cfg.top_p
            sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
            scores = torch.full_like(scores, float("-inf"))
            scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)
        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def _validate_next_token_logits(logits: torch.Tensor) -> None:
        """Reject malformed distributions before argmax/softmax hide the cause."""
        invalid = torch.isnan(logits) | torch.isposinf(logits)
        if bool(invalid.any()) or bool(torch.isneginf(logits).all(dim=-1).any()):
            raise RuntimeError(
                "decoder produced an invalid next-token distribution "
                "(NaN/+inf or an all-masked row)"
            )

    # -- tokenizer helpers ---------------------------------------------------
    def _eos_token_id(self) -> int | None:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return getattr(tokenizer, "eos_token_id", None)

    def _pad_token_id(self) -> int | None:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return getattr(tokenizer, "pad_token_id", None)

    def _decode(self, generated_ids: torch.Tensor) -> str | list[str]:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if hasattr(tokenizer, "batch_decode"):
            texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        elif hasattr(tokenizer, "decode"):
            texts = [tokenizer.decode(row, skip_special_tokens=True) for row in generated_ids]
        else:
            raise TypeError(
                "processor must implement batch_decode or decode to return text"
            )
        if len(texts) == 1:
            return texts[0]
        return texts
