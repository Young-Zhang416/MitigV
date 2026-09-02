"""OPERA — Over-trust Penalty and Retrospection-Allocation (Huang et al., CVPR 2024).

Beam-search decoding that detects the *knowledge aggregation pattern* — a
"columnar" self-attention map in which many subsequent tokens over-trust a single
summary token — and penalizes it, then rolls back and re-selects when the pattern
persists.

At each step the last layer's attention is reduced over heads by a max and
re-normalized. For each beam, the top-``num_attn_candidates`` next tokens are
temporarily appended (a cached look-ahead forward) and the over-trust penalty is
computed on the candidate's own attention as the raw column product over a local
window::

    phi = max_j  prod_{i=j}^{t-1} sigma * omega_{i,j}

The candidate's score is penalized by ``penalty_weight * phi``. Retrospection
rolls decoding back to a summary token (the mode of recent argmax-penalty
columns, when its count reaches ``threshold``) and re-selects a different next
token, excluding the previously chosen one.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any

import torch

from mitigv.backends.hf import HFMitigator, HFMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator

__all__ = ["OPERAConfig", "OPERA"]


class OPERAConfig(HFMitigatorConfig):
    """Hyper-parameters for OPERA (paper defaults).

    Attributes:
        num_beams: Beam width (must be >= 2; paper default 5).
        sigma: Attention scale factor (paper 50).
        penalty_weight: Over-trust penalty weight ``alpha`` (paper 1).
        num_attn_candidates: Top candidates per beam for the look-ahead (paper 5).
        window_size: Local attention window ``k`` (paper 5).
        threshold: Location-overlap count ``r`` that triggers retrospection (paper 15).
        retrospection_window: Number of recent argmax columns to consider ``l`` (> ``threshold``).
        max_rollback: Maximum rollback length ``beta``.
    """

    num_beams: int = 5
    sigma: float = 50.0
    penalty_weight: float = 1.0
    num_attn_candidates: int = 5
    window_size: int = 5
    threshold: int = 15
    retrospection_window: int = 20
    max_rollback: int = 30

    def validate(self) -> None:
        super().validate()
        if self.num_beams < 2:
            raise MitigatorConfigError("OPERA requires num_beams >= 2 (beam search)")
        if self.sigma <= 0:
            raise MitigatorConfigError("sigma must be > 0")
        if self.penalty_weight < 0:
            raise MitigatorConfigError("penalty_weight must be >= 0")
        if self.num_attn_candidates < 1:
            raise MitigatorConfigError("num_attn_candidates must be >= 1")
        if self.window_size < 1:
            raise MitigatorConfigError("window_size must be >= 1")
        if self.threshold < 1:
            raise MitigatorConfigError("threshold must be >= 1")
        if self.retrospection_window <= self.threshold:
            raise MitigatorConfigError("retrospection_window must be > threshold")
        if self.max_rollback < 1:
            raise MitigatorConfigError("max_rollback must be >= 1")


@register_mitigator("opera")
class OPERA(HFMitigator):
    """Over-trust Penalty and Retrospection-Allocation beam search."""

    algorithm_name = "opera"
    config_class = OPERAConfig

    # -- setup -----------------------------------------------------------------
    def _prepare_inputs(self, images: Any, prompt: str, cfg: OPERAConfig) -> dict[str, Any]:
        inputs = super()._prepare_inputs(images, prompt, cfg)
        self._response_start = inputs["input_ids"].shape[1]
        return inputs

    def _on_generate_start(self, cfg: OPERAConfig) -> None:
        self._force_eager_attention()
        layers = self._language_model_layers()
        self._attn_capture: torch.Tensor | None = None

        def hook(module: Any, args: Any, output: Any) -> Any:
            if output[1] is None:
                raise RuntimeError(
                    "OPERA needs real attention weights; force eager attention "
                    "did not take effect."
                )
            self._attn_capture = output[1].detach()
            return output

        self._attn_handle = layers[-1].self_attn.register_forward_hook(hook)
        self._attn_history: list[list[torch.Tensor]] = []
        self._penalty_columns: list[list[int]] = []
        self._chosen_tokens: list[list[int]] = []

    def _on_generate_end(self) -> None:
        if getattr(self, "_attn_handle", None) is not None:
            self._attn_handle.remove()
        self._attn_handle = None
        self._restore_attention_implementation()

    # -- attention -------------------------------------------------------------
    @staticmethod
    def _process_attention(attn: torch.Tensor) -> torch.Tensor:
        """Max over heads, then re-normalize each row (the paper's aggregation)."""
        attn = attn.max(dim=1).values  # (B, q, kv)
        return attn / (attn.sum(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def _align_row(row: torch.Tensor, kv_len: int) -> torch.Tensor:
        """Right-pad (or truncate) an attention row to the absolute KV length.

        Accepts a 1-D ``(kv,)`` row (returning ``(1, kv_len)``) or a 2-D batch.
        """
        if row.dim() == 1:
            row = row.unsqueeze(0)
        if row.shape[-1] < kv_len:
            row = torch.nn.functional.pad(row, (0, kv_len - row.shape[-1]))
        return row[:, :kv_len]

    @staticmethod
    def _compute_penalty(
        rows: list[torch.Tensor], kv_len: int, sigma: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw column-product penalty over a local window (paper Eq. 6), in FP32.

        ``rows`` are the last generated tokens' attention rows, absolute-aligned
        to ``kv_len``. Returns ``(penalty (B,), argmax_abs_col (B,))``.
        """
        k = len(rows)
        cols = torch.stack(
            [OPERA._align_row(r, kv_len)[:, -k:] for r in rows], dim=1
        ).to(torch.float32)
        window = cols * sigma  # (B, k, k)
        col_prod = torch.ones((window.shape[0], k), dtype=torch.float32, device=window.device)
        for j in range(k):
            col_prod[:, j] = window[:, j:, j].prod(dim=1)
        penalty, argmax_local = col_prod.max(dim=-1)
        return penalty, (kv_len - k) + argmax_local

    def _kv_len(self, past: Any, fallback: int) -> int:
        if past is not None and hasattr(past, "get_seq_length"):
            try:
                return int(past.get_seq_length())
            except Exception:
                pass
        return fallback

    def _clone_cache(self, cache: Any) -> Any:
        return copy.deepcopy(cache)

    def _lookahead_row(
        self, beam_idx: int, token: int, past: Any, kv_len: int, device: torch.device
    ) -> torch.Tensor:
        """Append ``token`` to beam ``beam_idx``'s cache and return its attention row."""
        cache = self._clone_cache(past)
        try:
            cache.batch_select_indices(torch.tensor([beam_idx], device=device))
        except Exception:
            pass
        self._attn_capture = None
        self.model(
            input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        attn = self._attn_capture
        if attn is None:
            return torch.zeros(kv_len + 1, device=device)
        row = self._process_attention(attn)[:, -1, :]  # (1, kv_len+1)
        return self._align_row(row, kv_len + 1)[0]  # (kv_len+1,)

    def _step_forward(self, input_ids, attention_mask, inputs, past_key_values):
        self._attn_capture = None
        logits, past = self._forward(input_ids, attention_mask, inputs, past_key_values)
        return logits, past, self._attn_capture

    def _detect_retrospection(self, cfg: OPERAConfig) -> tuple[int, int, int] | None:
        """Return ``(beam, target_len, exclude_token)`` or ``None``.

        ``target_len`` is the number of generated tokens to keep (roll back to
        just after the summary token). ``exclude_token`` is the previously chosen
        next token that must be excluded on re-allocation.
        """
        for b in range(len(self._penalty_columns)):
            cols = self._penalty_columns[b][-cfg.retrospection_window:]
            if len(cols) < cfg.threshold:
                continue
            pos, count = Counter(cols).most_common(1)[0]
            if count >= cfg.threshold and pos >= self._response_start:
                j = pos - self._response_start  # summary token's generation index
                target_len = j + 1
                exclude = (
                    self._chosen_tokens[b][j + 1]
                    if j + 1 < len(self._chosen_tokens[b])
                    else None
                )
                return b, target_len, exclude
        return None

    # -- beam search -----------------------------------------------------------
    def _beam_search_loop(self, inputs: dict[str, Any], cfg: OPERAConfig) -> torch.Tensor:
        batch_size = inputs["input_ids"].shape[0]
        num_beams = cfg.num_beams
        num_return = cfg.num_return_sequences
        device = self.device
        eos = self._eos_token_id()
        prompt_len = inputs["input_ids"].shape[1]
        n_beams = batch_size * num_beams

        beam_inputs = self._expand_inputs_for_beams(inputs, num_beams)
        input_ids = beam_inputs["input_ids"]
        attention_mask = beam_inputs.get("attention_mask")
        full_ids = input_ids

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        is_done = torch.zeros(n_beams, dtype=torch.bool, device=device)
        finished: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(batch_size)]
        past = None
        kv_len = prompt_len

        self._attn_history = [[] for _ in range(n_beams)]
        self._penalty_columns = [[] for _ in range(n_beams)]
        self._chosen_tokens = [[] for _ in range(n_beams)]
        excluded: list[set[int]] = [set() for _ in range(n_beams)]

        # snapshots for retrospection: n_generated_tokens -> full state
        snapshots: dict[int, tuple] = {
            0: self._snapshot(full_ids, beam_scores, past, is_done, attention_mask)
        }

        step = 0
        while step < cfg.max_new_tokens:
            if cfg.early_stopping and all(len(f) >= num_return for f in finished):
                break

            logits, past, attn = self._step_forward(input_ids, attention_mask, beam_inputs, past)
            kv_len = self._kv_len(past, kv_len + 1)
            row = self._align_row(self._process_attention(attn)[:, -1, :], kv_len)
            for b in range(row.shape[0]):
                self._attn_history[b].append(row[b].detach())

            log_probs = torch.log_softmax(logits, dim=-1)
            vocab_size = logits.shape[-1]

            # candidate-level penalty via cached look-ahead
            cand_scores: list[tuple[float, int, int]] = []
            for b in range(n_beams):
                if bool(is_done[b]):
                    continue
                top_scores, top_tokens = torch.topk(
                    log_probs[b], min(cfg.num_attn_candidates, vocab_size), largest=True
                )
                for sc, tok in zip(top_scores.tolist(), top_tokens.tolist()):
                    if tok in excluded[b]:
                        continue
                    c_row = self._lookahead_row(b, tok, past, kv_len, device)
                    penalty, _ = self._compute_penalty(
                        (self._attn_history[b] + [c_row])[-cfg.window_size:],
                        kv_len + 1,
                        cfg.sigma,
                    )
                    score = sc + float(beam_scores[b]) - cfg.penalty_weight * float(penalty[0])
                    cand_scores.append((score, b, tok))

            if not cand_scores:
                break
            cand_scores.sort(key=lambda t: t[0], reverse=True)

            # retrospection: roll back to the summary token and exclude the old next token
            rb = self._detect_retrospection(cfg)
            if rb is not None:
                b, target_len, exclude_token = rb
                if target_len in snapshots and target_len < len(self._chosen_tokens[b]):
                    full_ids, beam_scores, past, is_done, attention_mask = self._restore(
                        snapshots[target_len]
                    )
                    excluded = [set() for _ in range(n_beams)]
                    if exclude_token is not None:
                        excluded[b] = {exclude_token}
                    step = target_len
                    kv_len = self._response_start + target_len
                    snapshots = {k: v for k, v in snapshots.items() if k <= target_len}
                    input_ids = full_ids[:, -1:]
                    continue

            # select next beams (standard EOS semantics)
            new_beam_indices: list[int] = []
            new_token_ids: list[int] = []
            new_scores: list[float] = []
            seen: set[tuple[int, int]] = set()
            for score, b, tok in cand_scores:
                if len(new_token_ids) >= n_beams:
                    break
                if (b, tok) in seen:
                    continue
                seen.add((b, tok))
                if eos is not None and tok == eos:
                    batch = b // num_beams
                    seq = full_ids[b, prompt_len:].clone()
                    cur_len = seq.numel() + 1
                    finished[batch].append((score / (cur_len ** cfg.length_penalty), seq))
                    continue
                new_beam_indices.append(b)
                new_token_ids.append(tok)
                new_scores.append(score)

            if not new_beam_indices:
                break

            beam_idx = torch.tensor(new_beam_indices, dtype=torch.long, device=device)
            next_token_t = torch.tensor(new_token_ids, dtype=torch.long, device=device).unsqueeze(1)
            beam_scores = torch.tensor(new_scores, dtype=torch.float, device=device)

            full_ids = torch.cat([full_ids[beam_idx], next_token_t], dim=-1)
            if attention_mask is not None:
                ones = torch.ones((len(beam_idx), 1), dtype=attention_mask.dtype, device=device)
                attention_mask = torch.cat([attention_mask[beam_idx], ones], dim=-1)

            past = self._reorder_cache(past, beam_idx)
            self._reorder_aux_cache(beam_idx)
            is_done = is_done[beam_idx]

            self._attn_history = [self._attn_history[b] for b in new_beam_indices]
            self._penalty_columns = [self._penalty_columns[b] for b in new_beam_indices]
            self._chosen_tokens = [
                self._chosen_tokens[b] + [t] for b, t in zip(new_beam_indices, new_token_ids)
            ]

            input_ids = next_token_t
            kv_len += 1
            step += 1
            snapshots[step] = self._snapshot(full_ids, beam_scores, past, is_done, attention_mask)
            # prune snapshots older than max_rollback
            if len(snapshots) > cfg.max_rollback:
                oldest = min(snapshots)
                del snapshots[oldest]

        return self._finalize_beams(
            finished, full_ids, beam_scores, is_done, prompt_len,
            batch_size, num_beams, num_return, cfg, device,
        )

    def _snapshot(self, full_ids, beam_scores, past, is_done, attention_mask) -> tuple:
        return (
            full_ids.clone(),
            beam_scores.clone(),
            self._clone_cache(past),
            is_done.clone(),
            attention_mask.clone() if attention_mask is not None else None,
            [[r.clone() for r in h] for h in self._attn_history],
            [list(c) for c in self._penalty_columns],
            [list(t) for t in self._chosen_tokens],
        )

    def _restore(self, snap: tuple) -> tuple:
        (full_ids, beam_scores, past, is_done, attention_mask, history, columns, chosen) = snap
        self._attn_history = [[r.clone() for r in h] for h in history]
        self._penalty_columns = [list(c) for c in columns]
        self._chosen_tokens = [list(t) for t in chosen]
        return full_ids, beam_scores, past, is_done, attention_mask
