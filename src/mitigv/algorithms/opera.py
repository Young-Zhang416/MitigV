"""OPERA — Over-trust Penalty and Retrospection-Allocation (Huang et al., CVPR 2024).

Beam-search decoding that detects the *knowledge aggregation pattern* (a
"columnar" attention map where many subsequent tokens over-trust one summary
token) and penalizes it, with a rollback ("retrospection") when the pattern
persists. At each step, a local window of the self-attention map is scaled by
``sigma`` and reduced by a column-wise product; its maximum is the over-trust
penalty added to the beam score. If the penalty's argmax column overlaps across
steps, that beam's candidates are suppressed (score-level retrospection).
"""

from __future__ import annotations

from typing import Any

import torch

from mitigv.backends.hf import HFMitigator, HFMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator

__all__ = ["OPERAConfig", "OPERA"]


class OPERAConfig(HFMitigatorConfig):
    """Hyper-parameters for OPERA.

    Attributes:
        num_beams: Beam width (OPERA is beam search; paper default 5).
        sigma: Attention scale factor (paper default 50).
        penalty_weight: Weight of the over-trust penalty on the beam score (paper 1).
        window_size: Local attention window (number of recent tokens).
        retrospection_length: Rollback length used by the paper (kept for reference;
            the implementation uses score-level suppression instead of a cache rollback).
        retrospection_streak: Consecutive identical argmax-penalty columns that
            trigger retrospection.
    """

    num_beams: int = 5
    sigma: float = 50.0
    penalty_weight: float = 1.0
    window_size: int = 5
    retrospection_length: int = 15
    retrospection_streak: int = 2

    def validate(self) -> None:
        super().validate()
        if self.num_beams < 1:
            raise MitigatorConfigError("num_beams must be >= 1")
        if self.sigma <= 0:
            raise MitigatorConfigError("sigma must be > 0")
        if self.penalty_weight < 0:
            raise MitigatorConfigError("penalty_weight must be >= 0")
        if self.window_size < 1:
            raise MitigatorConfigError("window_size must be >= 1")
        if self.retrospection_length < 1:
            raise MitigatorConfigError("retrospection_length must be >= 1")
        if self.retrospection_streak < 1:
            raise MitigatorConfigError("retrospection_streak must be >= 1")


@register_mitigator("opera")
class OPERA(HFMitigator):
    """Over-trust Penalty and Retrospection-Allocation beam search."""

    algorithm_name = "opera"
    config_class = OPERAConfig

    # -- attention -------------------------------------------------------------
    def _on_generate_start(self, cfg: OPERAConfig) -> None:
        self._force_eager_attention()
        layers = self._language_model_layers()
        self._attn_capture: torch.Tensor | None = None

        def hook(module: Any, args: Any, output: Any) -> Any:
            # eager attention returns (attn_output, attn_weights); keep only the
            # last layer's weights to avoid materializing all layers (memory).
            self._attn_capture = output[1].detach()
            return output

        self._attn_handle = layers[-1].self_attn.register_forward_hook(hook)
        self._attn_buffer: list[torch.Tensor] = []
        self._penalty_columns: list[list[int]] = []

    def _on_generate_end(self) -> None:
        if getattr(self, "_attn_handle", None) is not None:
            self._attn_handle.remove()
        self._attn_handle = None
        self._restore_attention_implementation()

    def _step_forward(self, input_ids, attention_mask, inputs, past_key_values):
        self._attn_capture = None
        logits, past = self._forward(input_ids, attention_mask, inputs, past_key_values)
        return logits, past, self._attn_capture

    def _attention_row(self, attn: torch.Tensor | None, kv_len: int) -> torch.Tensor:
        """Mean (over heads) attention of the newest query, padded to ``kv_len``."""
        if attn is None:
            return torch.zeros(1, kv_len, device=self.device)
        attn = attn.mean(dim=1)  # (B, q, kv) over heads
        row = attn[:, -1, :]  # (B, kv)
        if row.shape[-1] < kv_len:
            row = torch.nn.functional.pad(row, (0, kv_len - row.shape[-1]))
        return row[:, :kv_len]

    @staticmethod
    def _compute_penalty(window: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Column-wise product over a ``(B, k, k)`` attention window (log space).

        Returns ``(log_penalty (B,), argmax_col (B,))``.
        """
        window = (window * sigma).clamp(min=1e-12)
        k = window.shape[-1]
        log_col = torch.stack(
            [window[:, j:, j].log().sum(dim=1) for j in range(k)], dim=1
        )  # (B, k)
        return log_col.max(dim=-1)

    def _penalty_for_step(self, cfg: OPERAConfig, kv_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the over-trust penalty from the accumulated attention window."""
        n_beams = self._attn_buffer[0].shape[0] if self._attn_buffer else 1
        if not self._attn_buffer:
            return (torch.zeros(n_beams, device=self.device),
                    torch.zeros(n_beams, dtype=torch.long, device=self.device))
        k = min(cfg.window_size, len(self._attn_buffer), kv_len)
        rows = self._attn_buffer[-k:]
        cols = []
        for row in rows:
            c = row[:, -k:]
            if c.shape[-1] < k:
                c = torch.nn.functional.pad(c, (k - c.shape[-1], 0))
            cols.append(c)
        window = torch.stack(cols, dim=1)  # (B, k, k)
        return OPERA._compute_penalty(window, cfg.sigma)

    def _retrospection_mask(self, cfg: OPERAConfig, n_beams: int) -> torch.Tensor:
        """Bool mask over beams: True = keep, False = retrospection triggered."""
        mask = torch.ones(n_beams, dtype=torch.bool, device=self.device)
        streak = cfg.retrospection_streak
        for b in range(n_beams):
            cols = self._penalty_columns[b] if b < len(self._penalty_columns) else []
            if len(cols) >= streak and len(set(cols[-streak:])) == 1:
                mask[b] = False
        return mask

    # -- beam search -----------------------------------------------------------
    def _beam_search_loop(self, inputs: dict[str, Any], cfg: OPERAConfig) -> torch.Tensor:
        batch_size = inputs["input_ids"].shape[0]
        num_beams = cfg.num_beams
        num_return = cfg.num_return_sequences
        device = self.device
        eos = self._eos_token_id()
        prompt_len = inputs["input_ids"].shape[1]

        beam_inputs = self._expand_inputs_for_beams(inputs, num_beams)
        input_ids = beam_inputs["input_ids"]
        attention_mask = beam_inputs.get("attention_mask")
        full_ids = input_ids

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        is_done = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)
        finished: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(batch_size)]
        batch_offset = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
        past = None
        kv_len = prompt_len
        self._penalty_columns = [[] for _ in range(batch_size * num_beams)]

        step = 0
        while step < cfg.max_new_tokens:
            if cfg.early_stopping and bool(is_done.all()):
                break

            logits, past, attn = self._step_forward(
                input_ids, attention_mask, beam_inputs, past
            )
            row = self._attention_row(attn, kv_len)
            self._attn_buffer.append(row.detach())

            log_penalty, argmax_col = self._penalty_for_step(cfg, kv_len)
            n_beams = log_penalty.shape[0]
            for b in range(n_beams):
                self._penalty_columns[b].append(int(argmax_col[b].item()))

            vocab_size = logits.shape[-1]
            scores = (
                logits
                + beam_scores.unsqueeze(1)
                - cfg.penalty_weight * log_penalty.unsqueeze(1)
            )
            retro_ok = self._retrospection_mask(cfg, n_beams)
            scores = scores.masked_fill((~retro_ok).unsqueeze(1), -float("inf"))
            scores = scores.masked_fill(is_done.unsqueeze(1), -float("inf"))
            scores = scores.view(batch_size, num_beams * vocab_size)
            topk_scores, topk_idx = torch.topk(scores, num_beams, dim=1, largest=True, sorted=True)

            parent = torch.div(topk_idx, vocab_size, rounding_mode="floor")
            token = topk_idx % vocab_size
            beam_idx = (batch_offset + parent).view(-1)
            next_tokens = token.view(-1, 1)

            full_ids = torch.cat([full_ids[beam_idx], next_tokens], dim=-1)
            if attention_mask is not None:
                ones = torch.ones((batch_size * num_beams, 1), dtype=attention_mask.dtype, device=device)
                attention_mask = torch.cat([attention_mask[beam_idx], ones], dim=-1)

            past = self._reorder_cache(past, beam_idx)
            self._reorder_aux_cache(beam_idx)
            beam_scores = topk_scores.view(-1)
            is_done = is_done[beam_idx]
            self._penalty_columns = [self._penalty_columns[b] for b in beam_idx.tolist()]

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

            input_ids = next_tokens
            kv_len += 1
            step += 1

        return self._finalize_beams(
            finished, full_ids, beam_scores, is_done, prompt_len,
            batch_size, num_beams, num_return, cfg, device,
        )
