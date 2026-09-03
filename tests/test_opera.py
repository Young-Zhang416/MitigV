"""Tests for :mod:`mitigv.algorithms.opera`."""

import pytest
import torch

from mitigv import MitigatorConfigError, build_mitigator, list_mitigators
from mitigv.algorithms.opera import OPERA, OPERAConfig


class DummyProcessor:
    def __init__(self):
        self.eos_token_id = None
        self.pad_token_id = 0

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["".join(str(int(i)) for i in s) for s in sequences]


class TestConfig:
    def test_defaults(self):
        cfg = OPERAConfig()
        assert cfg.num_beams == 5
        assert cfg.sigma == 50.0
        assert cfg.penalty_weight == 1.0
        assert cfg.num_attn_candidates == 5
        assert cfg.window_size == 5
        assert cfg.threshold == 15
        assert cfg.retrospection_window == 20
        assert cfg.max_rollback == 30

    def test_validation(self):
        with pytest.raises(MitigatorConfigError, match="num_beams"):
            OPERAConfig(num_beams=1)
        with pytest.raises(MitigatorConfigError, match="sigma"):
            OPERAConfig(sigma=0.0)
        with pytest.raises(MitigatorConfigError, match="window_size"):
            OPERAConfig(window_size=0)
        with pytest.raises(MitigatorConfigError, match="threshold"):
            OPERAConfig(threshold=0)
        with pytest.raises(MitigatorConfigError, match="retrospection_window"):
            OPERAConfig(retrospection_window=10)  # <= threshold (15)


class TestProcessAttention:
    def test_max_over_heads_and_renormalize(self):
        # (B=1, H=2, q=1, kv=2)
        attn = torch.tensor([[[[0.5, 0.5]], [[1.0, 0.0]]]])
        out = OPERA._process_attention(attn)
        # max over heads -> [1.0, 0.5]; renormalize -> [2/3, 1/3]
        assert torch.allclose(out, torch.tensor([[[2 / 3, 1 / 3]]]), atol=1e-6)


class TestPenalty:
    def test_columnar_column_detected(self):
        # One row (k=1): penalty is sigma * the max column value.
        row = torch.tensor([[0.4, 0.6, 0.0]])  # (1, kv=3)
        penalty, argmax = OPERA._compute_penalty([row], kv_len=3, sigma=2.0)
        # k=1 -> single column product = sigma * row[:, -1:] = 2*0.0 -> argmax col 2
        assert argmax.item() == 2

    def test_penalty_grows_with_sigma(self):
        rows = [torch.tensor([[0.4, 0.6, 0.5]])]
        p1, _ = OPERA._compute_penalty(rows, 3, sigma=1.0)
        p2, _ = OPERA._compute_penalty(rows, 3, sigma=2.0)
        assert p2.item() > p1.item()

    def test_two_row_column_product(self):
        # rows aligned to kv_len=2; window k=2
        rows = [
            torch.tensor([[0.5, 0.0]]),  # token t-1 attends to col 0
            torch.tensor([[0.5, 0.5]]),  # token t attends to cols 0,1
        ]
        penalty, argmax = OPERA._compute_penalty(rows, kv_len=2, sigma=1.0)
        # col 0: 0.5 * 0.5 = 0.25; col 1: 0.5 -> argmax col 1
        assert argmax.item() == 1
        assert torch.allclose(penalty, torch.tensor([0.5]), atol=1e-6)


class TestDetectRetrospection:
    def test_mode_reaches_threshold(self):
        opera = OPERA(torch.nn.Module(), DummyProcessor())
        opera._response_start = 10
        # summary token at abs position 12 repeats 15 times
        opera._penalty_columns = [[12] * 15 + [13, 14]]
        opera._chosen_tokens = [[99] * 20]
        rb = opera._detect_retrospection(
            OPERAConfig(threshold=15, retrospection_window=20)
        )
        assert rb == (
            0,
            3,
            99,
        )  # beam 0, target_len = 12-10+1 = 3, exclude chosen[3]=99

    def test_below_threshold_no_trigger(self):
        opera = OPERA(torch.nn.Module(), DummyProcessor())
        opera._response_start = 10
        opera._penalty_columns = [[12] * 14 + [13, 14]]
        opera._chosen_tokens = [[99] * 20]
        assert (
            opera._detect_retrospection(
                OPERAConfig(threshold=15, retrospection_window=20)
            )
            is None
        )


class TestOPERA:
    def test_registered_and_buildable(self):
        assert "opera" in list_mitigators()
        m = build_mitigator(
            "opera", torch.nn.Module(), DummyProcessor(), max_new_tokens=1
        )
        assert isinstance(m, OPERA)

    def test_num_beams_one_rejected(self):
        with pytest.raises(MitigatorConfigError, match="num_beams"):
            OPERAConfig(num_beams=1)

    def test_batch_is_rejected_instead_of_mixing_beam_groups(self):
        opera = OPERA(torch.nn.Module(), DummyProcessor())
        inputs = {
            "input_ids": torch.ones(2, 2, dtype=torch.long),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
        }
        with pytest.raises(NotImplementedError, match="batch_size=1"):
            opera._beam_search_loop(inputs, opera.config)

    def test_reads_sequence_length_from_legacy_cache(self):
        opera = OPERA(torch.nn.Module(), DummyProcessor())
        key = torch.zeros(2, 4, 7, 8)
        value = torch.zeros_like(key)
        assert opera._kv_len(((key, value),), fallback=99) == 7

    def test_selected_penalty_columns_are_persisted(self):
        class ScriptedOPERA(OPERA):
            def _step_forward(self, input_ids, attention_mask, inputs, past_key_values):
                batch_size = input_ids.shape[0]
                kv_len = attention_mask.shape[1]
                logits = torch.tensor([0.0, 3.0, 2.0, 1.0]).repeat(batch_size, 1)
                key = torch.zeros(batch_size, 1, kv_len, 1)
                cache = ((key, key.clone()),)
                attn = torch.full((batch_size, 1, 1, kv_len), 1.0 / kv_len)
                return logits, cache, attn

            def _lookahead_row(self, beam_idx, token, past, kv_len, device):
                row = torch.zeros(kv_len + 1, device=device)
                row[-1] = 1.0
                return row

        opera = ScriptedOPERA(
            torch.nn.Module(),
            DummyProcessor(),
            num_beams=2,
            num_attn_candidates=2,
            max_new_tokens=1,
        )
        inputs = opera._prepare_inputs(None, "ignored", opera.config)
        opera._beam_search_loop(inputs, opera.config)
        assert len(opera._penalty_columns) == 2
        assert all(len(columns) == 1 for columns in opera._penalty_columns)
