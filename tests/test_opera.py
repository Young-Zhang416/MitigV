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
        assert cfg.window_size == 5
        assert cfg.retrospection_length == 15

    def test_validation(self):
        with pytest.raises(MitigatorConfigError, match="num_beams"):
            OPERAConfig(num_beams=0)
        with pytest.raises(MitigatorConfigError, match="sigma"):
            OPERAConfig(sigma=0.0)
        with pytest.raises(MitigatorConfigError, match="window_size"):
            OPERAConfig(window_size=0)
        with pytest.raises(MitigatorConfigError, match="retrospection_streak"):
            OPERAConfig(retrospection_streak=0)


class TestPenalty:
    def test_columnar_column_detected(self):
        # 2x2 window: column 0 has a columnar pattern (both rows attend to it).
        window = torch.tensor([[
            [0.5, 0.0],
            [0.5, 0.1],
        ]])
        log_penalty, argmax = OPERA._compute_penalty(window, sigma=2.0)
        # col 0: (2*0.5)*(2*0.5)=1.0 -> log 0; col 1: (2*0.1)=0.2 -> log -1.6
        assert argmax.item() == 0
        assert torch.allclose(log_penalty, torch.tensor([0.0]), atol=1e-5)

    def test_penalty_grows_with_sigma(self):
        window = torch.tensor([[
            [0.5, 0.0],
            [0.5, 0.1],
        ]])
        p1, _ = OPERA._compute_penalty(window, sigma=1.0)
        p2, _ = OPERA._compute_penalty(window, sigma=2.0)
        assert p2.item() > p1.item()


class TestRetrospection:
    def test_streak_triggers(self):
        opera = OPERA(torch.nn.Module(), DummyProcessor())
        opera._penalty_columns = [[3, 3], [1, 2]]
        mask = opera._retrospection_mask(OPERAConfig(retrospection_streak=2), 2)
        assert mask.tolist() == [False, True]

    def test_no_streak_keeps_all(self):
        opera = OPERA(torch.nn.Module(), DummyProcessor())
        opera._penalty_columns = [[3, 1], [1, 2]]
        mask = opera._retrospection_mask(OPERAConfig(retrospection_streak=2), 2)
        assert mask.tolist() == [True, True]


class TestOPERA:
    def test_registered_and_buildable(self):
        assert "opera" in list_mitigators()
        m = build_mitigator("opera", torch.nn.Module(), DummyProcessor(), max_new_tokens=1)
        assert isinstance(m, OPERA)
