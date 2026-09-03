"""Tests for :mod:`mitigv.algorithms.only`."""

from types import SimpleNamespace

import pytest
import torch

from mitigv import MitigatorConfigError, build_mitigator, list_mitigators
from mitigv.algorithms.only import ONLY, ONLYConfig


VOCAB = {0: "<pad>", 1: "a", 2: "b"}


class DummyProcessor:
    def __init__(self, id_to_token):
        self.id_to_token = dict(id_to_token)
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.eos_token_id = None
        self.pad_token_id = 0

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        if isinstance(text, str):
            text = [text]
        batch = [[self.token_to_id[ch] for ch in t] for t in text]
        return {
            "input_ids": torch.tensor(batch, dtype=torch.long),
            "attention_mask": torch.ones((len(batch), len(batch[0])), dtype=torch.long),
        }

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [
            "".join(self.id_to_token.get(int(i), "<?") for i in s) for s in sequences
        ]


class TestConfig:
    def test_defaults(self):
        cfg = ONLYConfig()
        assert cfg.layer == 0
        assert cfg.alpha1 == 3.0
        assert cfg.alpha2 == 1.0
        assert cfg.gamma == 0.2

    def test_validation(self):
        with pytest.raises(MitigatorConfigError, match="layer"):
            ONLYConfig(layer=-1)
        with pytest.raises(MitigatorConfigError, match="alpha"):
            ONLYConfig(alpha1=-0.1)
        with pytest.raises(MitigatorConfigError, match="gamma"):
            ONLYConfig(gamma=-0.1)


class TestImageTokenSpan:
    def test_span(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(image_token_index=7)
        only = ONLY(model, DummyProcessor(VOCAB))
        inputs = {"input_ids": torch.tensor([[1, 7, 7, 7, 2]])}
        assert only._image_token_span(inputs) == (1, 4)


class TestSelectedMask:
    def test_low_tver_head_deactivated(self):
        # 2 heads, 1 query, 5 keys; image tokens at positions [1, 3).
        attn = torch.tensor(
            [
                [
                    [[0.05, 0.80, 0.05, 0.05, 0.05]],  # high text/image entropy ratio
                    [[0.70, 0.10, 0.10, 0.05, 0.05]],  # lower ratio
                ]
            ]
        )
        model = torch.nn.Module()
        model.config = SimpleNamespace(image_token_index=7)
        only = ONLY(model, DummyProcessor(VOCAB))
        only._img_start, only._img_end = 1, 3
        mask = only._compute_selected_mask(attn, only.config)
        assert mask.tolist() == [[False, True]]  # head 1 deactivated


class TestFuse:
    def test_collaborative_when_close(self):
        f = torch.tensor([[2.0, 1.0, 0.0]])
        ft = torch.tensor([[2.0, 1.0, 0.0]])  # identical -> d=0 < gamma
        out = ONLY._fuse(f, ft, alpha1=3.0, alpha2=1.0, gamma=0.2)
        assert torch.allclose(out, f + 3.0 * ft)  # [8, 4, 0]

    def test_contrastive_when_far(self):
        f = torch.tensor([[5.0, 0.0, 0.0]])
        ft = torch.tensor([[0.0, 0.0, 5.0]])  # very different -> d large
        out = ONLY._fuse(f, ft, alpha1=3.0, alpha2=1.0, gamma=0.2)
        assert torch.allclose(out, (1.0 + 1.0) * f - 1.0 * ft)  # [10, 0, -5]


class TestONLY:
    def test_registered_and_buildable(self):
        assert "only" in list_mitigators()
        m = build_mitigator(
            "only", torch.nn.Module(), DummyProcessor(VOCAB), max_new_tokens=1
        )
        assert isinstance(m, ONLY)
