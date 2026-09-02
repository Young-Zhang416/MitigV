"""Tests for :mod:`mitigv.algorithms.m3id`."""

from types import SimpleNamespace

import math
import pytest
import torch

from mitigv import MitigatorConfigError, build_mitigator, list_mitigators
from mitigv.algorithms.m3id import M3ID, M3IDConfig


VOCAB = {0: "<pad>", 1: "a", 2: "b", 3: "c"}


class M3idProcessor:
    """Inject ``pixel_values`` only when images are supplied."""

    def __init__(self, id_to_token):
        self.id_to_token = dict(id_to_token)
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.eos_token_id = None
        self.pad_token_id = 0

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        if isinstance(text, str):
            text = [text]
        batch = [[self.token_to_id[ch] for ch in t] for t in text]
        inputs = {
            "input_ids": torch.tensor(batch, dtype=torch.long),
            "attention_mask": torch.ones((len(batch), len(batch[0])), dtype=torch.long),
        }
        if images is not None:
            inputs["pixel_values"] = torch.zeros(1, 1)
        return inputs

    def batch_decode(self, sequences, skip_special_tokens=True):
        out = []
        for seq in sequences:
            out.append("".join(self.id_to_token.get(int(i), "<?") for i in seq))
        return out


class CondUncondModel(torch.nn.Module):
    """Cond (pixel_values present) vs uncond (absent) return different vectors."""

    def __init__(self, cond_vec, uncond_vec, vocab_size):
        super().__init__()
        self.cond_vec = torch.as_tensor(cond_vec, dtype=torch.float32)
        self.uncond_vec = torch.as_tensor(uncond_vec, dtype=torch.float32)
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.calls = []
        self.reorder_calls = []

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                use_cache=True, return_dict=True, **kwargs):
        self.calls.append("pixel_values" in kwargs)
        vec = self.cond_vec if "pixel_values" in kwargs else self.uncond_vec
        b, l = input_ids.shape
        logits = vec.view(1, 1, -1).expand(b, l, -1)
        step = 0 if past_key_values is None else int(past_key_values[0, 0].item()) + 1
        cache = torch.full((b, 1), step, dtype=torch.long)
        return SimpleNamespace(logits=logits, past_key_values=cache)

    def _reorder_cache(self, past_key_values, beam_idx):
        self.reorder_calls.append(beam_idx.clone())
        return past_key_values.index_select(0, beam_idx)


class TestM3IDConfig:
    def test_defaults(self):
        cfg = M3IDConfig()
        assert cfg.alpha == 0.3
        assert cfg.forgetting_rate == 0.02

    def test_validation(self):
        with pytest.raises(MitigatorConfigError, match="alpha"):
            M3IDConfig(alpha=1.5)
        with pytest.raises(MitigatorConfigError, match="alpha"):
            M3IDConfig(alpha=0.0)
        with pytest.raises(MitigatorConfigError, match="forgetting_rate"):
            M3IDConfig(forgetting_rate=-0.1)


class TestStepWeight:
    def test_zero_at_step_zero(self):
        assert M3ID._step_weight(0, 0.02) == 0.0

    def test_grows_with_step(self):
        w0 = M3ID._step_weight(0, 0.02)
        w1 = M3ID._step_weight(1, 0.02)
        w5 = M3ID._step_weight(5, 0.02)
        assert w0 < w1 < w5

    def test_zero_rate_is_always_zero(self):
        assert M3ID._step_weight(10, 0.0) == 0.0


class TestM3ID:
    def test_registered_and_buildable(self):
        assert "m3id" in list_mitigators()
        m = build_mitigator(
            "m3id", CondUncondModel([1, 0, 0, 0], [0, 1, 0, 0], 4),
            M3idProcessor(VOCAB), max_new_tokens=1,
        )
        assert isinstance(m, M3ID)

    def test_step_logits_contrast_when_gated(self):
        # forget_rate = ln(2) -> step 1 weight == 1.0
        model = CondUncondModel([0, 0, 0, 0], [0, 0, 1, 0], 4)
        m3id = M3ID(model, M3idProcessor(VOCAB), alpha=0.3,
                    forgetting_rate=math.log(2.0))

        inputs = {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
            "pixel_values": torch.zeros(1, 1),
        }
        m3id._uncond_inputs = {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }
        m3id._uncond_attention_mask = torch.ones((1, 1), dtype=torch.long)
        m3id._uncond_past = None

        # step 1: uniform cond (top-1=0.25 < 0.3) -> gate on, weight=1
        logits, _ = m3id._step_logits(
            inputs["input_ids"], inputs["attention_mask"], inputs, None, 1, m3id.config
        )
        # 2*l_c - l_u = 2*[0,0,0,0] - [0,0,1,0] = [0,0,-1,0]
        assert torch.allclose(logits, torch.tensor([[0.0, 0.0, -1.0, 0.0]]))

    def test_step_logits_no_contrast_when_confident(self):
        model = CondUncondModel([5, 0, 0, 0], [0, 0, 1, 0], 4)
        m3id = M3ID(model, M3idProcessor(VOCAB), alpha=0.3,
                    forgetting_rate=math.log(2.0))

        inputs = {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
            "pixel_values": torch.zeros(1, 1),
        }
        m3id._uncond_inputs = {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }
        m3id._uncond_attention_mask = torch.ones((1, 1), dtype=torch.long)
        m3id._uncond_past = None

        # top-1 ~ 0.98 > 0.3 -> gate off, return conditional logits unchanged
        logits, _ = m3id._step_logits(
            inputs["input_ids"], inputs["attention_mask"], inputs, None, 1, m3id.config
        )
        assert torch.allclose(logits, torch.tensor([[5.0, 0.0, 0.0, 0.0]]))

    def test_zero_forgetting_rate_skips_uncond(self):
        # identical branches -> deterministic output; only the call count matters.
        model = CondUncondModel([0, 5, 0, 0], [0, 5, 0, 0], 4)
        m3id = M3ID(model, M3idProcessor(VOCAB), forgetting_rate=0.0, max_new_tokens=2)
        assert m3id(torch.zeros(1, 1), "a") == "aa"
        assert len(model.calls) == 2  # 2 steps x 1 conditional branch (uncond skipped)

    def test_end_to_end_calls_both_after_step_zero(self):
        model = CondUncondModel([0, 5, 0, 0], [0, 0, 5, 0], 4)
        m3id = M3ID(model, M3idProcessor(VOCAB), alpha=0.3,
                    forgetting_rate=math.log(2.0), max_new_tokens=2)
        m3id(torch.zeros(1, 1), "a")
        # step 0 (weight 0): 1 cond call; step 1 (weight 1): cond + uncond = 3 total
        assert len(model.calls) == 3

    def test_beam_search_and_aux_reorder(self):
        model = CondUncondModel([5, 0, 0, 0], [0, 0, 5, 0], 4)

        class SpyM3ID(M3ID):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.aux_calls = []

            def _reorder_aux_cache(self, beam_idx):
                self.aux_calls.append(beam_idx.clone())
                super()._reorder_aux_cache(beam_idx)

        m3id = SpyM3ID(model, M3idProcessor(VOCAB), alpha=0.3,
                       forgetting_rate=math.log(2.0),
                       num_beams=2, num_return_sequences=2, max_new_tokens=2)
        out = m3id(torch.zeros(1, 1), "a")
        assert isinstance(out, list) and len(out) == 2
        assert len(m3id.aux_calls) >= 1
