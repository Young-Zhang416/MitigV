"""Tests for :mod:`mitigv.algorithms.icd`."""

from types import SimpleNamespace

import pytest
import torch

from mitigv import MitigatorConfigError, build_mitigator, list_mitigators
from mitigv.algorithms.icd import ICD, ICDConfig


VOCAB = {0: "<pad>", 1: "a", 2: "b", 3: "P"}


class LengthModel(torch.nn.Module):
    """Returns logits chosen by the initial prompt length (stored in the cache)."""

    def __init__(self, logits_by_len, vocab_size):
        super().__init__()
        self.logits_by_len = {
            int(k): torch.as_tensor(v, dtype=torch.float32)
            for k, v in logits_by_len.items()
        }
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.reorder_calls = []

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape
        if past_key_values is None:
            key = seq_len
            cache = torch.full((batch_size, 1), key, dtype=torch.long)
        else:
            key = int(past_key_values[0, 0].item())
            cache = past_key_values
        vec = self.logits_by_len[key]
        logits = vec.view(1, 1, -1).expand(batch_size, seq_len, -1)
        return SimpleNamespace(logits=logits, past_key_values=cache)

    def _reorder_cache(self, past_key_values, beam_idx):
        self.reorder_calls.append(beam_idx.clone())
        return past_key_values.index_select(0, beam_idx)


class DummyProcessor:
    def __init__(self, id_to_token, extra_inputs=None):
        self.id_to_token = dict(id_to_token)
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.pad_token_id = 0
        self.eos_token_id = None
        self.extra_inputs = extra_inputs or {}

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        if isinstance(text, str):
            text = [text]
        batch = [[self.token_to_id[ch] for ch in t if ch != " "] for t in text]
        inputs = {
            "input_ids": torch.tensor(batch, dtype=torch.long),
            "attention_mask": torch.ones((len(batch), len(batch[0])), dtype=torch.long),
        }
        inputs.update(self.extra_inputs)
        return inputs

    def batch_decode(self, sequences, skip_special_tokens=True):
        out = []
        for seq in sequences:
            out.append("".join(self.id_to_token.get(int(i), "<?") for i in seq))
        return out


class TestICDConfig:
    def test_defaults(self):
        cfg = ICDConfig()
        assert cfg.lam == 1.0
        assert cfg.alpha == 0.1
        assert "confused object detector" in cfg.disturbance_prefix

    def test_validation(self):
        with pytest.raises(MitigatorConfigError, match="lam"):
            ICDConfig(lam=-1.0)
        with pytest.raises(MitigatorConfigError, match="alpha"):
            ICDConfig(alpha=-0.1)
        with pytest.raises(MitigatorConfigError, match="disturbance_prefix"):
            ICDConfig(disturbance_prefix="  ")


class TestDisturbPrompt:
    def test_insert_after_image(self):
        p = "SYS USER: <image>\nQuestion here? ASSISTANT:"
        out = ICD._disturb_prompt_with_prefix(p, "P")
        assert out == "SYS USER: <image>\nP Question here? ASSISTANT:"

    def test_no_image_prepends(self):
        out = ICD._disturb_prompt_with_prefix("question", "P")
        assert out == "P question"


class TestICD:
    def test_registered_and_buildable(self):
        assert "icd" in list_mitigators()
        m = build_mitigator(
            "icd",
            LengthModel({1: [0.0, 1.0, 0.0, 0.0]}, 4),
            DummyProcessor(VOCAB),
            max_new_tokens=1,
        )
        assert isinstance(m, ICD)

    def test_adaptive_plausibility(self):
        std = torch.tensor([[1.0, 3.0, 2.0]])  # max 3.0
        contrast = torch.tensor([[7.0, 8.0, 9.0]])
        out = ICD._adaptive_plausibility(contrast, std, alpha=0.5)
        # cutoff = log(0.5)+3 ~= 2.307 -> tokens 0 (1.0) and 2 (2.0) masked
        assert out[0, 1] == 8.0
        assert out[0, 0] == float("-inf")
        assert out[0, 2] == float("-inf")

    def test_step_logits_contrast_formula(self):
        model = LengthModel(
            {
                1: torch.tensor([0.0, 5.0, 0.0, 0.0]),
                2: torch.tensor([0.0, 0.0, 5.0, 0.0]),
            },
            4,
        )
        processor = DummyProcessor(
            VOCAB, extra_inputs={"pixel_values": torch.zeros(1, 1)}
        )
        icd = ICD(model, processor, lam=1.0, alpha=0.0, disturbance_prefix="P")

        std_ids = torch.tensor([[1]])  # len 1 -> [0,5,0,0]
        icd._disturbed_inputs = {
            "input_ids": torch.tensor([[3, 1]]),  # len 2 -> [0,0,5,0]
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "pixel_values": torch.zeros(1, 1),
        }
        icd._disturbed_attention_mask = torch.ones((1, 2), dtype=torch.long)
        icd._disturbed_past = None

        inputs = {
            "input_ids": std_ids,
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
            "pixel_values": torch.zeros(1, 1),
        }
        logits, _ = icd._step_logits(
            std_ids, inputs["attention_mask"], inputs, None, 0, icd.config
        )
        # [0,5,0,0] - 1*[0,0,5,0] = [0,5,-5,0]
        assert torch.allclose(logits, torch.tensor([[0.0, 5.0, -5.0, 0.0]]))

    def test_end_to_end_greedy(self):
        model = LengthModel(
            {
                1: torch.tensor([0.0, 5.0, 0.0, 0.0]),  # std -> "a"
                2: torch.tensor([0.0, 0.0, 5.0, 0.0]),
            },
            4,
        )  # dst -> "b"
        processor = DummyProcessor(VOCAB)
        icd = ICD(
            model,
            processor,
            lam=1.0,
            alpha=0.0,
            disturbance_prefix="P",
            max_new_tokens=1,
        )
        # contrast [0,5,0,0] - [0,0,5,0] -> argmax "a"
        assert icd(None, "a") == "a"

    def test_end_to_end_lam_zero_is_plain(self):
        model = LengthModel(
            {
                1: torch.tensor([0.0, 5.0, 0.0, 0.0]),
                2: torch.tensor([0.0, 0.0, 5.0, 0.0]),
            },
            4,
        )
        processor = DummyProcessor(VOCAB)
        icd = ICD(
            model,
            processor,
            lam=0.0,
            alpha=0.0,
            disturbance_prefix="P",
            max_new_tokens=1,
        )
        assert icd(None, "a") == "a"  # lam=0 -> standard branch only

    def test_beam_search(self):
        model = LengthModel(
            {
                1: torch.tensor([0.0, 5.0, 0.0, 0.0]),
                2: torch.tensor([0.0, 0.0, 5.0, 0.0]),
            },
            4,
        )
        processor = DummyProcessor(
            VOCAB, extra_inputs={"pixel_values": torch.zeros(1, 1)}
        )
        icd = ICD(
            model,
            processor,
            lam=1.0,
            alpha=0.0,
            disturbance_prefix="P",
            num_beams=2,
            num_return_sequences=2,
            max_new_tokens=2,
        )
        out = icd(None, "a")
        assert isinstance(out, list) and len(out) == 2

    def test_beam_aux_reorder(self):
        model = LengthModel(
            {
                1: torch.tensor([0.0, 5.0, 0.0, 0.0]),
                2: torch.tensor([0.0, 0.0, 5.0, 0.0]),
            },
            4,
        )

        class SpyICD(ICD):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.aux_calls = []

            def _reorder_aux_cache(self, beam_idx):
                self.aux_calls.append(beam_idx.clone())
                super()._reorder_aux_cache(beam_idx)

        processor = DummyProcessor(
            VOCAB, extra_inputs={"pixel_values": torch.zeros(1, 1)}
        )
        icd = SpyICD(
            model,
            processor,
            lam=1.0,
            alpha=0.0,
            disturbance_prefix="P",
            num_beams=2,
            num_return_sequences=2,
            max_new_tokens=2,
        )
        icd(None, "a")
        assert len(icd.aux_calls) == 2  # reordered after both steps
