"""Tests for :mod:`mitigv.algorithms.vcd`."""

from types import SimpleNamespace

import pytest
import torch

from mitigv import MitigatorConfigError, build_mitigator, list_mitigators
from mitigv.algorithms.vcd import VCD, VCDConfig


VOCAB = {0: "<pad>", 1: "a", 2: "b", 3: "X", 4: "Y", 5: "<eos>"}


class DummyProcessor:
    def __init__(self, id_to_token, eos_token_id=None, extra_inputs=None):
        self.id_to_token = dict(id_to_token)
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.eos_token_id = eos_token_id
        self.extra_inputs = extra_inputs or {}

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        if isinstance(text, str):
            text = [text]
        batch = [[self.token_to_id.get(ch, 0) for ch in t] for t in text]
        inputs = {
            "input_ids": torch.tensor(batch, dtype=torch.long),
            "attention_mask": torch.ones((len(batch), len(batch[0])), dtype=torch.long),
        }
        inputs.update(self.extra_inputs)
        return inputs

    def batch_decode(self, sequences, skip_special_tokens=True):
        out = []
        for seq in sequences:
            chars = [
                self.id_to_token.get(int(i), "<?")
                for i in seq
                if not (skip_special_tokens and int(i) == self.eos_token_id)
            ]
            out.append("".join(chars))
        return out


class ScriptedModel(torch.nn.Module):
    """Ignores pixel_values and emits a scripted token sequence."""

    def __init__(self, script, vocab_size):
        super().__init__()
        self.script = list(script)
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        **kwargs,
    ):
        step = 0 if past_key_values is None else past_key_values + 1
        token = self.script[min(step, len(self.script) - 1)]
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size), -1e9, dtype=torch.float32
        )
        logits[..., token] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=step)


class VectorModel(torch.nn.Module):
    """Returns a fixed logits vector chosen by the sum of ``pixel_values``."""

    def __init__(self, logits_map, vocab_size):
        super().__init__()
        self.logits_map = logits_map
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        **kwargs,
    ):
        pv = kwargs.get("pixel_values")
        key = 0 if pv is None else int(pv.sum().item())
        vec = self.logits_map[key]
        batch_size, seq_len = input_ids.shape
        logits = vec.view(1, 1, -1).expand(batch_size, seq_len, -1)
        return SimpleNamespace(logits=logits, past_key_values=0)


def make_vcd(**cfg):
    model = ScriptedModel([3, 4], vocab_size=6)
    processor = DummyProcessor(VOCAB, extra_inputs={"pixel_values": torch.zeros(1, 1)})
    return VCD(model, processor, **cfg)


class TestVCD:
    def test_registered_and_buildable(self):
        assert "vcd" in list_mitigators()
        m = build_mitigator(
            "vcd",
            ScriptedModel([1], 6),
            DummyProcessor(VOCAB, extra_inputs={"pixel_values": torch.zeros(1, 1)}),
            max_new_tokens=1,
        )
        assert isinstance(m, VCD)

    def test_default_config(self):
        vcd = make_vcd()
        assert isinstance(vcd.config, VCDConfig)
        assert vcd.config.alpha == 1.0
        assert vcd.config.beta == 0.1
        assert vcd.config.distortion == "diffusion_noise"

    def test_config_validation(self):
        with pytest.raises(MitigatorConfigError, match="alpha"):
            VCDConfig(alpha=-1.0)
        with pytest.raises(MitigatorConfigError, match="beta"):
            VCDConfig(beta=-0.5)
        with pytest.raises(MitigatorConfigError, match="distortion_kwargs"):
            VCDConfig(distortion_kwargs=[])

    def test_end_to_end_identical_branches_cancel(self):
        # ScriptedModel ignores pixel_values, so both branches agree and the
        # contrast cancels -> same output as vanilla decoding.
        vcd = make_vcd(alpha=1.0, beta=0.0, max_new_tokens=2)
        assert vcd(None, "ab") == "XY"

    def test_distorts_pixel_values_only(self):
        torch.manual_seed(0)
        vcd = VCD(
            ScriptedModel([3], 6),
            DummyProcessor(VOCAB, extra_inputs={"pixel_values": torch.zeros(1, 1)}),
            distortion="gaussian_noise",
            distortion_kwargs={"std": 1.0},
        )
        inputs = vcd._prepare_inputs(None, "ab", vcd.config)
        assert not torch.equal(
            inputs["pixel_values"], vcd._distorted_inputs["pixel_values"]
        )
        assert torch.equal(inputs["input_ids"], vcd._distorted_inputs["input_ids"])

    def test_contrast_formula(self):
        model = VectorModel(
            {0: torch.tensor([1.0, 0.0, 0.0]), 1: torch.tensor([0.0, 1.0, 0.0])},
            vocab_size=3,
        )
        processor = DummyProcessor(
            {0: "<pad>", 1: "a", 2: "b"},
            extra_inputs={"pixel_values": torch.zeros(1, 1)},
        )
        vcd = VCD(model, processor, alpha=2.0, beta=0.0)

        inputs = {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
            "pixel_values": torch.zeros(1, 1),
        }
        vcd._distorted_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": torch.ones(1, 1),
        }
        vcd._distorted_past = None

        logits, _ = vcd._step_logits(
            inputs["input_ids"], inputs["attention_mask"], inputs, None, 0, vcd.config
        )
        expected = torch.tensor([[3.0, -2.0, 0.0]])  # (1+2)*[1,0,0] - 2*[0,1,0]
        assert torch.allclose(logits, expected)

    def test_adaptive_plausibility_masks(self):
        original = torch.tensor([[1.0, 3.0, 2.0]])  # max = 3.0
        contrast = torch.tensor([[7.0, 8.0, 9.0]])
        out = VCD._adaptive_plausibility(contrast, original, beta=0.5)
        # cutoff = log(0.5) + 3.0 ~= 2.307 -> tokens 0 (1.0) and 2 (2.0) masked.
        assert out[0, 1] == 8.0
        assert out[0, 0] == float("-inf")
        assert out[0, 2] == float("-inf")

    def test_adaptive_plausibility_beta_zero_is_noop(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        out = VCD._adaptive_plausibility(logits, logits, beta=0.0)
        assert torch.equal(out, logits)

    def test_step_logits_applies_apc_on_original_branch(self):
        model = VectorModel(
            {0: torch.tensor([0.0, 5.0, 0.0]), 1: torch.tensor([10.0, 0.0, 10.0])},
            vocab_size=3,
        )
        processor = DummyProcessor(
            {0: "<pad>", 1: "a", 2: "b"},
            extra_inputs={"pixel_values": torch.zeros(1, 1)},
        )
        vcd = VCD(model, processor, alpha=1.0, beta=0.1)

        inputs = {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
            "pixel_values": torch.zeros(1, 1),
        }
        vcd._distorted_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": torch.ones(1, 1),
        }
        vcd._distorted_past = None

        logits, _ = vcd._step_logits(
            inputs["input_ids"], inputs["attention_mask"], inputs, None, 0, vcd.config
        )
        # diffs = 2*[0,5,0] - 1*[10,0,10] = [-10, 10, -10]
        # APC on ORIGINAL [0,5,0]: cutoff = log(0.1)+5 ~= 2.7 -> tokens 0,2 masked.
        expected = torch.tensor([[float("-inf"), 10.0, float("-inf")]])
        assert torch.allclose(logits, expected, equal_nan=True)
