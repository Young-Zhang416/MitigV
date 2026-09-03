"""Tests for :mod:`mitigv.algorithms.probe_steer`."""

from types import SimpleNamespace

import pytest
import torch

from mitigv import MitigatorConfigError, build_mitigator, list_mitigators
from mitigv.algorithms.probe_steer import LinearProbeSteer, LinearProbeSteerConfig


VOCAB = {0: "<pad>", 1: "a", 2: "b"}


class IdentityLayer(torch.nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class TupleLayer(torch.nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states, "cache"


class FakeLm(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([IdentityLayer() for _ in range(n_layers)])


class FakeVlm(torch.nn.Module):
    def __init__(self, n_layers=4, hidden_dim=2, vocab_size=3):
        super().__init__()
        self.language_model = FakeLm(n_layers)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        h = torch.zeros(batch_size, seq_len, self.hidden_dim)
        for layer in self.language_model.layers:
            h = layer(h)
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        logits[..., 1] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=0)


class ProbeProcessor:
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
        out = []
        for seq in sequences:
            out.append("".join(self.id_to_token.get(int(i), "<?") for i in seq))
        return out


class TestConfig:
    def test_defaults(self):
        cfg = LinearProbeSteerConfig()
        assert cfg.beta == 5.0
        assert cfg.layer == 16

    def test_validation(self):
        with pytest.raises(MitigatorConfigError, match="beta"):
            LinearProbeSteerConfig(beta=-1.0)
        with pytest.raises(MitigatorConfigError, match="layer"):
            LinearProbeSteerConfig(layer=-1)


class TestLinearProbeSteer:
    def test_registered_and_buildable(self):
        assert "linear_probe_steer" in list_mitigators()
        m = build_mitigator(
            "linear_probe_steer", FakeVlm(), ProbeProcessor(VOCAB), max_new_tokens=1
        )
        assert isinstance(m, LinearProbeSteer)

    def test_injects_normalized_vector_at_single_layer(self):
        model = FakeVlm(n_layers=4, hidden_dim=2, vocab_size=3)
        # steering vector [3, 4] -> unit [0.6, 0.8]
        steer = LinearProbeSteer(
            model,
            ProbeProcessor(VOCAB),
            steering_vector=torch.tensor([3.0, 4.0]),
            beta=10.0,
            layer=1,
        )
        layers = steer._language_model_layers()
        steer._on_generate_start(steer.config)

        out = layers[1](torch.zeros(1, 1, 2))
        # 10 * [0.6, 0.8] = [6, 8]
        assert torch.allclose(out, torch.tensor([[[6.0, 8.0]]]))

        # other layers are untouched
        out0 = layers[0](torch.zeros(1, 1, 2))
        assert torch.equal(out0, torch.zeros(1, 1, 2))

        steer._on_generate_end()

    def test_zero_beta_no_hooks(self):
        model = FakeVlm(n_layers=4, hidden_dim=2, vocab_size=3)
        steer = LinearProbeSteer(
            model,
            ProbeProcessor(VOCAB),
            steering_vector=torch.tensor([3.0, 4.0]),
            beta=0.0,
            layer=1,
        )
        steer._on_generate_start(steer.config)
        assert steer._steer_hooks == []

    def test_tuple_layer_output_is_preserved(self):
        model = FakeVlm(n_layers=1, hidden_dim=2, vocab_size=3)
        model.language_model.layers[0] = TupleLayer()
        steer = LinearProbeSteer(
            model,
            ProbeProcessor(VOCAB),
            steering_vector=torch.tensor([3.0, 4.0]),
            beta=10.0,
            layer=0,
        )
        steer._on_generate_start(steer.config)
        try:
            hidden, cache = model.language_model.layers[0](torch.zeros(1, 1, 2))
            assert torch.allclose(hidden, torch.tensor([[[6.0, 8.0]]]))
            assert cache == "cache"
        finally:
            steer._on_generate_end()

    @pytest.mark.parametrize(
        "vector, message",
        [
            (torch.zeros(2), "non-zero"),
            (torch.tensor([1.0, float("nan")]), "finite"),
            (torch.ones(1, 2), "shape"),
        ],
    )
    def test_invalid_steering_vector_raises(self, vector, message):
        steer = LinearProbeSteer(
            FakeVlm(n_layers=1),
            ProbeProcessor(VOCAB),
            steering_vector=vector,
            beta=1.0,
            layer=0,
        )
        with pytest.raises(MitigatorConfigError, match=message):
            steer._on_generate_start(steer.config)

    def test_layer_out_of_range_raises(self):
        model = FakeVlm(n_layers=4, hidden_dim=2, vocab_size=3)
        steer = LinearProbeSteer(
            model,
            ProbeProcessor(VOCAB),
            steering_vector=torch.tensor([1.0, 0.0]),
            beta=1.0,
            layer=10,
        )
        with pytest.raises(MitigatorConfigError, match="out of range"):
            steer._on_generate_start(steer.config)

    def test_end_to_end_runs(self):
        model = FakeVlm(n_layers=4, hidden_dim=2, vocab_size=3)
        steer = LinearProbeSteer(
            model,
            ProbeProcessor(VOCAB),
            steering_vector=torch.tensor([1.0, 0.0]),
            beta=2.0,
            layer=1,
            max_new_tokens=2,
        )
        assert steer(None, "a") == "aa"
