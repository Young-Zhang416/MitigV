"""Tests for :mod:`mitigv.algorithms.vista`."""

from types import SimpleNamespace

import pytest
import torch

from mitigv import MitigatorConfigError, build_mitigator, list_mitigators
from mitigv.algorithms.vista import VISTA, VISTAConfig


VOCAB = {0: "<pad>", 1: "a", 2: "b", 3: "c"}


class IdentityLayer(torch.nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class FakeLm(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([IdentityLayer() for _ in range(n_layers)])


class FakeVlm(torch.nn.Module):
    """Residual stream is 2.0 with an image and 0.0 without; predicts token 1."""

    def __init__(self, n_layers=3, hidden_dim=2, vocab_size=4):
        super().__init__()
        self.language_model = FakeLm(n_layers)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                use_cache=True, return_dict=True, output_hidden_states=False, **kwargs):
        b, l = input_ids.shape
        base = 2.0 if pixel_values is not None else 0.0
        h = torch.full((b, l, self.hidden_dim), base)
        hidden_states = [h]
        for layer in self.language_model.layers:
            h = layer(h)
            hidden_states.append(h)
        logits = torch.zeros(b, l, self.vocab_size)
        logits[..., 1] = 1.0
        out = SimpleNamespace(logits=logits, past_key_values=0)
        if output_hidden_states:
            out.hidden_states = tuple(hidden_states)
        return out


class VistaProcessor:
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


class TestVISTAConfig:
    def test_defaults(self):
        cfg = VISTAConfig()
        assert cfg.steer_strength == 0.01

    def test_validation(self):
        with pytest.raises(MitigatorConfigError, match="steer_strength"):
            VISTAConfig(steer_strength=-0.1)


class TestRemoveImagePlaceholder:
    def test_removes_image_and_newline(self):
        p = "SYS USER: <image>\nQ? ASSISTANT:"
        assert VISTA._remove_image_placeholder(p) == "SYS USER: Q? ASSISTANT:"


class TestSteeringVectors:
    def test_compute_steering_vectors(self):
        model = FakeVlm(n_layers=3, hidden_dim=2, vocab_size=4)
        vista = VISTA(model, VistaProcessor(VOCAB), steer_strength=0.01)

        inputs = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
            "pixel_values": torch.zeros(1, 1),
        }
        vista._uncond_inputs = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }
        vectors = vista._compute_steering_vectors(inputs)
        assert len(vectors) == 3  # one per layer
        for v in vectors:
            assert torch.allclose(v, torch.full((1, 2), 2.0))  # 2.0 - 0.0


class TestInjection:
    def test_hooks_inject_and_remove(self):
        model = FakeVlm(n_layers=1, hidden_dim=2, vocab_size=4)
        vista = VISTA(model, VistaProcessor(VOCAB), steer_strength=0.01)
        vista._steer_vectors = [torch.tensor([[2.0, 2.0]])]

        layer = vista._language_model_layers()[0]
        vista._on_generate_start(vista.config)
        out = layer(torch.zeros(1, 1, 2))
        assert torch.allclose(out, torch.full((1, 1, 2), 0.02))  # 0 + 0.01 * 2.0

        vista._on_generate_end()
        out = layer(torch.zeros(1, 1, 2))
        assert torch.equal(out, torch.zeros(1, 1, 2))  # hook removed


class TestVISTA:
    def test_registered_and_buildable(self):
        assert "vista" in list_mitigators()
        m = build_mitigator("vista", FakeVlm(), VistaProcessor(VOCAB), max_new_tokens=1)
        assert isinstance(m, VISTA)

    def test_end_to_end_runs(self):
        model = FakeVlm(n_layers=3, hidden_dim=2, vocab_size=4)
        vista = VISTA(model, VistaProcessor(VOCAB), steer_strength=0.01, max_new_tokens=2)
        assert vista(torch.zeros(1, 1), "a") == "aa"

    def test_zero_strength_no_hooks(self):
        model = FakeVlm(n_layers=3, hidden_dim=2, vocab_size=4)
        vista = VISTA(model, VistaProcessor(VOCAB), steer_strength=0.0, max_new_tokens=1)
        vista._steer_vectors = [torch.tensor([[2.0, 2.0]])]
        vista._on_generate_start(vista.config)
        assert vista._steer_hooks == []
