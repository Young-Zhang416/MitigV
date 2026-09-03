"""Tests for :mod:`mitigv.algorithms.agla`."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from mitigv import MitigatorConfigError, build_mitigator, list_mitigators
from mitigv.algorithms.agla import AGLA, AGLAConfig


VOCAB = {0: "<pad>", 1: "a", 2: "b"}


class VectorModel(torch.nn.Module):
    """Returns a logits vector keyed by the sum of ``pixel_values``."""

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


class AglaProcessor:
    def __init__(self, id_to_token, add_pixel=True):
        self.id_to_token = dict(id_to_token)
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.eos_token_id = None
        self.pad_token_id = 0
        self.add_pixel = add_pixel

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        if isinstance(text, str):
            text = [text]
        batch = [[self.token_to_id[ch] for ch in t] for t in text]
        inputs = {
            "input_ids": torch.tensor(batch, dtype=torch.long),
            "attention_mask": torch.ones((len(batch), len(batch[0])), dtype=torch.long),
        }
        if images is not None and self.add_pixel:
            inputs["pixel_values"] = torch.zeros(1, 1)
        return inputs

    def batch_decode(self, sequences, skip_special_tokens=True):
        out = []
        for seq in sequences:
            out.append("".join(self.id_to_token.get(int(i), "<?") for i in seq))
        return out


class TestAGLAConfig:
    def test_defaults(self):
        cfg = AGLAConfig()
        assert cfg.alpha == 1.0
        assert cfg.crop_ratio == 0.5

    def test_validation(self):
        with pytest.raises(MitigatorConfigError, match="alpha"):
            AGLAConfig(alpha=-0.1)
        with pytest.raises(MitigatorConfigError, match="crop_ratio"):
            AGLAConfig(crop_ratio=1.5)


class TestCrop:
    def test_crop_centered_on_uniform_saliency(self):
        img = Image.new("RGB", (100, 100))
        sal = torch.ones(576)  # uniform -> centroid at center (50, 50)
        out = AGLA._crop_one(None, img, sal, AGLAConfig(crop_ratio=0.5))
        # cropped box should be a 50x50 square centered at (50, 50)
        assert out.size == (50, 50)

    def test_edge_saliency_keeps_requested_crop_size(self):
        img = Image.new("RGB", (120, 100))
        sal = torch.zeros(4)
        sal[0] = 1.0
        out = AGLA._crop_one(None, img, sal, AGLAConfig(crop_ratio=0.5))
        assert out.size == (50, 50)


class TestAGLA:
    def test_registered_and_buildable(self):
        assert "agla" in list_mitigators()
        m = build_mitigator(
            "agla",
            VectorModel({0: torch.tensor([1.0, 0.0, 0.0])}, 3),
            AglaProcessor(VOCAB),
            max_new_tokens=1,
        )
        assert isinstance(m, AGLA)

    def test_step_logits_adds_local(self):
        model = VectorModel(
            {0: torch.tensor([1.0, 0.0, 0.0]), 1: torch.tensor([0.0, 1.0, 0.0])}, 3
        )
        agla = AGLA(model, AglaProcessor(VOCAB), alpha=2.0)

        inputs = {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
            "pixel_values": torch.zeros(1, 1),
        }
        agla._local_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": torch.ones(1, 1),
        }
        agla._local_past = None

        logits, _ = agla._step_logits(
            inputs["input_ids"], inputs["attention_mask"], inputs, None, 0, agla.config
        )
        # [1,0,0] + 2*[0,1,0] = [1,2,0]
        assert torch.allclose(logits, torch.tensor([[1.0, 2.0, 0.0]]))

    def test_image_token_span(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(image_token_index=7)
        agla = AGLA(model, AglaProcessor(VOCAB))
        inputs = {"input_ids": torch.tensor([[1, 7, 7, 2, 3]])}
        assert agla._image_token_span(inputs) == (1, 3)

    def test_single_placeholder_expands_to_model_image_sequence(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(image_token_index=7, image_seq_length=576)
        agla = AGLA(model, AglaProcessor(VOCAB))
        inputs = {"input_ids": torch.tensor([[1, 7, 2, 3]])}
        assert agla._image_token_span(inputs) == (1, 577)
