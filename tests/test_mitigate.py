"""Tests for the :func:`mitigv.mitigate` context manager."""

from types import SimpleNamespace

import pytest
import torch

from mitigv import load_mitigator, mitigate
from mitigv.algorithms.vcd import VCD, VCDConfig
from mitigv.backends.qwen2_5_vl import Qwen2_5VLProcessorAdapter


VOCAB = {0: "<pad>", 1: "a", 2: "b"}


class ScriptedModel(torch.nn.Module):
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


class DummyProcessor:
    def __init__(self, id_to_token):
        self.id_to_token = dict(id_to_token)
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.pad_token_id = 0
        self.eos_token_id = None

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        if isinstance(text, str):
            text = [text]
        batch = [[self.token_to_id.get(ch, 0) for ch in t] for t in text]
        return {
            "input_ids": torch.tensor(batch, dtype=torch.long),
            "attention_mask": torch.ones((len(batch), len(batch[0])), dtype=torch.long),
        }

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [
            "".join(self.id_to_token.get(int(i), "<?") for i in seq)
            for seq in sequences
        ]


class QwenLikeProcessor(DummyProcessor):
    def __init__(self, id_to_token):
        super().__init__(id_to_token)
        self.templated_text = None

    def apply_chat_template(self, messages, **kwargs):
        self.templated_text = messages
        return "<|vision_start|><|image_pad|><|vision_end|>" + messages[0]["content"][1]["text"]


class TestMitigate:
    def test_load_mitigator_is_one_step_api_for_loaded_objects(self):
        decoder = load_mitigator(
            "vcd",
            model=ScriptedModel([1], 3),
            processor=DummyProcessor(VOCAB),
            alpha=0.0,
            beta=0.0,
            max_new_tokens=1,
        )
        assert isinstance(decoder, VCD)
        assert decoder(None, "a") == "a"

    def test_load_mitigator_auto_adapts_loaded_qwen_objects(self):
        model = ScriptedModel([1], 3)
        model.config = SimpleNamespace(model_type="qwen2_5_vl")
        processor = QwenLikeProcessor(VOCAB)
        decoder = load_mitigator(
            "vcd",
            model=model,
            processor=processor,
            alpha=0.0,
            beta=0.0,
            max_new_tokens=1,
        )
        assert isinstance(decoder.processor, Qwen2_5VLProcessorAdapter)
        assert decoder("image", "a") == "a"
        assert processor.templated_text[0]["content"][0]["type"] == "image"

    def test_load_mitigator_requires_a_complete_model_source(self):
        with pytest.raises(ValueError, match="provided together"):
            load_mitigator("vcd", model=ScriptedModel([1], 3))
        with pytest.raises(ValueError, match="model_type and model_id"):
            load_mitigator("vcd", model_type="llava")

    def test_load_mitigator_loads_checkpoint_and_algorithm_in_one_call(
        self, monkeypatch
    ):
        import mitigv.backends.factory as factory

        calls = []

        def fake_load(model_type, model_id, **kwargs):
            calls.append((model_type, model_id, kwargs))
            return ScriptedModel([1], 3), DummyProcessor(VOCAB)

        monkeypatch.setattr(factory, "load_vision_language", fake_load)
        decoder = load_mitigator(
            "vcd",
            model_type="llava",
            model_id="local/checkpoint",
            model_kwargs={"torch_dtype": "auto"},
            alpha=0.0,
            beta=0.0,
            max_new_tokens=1,
        )
        assert decoder(None, "a") == "a"
        assert calls == [
            (
                "llava",
                "local/checkpoint",
                {"model_kwargs": {"torch_dtype": "auto"}, "processor_kwargs": None},
            )
        ]

    def test_builds_and_yields_callable(self):
        with mitigate(
            "vcd",
            VCDConfig(alpha=0.0, beta=0.0, max_new_tokens=1),
            model=ScriptedModel([1], 3),
            processor=DummyProcessor(VOCAB),
        ) as f:
            assert isinstance(f, VCD)
            assert f(None, "a") == "a"

    def test_kwargs_as_config_overrides(self):
        with mitigate(
            "vcd",
            model=ScriptedModel([1], 3),
            processor=DummyProcessor(VOCAB),
            alpha=3.0,
            beta=0.0,
            max_new_tokens=1,
        ) as f:
            assert f.config.alpha == 3.0

    def test_accepts_class_instead_of_name(self):
        with mitigate(
            VCD,
            VCDConfig(max_new_tokens=1),
            model=ScriptedModel([1], 3),
            processor=DummyProcessor(VOCAB),
        ) as f:
            assert isinstance(f, VCD)

    def test_rejects_non_mitigator_class(self):
        with pytest.raises(TypeError, match="BaseMitigator"):
            with mitigate(
                int, model=ScriptedModel([1], 3), processor=DummyProcessor(VOCAB)
            ):
                pass

    def test_cleanup_frees_cuda_cache(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))

        with mitigate(
            "vcd",
            model=ScriptedModel([1], 3),
            processor=DummyProcessor(VOCAB),
            max_new_tokens=1,
        ):
            pass
        assert calls == [1]

    def test_cleanup_can_be_disabled(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))

        with mitigate(
            "vcd",
            model=ScriptedModel([1], 3),
            processor=DummyProcessor(VOCAB),
            cleanup=False,
        ):
            pass
        assert calls == []

    def test_import_mitigv_stays_torch_free(self):
        # mitigate must not force torch to be imported just by importing mitigv.
        import os
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = os.path.join(repo_root, "src")
        code = "import mitigv, sys; print('torch' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": src},
        )
        assert out.stdout.strip() == "False"
