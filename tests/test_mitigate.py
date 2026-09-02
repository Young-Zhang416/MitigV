"""Tests for the :func:`mitigv.mitigate` context manager."""

from types import SimpleNamespace

import pytest
import torch

from mitigv import mitigate
from mitigv.algorithms.vcd import VCD, VCDConfig


VOCAB = {0: "<pad>", 1: "a", 2: "b"}


class ScriptedModel(torch.nn.Module):
    def __init__(self, script, vocab_size):
        super().__init__()
        self.script = list(script)
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                use_cache=True, return_dict=True, **kwargs):
        step = 0 if past_key_values is None else past_key_values + 1
        token = self.script[min(step, len(self.script) - 1)]
        b, l = input_ids.shape
        logits = torch.full((b, l, self.vocab_size), -1e9, dtype=torch.float32)
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
        return ["".join(self.id_to_token.get(int(i), "<?") for i in seq) for seq in sequences]


class TestMitigate:
    def test_builds_and_yields_callable(self):
        with mitigate("vcd", VCDConfig(alpha=1.0, beta=0.0, max_new_tokens=1),
                  model=ScriptedModel([1], 3), processor=DummyProcessor(VOCAB),
                 ) as f:
            assert isinstance(f, VCD)
            assert f(None, "a") == "a"

    def test_kwargs_as_config_overrides(self):
        with mitigate("vcd", model=ScriptedModel([1], 3), processor=DummyProcessor(VOCAB),
                  alpha=3.0, beta=0.0, max_new_tokens=1,
                 ) as f:
            assert f.config.alpha == 3.0

    def test_accepts_class_instead_of_name(self):
        with mitigate(VCD, VCDConfig(max_new_tokens=1), model=ScriptedModel([1], 3),
                  processor=DummyProcessor(VOCAB)) as f:
            assert isinstance(f, VCD)

    def test_rejects_non_mitigator_class(self):
        with pytest.raises(TypeError, match="BaseMitigator"):
            with mitigate(int, model=ScriptedModel([1], 3), processor=DummyProcessor(VOCAB)):
                pass

    def test_cleanup_frees_cuda_cache(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))

        with mitigate("vcd", model=ScriptedModel([1], 3), processor=DummyProcessor(VOCAB),
                  max_new_tokens=1):
            pass
        assert calls == [1]

    def test_cleanup_can_be_disabled(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))

        with mitigate("vcd", model=ScriptedModel([1], 3), processor=DummyProcessor(VOCAB),
                  cleanup=False):
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
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": src},
        )
        assert out.stdout.strip() == "False"
