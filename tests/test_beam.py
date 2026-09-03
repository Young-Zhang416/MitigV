"""Tests for beam search in :mod:`mitigv.backends.generic`."""

from types import SimpleNamespace

import pytest
import torch

from mitigv import MitigatorConfigError
from mitigv.backends.generic import GenericMitigator


VOCAB = {0: "<pad>", 1: "a", 2: "b", 3: "<eos>"}


class BeamModel(torch.nn.Module):
    """Fake causal model with a reorderable cache and per-step logits.

    ``past_key_values`` is a (B, 1) tensor holding the step index, so cache
    reordering can be exercised (and recorded) without a real KV cache.
    """

    def __init__(self, step_logits, vocab_size):
        super().__init__()
        self.step_logits = [
            torch.as_tensor(v, dtype=torch.float32) for v in step_logits
        ]
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
            step = 0
            cache = torch.zeros(batch_size, 1, dtype=torch.long)
        else:
            step = int(past_key_values[0, 0].item()) + 1
            cache = torch.full((batch_size, 1), step, dtype=torch.long)
        vec = self.step_logits[min(step, len(self.step_logits) - 1)]
        logits = vec.view(1, 1, -1).expand(batch_size, seq_len, -1)
        return SimpleNamespace(logits=logits, past_key_values=cache)

    def _reorder_cache(self, past_key_values, beam_idx):
        self.reorder_calls.append(beam_idx.clone())
        return past_key_values.index_select(0, beam_idx)


class DummyProcessor:
    def __init__(self, id_to_token, eos_token_id=None, extra_inputs=None):
        self.id_to_token = dict(id_to_token)
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0
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
            chars = []
            for i in seq:
                i = int(i)
                if skip_special_tokens and (i == self.eos_token_id or i == 0):
                    continue
                chars.append(self.id_to_token.get(i, "<?"))
            out.append("".join(chars))
        return out


STEP_LOGITS = [
    torch.tensor([0.0, 2.0, 1.0, -10.0]),  # step 0: a=2 > b=1
    torch.tensor([0.0, 10.0, 20.0, -10.0]),
]  # step 1: b=20 > a=10


class TestBeamSearch:
    def test_returns_top_beams(self):
        model = BeamModel(STEP_LOGITS, 4)
        m = GenericMitigator(
            model,
            DummyProcessor(VOCAB),
            num_beams=2,
            num_return_sequences=2,
            max_new_tokens=2,
        )
        # top path: a->b (22); second: b->b (21)
        assert m(None, "a") == ["ab", "bb"]

    def test_single_return_sequence(self):
        model = BeamModel(STEP_LOGITS, 4)
        m = GenericMitigator(
            model,
            DummyProcessor(VOCAB),
            num_beams=2,
            num_return_sequences=1,
            max_new_tokens=2,
        )
        assert m(None, "a") == "ab"

    def test_eos_early_stopping(self):
        step_logits = [
            torch.tensor([0.0, 2.0, 1.0, -10.0]),
            torch.tensor([-10.0, -10.0, -10.0, 10.0]),
        ]  # eos dominates
        model = BeamModel(step_logits, 4)
        m = GenericMitigator(
            model,
            DummyProcessor(VOCAB, eos_token_id=3),
            num_beams=2,
            num_return_sequences=2,
            max_new_tokens=10,
        )
        out = m(None, "a")
        assert out == ["a", "b"]  # [a, eos] and [b, eos] -> eos stripped
        # early stopping: reorder once per completed step (2), not max_new_tokens
        assert len(model.reorder_calls) == 2

    def test_cache_reordering_is_invoked(self):
        model = BeamModel(STEP_LOGITS, 4)
        m = GenericMitigator(
            model,
            DummyProcessor(VOCAB),
            num_beams=2,
            num_return_sequences=2,
            max_new_tokens=2,
        )
        m(None, "a")
        assert len(model.reorder_calls) == 2
        assert model.reorder_calls[0].tolist() == [0, 0]  # both beams from beam 0
        assert model.reorder_calls[1].tolist() == [0, 1]

    def test_batch_beam_search(self):
        model = BeamModel(STEP_LOGITS, 4)
        m = GenericMitigator(
            model,
            DummyProcessor(VOCAB),
            num_beams=2,
            num_return_sequences=1,
            max_new_tokens=2,
        )
        assert m(None, ["a", "b"]) == ["ab", "ab"]

    def test_num_return_cannot_exceed_beams(self):
        model = BeamModel([torch.tensor([0.0, 0.0, 0.0, 0.0])], 4)
        with pytest.raises(MitigatorConfigError, match="num_return_sequences"):
            GenericMitigator(
                model, DummyProcessor(VOCAB), num_beams=1, num_return_sequences=2
            )

    def test_negative_length_penalty_raises(self):
        model = BeamModel([torch.tensor([0.0, 0.0, 0.0, 0.0])], 4)
        with pytest.raises(MitigatorConfigError, match="length_penalty"):
            GenericMitigator(model, DummyProcessor(VOCAB), num_beams=2, length_penalty=-1.0)

    def test_early_stopping_waits_for_full_finished_beam_set(self):
        step_logits = [
            torch.tensor([-10.0, 2.0, 1.0, 3.0]),
            torch.tensor([-10.0, -10.0, -10.0, 10.0]),
        ]
        model = BeamModel(step_logits, 4)
        m = GenericMitigator(
            model,
            DummyProcessor(VOCAB, eos_token_id=3),
            num_beams=2,
            num_return_sequences=1,
            max_new_tokens=5,
        )
        m(None, "a")
        assert len(model.reorder_calls) >= 2


class TestVCDBeamSearch:
    def test_aux_cache_is_reordered(self):
        from mitigv.algorithms.vcd import VCD

        class SpyVCD(VCD):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.aux_calls = []

            def _reorder_aux_cache(self, beam_idx):
                self.aux_calls.append(beam_idx.clone())
                super()._reorder_aux_cache(beam_idx)

        model = BeamModel(STEP_LOGITS, 4)
        processor = DummyProcessor(
            VOCAB, extra_inputs={"pixel_values": torch.zeros(1, 1)}
        )
        m = SpyVCD(
            model,
            processor,
            num_beams=2,
            num_return_sequences=2,
            max_new_tokens=2,
            alpha=1.0,
            beta=0.0,
            distortion="gaussian_noise",
            distortion_kwargs={"std": 0.0},
        )
        # identical branches (std=0, pixel-agnostic model) -> same as plain beam search
        assert m(None, "a") == ["ab", "bb"]
        assert len(m.aux_calls) == 2  # reordered after both steps

    def test_distorted_inputs_expanded_for_beams(self):
        from mitigv.algorithms.vcd import VCD

        model = BeamModel(STEP_LOGITS, 4)
        processor = DummyProcessor(
            VOCAB, extra_inputs={"pixel_values": torch.zeros(1, 1)}
        )
        m = VCD(
            model,
            processor,
            num_beams=3,
            distortion="gaussian_noise",
            distortion_kwargs={"std": 0.0},
        )
        inputs = m._prepare_inputs(None, "a", m.config)  # batch 1
        assert inputs["pixel_values"].shape[0] == 1
        m._expand_inputs_for_beams(inputs, 3)
        assert m._distorted_inputs["pixel_values"].shape[0] == 3
