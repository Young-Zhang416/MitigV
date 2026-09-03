"""Tests for :mod:`mitigv.backends.generic`."""

from types import SimpleNamespace

import pytest
import torch

from mitigv import MitigatorConfigError
from mitigv.backends.generic import GenericMitigator, GenericMitigatorConfig


# ---------------------------------------------------------------------------
# Fakes — a scripted causal model and a minimal processor/tokenizer.
# ---------------------------------------------------------------------------


class ScriptedModel(torch.nn.Module):
    """Emits a scripted token sequence; optionally records each forward call."""

    def __init__(self, script, vocab_size, record=False):
        super().__init__()
        self.script = list(script)
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))  # for device inference
        self.calls = [] if record else None

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
        if self.calls is not None:
            self.calls.append((sorted(kwargs.keys()), past_key_values))
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size), -1e9, dtype=torch.float32
        )
        logits[..., token] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=step)


class SoftModel(torch.nn.Module):
    """Returns a fixed (non-degenerate) logits vector regardless of input."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits
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
        batch_size, seq_len = input_ids.shape
        logits = self.logits.view(1, 1, -1).expand(batch_size, seq_len, -1)
        return SimpleNamespace(logits=logits, past_key_values=0)


class DummyProcessor:
    """Minimal processor/tokenizer double with a char->id vocabulary."""

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
            chars = []
            for token in seq:
                token = int(token)
                if skip_special_tokens and token == self.eos_token_id:
                    continue
                chars.append(self.id_to_token.get(token, "<?"))
            out.append("".join(chars))
        return out


VOCAB = {0: "<pad>", 1: "a", 2: "b", 3: "X", 4: "Y", 5: "<eos>"}


def make_mitigator(script, vocab_size=6, id_to_token=None, eos_token_id=None, **cfg):
    model = ScriptedModel(script, vocab_size)
    processor = DummyProcessor(id_to_token or VOCAB, eos_token_id=eos_token_id)
    return GenericMitigator(model, processor, **cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenericMitigator:
    def test_is_base_subclass_and_concrete(self):
        from mitigv import BaseMitigator

        assert issubclass(GenericMitigator, BaseMitigator)
        m = make_mitigator([1], max_new_tokens=1)
        assert isinstance(m, BaseMitigator)

    def test_default_config_class(self):
        m = make_mitigator([1])
        assert isinstance(m.config, GenericMitigatorConfig)

    def test_top_k_requires_an_integer(self):
        with pytest.raises(MitigatorConfigError, match="top_k"):
            GenericMitigatorConfig(top_k=1.5)

    def test_greedy_generation(self):
        m = make_mitigator(script=[3, 4], max_new_tokens=2)
        assert m(None, "ab") == "XY"

    def test_stops_at_eos(self):
        m = make_mitigator(script=[1, 5, 1], eos_token_id=5, max_new_tokens=10)
        assert m(None, "a") == "a"

    def test_no_eos_runs_full_length(self):
        m = make_mitigator(script=[1, 2, 1, 2], max_new_tokens=4)
        assert m(None, "a") == "abab"

    def test_max_new_tokens_bounds(self):
        m = make_mitigator(
            script=[1] * 10,
            vocab_size=2,
            id_to_token={0: "<pad>", 1: "a"},
            max_new_tokens=3,
        )
        assert m(None, "a") == "aaa"

    def test_generate_kwargs_override_config(self):
        m = make_mitigator(script=[1, 2, 1, 2], max_new_tokens=100)
        assert m(None, "a", max_new_tokens=2) == "ab"

    def test_unknown_generate_kwarg_raises(self):
        m = make_mitigator([1])
        with pytest.raises(MitigatorConfigError, match="unknown configuration key"):
            m(None, "a", bogus=1)

    def test_requires_model_and_processor(self):
        with pytest.raises(RuntimeError, match="requires a model"):
            GenericMitigator()(None, "hi")

    def test_device_inferred_from_model(self):
        m = make_mitigator([1])
        assert m.device == torch.device("cpu")

    def test_prepare_inputs_casts_float_to_model_dtype(self):
        class HalfModel(ScriptedModel):
            def __init__(self):
                super().__init__([1], 6)
                self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))

        model = HalfModel()
        processor = DummyProcessor(
            VOCAB, extra_inputs={"pixel_values": torch.zeros(1, 3)}
        )
        m = GenericMitigator(model, processor)
        inputs = m._prepare_inputs(None, "ab", m.config)
        assert inputs["pixel_values"].dtype == torch.float16
        assert inputs["input_ids"].dtype == torch.long

    def test_cache_and_visual_inputs_first_step_only(self):
        model = ScriptedModel([3, 4], vocab_size=6, record=True)
        processor = DummyProcessor(
            VOCAB, extra_inputs={"pixel_values": torch.zeros(1, 3)}
        )
        m = GenericMitigator(model, processor, max_new_tokens=2)

        assert m(None, "ab") == "XY"
        assert len(model.calls) == 2

        keys0, past0 = model.calls[0]
        keys1, past1 = model.calls[1]
        assert "pixel_values" in keys0 and past0 is None
        assert "pixel_values" not in keys1 and past1 == 0

    def test_step_logits_hook_is_used(self):
        class BanToken(GenericMitigator):
            algorithm_name = "ban"

            def _step_logits(self, input_ids, attention_mask, inputs, past, step, cfg):
                logits, past = super()._step_logits(
                    input_ids, attention_mask, inputs, past, step, cfg
                )
                logits = logits.clone()
                logits[:, 3] = -1e9  # ban token 3
                return logits, past

        model = ScriptedModel([3, 3], vocab_size=6)
        processor = DummyProcessor(VOCAB)
        m = BanToken(model, processor, max_new_tokens=2)

        # script predicts 3; banned -> argmax over all -1e9 ties -> token 0
        assert m(None, "ab") == "<pad><pad>"

    def test_sampling_reproducible_with_seed(self):
        logits = torch.tensor([0.1, 2.0, 0.3])  # soft, vocab=3

        def run(seed):
            model = SoftModel(logits)
            processor = DummyProcessor({0: "<pad>", 1: "a", 2: "b"})
            m = GenericMitigator(
                model, processor, do_sample=True, seed=seed, max_new_tokens=3
            )
            return m(None, "a")

        assert run(7) == run(7)

    def test_top_k_sampling_restricts_tokens(self):
        logits = torch.tensor([0.5, 3.0, 1.0])  # token 1 is the argmax
        model = SoftModel(logits)
        processor = DummyProcessor({0: "<pad>", 1: "a", 2: "b"})
        m = GenericMitigator(
            model, processor, do_sample=True, top_k=1, seed=0, max_new_tokens=3
        )
        assert m(None, "a") == "aaa"

    def test_batch_returns_list(self):
        # Batch of two prompts -> list of two strings.
        class BatchProcessor(DummyProcessor):
            def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
                inputs = super().__call__(
                    text=["a", "b"], return_tensors=return_tensors, **kwargs
                )
                return inputs

        model = ScriptedModel([3, 4], vocab_size=6)
        processor = BatchProcessor(VOCAB)
        m = GenericMitigator(model, processor, max_new_tokens=2)
        out = m(None, "ignored")
        assert out == ["XY", "XY"]

    def test_lazy_import_from_package(self):
        import mitigv

        assert mitigv.GenericMitigator is GenericMitigator
        assert mitigv.GenericMitigatorConfig is GenericMitigatorConfig

    def test_generate_lifecycle_hooks_called_in_order(self):
        events = []

        class Hooked(GenericMitigator):
            def _on_generate_start(self, cfg):
                events.append("start")

            def _on_generate_end(self):
                events.append("end")

        m = Hooked(ScriptedModel([1], 6), DummyProcessor(VOCAB), max_new_tokens=1)
        assert m(None, "a") == "a"
        assert events == ["start", "end"]

    def test_generate_lifecycle_end_runs_on_error(self):
        events = []

        class Boom(GenericMitigator):
            def _step_logits(self, *args, **kwargs):
                raise RuntimeError("boom")

            def _on_generate_end(self):
                events.append("end")

        m = Boom(ScriptedModel([1], 6), DummyProcessor(VOCAB), max_new_tokens=1)
        with pytest.raises(RuntimeError, match="boom"):
            m(None, "a")
        assert events == ["end"]

    def test_lifecycle_cleanup_runs_when_start_hook_fails(self):
        events = []

        class BrokenStart(GenericMitigator):
            def _on_generate_start(self, cfg):
                events.append("start")
                raise RuntimeError("partial patch failed")

            def _on_generate_end(self):
                events.append("end")

        m = BrokenStart(ScriptedModel([1], 6), DummyProcessor(VOCAB))
        with pytest.raises(RuntimeError, match="partial patch"):
            m(None, "a")
        assert events == ["start", "end"]

    def test_restores_model_training_mode(self):
        model = ScriptedModel([1], 6)
        model.train()
        m = GenericMitigator(model, DummyProcessor(VOCAB), max_new_tokens=1)
        assert m(None, "a") == "a"
        assert model.training is True

    def test_seeded_generation_restores_global_rng_state(self):
        torch.manual_seed(123)
        _ = torch.rand(1)
        expected = torch.rand(1)

        torch.manual_seed(123)
        _ = torch.rand(1)
        m = GenericMitigator(
            SoftModel(torch.tensor([0.0, 1.0, 2.0])),
            DummyProcessor({0: "x", 1: "a", 2: "b"}),
            do_sample=True,
            seed=999,
            max_new_tokens=2,
        )
        m(None, "a")
        assert torch.equal(torch.rand(1), expected)

    def test_unseeded_sampling_advances_global_rng_state(self):
        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()
        m = GenericMitigator(
            SoftModel(torch.tensor([0.0, 1.0, 2.0])),
            DummyProcessor({0: "x", 1: "a", 2: "b"}),
            do_sample=True,
            max_new_tokens=2,
        )
        m(None, "a")
        assert not torch.equal(torch.random.get_rng_state(), before)

    def test_left_pads_batched_token_inputs(self):
        inputs = {
            "input_ids": torch.tensor([[1, 0, 0], [1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 0, 0], [1, 1, 1]]),
            "token_type_ids": torch.tensor([[4, 0, 0], [4, 5, 6]]),
            "pixel_values": torch.zeros(2, 3, 4, 4),
        }
        out = GenericMitigator._left_pad_token_inputs(inputs)
        assert out["input_ids"].tolist() == [[0, 0, 1], [1, 2, 3]]
        assert out["attention_mask"].tolist() == [[0, 0, 1], [1, 1, 1]]
        assert out["token_type_ids"].tolist() == [[0, 0, 4], [4, 5, 6]]
        assert out["pixel_values"] is inputs["pixel_values"]

    def test_each_batch_row_stops_at_its_own_eos(self):
        class PerRowEosModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, input_ids=None, past_key_values=None, **kwargs):
                step = 0 if past_key_values is None else past_key_values + 1
                logits = torch.full((2, input_ids.shape[1], 6), -100.0)
                if step == 0:
                    logits[0, :, 5] = 0.0  # row 0 finishes immediately
                    logits[1, :, 1] = 0.0
                else:
                    logits[0, :, 2] = 0.0  # must be suppressed after EOS
                    logits[1, :, 5] = 0.0
                return SimpleNamespace(logits=logits, past_key_values=step)

        class BatchProcessor(DummyProcessor):
            def __call__(self, **kwargs):
                return {
                    "input_ids": torch.tensor([[1], [1]]),
                    "attention_mask": torch.ones(2, 1, dtype=torch.long),
                }

        m = GenericMitigator(
            PerRowEosModel(), BatchProcessor(VOCAB, eos_token_id=5), max_new_tokens=4
        )
        assert m(None, ["a", "a"]) == ["", "a"]

    @pytest.mark.parametrize(
        "logits",
        [
            torch.tensor([[0.0, float("nan")]]),
            torch.tensor([[0.0, float("inf")]]),
            torch.tensor([[float("-inf"), float("-inf")]]),
        ],
    )
    def test_rejects_invalid_next_token_distribution(self, logits):
        m = make_mitigator([1])
        with pytest.raises(RuntimeError, match="invalid next-token distribution"):
            m._sample_next(logits, m.config)

    def test_multi_token_generation_requires_cache(self):
        class NoCacheModel(ScriptedModel):
            def forward(self, **kwargs):
                output = super().forward(**kwargs)
                output.past_key_values = None
                return output

        m = GenericMitigator(
            NoCacheModel([1, 2], 6), DummyProcessor(VOCAB), max_new_tokens=2
        )
        with pytest.raises(RuntimeError, match="requires a cache"):
            m(None, "a")
