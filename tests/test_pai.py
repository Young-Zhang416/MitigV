"""Tests for :mod:`mitigv.algorithms.pai`."""

from types import SimpleNamespace

import pytest
import torch

from mitigv import MitigatorConfigError, build_mitigator, list_mitigators
from mitigv.algorithms.pai import PAI, PAIConfig


VOCAB = {0: "<pad>", 1: "a", 2: "b"}


class PaiProcessor:
    """Processor that only injects ``pixel_values`` when images are supplied."""

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


class PaiModel(torch.nn.Module):
    """Returns one logits vector for the with-image branch and another for the
    text-only branch, keyed on whether ``pixel_values`` was passed."""

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


# --- fake Llama stack for the attention patch lifecycle --------------------

class FakeAttn:
    def __init__(self):
        self.forward = lambda *a, **k: (None, None)


class FakeLayer:
    def __init__(self):
        self.self_attn = FakeAttn()


class FakeLmModel:
    def __init__(self, n):
        self.layers = [FakeLayer() for _ in range(n)]


class FakeLm:
    def __init__(self, n):
        self.model = FakeLmModel(n)


class FakeVlm(torch.nn.Module):
    def __init__(self, n_layers=4):
        super().__init__()
        self.language_model = FakeLm(n_layers)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        b, l = input_ids.shape
        logits = torch.zeros(b, l, 2)
        return SimpleNamespace(logits=logits, past_key_values=None)


class TestPAIConfig:
    def test_defaults(self):
        cfg = PAIConfig()
        assert cfg.alpha == 0.2
        assert cfg.gamma == 1.1
        assert cfg.beta == 0.1
        assert cfg.start_layer == 2
        assert cfg.end_layer == 32
        assert cfg.num_image_tokens is None

    def test_validation(self):
        with pytest.raises(MitigatorConfigError, match="alpha"):
            PAIConfig(alpha=-0.1)
        with pytest.raises(MitigatorConfigError, match="gamma"):
            PAIConfig(gamma=0.5)
        with pytest.raises(MitigatorConfigError, match="beta"):
            PAIConfig(beta=-1.0)
        with pytest.raises(MitigatorConfigError, match="end_layer"):
            PAIConfig(start_layer=4, end_layer=2)
        with pytest.raises(MitigatorConfigError, match="num_image_tokens"):
            PAIConfig(num_image_tokens=0)


class TestRemoveImagePlaceholder:
    def test_removes_image_and_newline(self):
        p = "SYS USER: <image>\nQuestion? ASSISTANT:"
        assert PAI._remove_image_placeholder(p) == "SYS USER: Question? ASSISTANT:"

    def test_no_image_is_unchanged(self):
        assert PAI._remove_image_placeholder("USER: question ASSISTANT:") == "USER: question ASSISTANT:"


class TestAmplifyAttention:
    def test_amplifies_last_query_image_span(self):
        w = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0]]]],  # (B=1, H=1, q=1, kv=4)
        )
        out = PAI._amplify_attention(w, start=1, end=3, alpha=0.5)
        # span = [2, 3] -> |w|*0.5 + w = [3, 4.5]
        assert torch.allclose(out[0, 0, -1, 1], torch.tensor(3.0))
        assert torch.allclose(out[0, 0, -1, 2], torch.tensor(4.5))
        # untouched positions
        assert out[0, 0, -1, 0] == 1.0
        assert out[0, 0, -1, 3] == 4.0

    def test_alpha_zero_is_noop(self):
        w = torch.tensor([[[[1.0, -2.0, 3.0]]]])
        assert torch.equal(PAI._amplify_attention(w, 0, 3, 0.0), w)


class TestPAI:
    def test_registered_and_buildable(self):
        assert "pai" in list_mitigators()
        m = build_mitigator(
            "pai", PaiModel([1, 0, 0], [0, 1, 0], 3), PaiProcessor(VOCAB),
            alpha=0.0, max_new_tokens=1,
        )
        assert isinstance(m, PAI)

    def test_locate_image_span_multiple_tokens(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(image_token_index=7)
        pai = PAI(model, PaiProcessor(VOCAB), alpha=0.0, gamma=1.0)
        inputs = {"input_ids": torch.tensor([[1, 7, 7, 7, 2, 3]])}
        start, end = pai._locate_image_span(inputs, pai.config)
        assert (start, end) == (1, 4)

    def test_locate_image_span_single_token_widens(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(image_token_index=7)
        pai = PAI(model, PaiProcessor(VOCAB), alpha=0.0, gamma=1.0)
        inputs = {"input_ids": torch.tensor([[1, 7, 2, 3]])}
        start, end = pai._locate_image_span(inputs, pai.config.copy(num_image_tokens=576))
        assert (start, end) == (1, 577)

    def test_adaptive_plausibility(self):
        image = torch.tensor([[1.0, 3.0, 2.0]])  # max 3.0
        guided = torch.tensor([[7.0, 8.0, 9.0]])
        out = PAI._adaptive_plausibility(guided, image, beta=0.5)
        # cutoff = log(0.5)+3 ~= 2.307 -> tokens 0 (1.0) and 2 (2.0) masked
        assert out[0, 1] == 8.0
        assert out[0, 0] == float("-inf")
        assert out[0, 2] == float("-inf")

    def test_step_logits_cfg_formula(self):
        model = PaiModel([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 3)
        pai = PAI(model, PaiProcessor(VOCAB), alpha=0.0, gamma=2.0, beta=0.0)

        inputs = {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
            "pixel_values": torch.zeros(1, 1),
        }
        pai._uncond_inputs = {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }
        pai._uncond_attention_mask = torch.ones((1, 1), dtype=torch.long)
        pai._uncond_past = None

        logits, _ = pai._step_logits(
            inputs["input_ids"], inputs["attention_mask"], inputs, None, 0, pai.config
        )
        # 2*([1,0,0] - [0,1,0]) + [0,1,0] = [2,-1,0]
        assert torch.allclose(logits, torch.tensor([[2.0, -1.0, 0.0]]))

    def test_gamma_one_skips_text_branch(self):
        model = PaiModel([5.0, 0.0, 0.0], [0.0, 5.0, 0.0], 3)
        pai = PAI(model, PaiProcessor(VOCAB), alpha=0.0, gamma=1.0, beta=0.0)

        inputs = {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
            "pixel_values": torch.zeros(1, 1),
        }
        pai._uncond_inputs = {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }
        pai._uncond_attention_mask = torch.ones((1, 1), dtype=torch.long)
        pai._uncond_past = None

        logits, _ = pai._step_logits(
            inputs["input_ids"], inputs["attention_mask"], inputs, None, 0, pai.config
        )
        assert torch.allclose(logits, torch.tensor([[5.0, 0.0, 0.0]]))
        assert model.calls == [True]  # only the with-image branch ran

    def test_end_to_end_greedy_with_cfg(self):
        model = PaiModel([0.0, 5.0, 0.0], [0.0, 0.0, 5.0], 3)
        pai = PAI(model, PaiProcessor(VOCAB), alpha=0.0, gamma=2.0, beta=0.0,
                  max_new_tokens=1)
        # guided logits = 2*[0,5,-5] + [0,0,5] = [0,10,-5] -> argmax "a"
        assert pai(torch.zeros(1, 1), "a") == "a"
        assert model.calls == [True, False]  # cond then uncond

    def test_attention_patch_lifecycle(self):
        vlm = FakeVlm(n_layers=4)
        pai = PAI(vlm, PaiProcessor(VOCAB), alpha=0.2, gamma=1.0)
        cfg = pai.config.copy(start_layer=1, end_layer=3)

        layers = pai._attention_layers()
        originals = [layers[i].self_attn.forward for i in range(4)]

        pai._on_generate_start(cfg)
        # only layers [1, 3) are patched
        assert layers[1].self_attn.forward is not originals[1]
        assert layers[2].self_attn.forward is not originals[2]
        assert layers[0].self_attn.forward is originals[0]
        assert layers[3].self_attn.forward is originals[3]

        pai._on_generate_end()
        for i in range(4):
            assert layers[i].self_attn.forward is originals[i]

    def test_alpha_zero_does_not_patch(self):
        vlm = FakeVlm(n_layers=4)
        pai = PAI(vlm, PaiProcessor(VOCAB), alpha=0.0, gamma=1.0)
        layers = pai._attention_layers()
        originals = [layers[i].self_attn.forward for i in range(4)]

        pai._on_generate_start(pai.config)
        for i in range(4):
            assert layers[i].self_attn.forward is originals[i]

    def test_beam_search(self):
        model = PaiModel([5.0, 0.0, 0.0], [0.0, 5.0, 0.0], 3)
        pai = PAI(model, PaiProcessor(VOCAB), alpha=0.0, gamma=2.0, beta=0.0,
                  num_beams=2, num_return_sequences=2, max_new_tokens=2)
        out = pai(torch.zeros(1, 1), "a")
        assert isinstance(out, list) and len(out) == 2
        assert len(model.reorder_calls) >= 1

    def test_beam_aux_reorder(self):
        model = PaiModel([5.0, 0.0, 0.0], [0.0, 5.0, 0.0], 3)

        class SpyPAI(PAI):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.aux_calls = []

            def _reorder_aux_cache(self, beam_idx):
                self.aux_calls.append(beam_idx.clone())
                super()._reorder_aux_cache(beam_idx)

        pai = SpyPAI(model, PaiProcessor(VOCAB), alpha=0.0, gamma=2.0, beta=0.0,
                     num_beams=2, num_return_sequences=2, max_new_tokens=2)
        pai(torch.zeros(1, 1), "a")
        assert len(pai.aux_calls) == 2
