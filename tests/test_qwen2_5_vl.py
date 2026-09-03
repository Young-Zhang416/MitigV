import torch
from types import SimpleNamespace

from mitigv.backends.generic import GenericMitigator
from mitigv.backends.qwen2_5_vl import (
    Qwen2_5VLModelAdapter,
    Qwen2_5VLProcessorAdapter,
    adapt_qwen2_5_vl,
)


class Model:
    device = "cpu"
    dtype = torch.float32

    def __call__(self, **kwargs):
        ids = kwargs["input_ids"]
        logits = torch.full((ids.shape[0], ids.shape[1], 4), -100.0)
        logits[..., 3] = 0
        return SimpleNamespace(logits=logits, past_key_values=object())


class Tokenizer:
    eos_token_id = None
    pad_token_id = 0

    def batch_decode(self, rows, **kwargs):
        return ["ok" for _ in rows]


class Processor:
    tokenizer = Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["content"][0]["type"] == "image"
        return "<|vision_start|><|image_pad|><|vision_end|>Describe"

    def __call__(self, **kwargs):
        assert "<|vision_start|>" in kwargs["text"]
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }


def test_qwen_adapter_formats_chat_and_decodes():
    model, processor = adapt_qwen2_5_vl(Model(), Processor())
    assert isinstance(model, Qwen2_5VLModelAdapter)
    assert isinstance(processor, Qwen2_5VLProcessorAdapter)
    assert GenericMitigator(model, processor, max_new_tokens=1)("image", "Describe") == "ok"


def test_qwen_model_forwards_rope_inputs():
    seen = {}

    class Recording(Model):
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return super().__call__(**kwargs)

    adapter = Qwen2_5VLModelAdapter(Recording())
    ids = torch.ones(1, 1, dtype=torch.long)
    adapter(input_ids=ids, image_grid_thw=torch.ones(1, 3, dtype=torch.long))
    assert "image_grid_thw" in seen
