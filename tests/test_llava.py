from types import SimpleNamespace

import torch

from mitigv import load_mitigator
from mitigv.backends.llava import LlavaProcessorAdapter


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="llava")
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **kwargs):
        assert int((input_ids == 32000).sum()) > 0
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)
        logits[..., 1] = 1
        return SimpleNamespace(logits=logits, past_key_values=object())


class Processor:
    class _Tokenizer(SimpleNamespace):
        eos_token_id = None
        pad_token_id = 0

        def batch_decode(self, rows, **kwargs):
            return ["ok" for _ in rows]

    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        return "USER: <image>\\n" + messages[0]["content"][1]["text"] + " ASSISTANT:"

    def __call__(self, text, images=None, **kwargs):
        assert "<image>" in text
        return {
            "input_ids": torch.tensor([[1, 32000, 2]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 2, 2),
        }

def test_llava_auto_inserts_image_placeholder_for_plain_prompt():
    model = Model()
    processor = Processor()
    decoder = load_mitigator(
        "vcd", model=model, processor=processor, alpha=0, beta=0, max_new_tokens=1
    )
    assert isinstance(decoder.processor, LlavaProcessorAdapter)
    assert decoder("image", "Describe the image") == "ok"
