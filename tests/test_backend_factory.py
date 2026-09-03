import torch

from mitigv.backends.factory import (
    adapt_vision_language,
    available_model_families,
    get_model_adapter,
    get_processor_adapter,
)
from mitigv.backends.hf_common import (
    VisionLanguageModelAdapter,
    VisionLanguageProcessorAdapter,
)
from mitigv.backends.llava import LlavaModelAdapter, LlavaProcessorAdapter
from mitigv.backends.qwen2_5_vl import Qwen2_5VLModelAdapter, Qwen2_5VLProcessorAdapter


class Model:
    def __call__(self, **kwargs):
        return {"logits": torch.zeros(1, 1, 2), "past_key_values": None}


class Processor:
    def __call__(self, **kwargs):
        return {"input_ids": torch.ones(1, 1, dtype=torch.long)}

    def batch_decode(self, rows, **kwargs):
        return ["ok" for _ in rows]


def test_adapters_share_common_parent_and_factory_selects_family():
    assert issubclass(LlavaModelAdapter, VisionLanguageModelAdapter)
    assert issubclass(Qwen2_5VLModelAdapter, VisionLanguageModelAdapter)
    assert issubclass(LlavaProcessorAdapter, VisionLanguageProcessorAdapter)
    assert issubclass(Qwen2_5VLProcessorAdapter, VisionLanguageProcessorAdapter)
    assert available_model_families() == ("llava", "qwen2.5-vl")
    assert get_model_adapter("qwen") is Qwen2_5VLModelAdapter
    assert get_processor_adapter("llava-next") is LlavaProcessorAdapter


def test_factory_instantiates_selected_adapters():
    model, processor = adapt_vision_language("qwen2.5-vl", Model(), Processor())
    assert isinstance(model, Qwen2_5VLModelAdapter)
    assert isinstance(processor, Qwen2_5VLProcessorAdapter)


def test_factory_rejects_unknown_model_type():
    try:
        get_model_adapter("unknown")
    except ValueError as error:
        assert "model_type" in str(error)
    else:
        raise AssertionError("unknown model type should fail")
