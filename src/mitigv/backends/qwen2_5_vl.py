"""HuggingFace Qwen2.5-VL adapters for the generic MitigV backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mitigv.backends.hf_common import (
    VisionLanguageModelAdapter,
    VisionLanguageProcessorAdapter,
)
from mitigv.core.interfaces import ModelProtocol, ProcessorProtocol

__all__ = [
    "Qwen2_5VLModelAdapter",
    "Qwen2_5VLProcessorAdapter",
    "Qwen2_5VLModel",
    "Qwen2_5VLProcessor",
    "adapt_qwen2_5_vl",
]


class Qwen2_5VLModelAdapter(VisionLanguageModelAdapter):
    """Normalize Qwen2.5-VL forward outputs and multimodal cache arguments."""

    model_family = "qwen2.5-vl"
    transformers_class_name = "Qwen2_5_VLForConditionalGeneration"

    def _prepare_model_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        # Qwen's model computes 3-D RoPE positions internally from
        # image_grid_thw/mm_token_type_ids. Do not synthesize position_ids; let
        # the model keep its rope_deltas state consistent.
        kwargs.setdefault("return_dict", True)
        return super()._prepare_model_kwargs(kwargs)


class Qwen2_5VLProcessorAdapter(VisionLanguageProcessorAdapter):
    """Adapt Qwen2.5-VL ``AutoProcessor`` to the generic processor contract."""

    def __init__(self, processor: Any, *, use_chat_template: bool = True) -> None:
        super().__init__(processor)
        self.use_chat_template = use_chat_template

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        use_chat_template: bool = True,
        **kwargs: Any,
    ) -> "Qwen2_5VLProcessorAdapter":
        return cls(
            cls._load_processor(model_id, **kwargs),
            use_chat_template=use_chat_template,
        )

    def _prepare_text(self, text: Any, images: Any) -> Any:
        if not self.use_chat_template or not hasattr(self.processor, "apply_chat_template"):
            return text
        if isinstance(text, str):
            return self._format_one(text, images)
        if isinstance(text, Sequence):
            image_items = images if isinstance(images, Sequence) and not isinstance(images, (str, bytes)) else None
            return [
                self._format_one(item, image_items[index] if image_items is not None and index < len(image_items) else images)
                for index, item in enumerate(text)
            ]
        return text

    def _format_one(self, text: str, images: Any) -> str:
        # Already-templated Qwen prompts contain the image placeholder and
        # should pass through unchanged.
        if "<|image_pad|>" in text or "<|video_pad|>" in text:
            return text
        if images is None:
            return text
        message = [{"role": "user", "content": [{"type": "image", "image": images}, {"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

def adapt_qwen2_5_vl(model: Any, processor: Any) -> tuple[ModelProtocol, ProcessorProtocol]:
    """Return generic model/processor adapters for Qwen2.5-VL."""

    return Qwen2_5VLModelAdapter(model), Qwen2_5VLProcessorAdapter(processor)


Qwen2_5VLModel = Qwen2_5VLModelAdapter
Qwen2_5VLProcessor = Qwen2_5VLProcessorAdapter
