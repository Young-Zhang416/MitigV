"""Factory for selecting a HuggingFace vision-language adapter by parameter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mitigv.backends.hf_common import (
    VisionLanguageModelAdapter,
    VisionLanguageProcessorAdapter,
)
from mitigv.backends.llava import LlavaModelAdapter, LlavaProcessorAdapter
from mitigv.backends.qwen2_5_vl import Qwen2_5VLModelAdapter, Qwen2_5VLProcessorAdapter

_FAMILIES: dict[str, tuple[type[VisionLanguageModelAdapter], type[VisionLanguageProcessorAdapter]]] = {
    "llava": (LlavaModelAdapter, LlavaProcessorAdapter),
    "llava-1.5": (LlavaModelAdapter, LlavaProcessorAdapter),
    "llava_next": (LlavaModelAdapter, LlavaProcessorAdapter),
    "llava-next": (LlavaModelAdapter, LlavaProcessorAdapter),
    "qwen": (Qwen2_5VLModelAdapter, Qwen2_5VLProcessorAdapter),
    "qwen2.5-vl": (Qwen2_5VLModelAdapter, Qwen2_5VLProcessorAdapter),
    "qwen2_5_vl": (Qwen2_5VLModelAdapter, Qwen2_5VLProcessorAdapter),
    "qwen2.5_vl": (Qwen2_5VLModelAdapter, Qwen2_5VLProcessorAdapter),
}

__all__ = [
    "available_model_families",
    "get_model_adapter",
    "get_processor_adapter",
    "adapt_vision_language",
    "load_vision_language",
]


def _normalize(model_type: str) -> str:
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("model_type must be a non-empty string")
    return model_type.strip().lower()


def available_model_families() -> tuple[str, ...]:
    """Return canonical model-family names accepted by the factory."""

    return ("llava", "qwen2.5-vl")


def get_model_adapter(model_type: str) -> type[VisionLanguageModelAdapter]:
    """Resolve a model adapter class from ``model_type``."""

    key = _normalize(model_type)
    try:
        return _FAMILIES[key][0]
    except KeyError as error:
        raise ValueError(
            f"unsupported model_type={model_type!r}; choose one of "
            f"{available_model_families()}"
        ) from error


def get_processor_adapter(model_type: str) -> type[VisionLanguageProcessorAdapter]:
    """Resolve a processor adapter class from ``model_type``."""

    key = _normalize(model_type)
    try:
        return _FAMILIES[key][1]
    except KeyError as error:
        raise ValueError(
            f"unsupported model_type={model_type!r}; choose one of "
            f"{available_model_families()}"
        ) from error


def adapt_vision_language(
    model_type: str,
    model: Any,
    processor: Any,
    *,
    processor_kwargs: Mapping[str, Any] | None = None,
) -> tuple[VisionLanguageModelAdapter, VisionLanguageProcessorAdapter]:
    """Instantiate adapters selected by ``model_type`` for loaded objects."""

    model_adapter = get_model_adapter(model_type)(model)
    processor_options = dict(processor_kwargs or {})
    processor_adapter = get_processor_adapter(model_type)(processor, **processor_options)
    return model_adapter, processor_adapter


def load_vision_language(
    model_type: str,
    model_id: str,
    *,
    model_kwargs: Mapping[str, Any] | None = None,
    processor_kwargs: Mapping[str, Any] | None = None,
) -> tuple[VisionLanguageModelAdapter, VisionLanguageProcessorAdapter]:
    """Load and adapt a checkpoint selected by ``model_type``.

    Model and processor keyword arguments are separate because Transformers
    model loading and processor loading accept different options.
    """

    model_cls = get_model_adapter(model_type)
    processor_cls = get_processor_adapter(model_type)
    model = model_cls.from_pretrained(model_id, **dict(model_kwargs or {}))
    processor = processor_cls.from_pretrained(model_id, **dict(processor_kwargs or {}))
    return model, processor
