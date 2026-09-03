"""Shared HuggingFace vision-language adapter primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

__all__ = [
    "VisionLanguageModelAdapter",
    "VisionLanguageProcessorAdapter",
]


class VisionLanguageModelAdapter:
    """Common adapter parent for HuggingFace multimodal causal models.

    Subclasses only provide model-specific loading and cache conventions. The
    wrapped model remains available through attribute forwarding, which keeps
    optional attention-based algorithms compatible with the native model
    internals.
    """

    model_family = "generic"
    transformers_class_name: str | None = None

    def __init__(self, model: Any) -> None:
        if not callable(model):
            raise TypeError(f"{type(self).__name__} requires a callable model")
        self.model = model

    @classmethod
    def from_pretrained(
        cls, model_id: str, **kwargs: Any
    ) -> "VisionLanguageModelAdapter":
        """Load the Transformers class declared by a concrete adapter."""

        model_class = cls._transformers_class()
        return cls(model_class.from_pretrained(model_id, **kwargs))

    @classmethod
    def _transformers_class(cls) -> type[Any]:
        if cls.transformers_class_name is None:
            raise TypeError(
                f"{cls.__name__} must define transformers_class_name to load checkpoints"
            )
        try:
            import transformers
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                f"{cls.__name__}.from_pretrained requires the 'transformers' extra"
            ) from error
        try:
            return getattr(transformers, cls.transformers_class_name)
        except AttributeError as error:
            raise ImportError(
                f"installed transformers does not provide {cls.transformers_class_name}"
            ) from error

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    @property
    def device(self) -> torch.device:
        value = getattr(self.model, "device", None)
        if value is not None:
            return torch.device(value)
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype | None:
        value = getattr(self.model, "dtype", None)
        if isinstance(value, torch.dtype):
            return value
        try:
            return next(self.model.parameters()).dtype
        except (AttributeError, StopIteration):
            return None

    def __call__(self, **kwargs: Any) -> Any:
        prepared = self._prepare_model_kwargs(dict(kwargs))
        try:
            return self.model(**prepared)
        except TypeError as error:
            if "cache_position" not in prepared:
                raise
            prepared.pop("cache_position", None)
            try:
                return self.model(**prepared)
            except TypeError:
                raise error

    def _prepare_model_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Hook for family-specific cache/position handling."""

        if kwargs.get("past_key_values") is not None:
            kwargs.setdefault("cache_position", self._cache_position(kwargs))
        return kwargs

    @staticmethod
    def _cache_position(kwargs: Mapping[str, Any]) -> torch.Tensor:
        ids = kwargs.get("input_ids")
        device = ids.device if isinstance(ids, torch.Tensor) else None
        length = ids.shape[1] if isinstance(ids, torch.Tensor) and ids.ndim > 1 else 1
        past = kwargs.get("past_key_values")
        past_length = 0
        if past is not None and hasattr(past, "get_seq_length"):
            try:
                past_length = int(past.get_seq_length())
            except (TypeError, ValueError):
                pass
        return torch.arange(past_length, past_length + length, device=device)


class VisionLanguageProcessorAdapter:
    """Common processor/tokenizer adapter parent."""

    def __init__(self, processor: Any) -> None:
        if not callable(processor):
            raise TypeError(f"{type(self).__name__} requires a callable processor")
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)

    @classmethod
    def from_pretrained(
        cls, model_id: str, **kwargs: Any
    ) -> "VisionLanguageProcessorAdapter":
        """Load an AutoProcessor and wrap it with the selected adapter."""

        return cls(cls._load_processor(model_id, **kwargs))

    @staticmethod
    def _load_processor(model_id: str, **kwargs: Any) -> Any:
        try:
            from transformers import AutoProcessor
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "loading a processor requires the 'transformers' extra"
            ) from error
        return AutoProcessor.from_pretrained(model_id, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.processor, name)

    def __call__(self, *, text: Any, images: Any = None,
                 return_tensors: str | None = "pt",
                 padding: bool | str | None = True, **kwargs: Any) -> Mapping[str, Any]:
        result = self.processor(
            text=self._prepare_text(text, images),
            images=images,
            return_tensors=return_tensors,
            padding=padding,
            **kwargs,
        )
        if not isinstance(result, Mapping):
            raise TypeError(f"{type(self).__name__} processor must return a mapping")
        return result

    def _prepare_text(self, text: Any, images: Any) -> Any:
        return text

    def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]:
        decoder = getattr(self.tokenizer, "batch_decode", None)
        if decoder is None:
            decoder = getattr(self.processor, "batch_decode", None)
        if decoder is None:
            raise TypeError(f"{type(self).__name__} does not provide batch_decode")
        return list(decoder(sequences, **kwargs))
