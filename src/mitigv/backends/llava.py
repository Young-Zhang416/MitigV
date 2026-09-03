"""HuggingFace LLaVA adapters for the generic MitigV interfaces.

The adapters deliberately contain the HuggingFace-specific details (not the
algorithms or decoding loop).  They work with ``LlavaForConditionalGeneration``
and ``AutoProcessor`` as well as small test doubles exposing the same calling
convention.  No HuggingFace import is performed at module import time.
"""

from __future__ import annotations

from typing import Any

from mitigv.backends.hf_common import (
    VisionLanguageModelAdapter,
    VisionLanguageProcessorAdapter,
)
from mitigv.core.interfaces import ModelProtocol, ProcessorProtocol

__all__ = [
    "LlavaModelAdapter",
    "LlavaProcessorAdapter",
    "LlavaModel",
    "LlavaProcessor",
    "adapt_llava",
]


class LlavaModelAdapter(VisionLanguageModelAdapter):
    """Adapt a HuggingFace LLaVA model to :class:`ModelProtocol`.

    LLaVA releases differ in whether ``cache_position`` is required and in the
    name used for the returned cache.  The adapter normalizes both differences
    while forwarding every other model feature unchanged.
    """

    model_family = "llava"
    transformers_class_name = "LlavaForConditionalGeneration"


class LlavaProcessorAdapter(VisionLanguageProcessorAdapter):
    """Adapt an ``AutoProcessor`` (or tokenizer-like object) to the protocol."""

    def __init__(self, processor: Any, **_: Any) -> None:
        super().__init__(processor)

def adapt_llava(model: Any, processor: Any) -> tuple[ModelProtocol, ProcessorProtocol]:
    """Return protocol adapters for a HuggingFace LLaVA model and processor."""

    return LlavaModelAdapter(model), LlavaProcessorAdapter(processor)


# Short names for callers treating adapters as their model/processor objects.
LlavaModel = LlavaModelAdapter
LlavaProcessor = LlavaProcessorAdapter
