"""HuggingFace LLaVA adapters for the generic MitigV interfaces.

The adapters deliberately contain the HuggingFace-specific details (not the
algorithms or decoding loop).  They work with ``LlavaForConditionalGeneration``
and ``AutoProcessor`` as well as small test doubles exposing the same calling
convention.  No HuggingFace import is performed at module import time.
"""

from __future__ import annotations

from collections.abc import Sequence
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

    def __init__(self, processor: Any, *, use_chat_template: bool = True, **_: Any) -> None:
        super().__init__(processor)
        self.use_chat_template = use_chat_template

    def _prepare_text(self, text: Any, images: Any) -> Any:
        """Ensure every image-bearing prompt contains LLaVA image tokens."""

        if not self.use_chat_template or images is None:
            return text
        if isinstance(text, str):
            return self._format_one(text)
        if isinstance(text, Sequence):
            return [self._format_one(item) for item in text]
        return text

    def _format_one(self, text: str) -> str:
        if "<image>" in text:
            return text
        apply_template = getattr(self.processor, "apply_chat_template", None)
        if callable(apply_template):
            try:
                return apply_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": text},
                            ],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (KeyError, TypeError, ValueError):
                # Older LLaVA checkpoints may not ship a usable chat template.
                pass
        return "<image>\n" + text

def adapt_llava(model: Any, processor: Any) -> tuple[ModelProtocol, ProcessorProtocol]:
    """Return protocol adapters for a HuggingFace LLaVA model and processor."""

    return LlavaModelAdapter(model), LlavaProcessorAdapter(processor)


# Short names for callers treating adapters as their model/processor objects.
LlavaModel = LlavaModelAdapter
LlavaProcessor = LlavaProcessorAdapter
