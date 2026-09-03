"""Small, framework-neutral interfaces used by MitigV backends.

The algorithms only need a very small slice of a vision-language model.  In
particular, they do *not* require a ``transformers.PreTrainedModel`` or an
``AutoProcessor``.  Applications can implement these protocols directly (or
adapt an existing runtime to them):

``ModelProtocol``
    A callable model returning an object or mapping with ``logits`` and,
    optionally, ``past_key_values``.  ``use_cache`` and ``past_key_values``
    follow the usual autoregressive semantics.

``ProcessorProtocol``
    Turns text/images into a mapping of model inputs and decodes generated
    token ids.  ``batch_decode`` is preferred, while a scalar ``decode`` is
    also accepted by the backend.

These are typing contracts, not inheritance requirements.  Structural
compatibility is checked at runtime only when a backend is instantiated, so
plain Python or custom runtime implementations can be adapted without
importing HuggingFace (the built-in decoding loop expects PyTorch tensors).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelProtocol(Protocol):
    """Minimal callable model contract for autoregressive decoding."""

    def __call__(self, **kwargs: Any) -> Any:
        """Run a forward pass and return logits plus an optional cache."""


@runtime_checkable
class ProcessorProtocol(Protocol):
    """Minimal text/image preprocessing and token decoding contract."""

    def __call__(
        self, *, text: Any, images: Any = None, return_tensors: str | None = None,
        padding: bool | str | None = None, **kwargs: Any
    ) -> Mapping[str, Any]:
        """Prepare model inputs from text and images."""

    def batch_decode(self, sequences: Sequence[Any], **kwargs: Any) -> Sequence[str]:
        """Decode a batch of generated token sequences."""


@runtime_checkable
class ModelOutputProtocol(Protocol):
    """Optional output shape understood by :class:`ModelMitigator`."""

    logits: Any
    past_key_values: Any


# ``*Interface`` aliases make the contracts discoverable to users who prefer
# that terminology while keeping the PEP 544 ``Protocol`` names explicit.
ModelInterface = ModelProtocol
ProcessorInterface = ProcessorProtocol
ModelOutputInterface = ModelOutputProtocol

__all__ = [
    "ModelProtocol",
    "ProcessorProtocol",
    "ModelOutputProtocol",
    "ModelInterface",
    "ProcessorInterface",
    "ModelOutputInterface",
]
