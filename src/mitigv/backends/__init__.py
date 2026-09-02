"""Model backends.

A backend adapts one framework's decoding loop to the framework-agnostic
:class:`~mitigv.core.base.BaseMitigator` interface. Currently only the
HuggingFace-transformers backend is provided.
"""

from mitigv.backends.hf import HFMitigator, HFMitigatorConfig

__all__ = ["HFMitigator", "HFMitigatorConfig"]
