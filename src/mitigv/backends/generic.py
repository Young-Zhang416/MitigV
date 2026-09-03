"""Framework-neutral autoregressive backend.

This is the preferred backend for algorithms and integrations. Concrete model
families connect to the decoding implementation through the public model and
processor interfaces.
"""

from mitigv.backends._generic_impl import GenericMitigator, GenericMitigatorConfig
from mitigv.backends.hf_common import (
    VisionLanguageModelAdapter,
    VisionLanguageProcessorAdapter,
)
from mitigv.core.interfaces import (
    ModelInterface,
    ModelOutputInterface,
    ModelOutputProtocol,
    ModelProtocol,
    ProcessorInterface,
    ProcessorProtocol,
)

__all__ = [
    "GenericMitigator",
    "MitigatorBackend",
    "ModelMitigator",
    "ModelMitigatorConfig",
    "GenericMitigatorConfig",
    "ModelProtocol",
    "ModelOutputProtocol",
    "ProcessorProtocol",
    "ModelInterface",
    "ModelOutputInterface",
    "ProcessorInterface",
    "VisionLanguageModelAdapter",
    "VisionLanguageProcessorAdapter",
]


ModelMitigator = GenericMitigator
MitigatorBackend = GenericMitigator
ModelMitigatorConfig = GenericMitigatorConfig
