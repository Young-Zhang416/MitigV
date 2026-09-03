"""Core abstractions shared by every MitigV algorithm."""

from mitigv.core.base import BaseMitigator, MitigatorConfig, MitigatorConfigError
from mitigv.core.interfaces import (
    ModelInterface,
    ModelOutputProtocol,
    ModelOutputInterface,
    ModelProtocol,
    ProcessorInterface,
    ProcessorProtocol,
)
from mitigv.core.registry import (
    MitigatorRegistry,
    build_mitigator,
    get_mitigator_class,
    is_registered,
    list_mitigators,
    register_mitigator,
    registry,
)

__all__ = [
    "BaseMitigator",
    "MitigatorConfig",
    "MitigatorConfigError",
    "ModelProtocol",
    "ModelOutputProtocol",
    "ProcessorProtocol",
    "ModelInterface",
    "ModelOutputInterface",
    "ProcessorInterface",
    "MitigatorRegistry",
    "build_mitigator",
    "get_mitigator_class",
    "is_registered",
    "list_mitigators",
    "register_mitigator",
    "registry",
]
