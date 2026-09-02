"""MitigV — training-free hallucination mitigation for large vision-language models.

The public API is intentionally small and grows one module at a time.

* Module 1 — ``core.base``: :class:`BaseMitigator` + :class:`MitigatorConfig`.
* Module 2 — ``core.registry``: name-based registration and instantiation.
* Module 3 — ``backends.hf``: :class:`HFMitigator` decoding skeleton.

    from mitigv import build_mitigator

    mitigator = build_mitigator("vcd", model, processor, alpha=2.0)
    text = mitigator(images, prompt)

Backend symbols (:class:`HFMitigator`, :class:`HFMitigatorConfig`) are imported
lazily so ``import mitigv`` stays free of the heavy torch/transformers deps.
"""

from mitigv.api import mitigate
from mitigv.core import (
    BaseMitigator,
    MitigatorConfig,
    MitigatorConfigError,
    build_mitigator,
    get_mitigator_class,
    is_registered,
    list_mitigators,
    register_mitigator,
)

__version__ = "0.1.0"

__all__ = [
    "BaseMitigator",
    "MitigatorConfig",
    "MitigatorConfigError",
    "build_mitigator",
    "get_mitigator_class",
    "is_registered",
    "list_mitigators",
    "register_mitigator",
    "mitigate",
    # Lazily imported (backends pull in torch/transformers only on use):
    "HFMitigator",
    "HFMitigatorConfig",
    "VCD",
    "VCDConfig",
    "ICD",
    "ICDConfig",
    "PAI",
    "PAIConfig",
    "M3ID",
    "M3IDConfig",
    "VISTA",
    "VISTAConfig",
    "LinearProbeSteer",
    "LinearProbeSteerConfig",
    "__version__",
]

_LAZY = {
    "HFMitigator": ("mitigv.backends", "HFMitigator"),
    "HFMitigatorConfig": ("mitigv.backends", "HFMitigatorConfig"),
    "VCD": ("mitigv.algorithms.vcd", "VCD"),
    "VCDConfig": ("mitigv.algorithms.vcd", "VCDConfig"),
    "ICD": ("mitigv.algorithms.icd", "ICD"),
    "ICDConfig": ("mitigv.algorithms.icd", "ICDConfig"),
    "PAI": ("mitigv.algorithms.pai", "PAI"),
    "PAIConfig": ("mitigv.algorithms.pai", "PAIConfig"),
    "M3ID": ("mitigv.algorithms.m3id", "M3ID"),
    "M3IDConfig": ("mitigv.algorithms.m3id", "M3IDConfig"),
    "VISTA": ("mitigv.algorithms.vista", "VISTA"),
    "VISTAConfig": ("mitigv.algorithms.vista", "VISTAConfig"),
    "LinearProbeSteer": ("mitigv.algorithms.probe_steer", "LinearProbeSteer"),
    "LinearProbeSteerConfig": ("mitigv.algorithms.probe_steer", "LinearProbeSteerConfig"),
}


def __getattr__(name: str):
    """Import backend symbols on first attribute access (PEP 562)."""
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
