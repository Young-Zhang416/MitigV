"""MitigV — training-free hallucination mitigation for large vision-language models.

The public API is intentionally small and grows one module at a time.

* Module 1 — ``core.base``: :class:`BaseMitigator` + :class:`MitigatorConfig`.
* Module 2 — ``core.registry``: name-based registration and instantiation.
* Module 3 — ``backends.generic``: framework-neutral decoding skeleton.
* Module 4 — ``backends.llava``: HuggingFace LLaVA adapters.

    from mitigv import build_mitigator

    mitigator = build_mitigator("vcd", model, processor, alpha=2.0)
    text = mitigator(images, prompt)

Backend symbols are imported lazily so ``import mitigv`` stays free of heavy
torch/transformers dependencies.
"""

from mitigv.api import load_mitigator, mitigate
from mitigv.core import (
    BaseMitigator,
    MitigatorConfig,
    MitigatorConfigError,
    ModelOutputProtocol,
    ModelProtocol,
    ModelInterface,
    ModelOutputInterface,
    ProcessorInterface,
    ProcessorProtocol,
    build_mitigator,
    get_mitigator_class,
    is_registered,
    list_mitigators,
    register_mitigator,
)

__version__ = "0.1.2"

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
    "build_mitigator",
    "get_mitigator_class",
    "is_registered",
    "list_mitigators",
    "register_mitigator",
    "mitigate",
    "load_mitigator",
    # Lazily imported (backends pull in torch/transformers only on use):
    "ModelMitigator",
    "ModelMitigatorConfig",
    "GenericMitigatorConfig",
    "MitigatorBackend",
    "GenericMitigator",
    "LlavaModelAdapter",
    "LlavaProcessorAdapter",
    "LlavaModel",
    "LlavaProcessor",
    "adapt_llava",
    "Qwen2_5VLModelAdapter",
    "Qwen2_5VLProcessorAdapter",
    "Qwen2_5VLModel",
    "Qwen2_5VLProcessor",
    "adapt_qwen2_5_vl",
    "VisionLanguageModelAdapter",
    "VisionLanguageProcessorAdapter",
    "available_model_families",
    "get_model_adapter",
    "get_processor_adapter",
    "adapt_vision_language",
    "load_vision_language",
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
    "AGLA",
    "AGLAConfig",
    "ONLY",
    "ONLYConfig",
    "OPERA",
    "OPERAConfig",
    "__version__",
]

_LAZY = {
    "ModelMitigator": ("mitigv.backends", "ModelMitigator"),
    "ModelMitigatorConfig": ("mitigv.backends", "ModelMitigatorConfig"),
    "GenericMitigatorConfig": ("mitigv.backends", "GenericMitigatorConfig"),
    "MitigatorBackend": ("mitigv.backends", "MitigatorBackend"),
    "GenericMitigator": ("mitigv.backends", "GenericMitigator"),
    "LlavaModelAdapter": ("mitigv.backends", "LlavaModelAdapter"),
    "LlavaProcessorAdapter": ("mitigv.backends", "LlavaProcessorAdapter"),
    "LlavaModel": ("mitigv.backends", "LlavaModel"),
    "LlavaProcessor": ("mitigv.backends", "LlavaProcessor"),
    "adapt_llava": ("mitigv.backends", "adapt_llava"),
    "Qwen2_5VLModelAdapter": ("mitigv.backends", "Qwen2_5VLModelAdapter"),
    "Qwen2_5VLProcessorAdapter": ("mitigv.backends", "Qwen2_5VLProcessorAdapter"),
    "Qwen2_5VLModel": ("mitigv.backends", "Qwen2_5VLModel"),
    "Qwen2_5VLProcessor": ("mitigv.backends", "Qwen2_5VLProcessor"),
    "adapt_qwen2_5_vl": ("mitigv.backends", "adapt_qwen2_5_vl"),
    "VisionLanguageModelAdapter": ("mitigv.backends", "VisionLanguageModelAdapter"),
    "VisionLanguageProcessorAdapter": ("mitigv.backends", "VisionLanguageProcessorAdapter"),
    "available_model_families": ("mitigv.backends", "available_model_families"),
    "get_model_adapter": ("mitigv.backends", "get_model_adapter"),
    "get_processor_adapter": ("mitigv.backends", "get_processor_adapter"),
    "adapt_vision_language": ("mitigv.backends", "adapt_vision_language"),
    "load_vision_language": ("mitigv.backends", "load_vision_language"),
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
    "LinearProbeSteerConfig": (
        "mitigv.algorithms.probe_steer",
        "LinearProbeSteerConfig",
    ),
    "AGLA": ("mitigv.algorithms.agla", "AGLA"),
    "AGLAConfig": ("mitigv.algorithms.agla", "AGLAConfig"),
    "ONLY": ("mitigv.algorithms.only", "ONLY"),
    "ONLYConfig": ("mitigv.algorithms.only", "ONLYConfig"),
    "OPERA": ("mitigv.algorithms.opera", "OPERA"),
    "OPERAConfig": ("mitigv.algorithms.opera", "OPERAConfig"),
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
