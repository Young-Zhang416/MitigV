"""Model backends and framework adapters.

Algorithms use the generic backend; framework-specific details live in
adapters such as :mod:`mitigv.backends.llava`.
"""

from mitigv.backends.generic import (
    GenericMitigator,
    GenericMitigatorConfig,
    MitigatorBackend,
    ModelMitigator,
    ModelMitigatorConfig,
)
from mitigv.core.interfaces import ModelProtocol, ProcessorProtocol
from mitigv.backends.llava import (
    LlavaModel,
    LlavaModelAdapter,
    LlavaProcessor,
    LlavaProcessorAdapter,
    adapt_llava,
)
from mitigv.backends.qwen2_5_vl import (
    Qwen2_5VLModel,
    Qwen2_5VLModelAdapter,
    Qwen2_5VLProcessor,
    Qwen2_5VLProcessorAdapter,
    adapt_qwen2_5_vl,
)
from mitigv.backends.hf_common import (
    VisionLanguageModelAdapter,
    VisionLanguageProcessorAdapter,
)
from mitigv.backends.factory import (
    adapt_vision_language,
    available_model_families,
    get_model_adapter,
    get_processor_adapter,
    load_vision_language,
)

__all__ = [
    "ModelMitigator",
    "MitigatorBackend",
    "GenericMitigator",
    "ModelMitigatorConfig",
    "GenericMitigatorConfig",
    "ModelProtocol",
    "ProcessorProtocol",
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
]
