"""High-level context-manager API for running mitigators."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from typing import Any, Iterator

from mitigv.core.base import BaseMitigator
from mitigv.core.registry import build_mitigator

__all__ = ["load_mitigator", "mitigate"]


def load_mitigator(
    algorithm: str,
    *,
    model: Any = None,
    processor: Any = None,
    model_type: str | None = None,
    model_id: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    processor_kwargs: Mapping[str, Any] | None = None,
    config: Any = None,
    **algorithm_kwargs: Any,
) -> BaseMitigator:
    """Create a ready-to-use mitigator from objects or a model checkpoint.

    Pass ``model`` and ``processor`` for any protocol-compatible runtime. Set
    ``model_type`` as well to wrap already-loaded LLaVA or Qwen objects. To load
    a HuggingFace checkpoint in one call, pass ``model_type`` and ``model_id``.
    """

    supplied_objects = model is not None or processor is not None
    if supplied_objects:
        if model is None or processor is None:
            raise ValueError("model and processor must be provided together")
        if model_id is not None:
            raise ValueError("model_id cannot be combined with loaded model objects")
        if model_kwargs:
            raise ValueError("model_kwargs cannot be applied to an already-loaded model")
        if model_type is not None:
            from mitigv.backends.factory import adapt_vision_language

            model, processor = adapt_vision_language(
                model_type,
                model,
                processor,
                processor_kwargs=processor_kwargs,
            )
        elif processor_kwargs:
            raise ValueError(
                "processor_kwargs require model_type when adapting loaded objects"
            )
    else:
        if model_type is None or model_id is None:
            raise ValueError(
                "provide model and processor, or provide both model_type and model_id"
            )
        from mitigv.backends.factory import load_vision_language

        model, processor = load_vision_language(
            model_type,
            model_id,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
        )
    return build_mitigator(
        algorithm,
        model=model,
        processor=processor,
        config=config,
        **algorithm_kwargs,
    )


@contextmanager
def mitigate(
    name: str | type[BaseMitigator],
    config: Any = None,
    *,
    model: Any = None,
    processor: Any = None,
    model_type: str | None = None,
    model_id: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    processor_kwargs: Mapping[str, Any] | None = None,
    device: str | Any | None = None,
    cleanup: bool = True,
    **kwargs: Any,
) -> Iterator[BaseMitigator]:
    """Run a hallucination-mitigation algorithm inside a context manager.

    Builds the algorithm ``name`` (or a :class:`BaseMitigator` class) and yields
    the mitigator, which is callable::

        from mitigv import mitigate
        from mitigv.algorithms.vcd import VCDConfig

        with mitigate("vcd", VCDConfig(alpha=2.0), model=model, processor=processor,
                      device="cuda") as f:
            text = f(images, prompt)

    ``config`` may be a config object, a mapping, or ``None``; extra keyword
    arguments are treated as config overrides. On exit, the model is restored to
    its original device (when ``device`` was given) and CUDA's cached memory is
    freed unless ``cleanup=False``.
    """
    if isinstance(name, type):
        if not issubclass(name, BaseMitigator):
            raise TypeError(
                f"name must be a mitigator name or a BaseMitigator subclass, got {name!r}"
            )
        if any(
            value is not None
            for value in (model_type, model_id, model_kwargs, processor_kwargs)
        ):
            raise ValueError("checkpoint loading requires an algorithm name, not a class")
        mitigator = name(model=model, processor=processor, config=config, **kwargs)
    else:
        mitigator = load_mitigator(
            name,
            model=model,
            processor=processor,
            model_type=model_type,
            model_id=model_id,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            config=config,
            **kwargs,
        )

    original_device = None
    try:
        if (
            device is not None
            and mitigator.model is not None
            and hasattr(mitigator.model, "to")
        ):
            original_device = getattr(mitigator, "device", None)
            mitigator.model.to(device)
        yield mitigator
    finally:
        if cleanup:
            if mitigator.model is not None and original_device is not None:
                if hasattr(mitigator.model, "to"):
                    mitigator.model.to(original_device)
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
