"""High-level context-manager API for running mitigators."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from mitigv.core.base import BaseMitigator
from mitigv.core.registry import build_mitigator

__all__ = ["mitigate"]


@contextmanager
def mitigate(
    name: str | type[BaseMitigator],
    config: Any = None,
    *,
    model: Any = None,
    processor: Any = None,
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
        mitigator = name(model=model, processor=processor, config=config, **kwargs)
    else:
        mitigator = build_mitigator(
            name, model=model, processor=processor, config=config, **kwargs
        )

    original_device = None
    try:
        if device is not None and mitigator.model is not None and hasattr(mitigator.model, "to"):
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
