"""Core abstractions for training-free hallucination mitigators.

This module defines the two contracts that every MitigV algorithm builds on:

* :class:`MitigatorConfig` — hyper-parameters, with validation, copying and
  (de)serialization. Algorithms subclass it to add their own knobs (e.g. the
  ``alpha``/``beta`` of VCD), and ``from_dict``/``copy`` keep working for free.
* :class:`BaseMitigator` — the abstract base class every algorithm implements.
  It owns the model/processor, resolves configuration, and pins down the
  ``generate`` contract. Subclasses override ``algorithm_name`` (the future
  registry key) and ``config_class``, then implement :meth:`~BaseMitigator.generate`.

Concrete algorithms (VCD, PAI, ...) will be thin subclasses that plug into this
root, which is what keeps the public API uniform and lets callers switch
algorithms polymorphically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar, Mapping

__all__ = ["MitigatorConfig", "MitigatorConfigError", "BaseMitigator"]


class MitigatorConfigError(ValueError):
    """Raised when a configuration is malformed or contains unknown keys."""


@dataclass
class MitigatorConfig:
    """Base hyper-parameter container shared by every mitigator.

    Subclasses add algorithm-specific fields as plain dataclass fields and may
    extend :meth:`validate`. The helpers below then operate on the *concrete*
    subclass, so ``from_dict`` and ``copy`` are inherited correctly.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (used when ``do_sample`` is true).
        do_sample: Whether to sample instead of greedy/beam decoding.
        top_p: Nucleus-sampling threshold, or ``None`` to disable.
        num_beams: Beam width (1 == greedy/sampling).
        seed: Random seed for reproducible sampling, or ``None``.
    """

    max_new_tokens: int = 512
    temperature: float = 1.0
    do_sample: bool = False
    top_p: float | None = None
    num_beams: int = 1
    seed: int | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-apply :func:`dataclasses.dataclass` to config subclasses.

        Config authors can write plain annotated fields (``alpha: float = 1.0``)
        without decorating their subclass with ``@dataclass``; the annotations
        are turned into real fields here so ``__init__``, :meth:`from_dict` and
        :meth:`copy` all see them. Subclasses that *are* explicitly decorated
        are left untouched.
        """
        super().__init_subclass__(**kwargs)
        if "__dataclass_fields__" not in cls.__dict__:
            dataclass(cls)

    def __post_init__(self) -> None:
        self.validate()

    # -- validation -----------------------------------------------------
    def validate(self) -> None:
        """Validate field values; raise :class:`MitigatorConfigError` if invalid.

        Subclasses should call ``super().validate()`` first, then check their
        own fields.
        """
        if self.max_new_tokens < 0:
            raise MitigatorConfigError("max_new_tokens must be >= 0")
        if self.num_beams < 1:
            raise MitigatorConfigError("num_beams must be >= 1")
        if self.temperature <= 0:
            raise MitigatorConfigError("temperature must be > 0")
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise MitigatorConfigError("top_p must be in (0, 1]")

    # -- (de)serialization & copying ------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return this config's fields as a plain ``dict``."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MitigatorConfig":
        """Build a config from a mapping, rejecting unknown keys (catches typos)."""
        if not isinstance(data, Mapping):
            raise TypeError(
                f"config must be a mapping, got {type(data).__name__!r}"
            )
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise MitigatorConfigError(
                f"unknown configuration key(s) for {cls.__name__}: "
                f"{sorted(unknown)}; valid keys are {sorted(valid)}"
            )
        return cls(**dict(data))

    def copy(self, **overrides: Any) -> "MitigatorConfig":
        """Return a new config equal to this one, with ``overrides`` applied.

        The original is never mutated, and overrides are validated against the
        concrete class's fields (and :meth:`validate` runs again).
        """
        valid = {f.name for f in fields(type(self))}
        unknown = set(overrides) - valid
        if unknown:
            raise MitigatorConfigError(
                f"unknown configuration key(s) for {type(self).__name__}: "
                f"{sorted(unknown)}; valid keys are {sorted(valid)}"
            )
        return replace(self, **overrides)


class BaseMitigator(ABC):
    """Abstract base class for every training-free hallucination mitigator.

    A mitigator wraps an LVLM (``model`` + ``processor``) and re-writes the
    decoding process at generation time to suppress hallucination — no training
    or fine-tuning involved.

    Subclass contract:

    * set ``algorithm_name`` (the future registry key / display name);
    * optionally set ``config_class`` to a :class:`MitigatorConfig` subclass
      (defaults to the base config);
    * implement :meth:`generate`.

    Configuration is resolved in :meth:`__init__` from ``config`` (a config
    instance, a mapping, or ``None``) merged with any extra keyword arguments,
    which are treated as config overrides and validated against ``config_class``.
    """

    algorithm_name: ClassVar[str] = ""
    config_class: ClassVar[type[MitigatorConfig]] = MitigatorConfig

    def __init__(
        self,
        model: Any = None,
        processor: Any = None,
        config: MitigatorConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.processor = processor
        self.config = self._resolve_config(config, **kwargs)

    # -- configuration resolution ---------------------------------------
    def _resolve_config(
        self,
        config: MitigatorConfig | Mapping[str, Any] | None,
        **kwargs: Any,
    ) -> MitigatorConfig:
        if config is None:
            cfg: MitigatorConfig = self.config_class()
        elif isinstance(config, self.config_class):
            cfg = config
        elif isinstance(config, MitigatorConfig):
            # A config from a different (e.g. base) hierarchy: reject rather than
            # silently drop algorithm-specific fields.
            raise TypeError(
                f"config must be a {self.config_class.__name__} instance, got "
                f"{type(config).__name__}"
            )
        elif isinstance(config, Mapping):
            cfg = self.config_class.from_dict(config)
        else:
            raise TypeError(
                f"config must be a {self.config_class.__name__}, a mapping, or "
                f"None; got {type(config).__name__!r}"
            )
        if kwargs:
            # copy (not mutate) so the caller's config object stays untouched.
            cfg = cfg.copy(**kwargs)
        return cfg

    # -- runtime guards ---------------------------------------------------
    def _ensure_ready(self) -> None:
        """Raise if the mitigator lacks the model/processor it needs to run."""
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__} requires a model; pass it via "
                f"{type(self).__name__}(model=...) before calling generate()."
            )
        if self.processor is None:
            raise RuntimeError(
                f"{type(self).__name__} requires a processor; pass it via "
                f"{type(self).__name__}(processor=...) before calling generate()."
            )

    # -- contract ----------------------------------------------------------
    @abstractmethod
    def generate(self, images: Any, prompt: str, **kwargs: Any) -> str | list[str]:
        """Run hallucination-mitigated generation.

        Args:
            images: One image or a sequence of images. The accepted forms (PIL
                images, file paths, or tensors) are defined by each backend;
                subclasses must document what they accept.
            prompt: Text prompt to condition generation on.
            **kwargs: Generation-time overrides (e.g. ``max_new_tokens``).

        Returns:
            Generated text, or a list of strings when a batch of independent
            (image, prompt) pairs is supplied.
        """
        raise NotImplementedError

    def __call__(self, images: Any, prompt: str, **kwargs: Any) -> str | list[str]:
        """Alias for :meth:`generate`."""
        return self.generate(images, prompt, **kwargs)

    # -- misc ---------------------------------------------------------------
    def __repr__(self) -> str:
        name = self.algorithm_name or type(self).__name__
        model_name = type(self.model).__name__ if self.model is not None else None
        return f"{type(self).__name__}(algorithm={name!r}, model={model_name!r})"
