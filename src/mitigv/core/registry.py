"""Algorithm registry for MitigV.

The registry is what makes the library polymorphic *by name*: an algorithm
decorates its class with :func:`register_mitigator`, and callers instantiate it
through :func:`build_mitigator` without importing the concrete module. Switching
algorithms is therefore a one-line change:

    from mitigv import build_mitigator

    mitigator = build_mitigator("vcd", model, processor, alpha=2.0)
    text = mitigator(images, prompt)
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from mitigv.core.base import BaseMitigator, MitigatorConfig

__all__ = [
    "MitigatorRegistry",
    "registry",
    "register_mitigator",
    "get_mitigator_class",
    "build_mitigator",
    "list_mitigators",
    "is_registered",
]


class MitigatorRegistry:
    """Maps algorithm names to :class:`BaseMitigator` subclasses.

    A small dict-like container (with ``__contains__``/``__len__``) kept as a
    module-level singleton so the public helpers and tests share one view.
    """

    def __init__(self) -> None:
        self._entries: dict[str, type[BaseMitigator]] = {}

    def register(
        self, name: str, cls: type[BaseMitigator], *, override: bool = False
    ) -> None:
        """Register ``cls`` under ``name``.

        Re-registering the *same* class under the same name is a no-op (so
        repeated imports / re-decoration are safe). Registering a *different*
        class under an existing name raises unless ``override`` is true.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"mitigator name must be a non-empty string, got {name!r}")
        if not isinstance(cls, type) or not issubclass(cls, BaseMitigator):
            raise TypeError(
                f"registered class must be a BaseMitigator subclass, got {cls!r}"
            )
        existing = self._entries.get(name)
        if existing is not None:
            if existing is cls:
                return
            if not override:
                raise ValueError(
                    f"mitigator name {name!r} is already registered by "
                    f"{existing.__name__}; pass override=True to replace it"
                )
        self._entries[name] = cls

    def get(self, name: str) -> type[BaseMitigator]:
        """Return the class registered under ``name``, or raise ``KeyError``."""
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"unknown mitigator {name!r}; registered mitigators: "
                f"{sorted(self._entries)}"
            ) from None

    def names(self) -> list[str]:
        """Return the sorted names of every registered mitigator."""
        return sorted(self._entries)

    def clear(self) -> None:
        """Remove every entry (mainly for tests / re-registration)."""
        self._entries.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"MitigatorRegistry(mitigators={self.names()!r})"


#: Process-wide registry singleton used by the public API.
registry = MitigatorRegistry()


def register_mitigator(
    name: str | type[BaseMitigator] | None = None, *, override: bool = False
) -> Callable[[type[BaseMitigator]], type[BaseMitigator]] | type[BaseMitigator]:
    """Decorator that registers a :class:`BaseMitigator` subclass.

    Usage (either form)::

        @register_mitigator("vcd")
        class VCD(BaseMitigator): ...

        @register_mitigator          # uses cls.algorithm_name as the key
        class PAI(BaseMitigator):
            algorithm_name = "pai"
            ...

    An explicit ``name`` argument always wins over ``algorithm_name``.
    """

    def decorator(cls: type[BaseMitigator]) -> type[BaseMitigator]:
        if not isinstance(cls, type) or not issubclass(cls, BaseMitigator):
            raise TypeError(
                "@register_mitigator can only decorate BaseMitigator subclasses, "
                f"got {cls!r}"
            )
        key = name if isinstance(name, str) else getattr(cls, "algorithm_name", "")
        if not key:
            raise ValueError(
                f"cannot register {cls.__name__}: set a non-empty "
                "`algorithm_name` or pass an explicit name to "
                "@register_mitigator(...)"
            )
        registry.register(key, cls, override=override)
        return cls

    # Bare usage: @register_mitigator
    if isinstance(name, type):
        cls, name = name, None
        return decorator(cls)
    return decorator


def get_mitigator_class(name: str) -> type[BaseMitigator]:
    """Return the registered mitigator class for ``name``."""
    return registry.get(name)


def build_mitigator(
    name: str,
    model: Any = None,
    processor: Any = None,
    config: MitigatorConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> BaseMitigator:
    """Instantiate a registered mitigator by name.

    If ``name`` is not registered yet, an attempt is made to auto-import the
    module ``mitigv.algorithms.<name>`` (whose decorator registers it).
    ``config`` and ``**kwargs`` are forwarded to the class constructor and
    validated against its ``config_class`` (see :class:`BaseMitigator`)::

        build_mitigator("vcd", model, processor, alpha=2.0, beta=0.1)
    """
    try:
        cls = registry.get(name)
    except KeyError:
        _import_algorithm_module(name)
        cls = registry.get(name)
    return cls(model=model, processor=processor, config=config, **kwargs)


def _import_algorithm_module(name: str) -> None:
    """Try to import ``mitigv.algorithms.<name>`` so its decorator registers it.

    A missing *algorithm* module is swallowed (the caller re-raises a helpful
    ``KeyError``); any other ``ModuleNotFoundError`` (e.g. a missing dependency
    of an existing algorithm) is propagated.
    """
    import importlib

    module_name = f"mitigv.algorithms.{name}"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return
        raise


def list_mitigators() -> list[str]:
    """Return the sorted names of all registered mitigators."""
    return registry.names()


def is_registered(name: str) -> bool:
    """Return whether ``name`` is currently registered."""
    return name in registry
