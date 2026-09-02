"""Tests for :mod:`mitigv.core.registry`."""

import pytest

from mitigv import (
    BaseMitigator,
    MitigatorConfig,
    build_mitigator,
    get_mitigator_class,
    is_registered,
    list_mitigators,
    register_mitigator,
)
from mitigv.core import registry


MODEL = object()
PROCESSOR = object()


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate each test, then restore the pre-test registry state.

    Snapshot/restore (rather than just clearing) matters because algorithms
    register themselves at import time (e.g. ``vcd``) and must survive other
    modules' tests.
    """
    snapshot = {name: registry.get(name) for name in registry.names()}
    registry.clear()
    yield
    registry.clear()
    for name, cls in snapshot.items():
        registry.register(name, cls)


def _class(name):
    """Create a fresh, minimal BaseMitigator subclass with ``algorithm_name``."""

    def generate(self, images, prompt, **kwargs):
        self._ensure_ready()
        return f"{name}:{prompt}"

    return type(name, (BaseMitigator,), {"algorithm_name": name, "generate": generate})


class TestRegistry:
    def test_register_and_build(self):
        cls = _class("demo")
        register_mitigator("demo")(cls)

        m = build_mitigator("demo", MODEL, PROCESSOR)
        assert isinstance(m, cls)
        assert m(None, "hi") == "demo:hi"

    def test_bare_decorator_uses_algorithm_name(self):
        @register_mitigator
        class M(BaseMitigator):
            algorithm_name = "bare"
            def generate(self, images, prompt, **kwargs):
                self._ensure_ready()
                return prompt

        assert "bare" in list_mitigators()
        assert get_mitigator_class("bare") is M

    def test_explicit_name_wins_over_algorithm_name(self):
        @register_mitigator("key")
        class M(BaseMitigator):
            algorithm_name = "other"
            def generate(self, images, prompt, **kwargs):
                return prompt

        assert get_mitigator_class("key") is M
        assert "other" not in registry

    def test_decorator_rejects_non_subclass(self):
        with pytest.raises(TypeError, match="BaseMitigator"):
            @register_mitigator("nope")
            class NotAMitigator:
                pass

    def test_requires_name(self):
        with pytest.raises(ValueError, match="algorithm_name"):
            @register_mitigator
            class M(BaseMitigator):
                algorithm_name = ""
                def generate(self, images, prompt, **kwargs):
                    return prompt

    def test_duplicate_name_raises(self):
        register_mitigator("dup")(_class("dup"))
        with pytest.raises(ValueError, match="already registered"):
            register_mitigator("dup")(_class("dup2"))

    def test_same_class_reregistration_is_noop(self):
        cls = _class("demo")
        register_mitigator("demo")(cls)
        register_mitigator("demo")(cls)  # must not raise

    def test_override_replaces(self):
        first = _class("demo")
        second = _class("demo2")
        register_mitigator("demo")(first)
        register_mitigator("demo", override=True)(second)
        assert get_mitigator_class("demo") is second

    def test_build_unknown_raises(self):
        with pytest.raises(KeyError, match="unknown mitigator"):
            build_mitigator("missing", MODEL, PROCESSOR)

    def test_build_forwards_config_and_kwargs(self):
        class Cfg(MitigatorConfig):
            alpha: float = 1.0

        @register_mitigator("cfg")
        class M(BaseMitigator):
            algorithm_name = "cfg"
            config_class = Cfg
            def generate(self, images, prompt, **kwargs):
                self._ensure_ready()
                return prompt

        m = build_mitigator("cfg", MODEL, PROCESSOR, config={"alpha": 5.0}, max_new_tokens=7)
        assert m.config.alpha == 5.0
        assert m.config.max_new_tokens == 7

    def test_list_is_registered_get(self):
        register_mitigator("zeta")(_class("zeta"))
        register_mitigator("alpha")(_class("alpha"))
        assert list_mitigators() == ["alpha", "zeta"]
        assert is_registered("alpha")
        assert not is_registered("nope")
        assert isinstance(get_mitigator_class("alpha"), type)
        assert "alpha" in registry

    def test_registry_len_and_clear(self):
        assert len(registry) == 0
        register_mitigator("a")(_class("a"))
        register_mitigator("b")(_class("b"))
        assert len(registry) == 2
        registry.clear()
        assert len(registry) == 0
        assert list_mitigators() == []

    def test_register_rejects_non_subclass(self):
        with pytest.raises(TypeError, match="BaseMitigator"):
            registry.register("x", object())

    def test_register_rejects_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            registry.register("", _class("x"))

    def test_repr(self):
        register_mitigator("a")(_class("a"))
        assert repr(registry) == "MitigatorRegistry(mitigators=['a'])"
