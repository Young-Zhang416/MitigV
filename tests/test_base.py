"""Tests for :mod:`mitigv.core.base`."""

import pytest

from mitigv import BaseMitigator, MitigatorConfig, MitigatorConfigError


# ---------------------------------------------------------------------------
# Test doubles — concrete subclasses used to exercise the abstract contract.
# ---------------------------------------------------------------------------

class ToyConfig(MitigatorConfig):
    """Config with one algorithm-specific field, plus its own validation."""

    alpha: float = 2.0

    def validate(self) -> None:
        super().validate()
        if self.alpha <= 0:
            raise MitigatorConfigError("alpha must be > 0")


class ToyMitigator(BaseMitigator):
    algorithm_name = "toy"
    config_class = ToyConfig

    def __init__(self, model=None, processor=None, config=None, generated="mitigated", **kwargs):
        super().__init__(model=model, processor=processor, config=config, **kwargs)
        self._generated = generated

    def generate(self, images, prompt, **kwargs):
        self._ensure_ready()
        return f"{prompt}->{self._generated}"


class BareMitigator(BaseMitigator):
    """Subclass that does NOT override ``config_class``."""

    algorithm_name = "bare"

    def generate(self, images, prompt, **kwargs):
        self._ensure_ready()
        return prompt


MODEL = object()
PROCESSOR = object()


# ---------------------------------------------------------------------------
# MitigatorConfig
# ---------------------------------------------------------------------------

class TestMitigatorConfig:
    def test_defaults(self):
        cfg = MitigatorConfig()
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 1.0
        assert cfg.do_sample is False
        assert cfg.top_p is None
        assert cfg.num_beams == 1
        assert cfg.seed is None

    def test_to_dict_roundtrip(self):
        cfg = MitigatorConfig(max_new_tokens=64, do_sample=True, top_p=0.9, seed=42)
        assert MitigatorConfig.from_dict(cfg.to_dict()) == cfg

    def test_subclass_to_dict_roundtrip(self):
        cfg = ToyConfig(alpha=3.0, max_new_tokens=16)
        assert ToyConfig.from_dict(cfg.to_dict()) == cfg
        assert isinstance(ToyConfig.from_dict(cfg.to_dict()), ToyConfig)

    def test_from_dict_rejects_unknown_key(self):
        with pytest.raises(MitigatorConfigError, match="unknown configuration key"):
            MitigatorConfig.from_dict({"nope": 1})

    def test_from_dict_rejects_non_mapping(self):
        with pytest.raises(TypeError):
            MitigatorConfig.from_dict([("max_new_tokens", 3)])

    def test_copy_returns_new_instance_and_keeps_original(self):
        cfg = ToyConfig(alpha=2.0)
        new = cfg.copy(alpha=5.0)
        assert new is not cfg
        assert new.alpha == 5.0
        assert cfg.alpha == 2.0

    def test_copy_rejects_unknown_key(self):
        with pytest.raises(MitigatorConfigError, match="unknown configuration key"):
            MitigatorConfig().copy(bogus=True)

    def test_copy_revalidates(self):
        with pytest.raises(MitigatorConfigError, match="alpha must be > 0"):
            ToyConfig().copy(alpha=-1.0)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"max_new_tokens": -1}, "max_new_tokens"),
            ({"num_beams": 0}, "num_beams"),
            ({"temperature": 0.0}, "temperature"),
            ({"top_p": 0.0}, "top_p"),
            ({"top_p": 1.5}, "top_p"),
        ],
    )
    def test_invalid_values_raise(self, kwargs, message):
        with pytest.raises(MitigatorConfigError, match=message):
            MitigatorConfig(**kwargs)


# ---------------------------------------------------------------------------
# BaseMitigator
# ---------------------------------------------------------------------------

class TestBaseMitigator:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseMitigator()  # type: ignore[abstract]

    def test_default_config_class(self):
        m = BareMitigator(MODEL, PROCESSOR)
        assert isinstance(m.config, MitigatorConfig)
        assert type(m.config) is MitigatorConfig

    def test_config_from_mapping(self):
        m = ToyMitigator(MODEL, PROCESSOR, config={"alpha": 4.0, "max_new_tokens": 10})
        assert m.config.alpha == 4.0
        assert m.config.max_new_tokens == 10

    def test_config_from_instance(self):
        cfg = ToyConfig(alpha=6.0)
        m = ToyMitigator(MODEL, PROCESSOR, config=cfg)
        assert m.config is cfg

    def test_config_kwargs_override(self):
        m = ToyMitigator(MODEL, PROCESSOR, alpha=3.5)
        assert m.config.alpha == 3.5
        assert m.config.max_new_tokens == 512  # untouched default

    def test_kwargs_do_not_mutate_passed_config(self):
        cfg = ToyConfig(alpha=4.0)
        ToyMitigator(MODEL, PROCESSOR, config=cfg, alpha=9.0)
        assert cfg.alpha == 4.0

    def test_unknown_kwarg_raises(self):
        with pytest.raises(MitigatorConfigError, match="unknown configuration key"):
            ToyMitigator(MODEL, PROCESSOR, beta=1.0)

    def test_unknown_config_key_in_mapping_raises(self):
        with pytest.raises(MitigatorConfigError, match="unknown configuration key"):
            ToyMitigator(MODEL, PROCESSOR, config={"beta": 1.0})

    def test_invalid_config_type_raises(self):
        with pytest.raises(TypeError, match="config must be"):
            ToyMitigator(MODEL, PROCESSOR, config=123)

    def test_config_from_wrong_hierarchy_raises(self):
        # ToyMitigator expects ToyConfig; a base MitigatorConfig must be rejected.
        with pytest.raises(TypeError, match="config must be a ToyConfig instance"):
            ToyMitigator(MODEL, PROCESSOR, config=MitigatorConfig())

    def test_generate_delegates_and_requires_deps(self):
        m = ToyMitigator()
        with pytest.raises(RuntimeError, match="requires a model"):
            m.generate(object(), "hi")

    def test_generate_runs_when_ready(self):
        m = ToyMitigator(MODEL, PROCESSOR)
        assert m.generate(object(), "hi") == "hi->mitigated"

    def test_call_is_generate_alias(self):
        m = ToyMitigator(MODEL, PROCESSOR, generated="x")
        assert m(object(), "q") == "q->x"

    def test_algorithm_name_and_repr(self):
        m = ToyMitigator(MODEL, PROCESSOR)
        assert m.algorithm_name == "toy"
        assert repr(m) == "ToyMitigator(algorithm='toy', model='object')"

    def test_repr_without_model(self):
        assert repr(ToyMitigator()) == "ToyMitigator(algorithm='toy', model=None)"
