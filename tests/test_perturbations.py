"""Tests for :mod:`mitigv.perturbations`."""

import pytest
import torch

from mitigv.perturbations import (
    DiffusionNoisePerturbation,
    GaussianNoisePerturbation,
    Perturbation,
    build_perturbation,
    list_perturbations,
    register_perturbation,
)


class TestGaussianNoise:
    def test_shape_dtype_and_non_mutation(self):
        p = GaussianNoisePerturbation(std=0.5)
        x = torch.zeros(2, 3, 4, 4)
        y = p(x)
        assert y.shape == x.shape and y.dtype == x.dtype
        assert torch.equal(x, torch.zeros_like(x))  # original untouched
        assert not torch.equal(x, y)  # noise was added

    def test_reproducible_with_seed(self):
        p = GaussianNoisePerturbation(std=1.0)
        x = torch.zeros(3, 3)
        torch.manual_seed(0)
        y1 = p(x)
        torch.manual_seed(0)
        y2 = p(x)
        assert torch.equal(y1, y2)

    def test_variance_scales_with_std(self):
        torch.manual_seed(0)
        x = torch.zeros(10_000)
        y = GaussianNoisePerturbation(std=2.0)(x)
        assert abs(y.var().item() - 4.0) < 0.5  # Var[std * N(0,1)] = std^2

    def test_clip(self):
        p = GaussianNoisePerturbation(std=1.0, clip=(0.0, 1.0))
        y = p(torch.full((100,), 0.5))
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0

    def test_invalid_std_raises(self):
        with pytest.raises(ValueError, match="std"):
            GaussianNoisePerturbation(std=-1.0)

    def test_non_finite_std_and_invalid_clip_raise(self):
        with pytest.raises(ValueError, match="std"):
            GaussianNoisePerturbation(std=float("nan"))
        with pytest.raises(ValueError, match="lower bound"):
            GaussianNoisePerturbation(std=0.1, clip=(1.0, 0.0))


class TestDiffusionNoise:
    def test_noise_increases_with_step(self):
        torch.manual_seed(0)
        x = torch.zeros(10_000)
        y0 = DiffusionNoisePerturbation(noise_step=0)(x)
        y999 = DiffusionNoisePerturbation(noise_step=999)(x)
        assert y0.var().item() < y999.var().item()

    def test_late_step_approaches_noise(self):
        torch.manual_seed(0)
        x = torch.zeros(10_000)
        y = DiffusionNoisePerturbation(noise_step=999)(x)
        assert abs(y.var().item() - 1.0) < 0.5

    def test_invalid_step_raises(self):
        with pytest.raises(ValueError, match="noise_step"):
            DiffusionNoisePerturbation(noise_step=1000, num_steps=1000)

        with pytest.raises(ValueError, match="num_steps"):
            DiffusionNoisePerturbation(noise_step=0, num_steps=0)


class TestRegistry:
    def test_builtins_registered(self):
        assert {"gaussian_noise", "diffusion_noise"} <= set(list_perturbations())

    def test_build_by_name(self):
        assert isinstance(
            build_perturbation("gaussian_noise", std=0.3), GaussianNoisePerturbation
        )
        assert isinstance(
            build_perturbation("diffusion_noise"), DiffusionNoisePerturbation
        )

    def test_build_unknown_raises(self):
        with pytest.raises(KeyError, match="unknown perturbation"):
            build_perturbation("nope")

    def test_register_custom(self):
        @register_perturbation
        class Zero(Perturbation):
            name = "zero"

            def __call__(self, image):
                return torch.zeros_like(image)

        assert "zero" in list_perturbations()
        assert torch.equal(build_perturbation("zero")(torch.ones(2)), torch.zeros(2))

    def test_register_non_subclass_raises(self):
        with pytest.raises(TypeError, match="Perturbation"):
            register_perturbation(object)
