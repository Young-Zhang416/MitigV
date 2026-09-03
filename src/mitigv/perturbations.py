"""Image perturbations for contrastive decoding (VCD, PAI, ...).

A :class:`Perturbation` turns an image *tensor* (typically ``pixel_values``)
into a distorted copy of the same shape/dtype/device. Contrastive mitigators
run the model on both the original and the distorted image and contrast the two
logits distributions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from numbers import Real
from typing import Any, Callable, ClassVar

import torch

__all__ = [
    "Perturbation",
    "GaussianNoisePerturbation",
    "DiffusionNoisePerturbation",
    "register_perturbation",
    "build_perturbation",
    "list_perturbations",
]


class Perturbation(ABC):
    """Abstract base class for image perturbations.

    Subclasses set a non-empty ``name`` and implement :meth:`__call__`.
    """

    name: ClassVar[str] = ""

    @abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Return a distorted copy of ``image`` (same shape/dtype/device)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class GaussianNoisePerturbation(Perturbation):
    """Add isotropic Gaussian noise: ``image + std * eps``, ``eps ~ N(0, I)``."""

    name = "gaussian_noise"

    def __init__(
        self, std: float = 0.1, clip: tuple[float, float] | None = None
    ) -> None:
        if (
            not isinstance(std, Real)
            or isinstance(std, bool)
            or not math.isfinite(float(std))
            or std < 0
        ):
            raise ValueError(f"std must be >= 0, got {std}")
        if clip is not None:
            if (
                not isinstance(clip, (tuple, list))
                or len(clip) != 2
                or not all(
                    isinstance(v, Real) and math.isfinite(float(v)) for v in clip
                )
            ):
                raise ValueError("clip must contain two finite bounds")
            if clip[0] > clip[1]:
                raise ValueError("clip lower bound cannot exceed upper bound")
        self.std = float(std)
        self.clip = clip

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        out = image + torch.randn_like(image, dtype=image.dtype) * self.std
        if self.clip is not None:
            out = out.clamp(self.clip[0], self.clip[1])
        return out


class DiffusionNoisePerturbation(Perturbation):
    """DDPM forward-process noise — the distortion used by VCD.

    At timestep ``noise_step`` the image becomes
    ``sqrt(alpha_bar_t) * image + sqrt(1 - alpha_bar_t) * eps``, using the
    linear-in-sigmoid-space beta schedule of the original DDPM.
    """

    name = "diffusion_noise"

    def __init__(self, noise_step: int = 500, num_steps: int = 1000) -> None:
        if not isinstance(noise_step, int) or isinstance(noise_step, bool):
            raise ValueError("noise_step must be an integer")
        if (
            not isinstance(num_steps, int)
            or isinstance(num_steps, bool)
            or num_steps < 1
        ):
            raise ValueError("num_steps must be a positive integer")
        if not (0 <= noise_step < num_steps):
            raise ValueError(
                f"noise_step must be in [0, {num_steps}), got {noise_step}"
            )
        self.noise_step = int(noise_step)
        self.num_steps = int(num_steps)
        self._alpha_bar = self._make_schedule(num_steps)

    @staticmethod
    def _make_schedule(num_steps: int) -> torch.Tensor:
        betas = torch.linspace(-6, 6, num_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        alphas = 1.0 - betas
        return torch.cumprod(alphas, dim=0)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        alpha_bar = self._alpha_bar[self.noise_step].to(
            device=image.device, dtype=image.dtype
        )
        eps = torch.randn_like(image, dtype=image.dtype)
        return torch.sqrt(alpha_bar) * image + torch.sqrt(1.0 - alpha_bar) * eps


# -- registry ---------------------------------------------------------------

_PERTURBATIONS: dict[str, type[Perturbation]] = {}


def register_perturbation(
    cls: type[Perturbation] | None = None, *, name: str | None = None
) -> type[Perturbation] | Callable[[type[Perturbation]], type[Perturbation]]:
    """Register a :class:`Perturbation` subclass under ``cls.name`` (or ``name``).

    Usable as ``@register_perturbation`` or ``@register_perturbation(name=...)``.
    """

    def _register(c: type[Perturbation]) -> type[Perturbation]:
        if not (isinstance(c, type) and issubclass(c, Perturbation)):
            raise TypeError(
                "@register_perturbation can only decorate Perturbation "
                f"subclasses, got {c!r}"
            )
        key = name or getattr(c, "name", "") or ""
        if not key:
            raise ValueError(f"cannot register {c.__name__}: set a non-empty `name`")
        existing = _PERTURBATIONS.get(key)
        if existing is not None and existing is not c:
            raise ValueError(
                f"perturbation {key!r} is already registered by {existing.__name__}"
            )
        _PERTURBATIONS[key] = c
        return c

    if isinstance(cls, type):
        return _register(cls)
    return _register


def build_perturbation(name: str, **kwargs: Any) -> Perturbation:
    """Instantiate a registered perturbation by name."""
    try:
        cls = _PERTURBATIONS[name]
    except KeyError:
        raise KeyError(
            f"unknown perturbation {name!r}; registered perturbations: "
            f"{sorted(_PERTURBATIONS)}"
        ) from None
    return cls(**kwargs)


def list_perturbations() -> list[str]:
    """Return the sorted names of all registered perturbations."""
    return sorted(_PERTURBATIONS)


register_perturbation(GaussianNoisePerturbation)
register_perturbation(DiffusionNoisePerturbation)
