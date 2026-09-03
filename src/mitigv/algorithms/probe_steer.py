"""LinearProbeSteer — steering with a linear probe's decision normal.

A self-implemented representative of the "representation steering" family
(alongside VISTA / SSL). A linear probe is trained on top of an intermediate
hidden state to classify *object presence / absence*; its decision normal
(the probe's weight vector) then defines a direction in the residual stream that
separates "visually grounded" from "ungrounded" continuations. At inference we
steer toward that direction::

    h_t^layer = h_t^layer + beta * (w / ||w||)

where ``w`` is the probe's weight vector and ``beta`` the injection strength
(scan ``{2, 5, 8, 12}``). Training does not update any model weight — only a
linear head is fit on frozen features — so the method stays training-free w.r.t.
the LVLM. Use the ``mitigv-train-probe`` command to fit the probe.
"""

from __future__ import annotations

from typing import Any

import torch

from mitigv.backends.generic import GenericMitigator, GenericMitigatorConfig
from mitigv.core.base import MitigatorConfigError
from mitigv.core.registry import register_mitigator

__all__ = ["LinearProbeSteerConfig", "LinearProbeSteer"]


class LinearProbeSteerConfig(GenericMitigatorConfig):
    """Hyper-parameters for LinearProbeSteer.

    Attributes:
        beta: Injection strength of the probe normal (scan ``{2, 5, 8, 12}``).
            ``beta=0`` disables steering (plain decoding).
        layer: Index of the decoder layer whose residual stream is steered. Must
            match the layer the probe was trained on.
    """

    beta: float = 5.0
    layer: int = 16

    def validate(self) -> None:
        super().validate()
        if self.beta < 0:
            raise MitigatorConfigError("beta must be >= 0")
        if self.layer < 0:
            raise MitigatorConfigError("layer must be >= 0")


@register_mitigator("linear_probe_steer")
class LinearProbeSteer(GenericMitigator):
    """Steer decoding with a linear probe's decision normal.

    ``steering_vector`` is the probe's (unnormalized or normalized) weight
    vector of shape ``(hidden_dim,)``; it is unit-normalized at injection time.
    """

    algorithm_name = "linear_probe_steer"
    config_class = LinearProbeSteerConfig

    def __init__(
        self,
        model: Any = None,
        processor: Any = None,
        config: Any = None,
        steering_vector: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, processor=processor, config=config, **kwargs)
        self.steering_vector = steering_vector

    def _on_generate_start(self, cfg: LinearProbeSteerConfig) -> None:
        self._steer_hooks: list[Any] = []
        if cfg.beta == 0.0 or self.steering_vector is None:
            return

        if not isinstance(self.steering_vector, torch.Tensor):
            raise MitigatorConfigError("steering_vector must be a torch.Tensor")
        if self.steering_vector.ndim != 1:
            raise MitigatorConfigError("steering_vector must have shape (hidden_dim,)")
        vec = self.steering_vector.to(
            device=self.device, dtype=self.dtype or torch.float32
        )
        if not bool(torch.isfinite(vec).all()):
            raise MitigatorConfigError(
                "steering_vector must contain only finite values"
            )
        norm = vec.norm()
        if float(norm) == 0.0:
            raise MitigatorConfigError("steering_vector must have a non-zero norm")
        vec = vec / norm  # unit decision normal

        layers = self._language_model_layers()
        if cfg.layer >= len(layers):
            raise MitigatorConfigError(
                f"layer {cfg.layer} out of range for {len(layers)} decoder layers"
            )
        strength = cfg.beta

        def hook(module: Any, args: Any, output: Any) -> Any:
            delta = strength * vec.view(1, 1, -1)
            return self._add_to_layer_output(output, delta)

        self._steer_hooks.append(layers[cfg.layer].register_forward_hook(hook))

    def _on_generate_end(self) -> None:
        for handle in getattr(self, "_steer_hooks", []):
            handle.remove()
        self._steer_hooks = []
