"""Direct radius shifts and parameter-matched controls."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from .scalestitch_encoder import ScaleStitchConfig, ScaleStitchEncoder
from .scalestitch_falsification import standardized_log_radius


def _validate_feature_dim(feature_dim: int) -> None:
    if (
        not isinstance(feature_dim, int)
        or isinstance(feature_dim, bool)
        or feature_dim <= 0
    ):
        raise ValueError("feature_dim must be a positive integer")


def _validity_column(
    radius: torch.Tensor,
    validity_mask: torch.Tensor,
    *,
    rows: int,
    device: torch.device,
) -> torch.Tensor:

    supplied_radius = torch.as_tensor(radius, device=device).reshape(-1, 1)
    requested = torch.as_tensor(validity_mask, device=device).reshape(-1, 1)
    if supplied_radius.shape[0] != rows or requested.shape[0] != rows:
        raise ValueError("radius/mask must contain one value per base row")
    if requested.is_floating_point():
        return torch.isfinite(requested) & (requested != 0)
    return requested != 0


class DirectRadiusShiftAdapter(nn.Module):

    architecture = "radius_linear_shift"

    def __init__(
        self,
        feature_dim: int,
        log_radius_mean: float,
        log_radius_std: float,
    ) -> None:
        super().__init__()
        _validate_feature_dim(feature_dim)
        self.feature_dim = feature_dim
        self.register_buffer(
            "log_radius_mean", torch.tensor(log_radius_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "log_radius_std", torch.tensor(log_radius_std, dtype=torch.float32)
        )
        self.descriptor_up = nn.Linear(1, feature_dim, bias=False)
        nn.init.zeros_(self.descriptor_up.weight)

    def forward(
        self,
        base: torch.Tensor,
        radius: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> torch.Tensor:
        if base.ndim != 2 or base.shape[-1] != self.feature_dim:
            raise ValueError(
                f"base must have shape [B,{self.feature_dim}], got {tuple(base.shape)}"
            )
        if not base.is_floating_point() or not torch.isfinite(base).all():
            raise ValueError("base must be finite floating-point features")
        descriptor, valid = standardized_log_radius(
            radius,
            validity_mask,
            log_radius_mean=float(self.log_radius_mean),
            log_radius_std=float(self.log_radius_std),
        )
        if descriptor.shape[0] != base.shape[0]:
            raise ValueError("radius/mask must contain one value per base row")
        residual = self.descriptor_up(descriptor)
        residual = residual * valid.to(residual.dtype)
        return base + residual.to(base.dtype)


class ConstantLinearShiftAdapter(nn.Module):

    architecture = "constant_linear_shift_no_factor"

    def __init__(
        self,
        feature_dim: int,
        log_radius_mean: float,
        log_radius_std: float,
    ) -> None:
        super().__init__()
        _validate_feature_dim(feature_dim)
        if not math.isfinite(log_radius_mean):
            raise ValueError("log_radius_mean must be finite")
        if not math.isfinite(log_radius_std) or log_radius_std <= 0.0:
            raise ValueError("log_radius_std must be finite and positive")
        self.feature_dim = feature_dim


        self.register_buffer(
            "log_radius_mean", torch.tensor(log_radius_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "log_radius_std", torch.tensor(log_radius_std, dtype=torch.float32)
        )
        self.descriptor_up = nn.Linear(1, feature_dim, bias=False)
        nn.init.zeros_(self.descriptor_up.weight)

    def forward(
        self,
        base: torch.Tensor,
        radius: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> torch.Tensor:
        if base.ndim != 2 or base.shape[-1] != self.feature_dim:
            raise ValueError(
                f"base must have shape [B,{self.feature_dim}], got {tuple(base.shape)}"
            )
        if not base.is_floating_point() or not torch.isfinite(base).all():
            raise ValueError("base must be finite floating-point features")
        valid = _validity_column(
            radius,
            validity_mask,
            rows=base.shape[0],
            device=base.device,
        )
        constant = valid.to(torch.float32)
        residual = self.descriptor_up(constant) * valid.to(torch.float32)
        return base + residual.to(base.dtype)


class ConstantDescriptorScaleStitchAdapter(nn.Module):

    architecture = "scalestitch_constant_descriptor_no_factor"

    def __init__(
        self,
        feature_dim: int,
        rank: int,
        log_radius_mean: float,
        log_radius_std: float,
        max_angle_degrees: float,
        tau: float,
        *,
        radius_epsilon: float = 1.0e-6,
        norm_epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.scalestitch = ScaleStitchEncoder(
            ScaleStitchConfig(
                feature_dim=feature_dim,
                rank=rank,
                log_radius_mean=log_radius_mean,
                log_radius_std=log_radius_std,
                max_angle_degrees=max_angle_degrees,
                tau=tau,
                radius_epsilon=radius_epsilon,
                norm_epsilon=norm_epsilon,
            )
        )

    @property
    def log_radius_mean(self) -> torch.Tensor:
        return self.scalestitch.log_radius_mean

    @property
    def log_radius_std(self) -> torch.Tensor:
        return self.scalestitch.log_radius_std

    @property
    def max_angle_radians(self) -> torch.Tensor:
        return self.scalestitch.max_angle_radians

    @property
    def tau(self) -> torch.Tensor:
        return self.scalestitch.tau

    @property
    def radius_epsilon(self) -> torch.Tensor:
        return self.scalestitch.radius_epsilon

    @property
    def norm_epsilon(self) -> torch.Tensor:
        return self.scalestitch.norm_epsilon

    def forward(
        self,
        base: torch.Tensor,
        radius: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> torch.Tensor:
        if base.ndim != 2 or base.shape[-1] != self.feature_dim:
            raise ValueError(
                f"base must have shape [B,{self.feature_dim}], got {tuple(base.shape)}"
            )
        if not base.is_floating_point() or not torch.isfinite(base).all():
            raise ValueError("base must be finite floating-point features")
        valid = _validity_column(
            radius,
            validity_mask,
            rows=base.shape[0],
            device=base.device,
        )
        constant_descriptor = valid.to(torch.float32)
        return self.scalestitch.apply_descriptor(
            base,
            constant_descriptor,
            validity_mask=valid,
        )


@dataclass(frozen=True)
class DirectAdapterBuild:
    module: nn.Module
    architecture: str
    expected_parameters: int
    observed_parameters: int


def build_direct_radius_adapter(
    feature_dim: int,
    log_radius_mean: float,
    log_radius_std: float,
) -> DirectAdapterBuild:
    module = DirectRadiusShiftAdapter(
        feature_dim, log_radius_mean, log_radius_std
    )
    observed = sum(parameter.numel() for parameter in module.parameters())
    if observed != feature_dim:
        raise AssertionError(
            f"direct radius control expected {feature_dim} parameters, got {observed}"
        )
    return DirectAdapterBuild(
        module=module,
        architecture=module.architecture,
        expected_parameters=feature_dim,
        observed_parameters=observed,
    )


def build_constant_linear_adapter(
    feature_dim: int,
    log_radius_mean: float,
    log_radius_std: float,
) -> DirectAdapterBuild:
    module = ConstantLinearShiftAdapter(
        feature_dim, log_radius_mean, log_radius_std
    )
    observed = sum(parameter.numel() for parameter in module.parameters())
    if observed != feature_dim:
        raise AssertionError(
            f"constant linear control expected {feature_dim} parameters, got {observed}"
        )
    return DirectAdapterBuild(
        module=module,
        architecture=module.architecture,
        expected_parameters=feature_dim,
        observed_parameters=observed,
    )


def build_constant_scalestitch_adapter(
    feature_dim: int,
    rank: int,
    log_radius_mean: float,
    log_radius_std: float,
    max_angle_degrees: float,
    tau: float,
    *,
    radius_epsilon: float = 1.0e-6,
    norm_epsilon: float = 1.0e-8,
) -> DirectAdapterBuild:
    module = ConstantDescriptorScaleStitchAdapter(
        feature_dim,
        rank,
        log_radius_mean,
        log_radius_std,
        max_angle_degrees,
        tau,
        radius_epsilon=radius_epsilon,
        norm_epsilon=norm_epsilon,
    )
    expected = 2 * feature_dim * rank + rank
    observed = sum(parameter.numel() for parameter in module.parameters())
    if observed != expected:
        raise AssertionError(
            f"constant ScaleStitch control expected {expected} parameters, got {observed}"
        )
    return DirectAdapterBuild(
        module=module,
        architecture=module.architecture,
        expected_parameters=expected,
        observed_parameters=observed,
    )
