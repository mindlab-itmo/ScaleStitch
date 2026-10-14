"""Bounded radius-conditioned feature correction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ScaleStitchConfig:

    feature_dim: int = 1024
    rank: int = 16
    log_radius_mean: float = 0.0
    log_radius_std: float = 1.0
    max_angle_degrees: float = 15.0
    tau: float = 1.0
    radius_epsilon: float = 1.0e-6
    norm_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if not math.isfinite(self.log_radius_mean):
            raise ValueError("log_radius_mean must be finite")
        if not math.isfinite(self.log_radius_std) or self.log_radius_std <= 0.0:
            raise ValueError("log_radius_std must be finite and positive")
        if (
            not math.isfinite(self.max_angle_degrees)
            or self.max_angle_degrees < 0.0
            or self.max_angle_degrees >= 90.0
        ):
            raise ValueError("max_angle_degrees must be in [0, 90)")
        if not math.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and positive")
        if not math.isfinite(self.radius_epsilon) or self.radius_epsilon <= 0.0:
            raise ValueError("radius_epsilon must be finite and positive")
        if not math.isfinite(self.norm_epsilon) or self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be finite and positive")


@dataclass
class ScaleStitchDiagnostics:

    descriptor: torch.Tensor
    standardized_log_radius: torch.Tensor
    validity_mask: torch.Tensor
    native_norm: torch.Tensor
    output_norm: torch.Tensor
    raw_residual: torch.Tensor
    tangent_residual: torch.Tensor
    bounded_residual: torch.Tensor
    raw_residual_norm: torch.Tensor
    tangent_residual_norm: torch.Tensor
    bounded_residual_norm: torch.Tensor
    residual_scale: torch.Tensor
    tangent_radial_component: torch.Tensor
    angular_change_radians: torch.Tensor
    angular_bound_radians: torch.Tensor
    zero_descriptor_mask: torch.Tensor
    identity_mask: torch.Tensor


class _LowRankConditionalResidual(nn.Module):

    descriptor_dim = 1

    def __init__(self, feature_dim: int, rank: int) -> None:
        super().__init__()
        self.feature_down = nn.Linear(feature_dim, rank, bias=False)
        self.descriptor_down = nn.Linear(self.descriptor_dim, rank, bias=False)
        self.output_up = nn.Linear(rank, feature_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.feature_down.weight, a=math.sqrt(5.0))
        nn.init.normal_(self.descriptor_down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_up.weight)

    def forward(self, native: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        hidden = self.feature_down(native) + self.descriptor_down(descriptor)
        return self.output_up(F.silu(hidden))


ScaleStitchReturn = Union[
    torch.Tensor,
    Tuple[torch.Tensor, ScaleStitchDiagnostics],
]


class ScaleStitchEncoder(nn.Module):

    descriptor_dim = 1

    def __init__(self, config: ScaleStitchConfig) -> None:
        super().__init__()
        self.config = config


        self.register_buffer(
            "log_radius_mean",
            torch.tensor(config.log_radius_mean, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "log_radius_std",
            torch.tensor(config.log_radius_std, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "max_angle_radians",
            torch.tensor(math.radians(config.max_angle_degrees), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "tau", torch.tensor(config.tau, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "radius_epsilon",
            torch.tensor(config.radius_epsilon, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "norm_epsilon",
            torch.tensor(config.norm_epsilon, dtype=torch.float32),
            persistent=True,
        )
        self.residual = _LowRankConditionalResidual(config.feature_dim, config.rank)

    @staticmethod
    def _flatten_column(
        value: torch.Tensor,
        leading_shape: torch.Size,
        name: str,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device)
        expected = math.prod(leading_shape)
        if tensor.numel() != expected:
            raise ValueError(
                f"{name} must contain one value per feature row: "
                f"expected {expected}, got {tensor.numel()}"
            )
        return tensor.reshape(expected, 1)

    def build_descriptor(
        self,
        native_features: torch.Tensor,
        object_radius: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self._validate_native(native_features)
        leading_shape = native_features.shape[:-1]
        radius = self._flatten_column(
            object_radius, leading_shape, "object_radius", native_features.device
        ).to(torch.float32)

        valid = torch.isfinite(radius) & (radius > self.radius_epsilon)
        if validity_mask is not None:
            requested = self._flatten_column(
                validity_mask, leading_shape, "validity_mask", native_features.device
            )
            if requested.is_floating_point():
                requested = torch.isfinite(requested) & (requested != 0)
            else:
                requested = requested != 0
            valid = valid & requested

        safe_radius = torch.where(valid, radius, torch.ones_like(radius))
        standardized = (
            torch.log(safe_radius) - self.log_radius_mean
        ) / self.log_radius_std


        canonical_radius = torch.exp(self.log_radius_mean)
        canonical_tolerance = 2.0 * torch.finfo(torch.float32).eps
        at_training_mean = valid & torch.isclose(
            radius,
            canonical_radius,
            rtol=canonical_tolerance,
            atol=0.0,
        )
        standardized = torch.where(
            at_training_mean, torch.zeros_like(standardized), standardized
        )
        standardized = torch.where(valid, standardized, torch.zeros_like(standardized))
        return standardized, standardized, valid

    def _validate_native(self, native_features: torch.Tensor) -> None:
        if not isinstance(native_features, torch.Tensor):
            raise TypeError("native_features must be a torch.Tensor")
        if native_features.ndim < 2:
            raise ValueError("native_features must have shape [..., feature_dim]")
        if native_features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"expected native feature width {self.config.feature_dim}, "
                f"got {native_features.shape[-1]}"
            )
        if not native_features.is_floating_point():
            raise TypeError("native_features must have a floating-point dtype")

    def _conditional_difference(
        self, native_unit: torch.Tensor, descriptor: torch.Tensor
    ) -> torch.Tensor:

        operator_dtype = self.residual.feature_down.weight.dtype
        stopped = native_unit.detach().to(dtype=operator_dtype)
        conditioned = descriptor.to(dtype=operator_dtype)
        zero_descriptor = torch.zeros_like(conditioned)
        return self.residual(stopped, conditioned) - self.residual(stopped, zero_descriptor)

    def _smooth_tanh_ratio(self, radius: torch.Tensor) -> torch.Tensor:

        scaled = radius / self.tau
        scaled_squared = scaled * scaled
        series = (
            1.0
            - scaled_squared / 3.0
            + 2.0 * scaled_squared * scaled_squared / 15.0
        ) / self.tau
        safe_radius = radius.clamp_min(self.norm_epsilon)
        direct = torch.tanh(scaled) / safe_radius
        return torch.where(radius <= self.norm_epsilon, series, direct)

    def apply_descriptor(
        self,
        native_features: torch.Tensor,
        descriptor: torch.Tensor,
        return_diagnostics: bool = False,
        standardized_log_radius: Optional[torch.Tensor] = None,
        validity_mask: Optional[torch.Tensor] = None,
    ) -> ScaleStitchReturn:

        self._validate_native(native_features)
        leading_shape = native_features.shape[:-1]
        batch_size = math.prod(leading_shape)
        descriptor_flat = torch.as_tensor(descriptor, device=native_features.device)
        if descriptor_flat.numel() != batch_size * self.descriptor_dim:
            raise ValueError(
                "descriptor must contain exactly one value per feature row "
                f"(expected {batch_size * self.descriptor_dim}, "
                f"got {descriptor_flat.numel()})"
            )
        descriptor_flat = descriptor_flat.reshape(batch_size, self.descriptor_dim).to(
            torch.float32
        )
        if not torch.isfinite(descriptor_flat).all():
            raise ValueError("descriptor must contain only finite values")

        native_flat = native_features.reshape(batch_size, self.config.feature_dim)
        native_fp32 = native_flat.to(torch.float32)
        base_geometry = native_fp32.detach()
        native_norm = torch.linalg.vector_norm(base_geometry, dim=-1, keepdim=True)
        safe_native_norm = native_norm.clamp_min(self.norm_epsilon)
        native_unit = base_geometry / safe_native_norm

        if validity_mask is None:
            validity_flat = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=native_features.device
            )
        else:
            validity_flat = self._flatten_column(
                validity_mask,
                leading_shape,
                "validity_mask",
                native_features.device,
            )
            if validity_flat.is_floating_point():
                validity_flat = torch.isfinite(validity_flat) & (validity_flat != 0)
            else:
                validity_flat = validity_flat != 0

        raw_residual = self._conditional_difference(native_unit, descriptor_flat).to(
            torch.float32
        )
        raw_residual = raw_residual * validity_flat.to(torch.float32)

        radial_component = (raw_residual * native_unit).sum(dim=-1, keepdim=True)
        tangent_residual = raw_residual - radial_component * native_unit
        tangent_radial_component = (tangent_residual * native_unit).sum(
            dim=-1, keepdim=True
        )
        tangent_norm = torch.linalg.vector_norm(tangent_residual, dim=-1, keepdim=True)


        angle_bound = self.max_angle_radians
        max_bounded_norm = torch.tan(angle_bound)
        residual_scale = max_bounded_norm * self._smooth_tanh_ratio(tangent_norm)
        bounded_residual = tangent_residual * residual_scale
        bounded_norm = torch.linalg.vector_norm(bounded_residual, dim=-1, keepdim=True)

        candidate = native_unit + bounded_residual
        candidate_unit = F.normalize(
            candidate, p=2.0, dim=-1, eps=self.norm_epsilon
        )
        updated_geometry = native_norm * candidate_unit

        zero_descriptor_mask = (descriptor_flat == 0).all(dim=-1, keepdim=True)
        zero_native_mask = native_norm <= self.norm_epsilon
        identity_mask = zero_descriptor_mask | (~validity_flat) | zero_native_mask
        updated_geometry = torch.where(identity_mask, base_geometry, updated_geometry)


        updated_fp32 = native_fp32 + (updated_geometry - base_geometry)
        updated = updated_fp32.to(dtype=native_features.dtype)
        updated = torch.where(identity_mask, native_flat, updated)
        output = updated.reshape(native_features.shape)

        if not return_diagnostics:
            return output

        output_norm = torch.linalg.vector_norm(
            output.reshape(batch_size, self.config.feature_dim).to(torch.float32),
            dim=-1,
            keepdim=True,
        )
        angular_change = torch.atan2(bounded_norm, torch.ones_like(bounded_norm))
        angular_change = torch.where(
            identity_mask, torch.zeros_like(angular_change), angular_change
        )

        if standardized_log_radius is None:
            standardized_flat = descriptor_flat[:, :1]
        else:
            standardized_flat = self._flatten_column(
                standardized_log_radius,
                leading_shape,
                "standardized_log_radius",
                native_features.device,
            ).to(torch.float32)
        diagnostics = ScaleStitchDiagnostics(
            descriptor=descriptor_flat,
            standardized_log_radius=standardized_flat,
            validity_mask=validity_flat,
            native_norm=native_norm,
            output_norm=output_norm,
            raw_residual=raw_residual,
            tangent_residual=tangent_residual,
            bounded_residual=bounded_residual,
            raw_residual_norm=torch.linalg.vector_norm(
                raw_residual, dim=-1, keepdim=True
            ),
            tangent_residual_norm=tangent_norm,
            bounded_residual_norm=bounded_norm,
            residual_scale=residual_scale,
            tangent_radial_component=tangent_radial_component,
            angular_change_radians=angular_change,
            angular_bound_radians=torch.ones_like(angular_change) * angle_bound,
            zero_descriptor_mask=zero_descriptor_mask,
            identity_mask=identity_mask,
        )
        return output, diagnostics

    def forward(
        self,
        native_features: torch.Tensor,
        object_radius: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> ScaleStitchReturn:
        descriptor, standardized, valid = self.build_descriptor(
            native_features, object_radius, validity_mask
        )
        return self.apply_descriptor(
            native_features,
            descriptor,
            return_diagnostics=return_diagnostics,
            standardized_log_radius=standardized,
            validity_mask=valid,
        )
