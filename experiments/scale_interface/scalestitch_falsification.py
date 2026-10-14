"""Linear, nonlinear and radius-conditioned feature controls."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scalestitch_encoder import ScaleStitchConfig, ScaleStitchEncoder


VARIANTS: Tuple[str, ...] = (
    "canonical_linear",
    "canonical_mlp",
    "radius_concat",
    "radius_film",
    "unconstrained_residual",
    "scalestitch",
)


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


class CanonicalLinearBridge(nn.Module):

    kind = "canonical_linear"

    def __init__(self, native_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(native_dim, feature_dim)

    def forward(self, native: torch.Tensor) -> torch.Tensor:
        return self.projection(native)


class CanonicalMLPBridge(nn.Module):

    kind = "canonical_mlp"

    def __init__(self, native_dim: int, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(native_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, native: torch.Tensor) -> torch.Tensor:
        return self.network(native)


def scale_adapter_parameter_budget(feature_dim: int, rank: int) -> int:

    if feature_dim <= 0 or rank <= 0:
        raise ValueError("feature_dim and rank must be positive")
    return 2 * feature_dim * rank + rank


def closest_parameter_matched_hidden_dim(
    native_dim: int,
    feature_dim: int,
    target_parameters: int,
) -> Tuple[int, int, float]:

    if native_dim <= 0 or feature_dim <= 0 or target_parameters <= 0:
        raise ValueError("dimensions and target_parameters must be positive")
    stride = native_dim + feature_dim + 1
    raw = (target_parameters - feature_dim) / stride
    candidates = sorted({max(1, math.floor(raw)), max(1, math.ceil(raw))})

    def count(width: int) -> int:
        return width * stride + feature_dim

    hidden = min(candidates, key=lambda width: (abs(count(width) - target_parameters), width))
    observed = count(hidden)
    mismatch = abs(observed - target_parameters) / target_parameters
    return hidden, observed, mismatch


@dataclass(frozen=True)
class BridgeBuild:
    module: nn.Module
    kind: str
    target_parameters: int
    observed_parameters: int
    relative_mismatch: float
    hidden_dim: int


def build_canonical_bridge(
    variant: str,
    native_dim: int,
    feature_dim: int,
    rank: int,
    *,
    maximum_parameter_mismatch: float = 0.01,
) -> BridgeBuild:

    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    linear = CanonicalLinearBridge(native_dim, feature_dim)
    linear_count = parameter_count(linear)
    target = linear_count + scale_adapter_parameter_budget(feature_dim, rank)
    if variant != "canonical_mlp":
        return BridgeBuild(
            module=linear,
            kind=linear.kind,
            target_parameters=linear_count,
            observed_parameters=linear_count,
            relative_mismatch=0.0,
            hidden_dim=0,
        )
    hidden, observed, mismatch = closest_parameter_matched_hidden_dim(
        native_dim, feature_dim, target
    )
    if mismatch > maximum_parameter_mismatch:
        raise RuntimeError(
            "canonical_mlp parameter matching failed: "
            f"target={target}, observed={observed}, mismatch={mismatch:.4%}"
        )
    bridge = CanonicalMLPBridge(native_dim, feature_dim, hidden)
    if parameter_count(bridge) != observed:
        raise AssertionError("canonical_mlp analytic parameter count is inconsistent")
    return BridgeBuild(
        module=bridge,
        kind=bridge.kind,
        target_parameters=target,
        observed_parameters=observed,
        relative_mismatch=mismatch,
        hidden_dim=hidden,
    )


def standardized_log_radius(
    radius: torch.Tensor,
    validity_mask: torch.Tensor,
    *,
    log_radius_mean: float,
    log_radius_std: float,
    radius_epsilon: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if not math.isfinite(log_radius_mean):
        raise ValueError("log_radius_mean must be finite")
    if not math.isfinite(log_radius_std) or log_radius_std <= 0.0:
        raise ValueError("log_radius_std must be finite and positive")
    radius_column = torch.as_tensor(radius).reshape(-1, 1).to(torch.float32)
    requested = torch.as_tensor(validity_mask, device=radius_column.device).reshape(-1, 1)
    if requested.numel() != radius_column.numel():
        raise ValueError("validity_mask must contain one value per radius")
    if requested.is_floating_point():
        requested = torch.isfinite(requested) & (requested != 0)
    else:
        requested = requested != 0
    valid = requested & torch.isfinite(radius_column) & (radius_column > radius_epsilon)
    safe = torch.where(valid, radius_column, torch.ones_like(radius_column))
    descriptor = (torch.log(safe) - log_radius_mean) / log_radius_std
    descriptor = torch.where(valid, descriptor, torch.zeros_like(descriptor))
    return descriptor, valid


class IdentityScaleAdapter(nn.Module):

    architecture = "identity_no_radius"

    def forward(
        self,
        base: torch.Tensor,
        radius: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> torch.Tensor:
        del radius, validity_mask
        return base


class _DescriptorAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        rank: int,
        log_radius_mean: float,
        log_radius_std: float,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or rank <= 0:
            raise ValueError("feature_dim and rank must be positive")
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.register_buffer(
            "log_radius_mean", torch.tensor(log_radius_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "log_radius_std", torch.tensor(log_radius_std, dtype=torch.float32)
        )

    def descriptor(
        self, radius: torch.Tensor, validity_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return standardized_log_radius(
            radius,
            validity_mask,
            log_radius_mean=float(self.log_radius_mean),
            log_radius_std=float(self.log_radius_std),
        )

    def _validate_base(self, base: torch.Tensor) -> None:
        if base.ndim != 2 or base.shape[-1] != self.feature_dim:
            raise ValueError(
                f"base must have shape [B,{self.feature_dim}], got {tuple(base.shape)}"
            )
        if not base.is_floating_point() or not torch.isfinite(base).all():
            raise ValueError("base must be finite floating-point features")


class RadiusConcatAdapter(_DescriptorAdapter):

    architecture = "radius_concat_low_rank"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_down = nn.Linear(self.feature_dim + 1, self.rank, bias=False)
        self.output_up = nn.Linear(self.rank, self.feature_dim, bias=False)
        nn.init.zeros_(self.output_up.weight)

    def forward(
        self, base: torch.Tensor, radius: torch.Tensor, validity_mask: torch.Tensor
    ) -> torch.Tensor:
        self._validate_base(base)
        descriptor, valid = self.descriptor(radius, validity_mask)
        conditioned = torch.cat((base.detach().to(torch.float32), descriptor), dim=-1)
        residual = self.output_up(F.silu(self.input_down(conditioned)))
        residual = residual * valid.to(residual.dtype)
        return base + residual.to(base.dtype)


class RadiusFiLMAdapter(_DescriptorAdapter):

    architecture = "radius_film_low_rank"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.descriptor_down = nn.Linear(1, self.rank, bias=False)
        self.affine_up = nn.Linear(self.rank, 2 * self.feature_dim, bias=False)
        nn.init.zeros_(self.affine_up.weight)

    def forward(
        self, base: torch.Tensor, radius: torch.Tensor, validity_mask: torch.Tensor
    ) -> torch.Tensor:
        self._validate_base(base)
        descriptor, valid = self.descriptor(radius, validity_mask)
        affine = self.affine_up(F.silu(self.descriptor_down(descriptor)))
        scale, shift = affine.chunk(2, dim=-1)
        scale = scale * valid.to(scale.dtype)
        shift = shift * valid.to(shift.dtype)
        return base * (1.0 + scale.to(base.dtype)) + shift.to(base.dtype)


class UnconstrainedResidualAdapter(_DescriptorAdapter):

    architecture = "unconstrained_conditional_residual"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feature_down = nn.Linear(self.feature_dim, self.rank, bias=False)
        self.descriptor_down = nn.Linear(1, self.rank, bias=False)
        self.output_up = nn.Linear(self.rank, self.feature_dim, bias=False)


        nn.init.kaiming_uniform_(self.feature_down.weight, a=math.sqrt(5.0))
        nn.init.normal_(self.descriptor_down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_up.weight)

    def _operator(self, base: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        hidden = self.feature_down(base.detach().to(torch.float32))
        hidden = hidden + self.descriptor_down(descriptor)
        return self.output_up(F.silu(hidden))

    def forward(
        self, base: torch.Tensor, radius: torch.Tensor, validity_mask: torch.Tensor
    ) -> torch.Tensor:
        self._validate_base(base)
        descriptor, valid = self.descriptor(radius, validity_mask)
        residual = self._operator(base, descriptor) - self._operator(
            base, torch.zeros_like(descriptor)
        )
        residual = residual * valid.to(residual.dtype)
        return base + residual.to(base.dtype)


class ScaleStitchDiagnosticAdapter(nn.Module):

    architecture = "scalestitch_tangent_bounded"

    def __init__(
        self,
        feature_dim: int,
        rank: int,
        log_radius_mean: float,
        log_radius_std: float,
        max_angle_degrees: float,
        tau: float,
    ) -> None:
        super().__init__()
        self.scalestitch = ScaleStitchEncoder(
            ScaleStitchConfig(
                feature_dim=feature_dim,
                rank=rank,
                log_radius_mean=log_radius_mean,
                log_radius_std=log_radius_std,
                max_angle_degrees=max_angle_degrees,
                tau=tau,
            )
        )

    def forward(
        self, base: torch.Tensor, radius: torch.Tensor, validity_mask: torch.Tensor
    ) -> torch.Tensor:
        descriptor, valid = standardized_log_radius(
            radius,
            validity_mask,
            log_radius_mean=float(self.scalestitch.log_radius_mean),
            log_radius_std=float(self.scalestitch.log_radius_std),
            radius_epsilon=float(self.scalestitch.radius_epsilon),
        )
        return self.scalestitch.apply_descriptor(
            base, descriptor, validity_mask=valid
        )


@dataclass(frozen=True)
class AdapterBuild:
    module: nn.Module
    architecture: str
    expected_parameters: int
    observed_parameters: int


def build_scale_adapter(
    variant: str,
    feature_dim: int,
    rank: int,
    log_radius_mean: float,
    log_radius_std: float,
    *,
    max_angle_degrees: float = 15.0,
    tau: float = 1.0,
) -> AdapterBuild:

    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    expected = 0
    common = (feature_dim, rank, log_radius_mean, log_radius_std)
    if variant in {"canonical_linear", "canonical_mlp"}:
        module: nn.Module = IdentityScaleAdapter()
    elif variant == "radius_concat":
        module = RadiusConcatAdapter(*common)
        expected = scale_adapter_parameter_budget(feature_dim, rank)
    elif variant == "radius_film":
        module = RadiusFiLMAdapter(*common)
        expected = scale_adapter_parameter_budget(feature_dim, rank)
    elif variant == "unconstrained_residual":
        module = UnconstrainedResidualAdapter(*common)
        expected = scale_adapter_parameter_budget(feature_dim, rank)
    else:
        module = ScaleStitchDiagnosticAdapter(
            feature_dim,
            rank,
            log_radius_mean,
            log_radius_std,
            max_angle_degrees,
            tau,
        )
        expected = scale_adapter_parameter_budget(feature_dim, rank)
    observed = parameter_count(module)
    if observed != expected:
        raise AssertionError(
            f"{variant} parameter contract failed: expected {expected}, observed {observed}"
        )
    return AdapterBuild(module, getattr(module, "architecture"), expected, observed)


class MetricRecoveryProbe(nn.Module):

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.regressor = nn.Linear(feature_dim, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.regressor(feature).squeeze(-1)


def semantic_alignment_loss(
    feature: torch.Tensor, teacher_feature: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    student = F.normalize(feature.to(torch.float32), dim=-1)
    teacher = F.normalize(teacher_feature.detach().to(torch.float32), dim=-1)
    cosine = 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
    normalized_mse = F.mse_loss(student, teacher)
    return cosine, {
        "cosine_loss": cosine,
        "normalized_mse_diagnostic": normalized_mse,
    }


def canonical_alignment_loss(
    feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    *,
    log_norm_loss_weight: float,
    norm_epsilon: float = 1.0e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    if not math.isfinite(log_norm_loss_weight) or log_norm_loss_weight < 0.0:
        raise ValueError("log_norm_loss_weight must be finite and nonnegative")
    if not math.isfinite(norm_epsilon) or norm_epsilon <= 0.0:
        raise ValueError("norm_epsilon must be finite and positive")
    angular, metrics = semantic_alignment_loss(feature, teacher_feature)
    student_norm = torch.linalg.vector_norm(feature.to(torch.float32), dim=-1)
    teacher_norm = torch.linalg.vector_norm(
        teacher_feature.detach().to(torch.float32), dim=-1
    )
    if torch.any(student_norm <= norm_epsilon) or torch.any(teacher_norm <= norm_epsilon):
        raise ValueError("canonical alignment requires nonzero feature norms")
    log_norm_error = torch.log(student_norm) - torch.log(teacher_norm)
    log_norm_loss = F.smooth_l1_loss(
        log_norm_error, torch.zeros_like(log_norm_error)
    )
    total = angular + float(log_norm_loss_weight) * log_norm_loss
    return total, {
        **metrics,
        "log_norm_smooth_l1_loss": log_norm_loss,
        "mean_abs_log_norm_error": log_norm_error.abs().mean(),
        "mean_norm_ratio": torch.exp(log_norm_error).mean(),
    }
