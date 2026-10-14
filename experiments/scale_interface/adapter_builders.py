from typing import Any, Mapping
from .scalestitch_encoder import ScaleStitchConfig, ScaleStitchEncoder
from .scalestitch_falsification import CanonicalLinearBridge, CanonicalMLPBridge, build_scale_adapter
from .factor_adapter_controls import build_constant_scalestitch_adapter, build_direct_radius_adapter, build_constant_linear_adapter
import torch.nn as nn

ADAPTER_KIND_TO_VARIANT = {
    "identity_no_radius": "canonical_linear",
    "radius_linear_shift": "radius_linear",
    "constant_linear_shift_no_factor": "constant_linear",
    "radius_concat_low_rank": "radius_concat",
    "radius_film_low_rank": "radius_film",
    "unconstrained_conditional_residual": "unconstrained_residual",
    "scalestitch_tangent_bounded": "scalestitch",
    "scalestitch_constant_descriptor_no_factor": "scalestitch_no_factor",
}

def expected_variant_for_architecture(adapter_kind: str, bridge_kind: str) -> str:

    if not isinstance(adapter_kind, str) or not adapter_kind:
        raise ValueError("adapter_kind must be a non-empty string")
    if not isinstance(bridge_kind, str) or not bridge_kind:
        raise ValueError("bridge_kind must be a non-empty string")
    if adapter_kind == "identity_no_radius":
        if bridge_kind not in {"canonical_linear", "canonical_mlp"}:
            raise ValueError(
                "identity_no_radius requires canonical_linear or canonical_mlp bridge"
            )
        return bridge_kind
    variant = ADAPTER_KIND_TO_VARIANT.get(adapter_kind)
    if variant is None:
        raise ValueError(f"unsupported factor-adapter kind: {adapter_kind!r}")
    return variant


def _build_bridge(architecture: Mapping[str, Any]) -> nn.Module:
    kind = architecture["bridge_kind"]
    native_dim = architecture["bridge_input_dim"]
    feature_dim = architecture["feature_dim"]
    hidden_dim = architecture["bridge_hidden_dim"]
    if kind == "canonical_linear":
        if hidden_dim != 0:
            raise ValueError("canonical_linear requires bridge_hidden_dim=0")
        return CanonicalLinearBridge(native_dim, feature_dim)
    if kind == "canonical_mlp":
        if hidden_dim <= 0:
            raise ValueError("canonical_mlp requires a positive bridge_hidden_dim")
        return CanonicalMLPBridge(native_dim, feature_dim, hidden_dim)
    raise ValueError(f"unsupported deployment bridge_kind: {kind!r}")


def _build_scalestitch(
    architecture: Mapping[str, Any], radius: Mapping[str, Any]
) -> nn.Module:

    kind = architecture["adapter_kind"]
    variant = expected_variant_for_architecture(kind, architecture["bridge_kind"])
    if variant == "scalestitch":
        return ScaleStitchEncoder(
            ScaleStitchConfig(
                feature_dim=architecture["feature_dim"],
                rank=architecture["rank"],
                log_radius_mean=radius["log_radius_mean"],
                log_radius_std=radius["log_radius_std"],
                max_angle_degrees=architecture["max_angle_degrees"],
                tau=architecture["tau"],
                radius_epsilon=architecture["radius_epsilon"],
                norm_epsilon=architecture["norm_epsilon"],
            )
        )
    if variant == "scalestitch_no_factor":
        return build_constant_scalestitch_adapter(
            architecture["feature_dim"],
            architecture["rank"],
            radius["log_radius_mean"],
            radius["log_radius_std"],
            architecture["max_angle_degrees"],
            architecture["tau"],
            radius_epsilon=architecture["radius_epsilon"],
            norm_epsilon=architecture["norm_epsilon"],
        ).module
    if architecture["radius_epsilon"] != 1.0e-6:
        raise ValueError(
            "non-ScaleStitch adapters currently require radius_epsilon=1e-6, "
            "matching their representation-training contract"
        )
    if variant == "radius_linear":
        return build_direct_radius_adapter(
            architecture["feature_dim"],
            radius["log_radius_mean"],
            radius["log_radius_std"],
        ).module
    if variant == "constant_linear":
        return build_constant_linear_adapter(
            architecture["feature_dim"],
            radius["log_radius_mean"],
            radius["log_radius_std"],
        ).module
    return build_scale_adapter(
        variant,
        architecture["feature_dim"],
        architecture["rank"],
        radius["log_radius_mean"],
        radius["log_radius_std"],
        max_angle_degrees=architecture["max_angle_degrees"],
        tau=architecture["tau"],
    ).module
