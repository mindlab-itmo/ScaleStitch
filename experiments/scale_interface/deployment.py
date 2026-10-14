from typing import Any, Mapping
import torch
from torch import nn
from torch.nn import functional as F
from .adapter_builders import _build_bridge, _build_scalestitch
SOURCE_INPUT_MODE = "source_feature"
INFERENCE_EXECUTION_MODE = "inference"
ADAPTER_TRAINING_EXECUTION_MODE = "adapter_training"
EXECUTION_MODES = {INFERENCE_EXECUTION_MODE, ADAPTER_TRAINING_EXECUTION_MODE}

class ScaleStitchDeployment(nn.Module):

    def __init__(self, payload: Mapping[str, Any], *, execution_mode: str) -> None:
        super().__init__()
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(
                "execution_mode must be exactly inference or adapter_training"
            )
        self.bridge = _build_bridge(payload["architecture"])
        self.scalestitch = _build_scalestitch(
            payload["architecture"], payload["radius_contract"]
        )
        self.bridge.load_state_dict(payload["state"]["bridge"], strict=True)
        self.scalestitch.load_state_dict(payload["state"]["scalestitch"], strict=True)
        self.input_mode = payload["execution"]["input_mode"]
        self.bridge_input_dim = int(payload["architecture"]["bridge_input_dim"])
        self.feature_dim = int(payload["architecture"]["feature_dim"])
        self.l2_epsilon = float(payload["normalization"]["l2_epsilon"])
        self._radius_epsilon_value = float(
            payload["architecture"]["radius_epsilon"]
        )
        self.execution_mode = execution_mode
        for parameter in self.parameters():
            parameter.requires_grad = False
        if self.execution_mode == ADAPTER_TRAINING_EXECUTION_MODE:
            for parameter in self.scalestitch.parameters():
                parameter.requires_grad = True
            self.train(True)
        else:
            self.train(False)

    @property
    def radius_epsilon(self) -> torch.Tensor:

        value = getattr(self.scalestitch, "radius_epsilon", None)
        if value is not None:
            return value
        parameter = next(iter(self.scalestitch.parameters()), None)
        device = (
            parameter.device
            if parameter is not None
            else next(self.bridge.parameters()).device
        )
        return torch.tensor(self._radius_epsilon_value, device=device)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")


        nn.Module.train(self, False)
        self.bridge.train(False)
        if self.execution_mode == ADAPTER_TRAINING_EXECUTION_MODE:
            self.scalestitch.train(mode)
        else:
            self.scalestitch.train(False)
        return self

    def forward(
        self,
        features: torch.Tensor,
        object_radius: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(features, torch.Tensor) or features.ndim < 2:
            raise TypeError("deployment features must be a rank-2-or-higher tensor")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("deployment features must be finite floating-point values")
        expected_width = (
            self.bridge_input_dim
            if self.input_mode == SOURCE_INPUT_MODE
            else self.feature_dim
        )
        if features.shape[-1] != expected_width:
            raise ValueError(
                f"deployment {self.input_mode} expects width {expected_width}, "
                f"got {features.shape[-1]}"
            )
        operator_dtype = next(self.bridge.parameters()).dtype
        if self.input_mode == SOURCE_INPUT_MODE:
            with torch.no_grad():
                canonical = self.bridge(features.to(dtype=operator_dtype))
        else:
            canonical = features.to(dtype=operator_dtype).detach()
        if not torch.isfinite(canonical).all():
            raise FloatingPointError("canonical bridge emitted non-finite features")
        leading_shape = canonical.shape[:-1]
        canonical_flat = canonical.reshape(-1, self.feature_dim)
        radius_flat = torch.as_tensor(
            object_radius, device=canonical.device
        ).reshape(-1)
        validity_flat = torch.as_tensor(
            validity_mask, device=canonical.device
        ).reshape(-1)
        if (
            radius_flat.numel() != canonical_flat.shape[0]
            or validity_flat.numel() != canonical_flat.shape[0]
        ):
            raise ValueError("radius/mask must contain one value per feature row")
        if self.execution_mode == INFERENCE_EXECUTION_MODE:
            with torch.no_grad():
                conditioned = self.scalestitch(
                    canonical_flat, radius_flat, validity_flat
                )
        else:
            conditioned = self.scalestitch(
                canonical_flat, radius_flat, validity_flat
            )
        conditioned_fp32 = conditioned.reshape(
            *leading_shape, self.feature_dim
        ).to(torch.float32)
        flat = conditioned_fp32.reshape(-1, self.feature_dim)
        valid = torch.as_tensor(validity_mask, device=flat.device).reshape(-1).to(torch.bool)
        if valid.numel() != flat.shape[0]:
            raise ValueError("validity_mask must contain one value per feature row")
        active_norm = torch.linalg.vector_norm(flat[valid], dim=-1)
        if active_norm.numel() and torch.any(active_norm <= self.l2_epsilon):
            raise ValueError("active deployed features must have non-zero norm")
        return F.normalize(conditioned_fp32, p=2.0, dim=-1, eps=self.l2_epsilon)
