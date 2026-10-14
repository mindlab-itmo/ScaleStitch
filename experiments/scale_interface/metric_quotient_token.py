"""Radius-conditioned tokens and constant controls."""
from __future__ import annotations

import math
import re
from typing import Tuple

import torch
import torch.nn as nn

from .scalestitch_falsification import standardized_log_radius


FACTOR_VARIANT = "anchored_quotient_token"
CONSTANT_VARIANT = "anchored_quotient_token_constant_no_factor"
VARIANTS = frozenset({FACTOR_VARIANT, CONSTANT_VARIANT})
INTERVENTIONS = frozenset({"true", "swapped", "constant"})
ROLE_BINDING_PATTERN = re.compile(
    r"Object A is <OBJ(?P<a>\d{3})>\. Object B is <OBJ(?P<b>\d{3})>\."
)


def _finite_vector(value: torch.Tensor, *, width: int, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().to(torch.float32).reshape(-1)
    if tensor.shape != (width,) or not torch.isfinite(tensor).all():
        raise ValueError(f"{label} must be one finite vector of width {width}")
    return tensor.clone()


def _finite_matrix(value: torch.Tensor, *, width: int, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().to(torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != width or not torch.isfinite(tensor).all():
        raise ValueError(f"{label} must be a finite rank-2 tensor of width {width}")
    if tensor.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one anchor token")
    return tensor.clone()


def parse_role_token_ids(question: str) -> Tuple[int, int]:

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    matches = list(ROLE_BINDING_PATTERN.finditer(question))
    if len(matches) != 1:
        raise ValueError("question must contain exactly one strict A/B object binding")
    a = int(matches[0].group("a"))
    b = int(matches[0].group("b"))
    if a == b:
        raise ValueError("A and B must bind different visible object tokens")
    return a, b


def intervene_scene_radii(
    radius: torch.Tensor,
    validity_mask: torch.Tensor,
    scene_mask: torch.Tensor,
    assigned_ids: torch.Tensor,
    question: str,
    *,
    intervention: str,
    constant_radius_m: float,
) -> torch.Tensor:

    if intervention not in INTERVENTIONS:
        raise ValueError(f"unsupported quotient-token intervention: {intervention!r}")
    if not math.isfinite(constant_radius_m) or constant_radius_m <= 0.0:
        raise ValueError("constant_radius_m must be finite and positive")
    values = torch.as_tensor(radius).reshape(-1).clone()
    valid = torch.as_tensor(validity_mask, device=values.device).reshape(-1).to(torch.bool)
    active = torch.as_tensor(scene_mask, device=values.device).reshape(-1).to(torch.bool)
    assignment = torch.as_tensor(assigned_ids, device=values.device).reshape(-1).to(torch.long)
    if not (values.numel() == valid.numel() == active.numel() == assignment.numel()):
        raise ValueError("radius, masks, and assigned_ids must have identical row counts")
    required = active
    if torch.any(required & ~valid):
        raise ValueError("every active quotient-token object requires a valid radius")
    if torch.any(required & (~torch.isfinite(values) | (values <= 0.0))):
        raise ValueError("every active quotient-token object requires a finite positive radius")
    if intervention == "true":
        return values
    if intervention == "constant":
        values[required] = float(constant_radius_m)
        return values

    visible_a, visible_b = parse_role_token_ids(question)
    for visible in (visible_a, visible_b):
        if visible >= active.numel() or not bool(active[visible].item()):
            raise ValueError("question role refers to an inactive visible object token")
    source_a = int(assignment[visible_a].item())
    source_b = int(assignment[visible_b].item())
    if (
        source_a == source_b
        or source_a < 0
        or source_b < 0
        or source_a >= values.numel()
        or source_b >= values.numel()
        or not bool(required[source_a].item())
        or not bool(required[source_b].item())
    ):
        raise ValueError("A/B role assignment does not map to two active source rows")
    original_a = values[source_a].clone()
    values[source_a] = values[source_b]
    values[source_b] = original_a
    return values


class AnchoredQuotientToken(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        log_radius_mean: float,
        log_radius_std: float,
        *,
        center_embedding: torch.Tensor,
        direction_initialization: torch.Tensor,
        prefix_embeddings: torch.Tensor,
        suffix_embeddings: torch.Tensor,
        variant: str = FACTOR_VARIANT,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if variant not in VARIANTS:
            raise ValueError(f"unsupported quotient-token variant: {variant!r}")
        if not math.isfinite(log_radius_mean):
            raise ValueError("log_radius_mean must be finite")
        if not math.isfinite(log_radius_std) or log_radius_std <= 0.0:
            raise ValueError("log_radius_std must be finite and positive")
        self.hidden_dim = hidden_dim
        self.variant = variant
        self.register_buffer(
            "log_radius_mean", torch.tensor(log_radius_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "log_radius_std", torch.tensor(log_radius_std, dtype=torch.float32)
        )
        self.register_buffer(
            "center_embedding",
            _finite_vector(center_embedding, width=hidden_dim, label="center_embedding"),
        )
        self.register_buffer(
            "prefix_embeddings",
            _finite_matrix(prefix_embeddings, width=hidden_dim, label="prefix_embeddings"),
        )
        self.register_buffer(
            "suffix_embeddings",
            _finite_matrix(suffix_embeddings, width=hidden_dim, label="suffix_embeddings"),
        )
        self.direction = nn.Parameter(
            _finite_vector(
                direction_initialization,
                width=hidden_dim,
                label="direction_initialization",
            )
        )

    @property
    def tokens_per_object(self) -> int:
        return int(self.prefix_embeddings.shape[0] + 1 + self.suffix_embeddings.shape[0])

    def factor_displacement(self, coordinate: torch.Tensor) -> torch.Tensor:
        coordinate = torch.as_tensor(coordinate, device=self.direction.device).to(torch.float32)
        return coordinate.unsqueeze(-1) * self.direction

    def forward(
        self,
        radius: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        descriptor, valid = standardized_log_radius(
            radius,
            validity_mask,
            log_radius_mean=float(self.log_radius_mean),
            log_radius_std=float(self.log_radius_std),
        )
        descriptor = descriptor.to(device=self.direction.device)
        valid = valid.to(device=self.direction.device)
        coefficient = (
            descriptor
            if self.variant == FACTOR_VARIANT
            else valid.to(torch.float32)
        )
        quotient = self.center_embedding + coefficient * self.direction
        rows = quotient.shape[0]
        prefix = self.prefix_embeddings.unsqueeze(0).expand(rows, -1, -1)
        suffix = self.suffix_embeddings.unsqueeze(0).expand(rows, -1, -1)
        tokens = torch.cat([prefix, quotient.unsqueeze(1), suffix], dim=1)
        tokens = tokens * valid.to(tokens.dtype).unsqueeze(-1)
        return tokens, valid.reshape(-1)
