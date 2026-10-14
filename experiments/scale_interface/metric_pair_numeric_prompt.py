"""Radius quantization in centimeters."""
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

def quantize_radius_cm(value: Any) -> int:

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("radius_m must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    return int((Decimal(str(numeric)) * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
