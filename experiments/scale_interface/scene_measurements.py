"""Question-independent radius packets for every object presented by the host.

The protocol host treats ``scene_mask`` as a mask of presented positions:
visible ID v refers to source row ``assigned_rows[v]`` when ``scene_mask[v]``
is active. Other host stages also mask source rows, so a supported permutation
must preserve the active source set; active positions cannot select masked
sources. Radii and validity flags are indexed by source row, not by v.
Inputs are plain CPU sequences; tensor callers should supply ``.tolist()``.

Only the existing metric radius is serialized. Height, width, bounding-box
volume and physical coordinate axes are outside this representation.
Eligibility additionally requires a positive radius after half-up centimeter
rounding. A positive upstream radius alone is insufficient: float32 0.005 m
is slightly below 0.005 and rounds to 0 cm. No tolerance is added. Preflight
this same eligibility rule for every comparison condition, including absent
packets; a caller must not recover excluded scenes only for the absent control.
"""
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
import math
from numbers import Integral, Real
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class SceneRadiusRecord:
    visible_id: int
    source_row: int
    value_source_row: Optional[int]
    radius_cm: int


@dataclass(frozen=True)
class SceneRadiusPacket:
    records: Tuple[SceneRadiusRecord, ...]
    parts: Tuple[str, ...]
    numerical_block_indices: Tuple[int, ...]
    unit: str
    mode: str

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def metadata(self) -> dict:
        """Return input-only lineage; no questions, categories or answers."""
        return {
            "measurement": "metric_radius",
            "unit": self.unit,
            "mode": self.mode,
            "objects": len(self.records),
            "records": [asdict(record) for record in self.records],
            "numerical_block_indices": list(self.numerical_block_indices),
        }


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _flags(values, name: str) -> Tuple[bool, ...]:
    result = []
    for value in values:
        if not isinstance(value, (bool, Integral)) or value not in (0, 1):
            raise ValueError(f"{name} must contain only boolean or integer 0/1 flags")
        result.append(bool(value))
    return tuple(result)


def _radius_cm(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("visible radius must be a positive finite number in meters")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("visible radius must be a positive finite number in meters")
    cm = int((Decimal(str(value)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if cm <= 0:
        raise ValueError("visible radius rounds to zero on the centimeter grid")
    return cm


def _value_text(cm: int, unit: str) -> str:
    if unit == "cm":
        return f" {cm} cm"
    if unit == "mm":
        return f" {cm * 10} mm"
    if unit == "m":
        return f" {cm // 100}.{cm % 100:02d} m"
    raise ValueError("unit must be cm, mm or m")


def serialize_scene_radius_records(
    records: Sequence[SceneRadiusRecord], *, unit: str = "cm", mode: str = "true"
) -> SceneRadiusPacket:
    """Serialize explicit records, exposing every numerical block for compression.

    The caller must still verify tokenizer boundary parity and the host's full
    sequence budget. This function does not truncate objects or tokenize text.
    """
    if mode not in ("true", "constant", "swapped"):
        raise ValueError("mode must be true, constant or swapped")
    _value_text(1, unit)
    records = tuple(records)
    if not records:
        raise ValueError("a scene packet requires at least one visible object")
    if any(not isinstance(record, SceneRadiusRecord) for record in records):
        raise TypeError("typed SceneRadiusRecord values are required")
    for record in records:
        if _integer(record.visible_id, "visible_id") > 999:
            raise ValueError("visible_id must fit the native three-digit object token")
        _integer(record.source_row, "source_row")
        _integer(record.radius_cm, "radius_cm", minimum=1)
        if record.value_source_row is not None:
            _integer(record.value_source_row, "value_source_row")
    if len({r.visible_id for r in records}) != len(records):
        raise ValueError("duplicate visible object IDs")
    sources = sorted(r.source_row for r in records)
    if len(set(sources)) != len(sources):
        raise ValueError("duplicate source rows")
    if mode == "swapped" and len(sources) < 2:
        raise ValueError("swapped radii require at least two visible objects")
    donors = {source: sources[index - 1] for index, source in enumerate(sources)}
    for record in records:
        expected = (None if mode == "constant" else
                    donors[record.source_row] if mode == "swapped" else record.source_row)
        if record.value_source_row != expected:
            raise ValueError("measurement donor does not match the declared condition")
    if mode == "constant" and len({r.radius_cm for r in records}) != 1:
        raise ValueError("constant condition requires equal radii")

    parts, numerical = [], []
    for index, record in enumerate(records):
        prefix = " Measured metric radius: " if index == 0 else "; "
        parts.append(f"{prefix}<OBJ{record.visible_id:03d}> =")
        numerical.append(len(parts))
        parts.append(_value_text(record.radius_cm, unit))
    parts.append(".")
    return SceneRadiusPacket(records, tuple(parts), tuple(numerical), unit, mode)


def build_scene_radius_packet(
    *,
    assigned_rows: Sequence[int],
    scene_mask: Sequence[bool],
    radius_m: Sequence[float],
    radius_valid: Sequence[bool],
    mode: str = "true",
    unit: str = "cm",
    constant_cm: int = 100,
    record_order: Optional[Sequence[int]] = None,
) -> SceneRadiusPacket:
    """Include all active host positions, without question-dependent selection.

    ``assigned_rows`` must be a permutation of every source row. ``record_order``
    optionally reorders all visible IDs without changing their bindings.
    ``swapped`` uses the previous sorted active source row as each value donor;
    for two objects this exchanges values. It is independent of displayed order.
    All visible source radii must be valid and round to positive centimeters,
    even for the constant control. Callers must preflight this same eligibility
    for the absent control too, so conditions share one evaluation cohort.
    """
    assigned = tuple(_integer(value, "assigned row") for value in assigned_rows)
    mask = _flags(scene_mask, "scene_mask")
    valid = _flags(radius_valid, "radius_valid")
    radii = tuple(radius_m)
    count = len(assigned)
    if not count or count > 1000 or not (len(mask) == len(valid) == len(radii) == count):
        raise ValueError("scene inputs require equal nonzero lengths of at most 1000")
    if sorted(assigned) != list(range(count)):
        raise ValueError("assigned_rows must be a complete permutation without duplicates")
    if mode not in ("true", "constant", "swapped"):
        raise ValueError("mode must be true, constant or swapped")
    _value_text(1, unit)
    constant_cm = _integer(constant_cm, "constant_cm", minimum=1)
    visible = [index for index, active in enumerate(mask) if active]
    if not visible:
        raise ValueError("a scene packet requires at least one visible object")
    measured = {}
    for index in visible:
        source = assigned[index]
        if not mask[source]:
            raise ValueError(
                f"visible object {index} selects masked source row {source}; "
                "assigned_rows must preserve the active source set"
            )
        if not valid[source]:
            raise ValueError(f"visible object {index} has an invalid radius at source row {source}")
        measured[source] = _radius_cm(radii[source])
    sources = sorted(measured)
    if mode == "swapped" and len(sources) < 2:
        raise ValueError("swapped radii require at least two visible objects")
    donors = {source: sources[index - 1] for index, source in enumerate(sources)}
    records = []
    for index in visible:
        source = assigned[index]
        donor = None if mode == "constant" else donors[source] if mode == "swapped" else source
        value = constant_cm if donor is None else measured[donor]
        records.append(SceneRadiusRecord(index, source, donor, value))
    if record_order is not None:
        order = tuple(_integer(value, "record_order visible ID") for value in record_order)
        if len(order) != len(visible) or sorted(order) != visible:
            raise ValueError("record_order must include every visible ID exactly once")
        by_visible = {record.visible_id: record for record in records}
        records = [by_visible[index] for index in order]
    return serialize_scene_radius_records(records, unit=unit, mode=mode)
