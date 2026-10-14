"""Scene-wide observed heights in an externally checked axis-aligned frame.

The caller supplies a measurement_source identifying the inspected inputs and
scenes. This module validates that declaration, not the underlying coordinate
transform. Height is the full available object cloud's Z extent; occluded parts
are not inferred. Values and metadata are never labelled as metric radii.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import re
from typing import Optional, Sequence, Tuple

from .scene_measurements import (
    SceneRadiusRecord, build_scene_radius_packet, serialize_scene_radius_records,
)


HEIGHT_MEASUREMENT = "full_object_axis_aligned_height"


def normalize_measurement_source(source):
    """Copy a caller's source declaration; do not read files or certify geometry."""
    required = {"measurement", "input_id", "source_reference", "verified_scenes"}
    if not isinstance(source, Mapping) or set(source) != required:
        raise ValueError("height requires measurement_source with measurement, input_id, source_reference, verified_scenes")
    if source["measurement"] != HEIGHT_MEASUREMENT:
        raise ValueError("measurement_source must describe full-object axis-aligned height")
    for key in ("input_id", "source_reference"):
        if not isinstance(source[key], str) or not source[key].strip():
            raise ValueError(f"measurement_source.{key} must be a nonempty string")
    scenes = source["verified_scenes"]
    if not isinstance(scenes, Mapping) or not scenes:
        raise ValueError("verified_scenes must map scene IDs to actual boolean flags")
    for scene, verified in scenes.items():
        if (not isinstance(scene, str) or re.fullmatch(r"scene[0-9]{4}_[0-9]{2}", scene) is None
                or type(verified) is not bool):
            raise ValueError("verified_scenes must map scene IDs to actual boolean flags")
    return {"measurement": HEIGHT_MEASUREMENT, "input_id": source["input_id"],
            "source_reference": source["source_reference"],
            "verified_scenes": dict(sorted(scenes.items()))}


def _source_for_scene(source, scene_id):
    normalized = normalize_measurement_source(source)
    if normalized["verified_scenes"].get(scene_id) is not True:
        raise ValueError(f"height source does not cover scene {scene_id}")
    return normalized


@dataclass(frozen=True)
class SceneHeightRecord:
    visible_id: int
    source_row: int
    value_source_row: Optional[int]
    height_cm: int


@dataclass(frozen=True)
class SceneHeightPacket:
    records: Tuple[SceneHeightRecord, ...]
    parts: Tuple[str, ...]
    numerical_block_indices: Tuple[int, ...]
    unit: str
    mode: str
    scene_id: str
    input_id: str
    source_reference: str

    @property
    def text(self):
        return "".join(self.parts)

    def metadata(self):
        return {"measurement": HEIGHT_MEASUREMENT, "unit": self.unit, "mode": self.mode,
                "objects": len(self.records), "records": [asdict(record) for record in self.records],
                "numerical_block_indices": list(self.numerical_block_indices),
                "measurement_source": {"scene_id": self.scene_id, "input_id": self.input_id,
                                       "source_reference": self.source_reference}}


def _height_packet(packet, scene_id, source):
    records = tuple(SceneHeightRecord(record.visible_id, record.source_row,
                                     record.value_source_row, record.radius_cm)
                    for record in packet.records)
    parts = (packet.parts[0].replace("Measured metric radius", "Measured axis-aligned object height", 1),
             *packet.parts[1:])
    return SceneHeightPacket(records, parts, packet.numerical_block_indices, packet.unit, packet.mode,
                             scene_id, source["input_id"], source["source_reference"])


def build_scene_height_packet(*, scene_id, measurement_source, assigned_rows, scene_mask,
                              height_m, height_valid, mode="true", unit="cm", constant_cm=100,
                              record_order=None):
    """Serialize every active object's checked Z extent with exact ID bindings.

    Reuse the length-unit quantization and permutation checks of the radius
    packet, but expose only height-typed records and height wording. Validate
    measured values even for constant/absent comparisons. No point selection,
    target category, reference answer, or implicit geometry reconstruction occurs.
    """
    source = _source_for_scene(measurement_source, scene_id)
    try:
        packet = build_scene_radius_packet(
            assigned_rows=assigned_rows, scene_mask=scene_mask, radius_m=height_m,
            radius_valid=height_valid, mode=mode, unit=unit, constant_cm=constant_cm,
            record_order=record_order,
        )
    except ValueError as error:
        raise ValueError(str(error).replace("radii", "heights").replace("radius", "height")) from error
    return _height_packet(packet, scene_id, source)


def serialize_scene_height_records(records: Sequence[SceneHeightRecord], *, scene_id,
                                    measurement_source, unit="cm", mode="true"):
    source = _source_for_scene(measurement_source, scene_id)
    records = tuple(records)
    if any(not isinstance(record, SceneHeightRecord) for record in records):
        raise TypeError("typed SceneHeightRecord values are required")
    equivalent = [SceneRadiusRecord(record.visible_id, record.source_row,
                                    record.value_source_row, record.height_cm) for record in records]
    try:
        packet = serialize_scene_radius_records(equivalent, unit=unit, mode=mode)
    except ValueError as error:
        raise ValueError(str(error).replace("radii", "heights").replace("radius", "height")) from error
    return _height_packet(packet, scene_id, source)
