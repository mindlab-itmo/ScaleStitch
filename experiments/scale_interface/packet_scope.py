"""Restrict an already constructed scene intervention to the queried pair."""
from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral

from .scene_extents import SceneHeightPacket
from .scene_measurements import SceneRadiusPacket


@dataclass(frozen=True)
class FocusedScenePacket:
    records: tuple
    parts: tuple
    numerical_block_indices: tuple
    source_packet: object

    @property
    def text(self):
        return "".join(self.parts)

    @property
    def unit(self):
        return self.source_packet.unit

    @property
    def mode(self):
        return self.source_packet.mode

    def metadata(self):
        result = self.source_packet.metadata()
        result.update(
            records=[asdict(record) for record in self.records],
            objects=len(self.records),
            numerical_block_indices=list(self.numerical_block_indices),
            packet_scope="queried_pair",
            full_scene_objects=len(self.source_packet.records),
            full_packet_sha256=hashlib.sha256(
                self.source_packet.text.encode("utf-8")
            ).hexdigest(),
        )
        return result


def focus_scene_packet(packet, *, candidate_source_rows):
    """Filter full-scene T/C/S records without changing their value donors.

    Candidates are source rows from the question, not visible IDs. Selection
    preserves the full packet's record order. Rebuilding a two-record swapped
    packet would change the original scene-wide reassignment intervention.
    """
    if not isinstance(packet, (SceneRadiusPacket, SceneHeightPacket)):
        raise TypeError("a constructed radius or height packet is required")
    rows = tuple(candidate_source_rows)
    if (len(rows) != 2 or any(isinstance(row, bool) or not isinstance(row, Integral)
                              or row < 0 for row in rows)
            or len(set(rows)) != 2):
        raise ValueError("exactly two distinct nonnegative candidate source rows are required")
    wanted = set(rows)
    sources = {record.source_row for record in packet.records}
    if len(sources) != len(packet.records) or not wanted <= sources:
        raise ValueError("candidate source rows must be present in the full packet")
    expected_indices = tuple(range(1, 2 * len(packet.records), 2))
    if (len(packet.parts) != 2 * len(packet.records) + 1
            or tuple(packet.numerical_block_indices) != expected_indices):
        raise ValueError("the full packet has an unsupported segmentation")
    first_suffix = f"<OBJ{packet.records[0].visible_id:03d}> ="
    if not packet.parts[0].endswith(first_suffix):
        raise ValueError("the full packet's heading and first identifier disagree")
    heading = packet.parts[0][:-len(first_suffix)]
    selected, parts, indices = [], [], []
    for record, numerical_index in zip(packet.records, expected_indices):
        if record.source_row not in wanted:
            continue
        prefix = heading if not selected else "; "
        parts.append(f"{prefix}<OBJ{record.visible_id:03d}> =")
        indices.append(len(parts))
        parts.append(packet.parts[numerical_index])
        selected.append(record)
    parts.append(packet.parts[-1])
    return FocusedScenePacket(tuple(selected), tuple(parts), tuple(indices), packet)
