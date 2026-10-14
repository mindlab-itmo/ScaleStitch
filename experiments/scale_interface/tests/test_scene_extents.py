"""Typed observed-height packets and caller-supplied geometry source declarations."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface.scene_extents import (
    HEIGHT_MEASUREMENT, SceneHeightRecord, build_scene_height_packet,
    normalize_measurement_source, serialize_scene_height_records,
)


def measurement_source(*scenes):
    return {"measurement": HEIGHT_MEASUREMENT, "input_id": "fixed-inputs",
            "source_reference": "geometry-audit", "verified_scenes": {scene: True for scene in scenes}}


class SceneExtentTests(unittest.TestCase):
    def packet(self, **changes):
        inputs = dict(scene_id="scene0000_00", measurement_source=measurement_source("scene0000_00"),
                      assigned_rows=[0, 1, 2], scene_mask=[True] * 3,
                      height_m=[1.23, .7, .9], height_valid=[True] * 3)
        inputs.update(changes)
        return build_scene_height_packet(**inputs)

    def test_height_wording_and_typed_metadata_never_relabel_radius(self):
        packet = self.packet()
        self.assertEqual(packet.records, (SceneHeightRecord(0, 0, 0, 123),
                                         SceneHeightRecord(1, 1, 1, 70), SceneHeightRecord(2, 2, 2, 90)))
        self.assertEqual(packet.text, " Measured axis-aligned object height: <OBJ000> = 123 cm; "
                         "<OBJ001> = 70 cm; <OBJ002> = 90 cm.")
        metadata = packet.metadata()
        self.assertEqual(metadata["measurement"], HEIGHT_MEASUREMENT)
        self.assertNotIn("radius", json.dumps(metadata))
        self.assertEqual(packet.numerical_block_indices, (1, 3, 5))

    def test_units_preserve_same_height(self):
        for unit, text in (("cm", " 123 cm"), ("mm", " 1230 mm"), ("m", " 1.23 m")):
            with self.subTest(unit=unit):
                packet = self.packet(unit=unit)
                self.assertEqual(packet.records[0].height_cm, 123)
                self.assertEqual(packet.parts[1], text)

    def test_row_assignment_and_display_order_preserve_bindings(self):
        forward = self.packet(assigned_rows=[2, 0, 1])
        reverse = self.packet(assigned_rows=[2, 0, 1], record_order=[2, 1, 0])
        self.assertEqual([record.height_cm for record in forward.records], [90, 123, 70])
        self.assertEqual(reverse.records, forward.records[::-1])

    def test_swapped_donors_are_source_bound(self):
        packet = self.packet(assigned_rows=[2, 0, 1], mode="swapped")
        self.assertEqual([(r.source_row, r.value_source_row, r.height_cm) for r in packet.records],
                         [(2, 1, 70), (0, 2, 90), (1, 0, 123)])

    def test_constant_still_requires_valid_measured_heights(self):
        packet = self.packet(mode="constant", constant_cm=55)
        self.assertEqual([r.height_cm for r in packet.records], [55, 55, 55])
        self.assertTrue(all(r.value_source_row is None for r in packet.records))
        for changes in ({"height_m": [0., .7, .9]}, {"height_valid": [False, True, True]}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "height"):
                self.packet(mode="constant", **changes)

    def test_sparse_masks_require_preserved_active_source_set(self):
        packet = self.packet(assigned_rows=[2, 1, 0], scene_mask=[True, False, True])
        self.assertEqual([r.height_cm for r in packet.records], [90, 123])
        with self.assertRaisesRegex(ValueError, "masked source"):
            self.packet(assigned_rows=[2, 0, 1], scene_mask=[True, False, True])

    def test_invalid_heights_and_centimeter_zero_are_rejected(self):
        small_float32 = struct.unpack("f", struct.pack("f", .005))[0]
        for value in (0, -.1, float("nan"), float("inf"), True, ".2", small_float32):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "height"):
                self.packet(height_m=[value, .7, .9])

    def test_source_declaration_required_and_scene_scoped(self):
        for value in (None, {}, measurement_source("scene0001_00")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.packet(measurement_source=value)
        source = measurement_source("scene0000_00")
        source["verified_scenes"]["scene0000_00"] = False
        with self.assertRaisesRegex(ValueError, "does not cover"):
            self.packet(measurement_source=source)

    def test_source_values_are_checked_without_claiming_geometry_verification(self):
        for field, value in (("measurement", "metric_radius"), ("input_id", ""),
                             ("source_reference", 123), ("verified_scenes", {"scene0000_00": 1}),
                             ("verified_scenes", {"scene0": True})):
            source = measurement_source("scene0000_00")
            source[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                normalize_measurement_source(source)

    def test_source_is_copied_and_packet_metadata_cannot_mutate_it(self):
        source = measurement_source("scene0000_00")
        copied = normalize_measurement_source(source)
        copied["verified_scenes"]["scene0000_00"] = False
        self.assertTrue(source["verified_scenes"]["scene0000_00"])
        packet = self.packet(measurement_source=source)
        packet.metadata()["measurement_source"]["input_id"] = "changed"
        self.assertEqual(packet.input_id, "fixed-inputs")
        with self.assertRaises(FrozenInstanceError):
            packet.records[0].height_cm = 1

    def test_explicit_serializer_checks_typed_records_and_donors(self):
        packet = self.packet()
        source = measurement_source("scene0000_00")
        actual = serialize_scene_height_records(packet.records, scene_id="scene0000_00", measurement_source=source)
        self.assertEqual(actual, packet)
        with self.assertRaises(TypeError):
            serialize_scene_height_records([{"height_cm": 10}], scene_id="scene0000_00", measurement_source=source)
        with self.assertRaises(ValueError):
            serialize_scene_height_records([replace(packet.records[0], value_source_row=1)],
                                           scene_id="scene0000_00", measurement_source=source)


if __name__ == "__main__":
    unittest.main()
