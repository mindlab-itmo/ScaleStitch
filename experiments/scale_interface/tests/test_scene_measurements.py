"""CPU tests for complete, question-independent scene radius packets."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface.scene_measurements import (
    SceneRadiusRecord, build_scene_radius_packet, serialize_scene_radius_records,
)


class SceneMeasurementTests(unittest.TestCase):
    def packet(self, **changes):
        inputs = dict(assigned_rows=[0, 1, 2], scene_mask=[1, 1, 1],
                      radius_m=[0.10, 0.20, 0.30], radius_valid=[True, True, True])
        inputs.update(changes)
        return build_scene_radius_packet(**inputs)

    def test_default_includes_every_visible_object(self):
        packet = self.packet()
        self.assertEqual(packet.records, (
            SceneRadiusRecord(0, 0, 0, 10), SceneRadiusRecord(1, 1, 1, 20),
            SceneRadiusRecord(2, 2, 2, 30)))
        self.assertEqual(packet.text, " Measured metric radius: <OBJ000> = 10 cm; "
                         "<OBJ001> = 20 cm; <OBJ002> = 30 cm.")
        self.assertEqual(packet.numerical_block_indices, (1, 3, 5))

    def test_adversarial_permutation_uses_source_rows(self):
        packet = self.packet(assigned_rows=[2, 0, 1])
        self.assertEqual([r.radius_cm for r in packet.records], [30, 10, 20])

    def test_sparse_mask_matches_host_presented_position_semantics(self):
        packet = self.packet(assigned_rows=[2, 1, 0], scene_mask=[1, 0, 1])
        self.assertEqual(packet.records, (SceneRadiusRecord(0, 2, 2, 30),
                                          SceneRadiusRecord(2, 0, 0, 10)))
        self.assertNotIn("<OBJ001>", packet.text)

    def test_sparse_permutation_cannot_select_a_masked_source(self):
        for mode in ("true", "constant", "swapped"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "masked source row 1"):
                self.packet(assigned_rows=[2, 0, 1], scene_mask=[1, 0, 1], mode=mode)

    def test_selected_source_radius_is_validated(self):
        with self.assertRaisesRegex(ValueError, "source row 0"):
            self.packet(assigned_rows=[2, 1, 0], scene_mask=[1, 0, 1],
                        radius_valid=[False, True, True])

    def test_invalid_unpresented_source_is_not_serialized(self):
        packet = self.packet(assigned_rows=[2, 1, 0], scene_mask=[1, 0, 1],
                             radius_m=[0.1, float("nan"), 0.3], radius_valid=[True, False, True])
        self.assertEqual(len(packet.records), 2)

    def test_cycles_are_bound_to_sorted_source_rows_not_visible_order(self):
        packet = self.packet(assigned_rows=[2, 0, 1], mode="swapped")
        self.assertEqual([(r.source_row, r.value_source_row, r.radius_cm) for r in packet.records],
                         [(2, 1, 20), (0, 2, 30), (1, 0, 10)])
        original = self.packet(mode="swapped")
        self.assertEqual({r.source_row: r.value_source_row for r in packet.records},
                         {r.source_row: r.value_source_row for r in original.records})

    def test_sparse_cycle_only_uses_presented_sources(self):
        packet = self.packet(assigned_rows=[2, 1, 0], scene_mask=[1, 0, 1], mode="swapped")
        self.assertEqual([(r.source_row, r.value_source_row) for r in packet.records], [(2, 0), (0, 2)])

    def test_reorder_preserves_bindings_and_all_objects(self):
        for mode in ("true", "constant", "swapped"):
            with self.subTest(mode=mode):
                ordinary = self.packet(mode=mode)
                reverse = self.packet(mode=mode, record_order=[2, 1, 0])
                self.assertEqual(reverse.records, ordinary.records[::-1])
                self.assertEqual(reverse.numerical_block_indices, (1, 3, 5))

    def test_sparse_visible_ids_can_be_reordered(self):
        packet = self.packet(scene_mask=[1, 0, 1], record_order=[2, 0])
        self.assertEqual([r.visible_id for r in packet.records], [2, 0])

    def test_unit_change_preserves_exact_quantized_quantity(self):
        expected = {"cm": " 123 cm", "mm": " 1230 mm", "m": " 1.23 m"}
        for unit, first in expected.items():
            with self.subTest(unit=unit):
                packet = self.packet(radius_m=[1.23, 0.2, 0.3], unit=unit)
                self.assertEqual(packet.records[0].radius_cm, 123)
                self.assertEqual(packet.parts[packet.numerical_block_indices[0]], first)

    def test_round_half_up_is_explicit(self):
        packet = self.packet(radius_m=[0.005, 0.125, 1.235])
        self.assertEqual([r.radius_cm for r in packet.records], [1, 13, 124])

    def test_float32_half_centimeter_is_below_the_quantization_boundary(self):
        radius = struct.unpack("f", struct.pack("f", 0.005))[0]
        self.assertGreater(radius, 0)
        self.assertLess(radius, 0.005)
        for mode in ("true", "constant", "swapped"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "rounds to zero"):
                self.packet(radius_m=[radius, 0.2, 0.3], mode=mode)

    def test_constant_keeps_object_bindings_and_records_no_data_donor(self):
        packet = self.packet(assigned_rows=[2, 0, 1], mode="constant", constant_cm=77)
        self.assertEqual([r.radius_cm for r in packet.records], [77, 77, 77])
        self.assertEqual([r.source_row for r in packet.records], [2, 0, 1])
        self.assertTrue(all(r.value_source_row is None for r in packet.records))

    def test_constant_does_not_bypass_visible_input_validation(self):
        for changes in ({"radius_valid": [False, True, True]}, {"radius_m": [0, 0.2, 0.3]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.packet(mode="constant", **changes)

    def test_single_object_allowed_except_swapped(self):
        for mode in ("true", "constant"):
            self.assertEqual(len(self.packet(scene_mask=[0, 1, 0], mode=mode).records), 1)
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.packet(scene_mask=[0, 1, 0], mode="swapped")

    def test_empty_or_all_masked_scenes_rejected(self):
        for changes in ({"scene_mask": [0, 0, 0]},
                        dict(assigned_rows=[], scene_mask=[], radius_m=[], radius_valid=[])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.packet(**changes)

    def test_malformed_assignments_rejected(self):
        for assigned in ([0, 0, 2], [0, 1, 3], [-1, 1, 2], [True, 1, 2], [0.0, 1, 2]):
            with self.subTest(assigned=assigned), self.assertRaises(ValueError):
                self.packet(assigned_rows=assigned)

    def test_incomplete_or_duplicate_record_order_rejected(self):
        for order in ([0, 0, 2], [0, 1], [0, 1, 3], [False, 1, 2]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                self.packet(record_order=order)

    def test_invalid_radii_rejected(self):
        for radius in (0, -1, float("nan"), float("inf"), "0.1", True, 0.004):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                self.packet(radius_m=[radius, 0.2, 0.3])

    def test_invalid_flags_lengths_and_options_rejected(self):
        cases = ({"scene_mask": [1, 2, 1]}, {"scene_mask": [1.0, 1, 1]},
                 {"radius_valid": [1, "yes", 1]}, {"radius_m": [0.1]},
                 {"unit": "ft"}, {"mode": "height"}, {"constant_cm": 0}, {"constant_cm": True})
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.packet(**changes)

    def test_metadata_is_input_only_json_and_frozen(self):
        packet = self.packet()
        metadata = json.loads(json.dumps(packet.metadata()))
        self.assertEqual(set(metadata), {"measurement", "unit", "mode", "objects",
                                         "records", "numerical_block_indices"})
        self.assertEqual(metadata["objects"], 3)
        with self.assertRaises(FrozenInstanceError):
            packet.records[0].radius_cm = 99

    def test_inputs_are_not_mutated(self):
        assigned, mask, radii, valid = [2, 1, 0], [1, 0, 1], [0.1, 0.2, 0.3], [True] * 3
        before = [list(value) for value in (assigned, mask, radii, valid)]
        build_scene_radius_packet(assigned_rows=assigned, scene_mask=mask, radius_m=radii,
                                  radius_valid=valid, mode="swapped", record_order=[2, 0])
        self.assertEqual([assigned, mask, radii, valid], before)

    def test_explicit_serializer_checks_duplicate_ids_and_bad_records(self):
        ordinary = self.packet().records
        bad = (
            [ordinary[0], replace(ordinary[1], visible_id=0)],
            [ordinary[0], replace(ordinary[1], source_row=0, value_source_row=0)],
            [replace(ordinary[0], radius_cm=0)],
            [replace(ordinary[0], visible_id=1000)],
            [replace(ordinary[0], value_source_row=2)],
            [],
        )
        for records in bad:
            with self.subTest(records=records), self.assertRaises(ValueError):
                serialize_scene_radius_records(records)
        with self.assertRaises(TypeError):
            serialize_scene_radius_records([{"visible_id": 0}])

    def test_every_numeric_record_has_one_explicit_block(self):
        packet = build_scene_radius_packet(assigned_rows=list(range(20)), scene_mask=[1] * 20,
                                           radius_m=[0.25] * 20, radius_valid=[True] * 20)
        self.assertEqual(len(packet.numerical_block_indices), 20)
        self.assertTrue(all(packet.parts[index] == " 25 cm" for index in packet.numerical_block_indices))
        self.assertEqual(len(packet.parts), 41)

    def test_serializer_rejects_misdeclared_controls(self):
        constant = self.packet(mode="constant").records
        swapped = self.packet(mode="swapped").records
        cases = (
            ([replace(constant[0], value_source_row=0)], "constant"),
            ([constant[0], replace(constant[1], radius_cm=101)], "constant"),
            ([replace(swapped[0], value_source_row=0), *swapped[1:]], "swapped"),
            ([swapped[0]], "swapped"),
        )
        for records, mode in cases:
            with self.subTest(records=records, mode=mode), self.assertRaises(ValueError):
                serialize_scene_radius_records(records, mode=mode)

    def test_equal_radii_are_not_discarded_for_swap_control(self):
        packet = self.packet(mode="swapped", radius_m=[0.2, 0.2, 0.2])
        self.assertEqual([r.radius_cm for r in packet.records], [20, 20, 20])
        self.assertTrue(all(r.source_row != r.value_source_row for r in packet.records))


if __name__ == "__main__":
    unittest.main()
