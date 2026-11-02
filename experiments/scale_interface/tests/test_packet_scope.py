import hashlib
import unittest

from experiments.scale_interface.packet_scope import focus_scene_packet
from experiments.scale_interface.scene_measurements import build_scene_radius_packet
from experiments.scale_interface.scene_extents import (
    HEIGHT_MEASUREMENT, build_scene_height_packet,
)


class PacketScopeTests(unittest.TestCase):
    def radius_packet(self, mode="swapped"):
        return build_scene_radius_packet(
            assigned_rows=[2, 0, 3, 1], scene_mask=[True] * 4,
            radius_m=[0.2, 0.5, 0.3, 1.4], radius_valid=[True] * 4,
            mode=mode, record_order=[3, 1, 0, 2],
        )

    def test_focus_keeps_full_scene_reassignment_donors(self):
        packet = self.radius_packet()
        focused = focus_scene_packet(packet, candidate_source_rows=[0, 2])
        self.assertEqual(
            [(r.visible_id, r.source_row, r.value_source_row, r.radius_cm)
             for r in focused.records],
            [(1, 0, 3, 140), (0, 2, 1, 50)],
        )
        self.assertEqual(focused.mode, "swapped")

    def test_focus_preserves_packet_order_not_candidate_argument_order(self):
        packet = self.radius_packet()
        first = focus_scene_packet(packet, candidate_source_rows=[0, 2])
        second = focus_scene_packet(packet, candidate_source_rows=[2, 0])
        self.assertEqual(first.text, second.text)
        self.assertEqual([r.visible_id for r in first.records], [1, 0])

    def test_first_retained_record_receives_original_heading(self):
        focused = focus_scene_packet(self.radius_packet(), candidate_source_rows=[0, 2])
        self.assertEqual(
            focused.text,
            " Measured metric radius: <OBJ001> = 140 cm; <OBJ000> = 50 cm.",
        )
        self.assertEqual(focused.numerical_block_indices, (1, 3))
        self.assertEqual([focused.parts[i] for i in (1, 3)], [" 140 cm", " 50 cm"])

    def test_height_heading_source_and_units_survive_focus(self):
        packet = build_scene_height_packet(
            scene_id="scene0001_00",
            measurement_source={"measurement": HEIGHT_MEASUREMENT,
                                "input_id": "height-fixture-v1",
                                "source_reference": "full-cloud-fixture",
                                "verified_scenes": {"scene0001_00": True}},
            assigned_rows=[2, 0, 3, 1], scene_mask=[True] * 4,
            height_m=[0.2, 0.5, 0.3, 1.4], height_valid=[True] * 4,
            mode="swapped", unit="mm", record_order=[3, 1, 0, 2],
        )
        focused = focus_scene_packet(packet, candidate_source_rows=[0, 2])
        self.assertEqual(
            focused.text,
            " Measured axis-aligned object height: <OBJ001> = 1400 mm; <OBJ000> = 500 mm.",
        )
        self.assertEqual(focused.metadata()["measurement_source"],
                         packet.metadata()["measurement_source"])
        self.assertNotIn("radius", focused.text)

    def test_metadata_retains_full_packet_provenance_and_selected_rows(self):
        packet = self.radius_packet()
        original = packet.metadata()
        focused = focus_scene_packet(packet, candidate_source_rows=[0, 2])
        meta = focused.metadata()
        self.assertEqual(meta["packet_scope"], "queried_pair")
        self.assertEqual(meta["objects"], 2)
        self.assertEqual(meta["full_scene_objects"], 4)
        self.assertEqual(meta["full_packet_sha256"],
                         hashlib.sha256(packet.text.encode("utf-8")).hexdigest())
        self.assertEqual([r["source_row"] for r in meta["records"]], [0, 2])
        self.assertEqual(packet.metadata(), original)
        meta["records"].clear()
        self.assertEqual(len(focused.metadata()["records"]), 2)

    def test_constant_stays_constant_without_creating_a_winner(self):
        focused = focus_scene_packet(self.radius_packet("constant"),
                                     candidate_source_rows=[0, 2])
        self.assertEqual([r.radius_cm for r in focused.records], [100, 100])
        self.assertTrue(all(r.value_source_row is None for r in focused.records))

    def test_invalid_or_missing_candidate_rows_fail(self):
        for rows in ([0], [0, 1, 2], [0, 0], [0, 99], [True, 2], ["0", 2]):
            with self.subTest(rows=rows), self.assertRaises((ValueError, TypeError)):
                focus_scene_packet(self.radius_packet(), candidate_source_rows=rows)

    def test_unknown_packet_type_is_rejected(self):
        with self.assertRaises(TypeError):
            focus_scene_packet({"text": "unverified"}, candidate_source_rows=[0, 2])


if __name__ == "__main__":
    unittest.main()
