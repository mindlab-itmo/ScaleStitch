"""Pure-metadata tests for matched object recognition and measured-radius probes."""
import copy
import json
from pathlib import Path
import re
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from scale_interface.controlled_semantic_probe import build_controlled_pairs


class ControlledSemanticProbeTests(unittest.TestCase):
    def scene(self):
        return {"scene0000_00": [
            {"source_id": 40, "category": "chair", "rho": .32, "radius_valid": True},
            {"source_id": 7, "category": "table", "rho": .8, "radius_valid": True},
            {"source_id": 91, "category": "cabinet", "rho": .56, "radius_valid": True},
            {"source_id": 15, "category": "lamp", "rho": .21, "radius_valid": True},
        ]}

    def test_fixed_pairs_maximum_and_separate_references(self):
        result = build_controlled_pairs(self.scene())
        self.assertEqual(len(result["direct_rows"]), 4)
        self.assertEqual(len(result["semantic_rows"]), 4)
        stats = result["summary"]["per_scene"]["scene0000_00"]
        self.assertEqual(stats["candidate_pairs"], 6)
        self.assertEqual(stats["unselected_by_pair_limit"], 2)
        refs = {ref["question_id"]: ref for ref in result["references"]}
        for direct, semantic, metadata in zip(result["direct_rows"], result["semantic_rows"], result["pair_metadata"]):
            self.assertEqual(direct["assigned_ids"], [0, 1, 2, 3])
            self.assertEqual(semantic["assigned_ids"], direct["assigned_ids"])
            self.assertNotIn("answers", direct)
            self.assertNotIn("caption", semantic)
            self.assertNotIn("roles", semantic)
            self.assertNotIn("radius_cm", semantic)
            self.assertEqual(refs[direct["question_id"]]["answers"], refs[semantic["question_id"]]["answers"])
            roles = metadata["roles"]
            expected = "A" if roles["A"]["radius_cm"] > roles["B"]["radius_cm"] else "B"
            self.assertEqual(metadata["answer"], expected)

    def test_direct_binding_uses_feature_row_not_logical_source_id(self):
        result = build_controlled_pairs(self.scene())
        for row, metadata in zip(result["direct_rows"], result["pair_metadata"]):
            for label in ("A", "B"):
                role = metadata["roles"][label]
                self.assertIn(f"Object {label} is <OBJ{role['row_id']:03d}>.", row["prompt"])
                self.assertEqual(self.scene()[row["scene_id"]][role["row_id"]]["source_id"], role["source_id"])
            self.assertNotIn("<OBJ040>", row["prompt"])

    def test_named_and_direct_questions_differ_only_in_binding(self):
        result = build_controlled_pairs(self.scene())
        for direct, semantic, metadata in zip(result["direct_rows"], result["semantic_rows"], result["pair_metadata"]):
            replaced = direct["prompt"]
            for label in ("A", "B"):
                role = metadata["roles"][label]
                replaced = replaced.replace(f"<OBJ{role['row_id']:03d}>", "the " + role["category"])
            self.assertEqual(replaced, semantic["prompt"])
            self.assertNotRegex(semantic["prompt"], r"<OBJ[0-9]{3}>")
            self.assertNotIn("volume", semantic["prompt"])
            self.assertNotIn("height", semantic["prompt"])

    def test_scene_measurement_packet_has_every_object_and_no_names(self):
        result = build_controlled_pairs(self.scene(), max_pairs_per_scene=1)
        records = result["scene_measurements"]["scene0000_00"]
        self.assertEqual([record["object_id"] for record in records], [0, 1, 2, 3])
        self.assertEqual([record["radius_cm"] for record in records], [32, 80, 56, 21])
        self.assertTrue(all(record["available"] for record in records))
        for category in ("chair", "table", "cabinet", "lamp"):
            self.assertNotIn(category, json.dumps(records))

    def test_recognition_is_deduplicated_and_candidates_cover_entire_scene(self):
        result = build_controlled_pairs(self.scene())
        self.assertEqual(len(result["recognition_rows"]), 4)
        self.assertEqual(result["recognition_candidates"]["scene0000_00"],
                         ["<OBJ000>", "<OBJ001>", "<OBJ002>", "<OBJ003>"])
        ids = {row["question_id"] for row in result["recognition_rows"]}
        for metadata in result["pair_metadata"]:
            for role in metadata["roles"].values():
                self.assertIn(role["recognition_question_id"], ids)
        for row in result["recognition_rows"]:
            self.assertNotRegex(row["prompt"], r"<OBJ[0-9]{3}>")

    def test_category_uniqueness_counts_invalid_radius_objects_too(self):
        scenes = self.scene()
        scenes["scene0000_00"].append({"source_id": 100, "category": " CHAIR ", "rho": 0., "radius_valid": False})
        result = build_controlled_pairs(scenes)
        stats = result["summary"]["per_scene"]["scene0000_00"]
        self.assertEqual(stats["excluded_objects_by_first_reason"]["non_unique_category"], 2)
        self.assertEqual(stats["eligible_objects"], 3)
        self.assertFalse(any("chair" in row["prompt"] for row in result["semantic_rows"]))
        self.assertEqual(len(result["recognition_candidates"]["scene0000_00"]), 5)
        self.assertFalse(result["scene_measurements"]["scene0000_00"][-1]["available"])

    def test_normalized_category_aliases_are_not_unique(self):
        scenes = self.scene()
        scenes["scene0000_00"][0]["category"] = "night_stand"
        scenes["scene0000_00"][1]["category"] = "Night Stand"
        result = build_controlled_pairs(scenes)
        self.assertEqual(result["summary"]["per_scene"]["scene0000_00"]["eligible_objects"], 2)
        self.assertEqual(result["summary"]["selected_pairs"], 1)

    def test_ties_after_centimeter_quantization_count_before_pair_cap(self):
        scenes = self.scene()
        for obj, rho in zip(scenes["scene0000_00"], (.101, .102, .2, .3)):
            obj["rho"] = rho
        result = build_controlled_pairs(scenes, max_pairs_per_scene=1)
        stats = result["summary"]["per_scene"]["scene0000_00"]
        self.assertEqual(stats["candidate_pairs"], 6)
        self.assertEqual(stats["centimeter_ties"], 1)
        self.assertEqual(stats["non_tied_pairs"], 5)
        self.assertEqual(stats["selected_pairs"], 1)
        self.assertEqual(stats["unselected_by_pair_limit"], 4)

    def test_all_ties_produce_no_fake_answer(self):
        scenes = self.scene()
        for obj in scenes["scene0000_00"]:
            obj["rho"] = .5
        result = build_controlled_pairs(scenes)
        self.assertEqual(result["summary"]["centimeter_ties"], 6)
        self.assertEqual(result["direct_rows"], [])
        self.assertEqual(result["semantic_rows"], [])
        self.assertEqual(result["recognition_rows"], [])
        self.assertEqual(result["references"], [])

    def test_missing_generic_invalid_and_tiny_radii_have_explicit_counts(self):
        scenes = self.scene()
        scenes["scene0000_00"][0]["category"] = None
        scenes["scene0000_00"][1]["category"] = "other furniture"
        scenes["scene0000_00"][2].update(rho=0., radius_valid=False)
        scenes["scene0000_00"][3]["rho"] = .004
        result = build_controlled_pairs(scenes)
        counts = result["summary"]["per_scene"]["scene0000_00"]["excluded_objects_by_first_reason"]
        self.assertEqual(counts, {"missing_or_generic_category": 2, "non_unique_category": 0,
                                  "invalid_radius": 1, "nonpositive_centimeter_radius": 1})
        self.assertEqual(result["summary"]["selected_pairs"], 0)

    def test_role_orientation_and_selection_ignore_values_except_ties(self):
        scenes = self.scene()
        before = build_controlled_pairs(scenes)
        for obj in scenes["scene0000_00"]:
            obj["rho"] = 2.0 - obj["rho"]
        after = build_controlled_pairs(scenes)
        self.assertEqual(before["direct_rows"], after["direct_rows"])
        self.assertEqual(before["semantic_rows"], after["semantic_rows"])
        for first, second in zip(before["pair_metadata"], after["pair_metadata"]):
            self.assertNotEqual(first["answer"], second["answer"])
            self.assertEqual([first["roles"][label]["source_id"] for label in ("A", "B")],
                             [second["roles"][label]["source_id"] for label in ("A", "B")])

    def test_pair_selection_uses_stable_source_ids_not_feature_row_order(self):
        scenes = self.scene()
        before = build_controlled_pairs(scenes)
        scenes["scene0000_00"].reverse()
        after = build_controlled_pairs(scenes)
        self.assertEqual(before["semantic_rows"], after["semantic_rows"])
        for first, second in zip(before["pair_metadata"], after["pair_metadata"]):
            self.assertEqual([first["roles"][label]["source_id"] for label in ("A", "B")],
                             [second["roles"][label]["source_id"] for label in ("A", "B")])
            self.assertEqual(first["answer"], second["answer"])

    def test_role_hash_is_repeatable_and_not_always_ascending(self):
        scenes = self.scene()
        result = build_controlled_pairs(scenes)
        self.assertEqual(result, build_controlled_pairs(scenes))
        orders = [pair["roles"]["A"]["source_id"] < pair["roles"]["B"]["source_id"]
                  for pair in result["pair_metadata"]]
        self.assertIn(True, orders)
        self.assertIn(False, orders)

    def test_inputs_not_mutated_and_extra_locs_predictions_answers_are_unused(self):
        scenes = self.scene()
        before = copy.deepcopy(scenes)
        result = build_controlled_pairs(scenes)
        self.assertEqual(scenes, before)
        for obj in scenes["scene0000_00"]:
            obj.update(locs=[999.] * 6, answers=["SECRET"], prediction="SECRET")
        self.assertEqual(result, build_controlled_pairs(scenes))
        self.assertNotIn("SECRET", json.dumps(result))

    def test_invalid_metadata_and_duplicate_id_aliases_fail(self):
        for field, value in (("source_id", True), ("source_id", 1.5), ("source_id", -1),
                             ("radius_valid", 1), ("radius_valid", "true"),
                             ("rho", 0.), ("rho", -1.), ("rho", float("nan")),
                             ("category", "chair. Answer A")):
            scenes = self.scene()
            scenes["scene0000_00"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                build_controlled_pairs(scenes)
        scenes = self.scene()
        scenes["scene0000_00"][0]["source_id"] = "07"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_controlled_pairs(scenes)

    def test_pair_limit_and_empty_input(self):
        for value in (0, 5, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_controlled_pairs(self.scene(), max_pairs_per_scene=value)
        result = build_controlled_pairs({})
        self.assertEqual(result["summary"]["scenes"], 0)
        self.assertEqual(result["references"], [])


if __name__ == "__main__":
    unittest.main()
