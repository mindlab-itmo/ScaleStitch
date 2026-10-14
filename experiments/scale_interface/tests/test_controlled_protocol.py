"""CPU contract checks for complete native-ID controlled comparisons."""
import ast
import copy
from pathlib import Path
import unittest

from experiments.scale_interface.controlled_protocol import (append_instruction, audit_coverage, build_full_sqa3d_inputs,
                                build_native_id_pairs, four_conditions, score_native_id)

SCENE = "scene0001_00"


def objects(n=6):
    return {SCENE: [dict(source_id=10+i*2, category="category " + str(i), rho=(i+1)/10,
                        radius_valid=True, height_m=(i+1)/5, height_valid=True) for i in range(n)]}


def height_source():
    return dict(measurement="full_object_axis_aligned_height", input_id="test-verified-height",
                source_reference="test fixture only", verified_scenes={SCENE: True})


class FullPairsTests(unittest.TestCase):
    def test_all_fifteen_pairs_not_four_or_selected_sixty_four(self):
        source = objects()
        original = copy.deepcopy(source)
        built = build_native_id_pairs(source)
        self.assertEqual(source, original)
        self.assertEqual(len(built["pair_metadata"]), 15)
        self.assertEqual(len(built["direct_rows"]), 15)
        self.assertEqual(len(built["named_rows"]), 15)
        self.assertEqual(built["summary"]["per_scene"][SCENE]["unselected_by_pair_limit"], 0)
        self.assertEqual(len(build_native_id_pairs(objects(13))["pair_metadata"]), 78)
        allowed = {"qid", "question_id", "scene_id", "pair_id", "question", "prompt", "assigned_ids"}
        for row in built["named_rows"]:
            self.assertEqual(set(row), allowed)
            self.assertNotIn("<OBJ", row["prompt"])
            self.assertEqual(row["assigned_ids"], list(range(6)))
        self.assertEqual({r["pair_id"] for r in built["direct_rows"]},
                         {r["pair_id"] for r in built["named_rows"]})
        self.assertEqual({r["question_id"] for r in built["references"]},
                         {r["qid"] for r in built["direct_rows"]+built["named_rows"]})

    def test_height_not_conditioned_on_radius_validity_or_ties(self):
        source = objects(3)
        for obj in source[SCENE]:
            obj["rho"] = 0
            obj["radius_valid"] = False
        self.assertEqual(len(build_native_id_pairs(source)["pair_metadata"]), 0)
        built = build_native_id_pairs(source, measurement="height", measurement_source=height_source())
        self.assertEqual(len(built["pair_metadata"]), 3)
        source[SCENE][0]["height_m"] = source[SCENE][1]["height_m"]
        built = build_native_id_pairs(source, measurement="height", measurement_source=height_source())
        self.assertEqual(len(built["pair_metadata"]), 2)
        self.assertEqual(built["summary"]["per_scene"][SCENE]["centimeter_ties"], 1)
        with self.assertRaises(ValueError):
            build_native_id_pairs(source, measurement="height")

    def test_complete_object_exclusion_ledger_and_quantized_ties(self):
        source = objects(8)
        source[SCENE][0]["category"] = "other"
        source[SCENE][1]["category"] = source[SCENE][2]["category"] = "chair"
        source[SCENE][3].update(rho=0, radius_valid=False)
        source[SCENE][4]["rho"] = .004
        source[SCENE][5]["rho"] = .6001
        source[SCENE][6]["rho"] = .6002
        ledger = build_native_id_pairs(source)["summary"]["per_scene"][SCENE]
        self.assertEqual(ledger["excluded_objects_by_first_reason"], {
            "missing_or_generic_category": 1, "non_unique_category": 2,
            "invalid_measurement": 1, "nonpositive_centimeter_measurement": 1})
        self.assertEqual(ledger["candidate_pairs"], 3)
        self.assertEqual(ledger["selected_pairs"] + ledger["centimeter_ties"], 3)
        self.assertFalse(ledger["whole_scene_measurements_available"])

    def test_prompt_and_identity_do_not_change_with_target_answer(self):
        source = objects(2)
        before = build_native_id_pairs(source)
        source[SCENE][0]["rho"], source[SCENE][1]["rho"] = source[SCENE][1]["rho"], source[SCENE][0]["rho"]
        after = build_native_id_pairs(source)
        self.assertEqual(before["direct_rows"], after["direct_rows"])
        self.assertEqual(before["named_rows"], after["named_rows"])
        self.assertNotEqual(before["references"], after["references"])

    def test_bad_scene_identity_and_duplicate_source_rejected(self):
        for corrupt in (lambda s: s[SCENE][1].update(source_id=10),
                        lambda s: s[SCENE][0].update(radius_valid=1),
                        lambda s: s[SCENE][0].update(rho=float("nan"))):
            source = objects(2)
            corrupt(source)
            with self.assertRaises(ValueError):
                build_native_id_pairs(source)

    def test_four_condition_contract_matches_actual_runtime_class_without_torch(self):
        path = Path(__file__).resolve().parents[1] / "natural_qa.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        keep = [node for node in tree.body if
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                {"OBJECT_CONTROLS", "REPRESENTATIONS"} for t in node.targets) or
                isinstance(node, ast.ClassDef) and node.name == "NaturalQACondition"]
        from dataclasses import dataclass
        namespace = {"dataclass": dataclass}
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), namespace)
        for measurement in ("radius", "height"):
            conditions = [namespace["NaturalQACondition"](**c) for c in four_conditions(measurement)]
            self.assertEqual(len({c.name for c in conditions}), 4)
            self.assertEqual(conditions[0].measurement, "radius")


class NativeScorerTests(unittest.TestCase):
    def score(self, raw, **kwargs):
        options = dict(assigned_ids=[2, 0, 1], scene_mask=[True]*3, end_sym="</s>",
                       pair_source_rows=[0, 2], target_source_row=2)
        options.update(kwargs)
        return score_native_id(raw, **options)

    def test_visible_to_source_mapping_exactly_once(self):
        result = self.score(" <OBJ000></s>ignored decoder suffix")
        self.assertEqual(result["exact_match"], 1)
        self.assertEqual(result["scorer_prediction"], "<OBJ002>")
        self.assertEqual(self.score("<OBJ001>")["status"], "other_pair_object")
        self.assertEqual(self.score("<OBJ002>")["status"], "outside_queried_pair")

    def test_invalid_outputs_never_extracted_or_repaired(self):
        for raw in ("A", "B", "<OBJ0>", "<obj000>", "It is <OBJ000>",
                    "<OBJ000> <OBJ001>", "", "<OBJ999>", "<OBJ002>"):
            result = self.score(raw)
            self.assertEqual(result["exact_match"], 0, raw)
            self.assertEqual(result["scorer_prediction"], "__native_id_error__", raw)

    def test_terminal_punctuation_matches_frozen_id_rule(self):
        self.assertEqual(self.score("<OBJ000>.</s>")["exact_match"], 1)
        self.assertEqual(self.score("<OBJ000>. extra")["exact_match"], 0)

    def test_mask_mapping_and_reference_validation(self):
        for values in (dict(assigned_ids=[2, 2, 1]), dict(end_sym=""),
                       dict(scene_mask=[True, True, False]), dict(target_source_row=1)):
            with self.assertRaises(ValueError):
                self.score("<OBJ000>", **values)


class CoverageTests(unittest.TestCase):
    def test_sqa_all_questions_not_eight_per_scene_and_no_answer_access(self):
        class InputsOnly(dict):
            def __getitem__(self, key):
                if key not in {"question_id", "scene_id", "situation", "question"}:
                    raise AssertionError("answer leakage")
                return super().__getitem__(key)
        records = [InputsOnly(question_id=i, scene_id=SCENE, situation=" I am here. ",
                              question="What color is it?", answers="must not read") for i in range(20)]
        result = build_full_sqa3d_inputs(records, scene_object_counts={SCENE: 3})
        self.assertEqual(len(result["rows"]), 20)
        self.assertTrue(result["complete_input_coverage"])
        self.assertEqual(result["rows"][0]["prompt"], "Situation:\n I am here. \n\nQuestion:\nWhat color is it?")
        missing = build_full_sqa3d_inputs(records, scene_object_counts={})
        self.assertEqual(len(missing["exclusions"]), 20)
        self.assertFalse(missing["complete_input_coverage"])

    def test_exact_candidate_partition_and_complete_prediction_matrix(self):
        built = build_native_id_pairs(objects(2))
        rows = built["direct_rows"] + built["named_rows"]
        selected = [r["qid"] for r in rows]
        conditions = ["absent", "true", "constant", "swapped"]
        predictions = [dict(qid=q, scene_id=SCENE, condition=c) for q in selected for c in conditions]
        def check(chosen=selected, excluded=(), outputs=predictions):
            return audit_coverage(rows, chosen, excluded, conditions=conditions,
                                  predictions=outputs, matched_pairs=True)
        self.assertTrue(check()["matrix_verified"])
        for outputs in (predictions[:-1], predictions+predictions[:1]):
            with self.assertRaises(ValueError):
                check(outputs=outputs)
        with self.assertRaisesRegex(ValueError, "partition"):
            check(chosen=selected[:1], outputs=None)
        with self.assertRaisesRegex(ValueError, "partial"):
            check(chosen=selected[:1], excluded=[dict(qid=selected[1], reason="context limit")], outputs=None)
        result = check(chosen=[], excluded=[dict(qid=q, reason="paired context exclusion") for q in selected], outputs=[])
        self.assertEqual(result["excluded_questions"], 2)



class InstructionTests(unittest.TestCase):
    def test_candidate_exact_suffix_preserves_identity_and_source(self):
        rows = build_native_id_pairs(objects(2))["direct_rows"]
        original = copy.deepcopy(rows)
        changed = append_instruction(rows, "candidate")
        self.assertEqual(rows, original)
        self.assertEqual(changed[0]["prompt"], rows[0]["prompt"] + " " +
                         "Your answer must be one of the two object identifiers in the question. "
                         "Return exactly that identifier and no other text.")
        self.assertEqual(changed[0]["question"], changed[0]["prompt"])
        for key in ("qid", "question_id", "scene_id", "pair_id", "assigned_ids"):
            self.assertEqual(changed[0][key], rows[0][key])

    def test_natural_preserves_separate_question_and_none_preserves_all(self):
        rows = [dict(qid="17", scene_id=SCENE, question="What color is it?",
                     prompt="Situation:\nI am here.\n\nQuestion:\nWhat color is it?",
                     assigned_ids=[0, 1])]
        changed = append_instruction(rows, "natural")
        self.assertEqual(changed[0]["question"], rows[0]["question"])
        self.assertEqual(changed[0]["prompt"], rows[0]["prompt"] + " " +
                         "Answer the question with a short natural-language answer. "
                         "Do not output object identifiers.")
        unchanged = append_instruction(rows)
        self.assertEqual(unchanged, rows)
        unchanged[0]["assigned_ids"].append(2)
        self.assertEqual(rows[0]["assigned_ids"], [0, 1])

    def test_reference_fields_rejected_without_reading_or_copying_values(self):
        class ForbiddenValue:
            def __deepcopy__(self, memo):
                raise AssertionError("read a reference answer")
        row = build_native_id_pairs(objects(2))["direct_rows"][0]
        for key in ("answer", "answers", "references", "target", "target_source_row",
                    "target_object_id"):
            with self.subTest(key=key):
                contaminated = dict(row, **{key: ForbiddenValue()})
                with self.assertRaisesRegex(ValueError, "reference answers"):
                    append_instruction([contaminated], "candidate")

    def test_named_questions_cannot_receive_candidate_instruction(self):
        rows = build_native_id_pairs(objects(2))["named_rows"]
        with self.assertRaisesRegex(ValueError, "two distinct"):
            append_instruction(rows, "candidate")

    def test_double_append_duplicates_and_mismatched_identity_rejected(self):
        rows = build_native_id_pairs(objects(2))["direct_rows"]
        with self.assertRaisesRegex(ValueError, "twice"):
            append_instruction(append_instruction(rows, "candidate"), "candidate")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            append_instruction(rows + rows, "candidate")
        bad = copy.deepcopy(rows)
        bad[0]["question_id"] += "/different"
        with self.assertRaisesRegex(ValueError, "disagree"):
            append_instruction(bad, "candidate")



if __name__ == "__main__":
    unittest.main()
