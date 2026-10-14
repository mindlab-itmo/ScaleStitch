"""Synthetic matched-task checks; statistics stay in the shared QA evaluator."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface import matched_question_metrics as matched
from scale_interface.natural_qa_metrics import evaluate_natural_qa


class MatchedQuestionMetricTests(unittest.TestCase):
    def fixtures(self):
        rows = {task: [] for task in ("direct", "semantic")}
        refs, predictions = [], []
        texts = {("direct", "A"): ["A", "B"], ("direct", "T"): ["A", "A"],
                 ("semantic", "A"): ["B", "B"], ("semantic", "T"): ["A", "B"]}
        for task in rows:
            for i, sid in enumerate(("scene0001_00", "scene0002_00")):
                qid, pair = task + str(i), "pair" + str(i)
                rows[task].append(dict(qid=qid, question_id=qid, pair_id=pair, scene_id=sid))
                refs.append(dict(question_id=qid, scene_id=sid, answers=["A"]))
                for condition in ("A", "T"):
                    predictions.append(dict(qid=qid, scene_id=sid, condition=condition,
                                            prediction=texts[(task, condition)][i],
                                            raw_prediction=texts[(task, condition)][i] + "</s>",
                                            output_token_ids=[10, 20], invalid_object_id=False))
        return predictions, refs, rows

    def score(self, predictions=None, refs=None, rows=None, **options):
        default_predictions, default_refs, default_rows = self.fixtures()
        arguments = dict(conditions=["A", "T"], cohort_pair_ids=["pair0", "pair1"],
                         contrasts={"true_task_gap": ("semantic|T", "direct|T")},
                         exploratory_linear_contrasts={"utility_interaction": {
                             "semantic|T": 1, "semantic|A": -1, "direct|T": -1, "direct|A": 1}},
                         replicates=120, seed=42, family_size=None)
        arguments.update(options)
        return matched.evaluate_matched_questions(
            default_predictions if predictions is None else predictions,
            default_refs if refs is None else refs,
            rows_by_task=default_rows if rows is None else rows, **arguments)

    def test_known_task_gap_and_scene_clustered_interaction(self):
        result = self.score()
        metric = "protocol_normalized_exact"
        gap = result["comparisons"]["true_task_gap"]["metrics"][metric]
        interaction = result["exploratory_linear_contrasts"]["utility_interaction"]["metrics"][metric]
        self.assertEqual(gap["delta"], -.5)
        self.assertEqual(interaction["delta"], 0.)
        self.assertEqual(interaction["ci95"], [-1., 1.])
        self.assertIsNone(interaction["ci_bonferroni"])
        self.assertEqual(result["cohort"]["unit"], "matched_pair")
        self.assertEqual(result["matched_questions"]["original_questions"], 4)
        self.assertEqual(result["bootstrap"]["cluster"], "sceneNNNN family")
        self.assertIn("do not isolate object recognition", result["interpretation"])

    def test_same_scorer_receives_pair_ids_and_preserved_raw_fields(self):
        predictions, refs, rows = self.fixtures()
        original = deepcopy((predictions, refs, rows))
        with patch.object(matched, "evaluate_natural_qa", wraps=evaluate_natural_qa) as shared:
            self.score(predictions, refs, rows)
        self.assertEqual(shared.call_count, 1)
        converted, pair_refs = shared.call_args.args
        self.assertEqual(len(pair_refs), 2)
        self.assertEqual(converted[0]["qid"], "pair0")
        self.assertEqual(converted[0]["condition"], "direct|A")
        for field in ("raw_prediction", "output_token_ids", "invalid_object_id", "prediction"):
            self.assertEqual(converted[0][field], predictions[0][field])
        self.assertEqual((predictions, refs, rows), original)

    def test_tuple_endpoints_match_serialized_endpoints(self):
        result = self.score(contrasts={"true_task_gap": (("semantic", "T"), ("direct", "T"))},
                            exploratory_linear_contrasts={"utility_interaction": {
                                ("semantic", "T"): 1, ("semantic", "A"): -1,
                                ("direct", "T"): -1, ("direct", "A"): 1}})
        self.assertEqual(result, self.score())

    def test_known_nonzero_interaction_and_joint_condition_gap(self):
        predictions, refs, rows = self.fixtures()
        for row in predictions:
            if row["condition"] == "T" and row["qid"].startswith("semantic"):
                row["prediction"] = "A"
        result = self.score(predictions, refs, rows)
        interaction = result["exploratory_linear_contrasts"]["utility_interaction"]
        self.assertEqual(interaction["metrics"]["protocol_normalized_exact"]["delta"], .5)
        for row in predictions:
            if row["condition"] == "T":
                row["condition"] = "joint_permutation"
        joint = self.score(predictions, refs, rows, conditions=["A", "joint_permutation"],
                           contrasts={"joint_gap": ("semantic|joint_permutation", "direct|joint_permutation")},
                           exploratory_linear_contrasts=None)
        self.assertEqual(joint["comparisons"]["joint_gap"]["metrics"]["raw_exact"]["delta"], 0.)

    def test_invalid_and_blank_predictions_remain_errors(self):
        predictions, refs, rows = self.fixtures()
        predictions[0]["invalid_object_id"] = True
        predictions[1]["prediction"] = ""
        result = self.score(predictions, refs, rows)
        self.assertEqual(result["matched_questions"]["pairs"], 2)
        self.assertEqual(result["conditions"]["direct|A"]["invalid_object_id_predictions"], 1)
        self.assertEqual(result["conditions"]["direct|A"]["metrics"]["raw_exact"]["correct"], 0)
        self.assertEqual(result["conditions"]["direct|T"]["blank_predictions"], 1)

    def test_missing_selected_prediction_never_changes_the_cohort(self):
        predictions, refs, rows = self.fixtures()
        with self.assertRaisesRegex(ValueError, "incomplete pairing"):
            self.score(predictions[:-1], refs, rows)

    def test_valid_outside_pairs_are_ignored_without_requiring_complete_predictions(self):
        predictions, refs, rows = self.fixtures()
        result = self.score(predictions[:-1], refs, rows, cohort_pair_ids=["pair0"])
        self.assertEqual(result["matched_questions"]["pairs"], 1)
        self.assertEqual(result["cohort"]["ignored_reference_questions"], 1)
        self.assertEqual(result["cohort"]["ignored_prediction_questions"]["direct|A"], 1)
        self.assertIsNone(result["comparisons"]["true_task_gap"]["metrics"]["raw_exact"]["ci95"])

    def test_duplicate_or_malformed_outside_predictions_fail(self):
        predictions, refs, rows = self.fixtures()
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            self.score(predictions + [predictions[-1]], refs, rows, cohort_pair_ids=["pair0"])
        for change in ({"invalid_object_id": 1}, {"prediction": None}, {"condition": "unknown"},
                       {"scene_id": "scene9999_00"}, {"qid": "recognition"}):
            predictions, refs, rows = self.fixtures()
            predictions[-1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.score(predictions, refs, rows, cohort_pair_ids=["pair0"])

    def test_reference_superset_is_validated_but_not_recognition_gating(self):
        predictions, refs, rows = self.fixtures()
        refs.append(dict(question_id="recognition", scene_id="scene0001_00", answers=["<OBJ000>"]))
        result = self.score(predictions, refs, rows)
        self.assertEqual(result["matched_questions"]["ignored_unmapped_reference_questions"], 1)
        refs[-1]["answers"] = [""]
        with self.assertRaises(ValueError):
            self.score(predictions, refs, rows)

    def test_missing_reference_fails_even_outside_selected_pair(self):
        predictions, refs, rows = self.fixtures()
        with self.assertRaisesRegex(ValueError, "missing paired reference"):
            self.score(predictions, refs[:-1], rows, cohort_pair_ids=["pair0"])

    def test_pair_references_require_identical_raw_sets_not_only_normalized_answers(self):
        predictions, refs, rows = self.fixtures()
        refs[2]["answers"] = ["a"]
        with self.assertRaisesRegex(ValueError, "same exact accepted"):
            self.score(predictions, refs, rows)
        refs[0]["answers"] = ["A", "yes"]
        refs[2]["answers"] = ["yes", "A", "A"]
        self.assertEqual(self.score(predictions, refs, rows)["matched_questions"]["pairs"], 2)

    def test_duplicate_references_and_scene_conflicts_fail(self):
        predictions, refs, rows = self.fixtures()
        with self.assertRaisesRegex(ValueError, "duplicate reference"):
            self.score(predictions, refs + [refs[-1]], rows, cohort_pair_ids=["pair0"])
        refs[-1]["scene_id"] = "scene0003_00"
        with self.assertRaisesRegex(ValueError, "reference/row scene"):
            self.score(predictions, refs, rows)

    def test_one_row_per_task_pair_and_same_scene_are_required(self):
        for change in ("missing", "duplicate", "scene", "qid_collision", "identity_alias"):
            predictions, refs, rows = self.fixtures()
            if change == "missing":
                rows["semantic"].pop()
            elif change == "duplicate":
                rows["direct"].append(dict(rows["direct"][0], qid="other", question_id="other"))
            elif change == "scene":
                rows["semantic"][0]["scene_id"] = "scene0003_00"
            elif change == "qid_collision":
                rows["semantic"][0].update(qid="direct0", question_id="direct0")
            else:
                rows["direct"][0]["question_id"] = "other"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.score(predictions, refs, rows)

    def test_explicit_cohort_and_condition_names_are_strict(self):
        for cohort in (None, [], ["pair0", "pair0"], ["missing"], [True], "pair0"):
            with self.subTest(cohort=cohort), self.assertRaises(ValueError):
                self.score(cohort_pair_ids=cohort)
        for conditions in ([], ["A", "A"], [True], "A"):
            with self.subTest(conditions=conditions), self.assertRaises(ValueError):
                self.score(conditions=conditions)

    def test_invalid_contrasts_and_duplicate_endpoint_aliases_fail(self):
        for contrasts in ({}, {"x": ("recognition|T", "direct|T")},
                          {"x": ("semantic|unknown", "direct|T")},
                          {"x": ("semantic|T", "semantic|T")}, {"x": ("semantic|T",)}):
            with self.subTest(contrasts=contrasts), self.assertRaises(ValueError):
                self.score(contrasts=contrasts)
        with self.assertRaisesRegex(ValueError, "duplicate aliases"):
            self.score(exploratory_linear_contrasts={"bad": {
                "semantic|T": 1, ("semantic", "T"): 1, "direct|T": -2}})
        with self.assertRaises(ValueError):
            self.score(exploratory_linear_contrasts={"bad": {"semantic|T": 1, "direct|T": 1}})

    def test_order_and_json_compatible_endpoint_forms_do_not_change_statistics(self):
        predictions, refs, rows = self.fixtures()
        reversed_rows = {task: values[::-1] for task, values in rows.items()}
        self.assertEqual(self.score(), self.score(predictions[::-1], refs[::-1], reversed_rows,
                                                  cohort_pair_ids=["pair1", "pair0"]))
        self.assertEqual(self.score(), self.score(contrasts={"true_task_gap": [["semantic", "T"], ["direct", "T"]]}))

    def test_multiple_captures_share_existing_scene_family_bootstrap(self):
        predictions, refs, rows = self.fixtures()
        for collection in (predictions, refs, *rows.values()):
            for row in collection:
                if row["scene_id"] == "scene0002_00":
                    row["scene_id"] = "scene0001_01"
        result = self.score(predictions, refs, rows)
        self.assertEqual(result["cohort"]["scenes"], 2)
        self.assertEqual(result["cohort"]["scene_families"], 1)
        self.assertFalse(result["bootstrap"]["intervals_available"])


if __name__ == "__main__":
    unittest.main()
