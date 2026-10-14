"""Synthetic-only checks for strict pairing, exact matches and clustered effects."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface.natural_qa_metrics import evaluate_natural_qa, normalize_answer


class NaturalQAMetricTests(unittest.TestCase):
    def fixtures(self):
        references = [
            {"question_id": "q1", "scene_id": "scene0001_00", "answers": ["chair", "a chair"]},
            {"question_id": "q2", "scene_id": "scene0001_00", "answers": ["two"]},
            {"question_id": "q3", "scene_id": "scene0002_00", "answers": ["1.2"]},
            {"question_id": "q4", "scene_id": "scene0002_01", "answers": ["-2"]},
        ]
        predictions = []
        for condition, texts in (("control", ["chair", "one", "wrong", "-2"]),
                                 ("treatment", ["The chair.", "two", "1.2", "wrong"])):
            predictions.extend({"qid": row["question_id"], "scene_id": row["scene_id"],
                                "condition": condition, "prediction": text}
                               for row, text in zip(references, texts))
        return predictions, references

    def score(self, predictions=None, references=None, **options):
        original_predictions, original_references = self.fixtures()
        arguments = dict(conditions=["control", "treatment"], contrasts={"effect": ("treatment", "control")},
                         replicates=120, seed=42)
        arguments.update(options)
        return evaluate_natural_qa(original_predictions if predictions is None else predictions,
                                   original_references if references is None else references, **arguments)

    def test_three_distinct_normalizations_and_no_substring(self):
        self.assertEqual(normalize_answer(" The  chair. ", metric="raw_exact"), " The  chair. ")
        self.assertEqual(normalize_answer(" The  chair. ", metric="lower_whitespace_exact"), "the chair.")
        self.assertEqual(normalize_answer(" The  chair. "), "chair")
        self.assertNotEqual(normalize_answer("armchair"), normalize_answer("chair"))
        self.assertNotEqual(normalize_answer("chair beside table"), normalize_answer("chair"))

    def test_numbers_minus_and_unicode_are_preserved(self):
        self.assertNotEqual(normalize_answer("1.2"), normalize_answer("12"))
        self.assertNotEqual(normalize_answer("-2"), normalize_answer("2"))
        self.assertNotEqual(normalize_answer("2"), normalize_answer("two"))
        self.assertNotEqual(normalize_answer("none"), normalize_answer("zero"))
        self.assertEqual(normalize_answer("  ÉTAGÈRE\t"), "étagère")
        self.assertEqual(normalize_answer("-1.2."), "-1.2")

    def test_expected_hits_changes_and_pairwise_effect(self):
        result = self.score()
        self.assertEqual(result["primary_metric"], "protocol_normalized_exact")
        self.assertEqual(result["conditions"]["treatment"]["metrics"]["protocol_normalized_exact"]["correct"], 3)
        self.assertEqual(result["conditions"]["treatment"]["metrics"]["raw_exact"]["correct"], 2)
        effect = result["comparisons"]["effect"]["metrics"]["protocol_normalized_exact"]
        self.assertEqual((effect["wins"], effect["losses"], effect["ties"]), (2, 1, 1))
        self.assertEqual(effect["delta"], .25)
        self.assertEqual(result["comparisons"]["effect"]["answer_changes"]["raw_exact"], 4)
        self.assertEqual(result["comparisons"]["effect"]["answer_changes"]["protocol_normalized_exact"], 3)

    def test_scene_family_clustering_retains_multiple_captures(self):
        result = self.score()
        self.assertEqual(result["cohort"]["scenes"], 3)
        self.assertEqual(result["cohort"]["scene_families"], 2)
        effect = result["comparisons"]["effect"]["metrics"]["protocol_normalized_exact"]
        # family1 contributes +1/2; family2's two captures cancel (+1 and -1).
        self.assertEqual(effect["ci95"], [0., .5])
        self.assertAlmostEqual(effect["scene_equal_delta"], 1 / 6)

    def test_question_and_scene_equal_weights_are_distinct(self):
        effect = self.score()["comparisons"]["effect"]["metrics"]["protocol_normalized_exact"]
        self.assertEqual(effect["delta"], .25)
        self.assertAlmostEqual(effect["scene_equal_delta"], 1 / 6)
        self.assertNotEqual(effect["delta"], effect["scene_equal_delta"])

    def test_input_order_does_not_change_seeded_result(self):
        predictions, references = self.fixtures()
        self.assertEqual(self.score(), self.score(predictions[::-1], references[::-1]))
        self.assertEqual(self.score(), self.score())

    def test_bonferroni_intervals_are_not_narrower_than_pointwise(self):
        effect = self.score()["comparisons"]["effect"]["metrics"]["protocol_normalized_exact"]
        self.assertAlmostEqual(effect["bonferroni_confidence"], 1 - .05 / 3)
        self.assertLessEqual(effect["ci_bonferroni"][0], effect["ci95"][0])
        self.assertGreaterEqual(effect["ci_bonferroni"][1], effect["ci95"][1])

    def test_multiple_comparison_family_is_only_for_primary_normalization(self):
        result = self.score()
        self.assertTrue(result["bootstrap"]["bonferroni_primary_metric_only"])
        metrics = result["comparisons"]["effect"]["metrics"]
        self.assertIsNone(metrics["raw_exact"]["ci_bonferroni"])
        self.assertIsNone(metrics["lower_whitespace_exact"]["ci_bonferroni"])

    def test_explicit_subset_accepts_valid_supersets_and_reports_ignored(self):
        result = self.score(cohort_qids=["q1", "q3"])
        self.assertEqual(result["cohort"]["questions"], 2)
        self.assertEqual(result["cohort"]["ignored_reference_questions"], 2)
        self.assertEqual(result["cohort"]["ignored_prediction_questions"], {"control": 2, "treatment": 2})

    def test_missing_prediction_never_silently_changes_cohort(self):
        predictions, references = self.fixtures()
        with self.assertRaisesRegex(ValueError, "incomplete pairing"):
            self.score(predictions[:-1], references)
        self.assertEqual(self.score(predictions[:-1], references, cohort_qids=["q1", "q2"])["cohort"]["questions"], 2)

    def test_duplicate_predictions_rejected_even_outside_cohort(self):
        predictions, references = self.fixtures()
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            self.score(predictions + [predictions[-1]], references, cohort_qids=["q1"])

    def test_scene_mismatch_rejected_even_outside_cohort(self):
        predictions, references = self.fixtures()
        predictions[-1]["scene_id"] = "scene9999_00"
        with self.assertRaisesRegex(ValueError, "scene mismatch"):
            self.score(predictions, references, cohort_qids=["q1"])

    def test_duplicate_references_rejected_even_outside_cohort(self):
        predictions, references = self.fixtures()
        with self.assertRaisesRegex(ValueError, "duplicate reference"):
            self.score(predictions, references + [references[-1]], cohort_qids=["q1"])

    def test_numeric_and_string_ids_pair_canonically_but_duplicates_fail(self):
        predictions, references = self.fixtures()
        references[0]["question_id"] = 1
        for row in predictions:
            if row["qid"] == "q1":
                row["qid"] = "1"
        self.assertEqual(self.score(predictions, references)["cohort"]["questions"], 4)
        references.append({**references[0], "question_id": "1"})
        with self.assertRaisesRegex(ValueError, "duplicate reference"):
            self.score(predictions, references)

    def test_bad_qids_scenes_conditions_and_prediction_types_fail(self):
        for change in ({"qid": True}, {"qid": ""}, {"scene_id": "other"},
                       {"condition": "undeclared"}, {"prediction": None}, {"invalid_object_id": 1}):
            predictions, references = self.fixtures()
            predictions[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.score(predictions, references)

    def test_blank_and_invalid_object_predictions_are_errors_not_exclusions(self):
        predictions, references = self.fixtures()
        predictions[0].update(prediction="chair", invalid_object_id=True)
        predictions[4]["prediction"] = "  "
        result = self.score(predictions, references)
        self.assertEqual(result["cohort"]["questions"], 4)
        self.assertEqual(result["conditions"]["control"]["invalid_object_id_predictions"], 1)
        self.assertEqual(result["conditions"]["treatment"]["blank_predictions"], 1)
        self.assertEqual(result["conditions"]["control"]["metrics"]["raw_exact"]["correct"], 1)

    def test_blank_reference_strings_and_normalized_empty_references_fail(self):
        for answers in ([], [""], [" "], ["!!!"], "chair", [123]):
            predictions, references = self.fixtures()
            references[0]["answers"] = answers
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                self.score(predictions, references)

    def test_every_original_reference_can_match_without_ranking(self):
        predictions, references = self.fixtures()
        references[0]["answers"] = ["armchair", "stool", "chair"]
        self.assertEqual(self.score(predictions, references)["conditions"]["control"]["metrics"]["raw_exact"]["correct"], 2)

    def test_one_family_has_no_inferential_interval(self):
        effect = self.score(cohort_qids=["q1", "q2"])["comparisons"]["effect"]["metrics"]["protocol_normalized_exact"]
        self.assertEqual(effect["delta"], .5)
        self.assertIsNone(effect["ci95"])
        self.assertIsNone(effect["ci_bonferroni"])

    def test_all_zero_is_marked_degenerate_not_equivalent(self):
        predictions, references = self.fixtures()
        for row in predictions:
            row["prediction"] = "wrong"
        result = self.score(predictions, references)
        effect = result["comparisons"]["effect"]["metrics"]["protocol_normalized_exact"]
        self.assertTrue(effect["bootstrap_degenerate"])
        self.assertTrue(effect["both_conditions_zero_accuracy"])
        self.assertEqual(effect["ci95"], [0., 0.])
        self.assertIn("do not establish equivalence", result["interpretation"])

    def test_invalid_cohorts_and_bootstrap_arguments_fail(self):
        for change in ({"cohort_qids": []}, {"cohort_qids": ["q1", "q1"]},
                       {"cohort_qids": ["absent"]}, {"replicates": 0}, {"seed": True},
                       {"family_size": 0}, {"conditions": ["control", "control"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.score(**change)

    def test_main_family_cannot_be_smaller_than_number_of_contrasts(self):
        with self.assertRaisesRegex(ValueError, "family_size"):
            self.score(contrasts={"one": ("treatment", "control"), "two": ("control", "treatment")}, family_size=1)

    def test_exploratory_analysis_has_no_automatic_bonferroni_claim(self):
        metric = self.score(family_size=None)["comparisons"]["effect"]["metrics"]["protocol_normalized_exact"]
        self.assertIsNone(metric["ci_bonferroni"])
        self.assertIsNone(metric["bonferroni_confidence"])

    def test_four_condition_interaction_uses_paired_cluster_resamples(self):
        predictions, references = self.fixtures()
        extra = []
        for row in predictions:
            extra.append({**row, "condition": "perturbed_" + row["condition"], "prediction": "wrong"})
        result = self.score(predictions + extra, references,
                            conditions=["control", "treatment", "perturbed_control", "perturbed_treatment"],
                            exploratory_linear_contrasts={"interaction": {
                                "treatment": 1, "control": -1, "perturbed_treatment": -1, "perturbed_control": 1}})
        main = result["comparisons"]["effect"]["metrics"]["protocol_normalized_exact"]
        interaction = result["exploratory_linear_contrasts"]["interaction"]["metrics"]["protocol_normalized_exact"]
        self.assertEqual(interaction["delta"], main["delta"])
        self.assertEqual(interaction["ci95"], main["ci95"])
        self.assertIsNone(interaction["ci_bonferroni"])
        self.assertNotIn("wins", interaction)

    def test_invalid_linear_contrast_weights_fail(self):
        for weights in ({"treatment": 1, "control": 1}, {"treatment": True, "control": -1},
                        {"treatment": float("nan"), "control": -1}, {"missing": 1, "control": -1}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.score(exploratory_linear_contrasts={"bad": weights})

    def test_input_records_are_unchanged(self):
        predictions, references = self.fixtures()
        before = deepcopy((predictions, references))
        self.score(predictions, references)
        self.assertEqual((predictions, references), before)


if __name__ == "__main__":
    unittest.main()
