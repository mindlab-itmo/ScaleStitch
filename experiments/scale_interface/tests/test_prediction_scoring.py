"""Independent parity and integrity tests for compact public prediction scoring."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.scale_interface import natural_qa_metrics as metrics
from experiments.scale_interface.controlled_protocol import build_native_id_pairs, score_native_id
from experiments.scale_interface.prediction_scoring import score_predictions
from experiments.scale_interface.score_natural_qa import score


SCENE = "scene0001_00"


def controlled_fixture():
    built = build_native_id_pairs({SCENE: [
        dict(source_id=10, category="chair", rho=.1, radius_valid=True),
        dict(source_id=20, category="table", rho=.3, radius_valid=True),
        dict(source_id=30, category="other", rho=.2, radius_valid=True),
    ]})
    rows = built["direct_rows"] + built["named_rows"]
    for row in rows:
        row["assigned_ids"] = [2, 0, 1]
    return rows, built["references"], built["pair_metadata"]


class PredictionScoringTests(unittest.TestCase):
    def setUp(self):
        self.questions, self.references, self.metadata = controlled_fixture()
        self.qid = self.questions[0]["qid"]

    def prediction(self, raw="<OBJ002></s>", mapped="<OBJ001>", invalid=False):
        return dict(qid=self.qid, scene_id=SCENE, condition="T",
                    raw_prediction=raw, prediction=mapped, invalid_object_id=invalid)

    def native(self, prediction=None, **overrides):
        options = dict(cohort=[self.qid], end_sym="</s>", scoring="native_id",
                       questions=self.questions, pair_metadata=self.metadata)
        options.update(overrides)
        return score_predictions([prediction or self.prediction()], self.references, **options)

    def test_strict_native_hits_mapping_and_candidate_compliance(self):
        # Expectations from the frozen collector: no token extraction or second remap.
        cases = [
            (" <OBJ002>!? </s> trailing", " <OBJ001>!? ", False, 1, 1, 1),
            ("<OBJ001>", "<OBJ000>", False, 0, 1, 1),
            ("<OBJ000>", "<OBJ002>", False, 0, 0, 1),
            ("the <OBJ002></s>", "<OBJ001>", False, 0, 0, 0),
            ("<OBJ002> <OBJ000>", "<OBJ001>", False, 0, 0, 0),
            ("<OBJ2>", "<OBJ001>", False, 0, 0, 0),
            ("<obj002>", "<OBJ001>", False, 0, 0, 0),
            ("<OBJ002>", "object <OBJ001>", False, 0, 0, 0),
            ("<OBJ002>", "<OBJ001>", True, 0, 0, 0),
            ("<OBJ999>", "<OBJ999>", False, 0, 0, 1),
            ("", "", False, 0, 0, 0),
        ]
        for raw, mapped, invalid, hit, in_pair, native in cases:
            with self.subTest(raw=raw, mapped=mapped, invalid=invalid):
                row = self.prediction(raw, mapped, invalid)
                original = copy.deepcopy(row)
                _, outcomes = self.native(row)
                result = outcomes["conditions"]["T"][self.qid]
                self.assertEqual((result["correct"], result["candidate_pair_compliant"],
                                  result["native_id_only"]), (hit, in_pair, native))
                self.assertEqual(row, original)
                if not invalid:
                    pure = score_native_id(raw, assigned_ids=[2, 0, 1],
                                           scene_mask=[True] * 3, end_sym="</s>",
                                           pair_source_rows=[0, 1], target_source_row=1)
                    if mapped == "<OBJ001>" or native:
                        self.assertEqual(result["correct"], pure["exact_match"])

    def test_guarded_text_preserves_frozen_supplementary_metrics(self):
        # strict_native retained every valid mapped text verbatim and blanked invalid syntax.
        cases = [
            ("<OBJ002>!?</s>", " <OBJ001>!? ", False, " <OBJ001>!? "),
            ("<OBJ001>", "<OBJ000>", False, "<OBJ000>"),
            ("<OBJ000>", "<OBJ002>", False, "<OBJ002>"),
            ("the <OBJ002>", "<OBJ001>", False, ""),
            ("<OBJ002>", "<OBJ001>", True, ""),
        ]
        for raw, mapped, invalid, expected in cases:
            with self.subTest(raw=raw, mapped=mapped, invalid=invalid):
                guarded, _ = self.native(self.prediction(raw, mapped, invalid))
                self.assertEqual(guarded[0]["prediction"], expected)

    def test_raw_and_mapped_ids_must_agree_exactly_once(self):
        with self.assertRaisesRegex(ValueError, "mapping mismatch"):
            self.native(self.prediction("<OBJ002>", "<OBJ000>"))
        with self.assertRaisesRegex(ValueError, "scorer disagree"):
            self.native(self.prediction("<OBJ999>", "<OBJ001>"))

    def test_invalid_flags_and_raw_mapped_types_are_rejected(self):
        for row in (self.prediction(invalid=1), self.prediction(raw=None),
                    self.prediction(mapped=7)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.native(row)

    def test_target_reference_pair_scene_and_assignment_mismatches_rejected(self):
        mutations = [
            lambda: self.references[0].update(answers=["<OBJ000>"]),
            lambda: self.metadata[0].update(target_source_row=0),
            lambda: self.metadata[0].update(scene_id="scene0002_00"),
            lambda: self.questions[0].update(pair_id="unrelated-pair"),
            lambda: self.questions[0].update(assigned_ids=[2, 2, 1]),
        ]
        for mutate in mutations:
            self.setUp()
            mutate()
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.native()

    def test_both_candidates_must_be_visible_even_when_target_is_visible(self):
        for role in self.metadata[0]["roles"].values():
            if role["row_id"] != self.metadata[0]["target_source_row"]:
                role["row_id"] = 99
        with self.assertRaises(ValueError):
            self.native()

    def test_incomplete_duplicate_or_missing_controlled_cohort_rejected(self):
        qids = [row["qid"] for row in self.questions]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.native(cohort=qids)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.native(questions=[])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_predictions([self.prediction(), self.prediction()], self.references,
                              cohort=[self.qid], end_sym="</s>", scoring="native_id",
                              questions=self.questions, pair_metadata=self.metadata)

    def test_natural_vectors_retain_all_failures_and_export_no_text(self):
        examples = [
            ("The Chair.</s>", "The Chair.", False, "chair", 1, 0),
            ("<OBJ001></s>", "<OBJ001>", False, "<OBJ001>", 1, 1),
            ("red</s>", "red", True, "red", 0, 0),
            ("", "", False, "blue", 0, 0),
        ]
        predictions, references = [], []
        for i, (raw, mapped, invalid, answer, _, _) in enumerate(examples):
            predictions.append(dict(qid=str(i), scene_id=SCENE, condition="T",
                                    raw_prediction=raw, prediction=mapped,
                                    invalid_object_id=invalid))
            references.append(dict(question_id=str(i), scene_id=SCENE, answers=[answer]))
        before = copy.deepcopy(predictions)
        guarded, outcomes = score_predictions(predictions, references,
                                              cohort=[str(i) for i in range(4)], end_sym="</s>")
        self.assertEqual(guarded, before)
        for i, (_, _, _, _, hit, native) in enumerate(examples):
            row = outcomes["conditions"]["T"][str(i)]
            self.assertEqual(set(row), {"scene_id", "correct", "candidate_pair_compliant",
                                        "native_id_only", "normalized_without_native_guard"})
            self.assertEqual((row["correct"], row["native_id_only"]), (hit, native))
            self.assertIsNone(row["candidate_pair_compliant"])

    def test_punctuation_only_keeps_articles_and_internal_numeric_punctuation(self):
        cases = [("  The   CHAIR.!? ", "the chair", "chair"),
                 ("An apple.", "an apple", "apple"),
                 ("-1.25 meters?!", "-1.25 meters", "-1.25 meters"),
                 ("a/b, c: d", "a/b, c: d", "a/b, c: d")]
        for text, punctuation_only, primary in cases:
            self.assertEqual(metrics.normalize_answer(text, metric="punctuation_only_exact"),
                             punctuation_only)
            self.assertEqual(metrics.normalize_answer(text), primary)
            self.assertEqual(metrics.normalize_answer(text, metric="raw_exact"), text)
        with self.assertRaises(ValueError):
            metrics.normalize_answer("chair", metric="unknown")


class ScoringIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.questions, self.references, self.metadata = controlled_fixture()
        self.qids = [row["qid"] for row in self.questions]
        self.conditions = ["A", "T"]
        self.write("questions.json", self.questions)
        self.write("references.json", self.references)
        self.write("pairs.json", self.metadata)
        self.write("cohort.json", self.qids)
        self.write("run/plan.json", dict(selected_qids=self.qids,
                   conditions=[{"name": name} for name in self.conditions]))
        self.write("run/result.json", dict(status="complete", question_ids=self.qids,
                   condition_names=self.conditions, completed_predictions=4))
        self.write("run/effective_config.json", {"host": {"model": {"end_sym": "</s>"}}})
        self.predictions = [dict(qid=qid, scene_id=SCENE, condition=condition,
                                raw_prediction="<OBJ001>" if condition == "A" else "<OBJ002></s>",
                                prediction="<OBJ000>" if condition == "A" else "<OBJ001>",
                                invalid_object_id=False)
                            for condition in self.conditions for qid in self.qids]
        self.write_predictions()
        self.seal()
        self.config = dict(run=str(self.run), references=str(self.root / "references.json"),
                           cohort=str(self.root / "cohort.json"), conditions=self.conditions,
                           contrasts={"utility": ["T", "A"]}, seed=42, family_size=1,
                           replicates=20, scoring="native_id", export_scored_rows=True,
                           questions=str(self.root / "questions.json"),
                           pair_metadata=str(self.root / "pairs.json"))

    def write(self, relative, value):
        path = self.root / relative
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def read(self, relative):
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def write_predictions(self):
        (self.run / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.predictions), encoding="utf-8")

    def seal(self, effective=True):
        names = ["plan.json", "result.json", "predictions.jsonl"]
        if effective:
            names.append("effective_config.json")
        hashes = {name: hashlib.sha256((self.run / name).read_bytes()).hexdigest() for name in names}
        rows_path = str((self.root / "questions.json").resolve())
        return self.write("run/status.json", dict(
            schema_version="scale_interface_run/v1", status="complete", action="generate-scene",
            artifacts=hashes, input_roles={"data.rows": rows_path},
            inputs={rows_path: hashlib.sha256((self.root / "questions.json").read_bytes()).hexdigest()}))

    def score(self):
        path = self.write("config.json", self.config)
        return score(path, self.root / "scored")

    def test_sealed_native_run_scores_and_exports_compact_outcomes(self):
        manifest = self.score()
        result = self.read("scored/metrics.json")
        self.assertEqual(result["comparisons"]["utility"]["metrics"]["protocol_normalized_exact"]["delta"], 1)
        self.assertEqual(manifest["scoring"], "native_id")
        self.assertEqual(manifest["scored_rows"], "scored_rows.json")
        self.assertIn("effective_config", manifest["inputs"])
        self.assertIn("questions", manifest["inputs"])
        outcomes = self.read("scored/scored_rows.json")["conditions"]
        self.assertEqual(set(outcomes), {"A", "T"})
        self.assertEqual(set(outcomes["A"]), set(self.qids))
        self.assertTrue(all(row["candidate_pair_compliant"] == 1 for row in outcomes["A"].values()))

    def test_effective_config_must_be_sealed_and_unchanged(self):
        self.seal(effective=False)
        with self.assertRaisesRegex(ValueError, "effective_config.json"):
            self.score()
        self.seal()
        self.write("run/effective_config.json", {"host": {"model": {"end_sym": "different"}}})
        with self.assertRaisesRegex(ValueError, "effective_config.json"):
            self.score()
        self.assertFalse((self.root / "scored").exists())

    def test_controlled_rows_must_match_recorded_data_rows_hash(self):
        status = self.read("run/status.json")
        del status["input_roles"]["data.rows"]
        self.write("run/status.json", status)
        with self.assertRaisesRegex(ValueError, "recorded inference input"):
            self.score()
        self.seal()
        self.questions[0]["prompt"] += " changed"
        self.write("questions.json", self.questions)
        with self.assertRaisesRegex(ValueError, "recorded inference input"):
            self.score()

    def test_sealed_partial_prediction_matrix_is_still_rejected(self):
        self.predictions.pop()
        self.write_predictions()
        self.seal()
        with self.assertRaisesRegex(ValueError, "incomplete pairing"):
            self.score()

    def test_default_natural_scorer_still_accepts_legacy_prediction_rows(self):
        for key in ("scoring", "export_scored_rows", "questions", "pair_metadata"):
            del self.config[key]
        for row in self.predictions:
            del row["raw_prediction"]
            del row["invalid_object_id"]
        self.write_predictions()
        self.seal(effective=False)
        manifest = self.score()
        self.assertEqual(manifest["scoring"], "natural_qa")
        self.assertIsNone(manifest["scored_rows"])
        self.assertNotIn("effective_config", manifest["inputs"])
        self.assertFalse((self.root / "scored/scored_rows.json").exists())


if __name__ == "__main__":
    unittest.main()
