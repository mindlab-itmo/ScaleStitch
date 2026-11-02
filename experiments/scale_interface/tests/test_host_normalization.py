"""Pinned published-cleaner fixtures and fixed-cohort host-normalized EM1 checks.

The first twelve pairs and the invalid/empty policies are the fourteen fixtures
in the frozen revision analysis. Expected hits are literal, not scorer-derived.
"""
import ast
import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.scale_interface import natural_qa_metrics as metrics
from experiments.scale_interface.prediction_scoring import score_predictions
from experiments.scale_interface.score_natural_qa import score


METRIC = "host_normalized_em1_3dgraphllm_19d65cd9_v1"
SOURCE_SHA = "4713ad135a7a3abd20771b007b193639f868b9e506b87edda8a9b34a571bcbd8"
SCENE = "scene0001_00"
FROZEN_PAIRS = [
    ("2", "two", 1), ("21", "twenty-one", 0), ("23", "twenty-three", 1),
    ("none", "zero", 1), ("there are two", "two", 0), ("two chairs", "two", 0),
    ("fourteen", "four", 0), ("the red chair", "red chair", 1),
    ("red the chair", "red chair", 1), ("tehre", "there", 1),
    ("002", "two", 0), ("one hundred", "100", 0),
]


def paired_records(cases):
    references, predictions = [], []
    for i, (prediction, answers, invalid) in enumerate(cases):
        qid, scene = str(i), "scene%04d_00" % (1 + i % 2)
        references.append(dict(question_id=qid, scene_id=scene, answers=answers))
        for condition, text, flag in (("T", prediction, invalid), ("A", "wrong", False)):
            predictions.append(dict(qid=qid, scene_id=scene, condition=condition,
                                    prediction=text, raw_prediction=text,
                                    invalid_object_id=flag))
    return predictions, references


class HostNormalizationTests(unittest.TestCase):
    def host(self):
        # A missing new module is an explicit RED assertion, not an import error.
        spec = importlib.util.find_spec("experiments.scale_interface.host_normalization")
        self.assertIsNotNone(spec, "the pinned published normalization is not implemented")
        return importlib.import_module(spec.name)

    def test_published_function_is_ast_identical_to_pinned_source(self):
        host = self.host()
        tree = ast.parse(inspect.getsource(host.clean_answer))
        digest = hashlib.sha256(ast.dump(tree.body[0], include_attributes=False).encode()).hexdigest()
        # Derived from the audited helper, whose whole-file SHA is SOURCE_SHA.
        self.assertEqual(digest, "c39a43084e3e42c2836457982d1bfade0e3b6890caf8c913ed8e711bc2a87836")
        self.assertEqual(host.provenance()["source_sha256"], SOURCE_SHA)

    def test_frozen_twelve_pairs_require_nonempty_exact_equality(self):
        clean = self.host().clean_answer
        for prediction, reference, expected in FROZEN_PAIRS:
            with self.subTest(prediction=prediction, reference=reference):
                self.assertEqual(int(bool(clean(prediction)) and clean(prediction) == clean(reference)), expected)

    def test_exact_published_digits_typos_and_article_rules(self):
        clean = self.host().clean_answer
        words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                 "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                 "seventeen", "eighteen", "nineteen", "twenty"]
        for number, word in enumerate(words):
            self.assertEqual(clean(str(number)), word)
        for text, expected in [("red a chair", "red chair"), ("red an apple", "red apple"),
                               ("letf rigth behine", "left right behind"), ("tv", "TV"),
                               ("no1 mat2", "no mat"), ("backwards", "backward"),
                               ("21", "21"), ("100", "100"), ("002", "002")]:
            self.assertEqual(clean(text), expected)

    def evaluate(self, predictions, references, **options):
        self.host()
        return metrics.evaluate_natural_qa(
            predictions, references, conditions=["T", "A"], contrasts={"effect": ["T", "A"]},
            primary_metric=METRIC, seed=42, replicates=30, family_size=1, **options)

    def test_all_reference_variants_are_cleaned_and_failures_retain_full_denominator(self):
        cases = [(prediction, [reference], False) for prediction, reference, _ in FROZEN_PAIRS]
        cases += [("2", ["two"], True), ("", ["two"], False),
                  ("two", ["unrelated", "2"], False), ("?!", ["two"], False),
                  ("\t", ["é"], False)]
        predictions, references = paired_records(cases)
        before = copy.deepcopy((predictions, references))
        result = self.evaluate(predictions, references)
        self.assertEqual(result["primary_metric"], METRIC)
        self.assertEqual(result["cohort"]["questions"], 17)
        condition = result["conditions"]["T"]
        self.assertEqual(condition["questions"], 17)
        self.assertEqual(condition["metrics"][METRIC]["correct"], 7)
        self.assertEqual(condition["metrics"][METRIC]["accuracy"], 7 / 17)
        self.assertEqual(condition["invalid_object_id_predictions"], 1)
        self.assertEqual(condition["blank_predictions"], 2)
        effect = result["comparisons"]["effect"]["metrics"][METRIC]
        self.assertEqual((effect["wins"], effect["losses"], effect["ties"]), (7, 0, 10))
        self.assertEqual(effect["delta"], 7 / 17)
        self.assertIsNotNone(effect["ci_bonferroni"])
        self.assertIsNone(result["comparisons"]["effect"]["metrics"]["protocol_normalized_exact"]["ci_bonferroni"])
        self.assertEqual(result["metric_provenance"]["source_sha256"], SOURCE_SHA)
        self.assertEqual((predictions, references), before)

    def test_legacy_counts_and_intervals_are_unchanged_by_default(self):
        predictions, references = paired_records([("2", ["two"], False), ("tehre", ["there"], False)])
        options = dict(conditions=["T", "A"], contrasts={"effect": ["T", "A"]},
                       seed=42, replicates=30, family_size=1)
        old = metrics.evaluate_natural_qa(predictions, references, **options)
        revised = self.evaluate(predictions, references)
        self.assertEqual(old["primary_metric"], "protocol_normalized_exact")
        self.assertNotIn(METRIC, old["conditions"]["T"]["metrics"])
        self.assertEqual(old["conditions"]["T"]["metrics"]["protocol_normalized_exact"]["correct"], 0)
        self.assertEqual(revised["conditions"]["T"]["metrics"][METRIC]["correct"], 2)
        for condition in ("T", "A"):
            for name in metrics.METRICS:
                self.assertEqual(old["conditions"][condition]["metrics"][name],
                                 revised["conditions"][condition]["metrics"][name])
        # Changing the primary endpoint changes only the planned multiplicity adjustment.
        for name in metrics.METRICS:
            old_effect = old["comparisons"]["effect"]["metrics"][name]
            new_effect = revised["comparisons"]["effect"]["metrics"][name]
            for field in ("delta", "ci95", "scene_equal_ci95", "wins", "losses", "ties"):
                self.assertEqual(old_effect[field], new_effect[field])

    def test_missing_or_duplicate_rows_cannot_shrink_the_fixed_host_cohort(self):
        predictions, references = paired_records([("2", ["two"], False), ("red", ["red"], True)])
        self.host()
        for rows, message in ((predictions[:-1], "incomplete pairing"),
                              (predictions + [predictions[0]], "duplicate prediction")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.evaluate(rows, references)

    def test_host_outcome_vectors_use_same_rule_and_never_salvage_invalid(self):
        predictions, references = paired_records([("2", ["two"], False), ("2", ["two"], True),
                                                  ("", ["two"], False)])
        self.host()
        guarded, outcomes = score_predictions(predictions, references,
            cohort=["0", "1", "2"], end_sym="</s>", scoring="host_normalized_em1")
        self.assertEqual(guarded, predictions)
        self.assertEqual([outcomes["conditions"]["T"][str(i)]["correct"] for i in range(3)], [1, 0, 0])
        self.assertEqual(outcomes["primary_metric"], METRIC)
        self.assertEqual(outcomes["metric_provenance"]["source_sha256"], SOURCE_SHA)


class HostScoringCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.predictions, self.references = paired_records([
            ("2", ["two"], False), ("2", ["two"], True), ("", ["two"], False),
            ("red the chair", ["red chair"], False)])
        self.qids = [str(i) for i in range(4)]
        self.write("references.json", self.references)
        self.write("cohort.json", self.qids)
        self.write("run/plan.json", dict(selected_qids=self.qids, conditions=[{"name": "T"}, {"name": "A"}]))
        self.write("run/result.json", dict(status="complete", question_ids=self.qids,
                   condition_names=["T", "A"], completed_predictions=8))
        self.write("run/effective_config.json", {"host": {"model": {"end_sym": "</s>"}}})
        (self.run / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.predictions), encoding="utf-8")
        self.seal()
        self.config = dict(run=str(self.run), references=str(self.root / "references.json"),
            cohort=str(self.root / "cohort.json"), conditions=["T", "A"], contrasts={"effect": ["T", "A"]},
            seed=42, family_size=1, replicates=30, scoring="host_normalized_em1", export_scored_rows=True)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def read(self, name):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def seal(self):
        hashes = {name: hashlib.sha256((self.run / name).read_bytes()).hexdigest()
                  for name in ("plan.json", "result.json", "predictions.jsonl", "effective_config.json")}
        self.write("run/status.json", dict(schema_version="scale_interface_run/v1", status="complete",
                                          action="generate-scene", artifacts=hashes))

    def test_cli_selects_host_primary_and_seals_provenance_and_vectors(self):
        config = self.write("config.json", self.config)
        process = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m",
            "experiments.scale_interface.score_natural_qa", "--config", str(config),
            "--output", str(self.root / "scored")], capture_output=True, text=True,
            encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"},
            cwd=Path(__file__).resolve().parents[3])
        self.assertEqual(process.returncode, 0, process.stderr[-2000:])
        result, manifest = self.read("scored/metrics.json"), self.read("scored/manifest.json")
        self.assertEqual(result["primary_metric"], METRIC)
        self.assertEqual(result["conditions"]["T"]["metrics"][METRIC]["correct"], 2)
        self.assertEqual(result["conditions"]["T"]["questions"], 4)
        self.assertEqual(manifest["scoring"], "host_normalized_em1")
        self.assertEqual(manifest["metric_provenance"]["source_sha256"], SOURCE_SHA)
        self.assertIn("host_normalization.py", manifest["code_sha256"])
        vectors = self.read("scored/scored_rows.json")
        self.assertEqual(vectors["primary_metric"], METRIC)
        self.assertEqual([vectors["conditions"]["T"][str(i)]["correct"] for i in range(4)], [1, 0, 0, 1])

    def test_nonexport_path_scores_legacy_rows_without_raw_prediction(self):
        self.config["export_scored_rows"] = False
        for row in self.predictions:
            del row["raw_prediction"]
        (self.run / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.predictions), encoding="utf-8")
        self.seal()
        manifest = score(self.write("config.json", self.config), self.root / "scored")
        self.assertEqual(self.read("scored/metrics.json")["primary_metric"], METRIC)
        self.assertIsNone(manifest["scored_rows"])

    def test_completed_run_hash_guard_still_applies_to_host_path(self):
        # A new endpoint must not bypass the existing sealed-run input checks.
        with (self.run / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**self.predictions[0], "qid": "extra"}) + "\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            score(self.write("config.json", self.config), self.root / "scored")
        self.assertFalse((self.root / "scored").exists())


if __name__ == "__main__":
    unittest.main()
