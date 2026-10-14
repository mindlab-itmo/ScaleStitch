"""Synthetic CPU checks for public preparation/scoring and provenance failures."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface.prepare_questions import prepare
from scale_interface.score_natural_qa import score


class PublicProtocolCommands(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.questions = [
            dict(question_id="q1", scene_id="scene0001_00", question=" What color is the chair? ",
                 answers=["red", "a red chair"], object_ids=[1], object_names=["SECRET"]),
            dict(question_id="q2", scene_id="scene0002_00", question="Which object is taller?",
                 answers=["table"], object_ids=[0]),
            dict(question_id="q3", scene_id="scene0003_00", question="What is there?", answers=["excluded"]),
        ]
        self.write("source.json", self.questions)
        self.write("ids.json", {"scene0001_00": [0, 1, 2], "scene0002_00": [0, 1]})
        self.config = dict(dataset="scanqa", split="val", seed=42, questions="source.json",
                           scene_object_ids="ids.json", cohort={"mode": "all_available"})
        # Config directory deliberately differs from cwd: inference path semantics.
        (self.root / "configs").mkdir()
        self.config_path = self.root / "configs/prepare.json"

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def read(self, name):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def prepare(self, name="prepared", config=None):
        self.write("configs/prepare.json", self.config if config is None else config)
        with mock.patch("pathlib.Path.cwd", return_value=self.root):
            return prepare(self.config_path, self.root / name)

    def fixture_run(self):
        self.prepare()
        conditions, qids = ["absent", "true"], ["q1", "q2"]
        self.write("run/result.json", dict(status="complete", question_ids=qids,
                                           condition_names=conditions, completed_predictions=4))
        self.write("run/plan.json", dict(selected_qids=qids, conditions=[{"name": c} for c in conditions],
                                         requested_generation={"question_ids": None, "condition_names": None}))
        predictions = [dict(qid=qid, scene_id=scene, condition=condition, prediction=answer)
                       for condition, answers in (("absent", ["red", "wrong"]), ("true", ["red", "table"]))
                       for qid, scene, answer in zip(qids, ["scene0001_00", "scene0002_00"], answers)]
        (self.root / "run/predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
        self.seal()
        self.score_config = dict(run="run", references="prepared/scoring/references.json",
                                 cohort="prepared/cohort.json", conditions=conditions,
                                 contrasts={"size": ["true", "absent"]}, seed=42,
                                 family_size=3, replicates=40)
        return predictions

    def seal(self):
        hashes = {name: hashlib.sha256((self.root / "run" / name).read_bytes()).hexdigest()
                  for name in ("predictions.jsonl", "plan.json", "result.json")}
        self.write("run/status.json", dict(schema_version="scale_interface_run/v1", status="complete",
                                           action="generate-scene", artifacts=hashes,
                                           predictions_sha256=hashes["predictions.jsonl"]))

    def score(self, name="scored"):
        path = self.write("configs/score.json", self.score_config)
        with mock.patch("pathlib.Path.cwd", return_value=self.root):
            return score(path, self.root / name)

    def test_scanqa_export_has_complete_input_ids_and_separate_answers(self):
        manifest = self.prepare()
        rows = self.read("prepared/questions.json")
        self.assertEqual([row["qid"] for row in rows], ["q1", "q2"])
        self.assertEqual(rows[0]["assigned_ids"], [0, 1, 2])
        self.assertEqual(rows[0]["prompt"], self.questions[0]["question"])
        self.assertFalse(any(key in row for row in rows for key in ("answers", "object_ids", "object_names")))
        self.assertEqual(self.read("prepared/scoring/references.json")[0]["answers"], ["red", "a red chair"])
        self.assertFalse(manifest["selection_uses_answers"])
        self.assertEqual(manifest["inputs"]["questions"]["sha256"],
                         hashlib.sha256((self.root / "source.json").read_bytes()).hexdigest())
        self.assertEqual(manifest["outputs_sha256"]["questions.json"],
                         hashlib.sha256((self.root / "prepared/questions.json").read_bytes()).hexdigest())

    def test_changing_answers_cannot_change_inference_or_cohort(self):
        self.prepare()
        for row in self.questions:
            row["answers"] = ["different answer"]
            row["object_ids"] = [999]
        self.write("source.json", self.questions)
        self.prepare("changed")
        for name in ("questions.json", "cohort.json"):
            self.assertEqual(self.read("prepared/" + name), self.read("changed/" + name))
        self.assertNotEqual(self.read("prepared/scoring/references.json"), self.read("changed/scoring/references.json"))

    def test_explicit_ids_preserve_source_order_and_reject_unavailable_questions(self):
        self.write("fixed.json", ["q2", "q1"])
        self.config["cohort"] = {"mode": "question_ids", "path": "fixed.json"}
        self.prepare()
        self.assertEqual(self.read("prepared/cohort.json"), ["q1", "q2"])
        self.write("fixed.json", ["q3"])
        with self.assertRaisesRegex(ValueError, "outside available input scenes"):
            self.prepare("bad")
        self.assertFalse((self.root / "bad").exists())

    def test_nonsequential_or_target_only_identity_mapping_is_not_reindexed(self):
        for ids in ([1, 2], [0, 2], [1, 0], [True], []):
            self.write("ids.json", {"scene0001_00": ids})
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "complete sequential"):
                self.prepare()

    def test_sqa3d_official_wrappers_keep_original_situation(self):
        questions = [dict(question_id="q1", scene_id="scene0001_00", question="What color is it?",
                          situation=" I am next to the chair. ", alternative_situation=["SECRET"]),
                     dict(question_id="q2", scene_id="scene0001_00", question="Which object is taller?",
                          situation="I face a table.")]
        self.write("source.json", {"questions": questions})
        self.write("answers.json", {"annotations": [dict(question_id=row["question_id"], scene_id=row["scene_id"],
                                                           answers=[{"answer": "red"}], position={"secret": 1})
                                                      for row in questions]})
        self.config.update(dataset="sqa3d", annotations="answers.json", cohort={"mode": "sqa3d_per_scene", "per_scene": 1})
        self.prepare()
        rows = self.read("prepared/questions.json")
        self.assertIn("q2", [row["qid"] for row in rows])  # Required lexical row is retained.
        for row in rows:
            original = next(q for q in questions if q["question_id"] == row["qid"])
            self.assertEqual(row["prompt"], "Situation:\n" + original["situation"] + "\n\nQuestion:\n" + original["question"])
            self.assertNotIn("alternative_situation", row)
            self.assertNotIn("position", row)

    def test_complete_sealed_run_scores_paired_effect_with_traceable_inputs(self):
        self.fixture_run()
        manifest = self.score()
        metrics = self.read("scored/metrics.json")
        self.assertEqual(metrics["comparisons"]["size"]["metrics"]["protocol_normalized_exact"]["delta"], .5)
        self.assertEqual(metrics["cohort"]["questions"], 2)
        self.assertEqual(manifest["inputs"]["predictions"]["sha256"], self.read("run/status.json")["artifacts"]["predictions.jsonl"])
        self.assertIn("natural_qa_metrics.py", manifest["code_sha256"])
        self.assertEqual(set(path.name for path in (self.root / "scored").iterdir()), {"metrics.json", "manifest.json"})

    def test_partial_run_rejected_even_when_fixed_cohort_happens_to_be_complete(self):
        self.fixture_run()
        lines = (self.root / "run/predictions.jsonl").read_text().splitlines()
        (self.root / "run/predictions.jsonl").write_text("\n".join(lines[:-1]) + "\n")
        self.write("subset.json", ["q1"])
        self.score_config["cohort"] = "subset.json"
        self.seal()
        with self.assertRaisesRegex(ValueError, "incomplete pairing"):
            self.score()
        self.assertFalse((self.root / "scored").exists())

    def test_modified_predictions_fail_seal_before_metrics(self):
        self.fixture_run()
        path = self.root / "run/predictions.jsonl"
        path.write_text(path.read_text().replace("wrong", "table"))
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self.score()

    def test_running_status_and_completion_count_disagreement_fail(self):
        self.fixture_run()
        status = self.read("run/status.json")
        self.write("run/status.json", {**status, "status": "running"})
        with self.assertRaisesRegex(ValueError, "complete generate-scene"):
            self.score()
        result = self.read("run/result.json")
        self.write("run/result.json", {**result, "completed_predictions": 3})
        self.seal()
        with self.assertRaisesRegex(ValueError, "completion count"):
            self.score()

    def test_fixed_cohort_never_silently_intersects_run(self):
        self.fixture_run()
        self.write("subset.json", ["q1", "missing"])
        self.score_config["cohort"] = "subset.json"
        with self.assertRaisesRegex(ValueError, "absent from the completed run"):
            self.score()

    def test_runtime_cohort_must_match_its_sealed_completion(self):
        self.fixture_run()
        path = self.write("run/cohort.json", ["q1", "q2"])
        status = self.read("run/status.json")
        status["artifacts"]["cohort.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write("run/status.json", status)
        self.score_config["cohort"] = "run/cohort.json"
        self.score()
        self.write("run/cohort.json", ["q1"])
        with self.assertRaisesRegex(ValueError, "sealed completed run"):
            self.score("changed")

    def test_new_directory_only_for_both_commands(self):
        self.fixture_run()
        self.score()
        with self.assertRaises(FileExistsError):
            self.prepare()
        with self.assertRaises(FileExistsError):
            self.score()

    def test_portable_entrypoints_do_not_import_torch(self):
        # Locate the implementation, since packaging relocates the test into
        # experiments/scale_interface/tests while retaining the package layout.
        repo = Path(sys.modules[prepare.__module__].__file__).resolve().parents[2]
        program = (
            "import sys; "
            "from experiments.scale_interface import prepare_questions, score_natural_qa; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", program], cwd=repo, check=True, capture_output=True, text=True)
        for module in ("prepare_questions", "score_natural_qa"):
            result = subprocess.run([sys.executable, "-m", "experiments.scale_interface." + module, "--help"],
                                    cwd=repo, check=True, capture_output=True, text=True)
            self.assertIn("--config", result.stdout)



    def test_natural_instruction_preserves_sqa_question_and_annotation_independence(self):
        questions = [dict(question_id="q1", scene_id="scene0001_00",
                          question="What color is it?", situation=" I am next to the chair. ")]
        annotations = [dict(question_id="q1", scene_id="scene0001_00",
                            answers=[{"answer": "red"}])]
        self.write("source.json", {"questions": questions})
        self.write("answers.json", {"annotations": annotations})
        self.config.update(dataset="sqa3d", annotations="answers.json", instruction="natural")
        manifest = self.prepare()
        rows = self.read("prepared/questions.json")
        expected = ("Situation:\n I am next to the chair. \n\nQuestion:\nWhat color is it? "
                    "Answer the question with a short natural-language answer. "
                    "Do not output object identifiers.")
        self.assertEqual(rows[0]["prompt"], expected)
        self.assertEqual(rows[0]["question"], questions[0]["question"])
        self.assertEqual(manifest["instruction"], "natural")
        self.assertNotIn("answers", rows[0])
        annotations[0]["answers"] = [{"answer": "blue"}]
        self.write("answers.json", {"annotations": annotations})
        self.prepare("changed")
        self.assertEqual(self.read("prepared/questions.json"), self.read("changed/questions.json"))
        self.assertEqual(self.read("prepared/cohort.json"), self.read("changed/cohort.json"))
        self.assertNotEqual(self.read("prepared/scoring/references.json"),
                            self.read("changed/scoring/references.json"))

    def test_sqa_full_input_exclusion_ledger_preserves_missing_scene_question(self):
        questions = [dict(question_id="q" + str(i), scene_id=scene,
                          question="What color is it?", situation="I am here.")
                     for i, scene in enumerate(("scene0001_00", "scene0002_00", "scene0003_00"), 1)]
        self.write("source.json", {"questions": questions})
        self.write("answers.json", {"annotations": [
            dict(question_id=row["question_id"], scene_id=row["scene_id"],
                 answers=[{"answer": "red"}]) for row in questions]})
        self.config.update(dataset="sqa3d", annotations="answers.json")
        manifest = self.prepare()
        self.assertEqual(manifest["source_questions"], 3)
        self.assertEqual(manifest["questions"], 2)
        self.assertFalse(manifest["complete_input_coverage"])
        self.assertEqual(self.read("prepared/input_exclusions.json"), [
            dict(qid="q3", scene_id="scene0003_00", stage="input_assets",
                 reason="scene_not_in_input_assets")])
        self.assertIn("input_exclusions.json", manifest["outputs_sha256"])
        self.assertEqual(self.read("prepared/cohort.json"), ["q1", "q2"])


if __name__ == "__main__":
    unittest.main()
