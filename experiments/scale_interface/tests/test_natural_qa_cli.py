"""CPU command-line checks with loading and generation replaced by small stubs."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface import __main__ as cli
from scale_interface import natural_qa as qa


def make_plan(conditions):
    return SimpleNamespace(
        conditions=conditions, rows=[SimpleNamespace(qid="q1"), SimpleNamespace(qid="q2")],
        metadata=lambda: {"selected_qids": ["q1", "q2"], "exclusions": [{"qid": "excluded"}]})


class NaturalCLI(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.conditions = [qa.NaturalQACondition("absent"), qa.NaturalQACondition()]
        self.options = {"conditions": [{"representation": "absent"}, {}]}
        self.plan = make_plan(self.conditions)
        self.core, self.data = object(), object()

    def run_scene(self, *, options=None, produce=None, plan=None):
        if produce is None:
            def produce(core, data, plan, **kwargs):
                for qid in kwargs["question_ids"]:
                    for condition in kwargs["condition_names"]:
                        yield {"qid": qid, "condition": condition, "prediction": "answer"}
        with mock.patch.object(qa, "preflight_natural_qa", return_value=plan or self.plan) as prepare, \
             mock.patch.object(qa, "iter_natural_qa", side_effect=produce) as generate:
            result = cli._generate_scene(self.core, self.data, {}, self.output,
                                         self.options if options is None else options)
        return result, prepare, generate

    def test_all_conditions_are_checked_before_streaming_filtered_questions(self):
        options = {**self.options, "question_ids": ["q2"], "condition_names": [self.conditions[1].name]}

        def produce(core, data, plan, **kwargs):
            self.assertTrue((self.output / "plan.json").exists())
            self.assertEqual(kwargs["question_ids"], ["q2"])
            yield {"qid": "q2", "prediction": "стул"}

        result, prepare, _ = self.run_scene(options=options, produce=produce)
        self.assertEqual(prepare.call_args.args[2], self.conditions)
        self.assertEqual(prepare.call_args.kwargs["max_new_tokens"], 32)
        self.assertEqual(result["completed_predictions"], 1)
        self.assertEqual(result["status"], "complete")
        saved = json.loads((self.output / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["selected_qids"], ["q1", "q2"])
        self.assertEqual(saved["requested_generation"]["question_ids"], ["q2"])
        self.assertEqual(json.loads((self.output / "predictions.jsonl").read_text(encoding="utf-8"))["prediction"], "стул")

    def test_default_generation_uses_whole_plan(self):
        result, _, _ = self.run_scene()
        self.assertEqual(result["completed_predictions"], 4)
        self.assertEqual(len((self.output / "predictions.jsonl").read_text().splitlines()), 4)

    def test_excluded_question_fails_after_saving_plan_before_generation(self):
        with mock.patch.object(qa, "preflight_natural_qa", return_value=self.plan), \
             mock.patch.object(qa, "iter_natural_qa") as generate:
            with self.assertRaisesRegex(ValueError, "unknown or excluded.*excluded"):
                cli._generate_scene(self.core, self.data, {}, self.output,
                                    {**self.options, "question_ids": ["excluded"]})
        self.assertTrue((self.output / "plan.json").exists())
        self.assertFalse((self.output / "predictions.jsonl").exists())
        generate.assert_not_called()

    def test_empty_plan_does_not_create_predictions(self):
        self.plan.rows = []
        with self.assertRaisesRegex(ValueError, "no questions passed"):
            self.run_scene()
        self.assertFalse((self.output / "predictions.jsonl").exists())

    def test_bad_options_are_rejected_without_preflight(self):
        for options in ({}, {"conditions": []}, {"conditions": [["absent"]]},
                        {**self.options, "suite": "necessity"}):
            with self.subTest(options=options), \
                 mock.patch.object(qa, "preflight_natural_qa") as prepare, \
                 self.assertRaises(ValueError):
                cli._generate_scene(self.core, self.data, {}, self.output, options)
            prepare.assert_not_called()

    def test_malformed_filters_are_refused(self):
        for field, value in (("question_ids", []), ("question_ids", ["q1", "q1"]),
                             ("question_ids", [True]), ("condition_names", "full"),
                             ("condition_names", ["unknown"])):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.run_scene(options={**self.options, field: value})
        self.assertFalse((self.output / "predictions.jsonl").exists())

    def test_rows_are_flushed_and_iterator_is_closed_on_failure(self):
        closed = []

        def produce(core, data, plan, **kwargs):
            try:
                yield {"qid": "q1", "prediction": "first"}
                self.assertEqual(len((self.output / "predictions.jsonl").read_text().splitlines()), 1)
                raise RuntimeError("model failed")
            finally:
                closed.append(True)

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            self.run_scene(produce=produce)
        self.assertEqual(closed, [True])
        self.assertFalse((self.output / "result.json").exists())

    def test_truncated_matrix_cannot_be_reported_complete(self):
        def produce(core, data, plan, **kwargs):
            yield {"qid": "q1", "prediction": "one"}

        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.run_scene(produce=produce)

    def test_main_dispatch_saves_completion(self):
        config_path = self.output / "config.json"
        questions_path = self.output / "questions.json"
        questions_path.write_text("[]", encoding="utf-8")
        config_path.write_text(json.dumps({"data": {"rows": str(questions_path)}, "host": {"model": {}},
                                          "device": "cpu", "run": self.options}))
        target = self.output / "run"
        with mock.patch.object(sys, "argv", ["scale_interface", "generate-scene", "--config", str(config_path), "--output", str(target)]), \
             mock.patch.object(cli, "seed_everything"), \
             mock.patch.object(cli, "SceneDataset", return_value=self.data), \
             mock.patch.object(cli, "load_host", return_value=self.core) as load, \
             mock.patch.object(cli, "_generate_scene", return_value={"status": "complete"}) as scene:
            cli.main()
        load.assert_called_once_with({"model": {}}, device="cpu")
        scene.assert_called_once_with(self.core, self.data, mock.ANY, target, self.options)
        self.assertEqual(json.loads((target / "result.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
