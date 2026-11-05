"""Public exported-host generation preserves learned complete answer endings."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from transformers import StoppingCriteriaList

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface import __main__ as cli
from scale_interface import natural_qa as qa
from scale_interface import packet_lora_cli as lora_cli
import test_natural_qa as natural_fixtures
from test_packet_lora_stopping import PartialEndingStop, stopping_host


class SceneCompleteEndingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.core = stopping_host()
        self.original_generate = self.core.llama_model.generate
        self.legacy = PartialEndingStop()
        self.original_stops = [value.clone() for value in self.legacy.stops]
        self.cpu = {"assigned_ids": torch.tensor([[2, 0, 1]]),
                    "scene_mask": torch.ones(1, 3, dtype=torch.bool)}
        self.options = {"conditions": [{"representation": "absent"}, {"representation": "lexical"}]}
        self.closed = False

    def prepare(self, core, data, conditions, **options):
        self.assertNotIn("complete_host_ending", options)
        return SimpleNamespace(conditions=conditions, rows=[SimpleNamespace(qid="q0")],
                               metadata=lambda: {"generation": {"max_new_tokens": 32}})

    def produce(self, core, data, plan, **selected):
        try:
            for qid in selected["question_ids"]:
                for condition in selected["condition_names"]:
                    tokens = core.llama_model.generate(stopping_criteria=StoppingCriteriaList([self.legacy]))
                    yield dict(qa._decoded_prediction(core, tokens[0], self.cpu), qid=qid, condition=condition)
        finally:
            self.closed = True

    def run_scene(self, options, *, produce=None, name="output"):
        output = self.root / name
        output.mkdir()
        with mock.patch.object(qa, "preflight_natural_qa", side_effect=self.prepare), \
             mock.patch.object(qa, "iter_natural_qa", side_effect=produce or self.produce):
            try:
                result = cli._generate_scene(self.core, object(), {}, output, options)
            except ValueError as error:
                if "unknown generate-scene options" in str(error):
                    self.fail("public generate-scene rejects the complete-host-ending option")
                raise
        return result, output

    def assert_restored(self):
        self.assertEqual(self.core.llama_model.generate, self.original_generate)
        self.assertTrue(self.closed)
        self.assertTrue(all(torch.equal(before, after)
                            for before, after in zip(self.original_stops, self.legacy.stops)))
        legacy = self.core.llama_model.generate(stopping_criteria=StoppingCriteriaList([self.legacy]))
        self.assertEqual(legacy.tolist(), [[128256, 524]])

    def test_explicit_complete_ending_is_recorded_and_keeps_native_identifiers(self):
        result, output = self.run_scene({**self.options, "complete_host_ending": True})
        rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(result["completed_predictions"], 2)
        self.assertEqual(result["answer_stopping"]["protocol"], "host_stops_without_partial_ending/v2")
        self.assertEqual(result["answer_stopping"]["ending_token_ids"], [524, 82, 29])
        self.assertEqual(json.loads((output / "plan.json").read_text())["answer_stopping"], result["answer_stopping"])
        for row in rows:
            self.assertEqual(row["output_token_ids"], [128256, 524, 82, 29])
            self.assertEqual(row["raw_prediction"], "<OBJ000></s>")
            self.assertEqual(row["prediction"], "<OBJ002>")
            self.assertEqual(lora_cli.native_hit(row, 2, [2, 0, 1], "</s>"), 1)
        self.assert_restored()

    def test_missing_and_explicit_false_retain_identical_legacy_outputs(self):
        first, a = self.run_scene(self.options, name="default")
        second, b = self.run_scene({**self.options, "complete_host_ending": False}, name="false")
        self.assertEqual((a / "predictions.jsonl").read_bytes(), (b / "predictions.jsonl").read_bytes())
        self.assertEqual(first["answer_stopping"], {"protocol": "legacy_host_stops/v1"})
        self.assertEqual(first, second)
        rows = [json.loads(line) for line in (a / "predictions.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["output_token_ids"] == [128256, 524] for row in rows))
        self.assert_restored()

    def test_nonboolean_option_is_refused_before_preflight(self):
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), mock.patch.object(qa, "preflight_natural_qa") as prepare, \
                 self.assertRaisesRegex(ValueError, "complete_host_ending must be a boolean"):
                cli._generate_scene(object(), object(), {}, self.root,
                                    {**self.options, "complete_host_ending": value})
            prepare.assert_not_called()

    def test_stream_failure_closes_iterator_and_restores_legacy_stops(self):
        def interrupted(*args, **kwargs):
            source = self.produce(*args, **kwargs)
            try:
                yield next(source)
                raise RuntimeError("generation interrupted")
            finally:
                source.close()
        with self.assertRaisesRegex(RuntimeError, "generation interrupted"):
            self.run_scene({**self.options, "complete_host_ending": True}, produce=interrupted)
        rows = (self.root / "output/predictions.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])["output_token_ids"], [128256, 524, 82, 29])
        self.assert_restored()

    def test_main_records_requested_and_effective_ending_policy(self):
        rows = self.root / "rows.json"
        rows.write_text("[]", encoding="utf-8")
        config = self.root / "config.json"
        config.write_text(json.dumps(dict(data={"rows": str(rows)}, host={"model": {}},
            device="cpu", run={**self.options, "complete_host_ending": True})), encoding="utf-8")
        output = self.root / "main"
        with mock.patch.object(cli, "seed_everything"), \
             mock.patch.object(cli, "SceneDataset", return_value=object()), \
             mock.patch.object(cli, "load_host", return_value=self.core), \
             mock.patch.object(qa, "preflight_natural_qa", side_effect=self.prepare), \
             mock.patch.object(qa, "iter_natural_qa", side_effect=self.produce):
            try:
                cli.main(["generate-scene", "--config", str(config), "--output", str(output)])
            except ValueError as error:
                if "unknown generate-scene options" in str(error):
                    self.fail("main cannot run the requested complete host ending")
                raise
        self.assertTrue(json.loads((output / "effective_config.json").read_text())["run"]["complete_host_ending"])
        policy = json.loads((output / "result.json").read_text())["answer_stopping"]
        self.assertEqual(policy, json.loads((output / "plan.json").read_text())["answer_stopping"])
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")
        self.assert_restored()

    def test_complete_ending_works_with_real_preflight_and_stream(self):
        core = natural_fixtures.NaturalHost()
        core.end_sym = "</s>"
        core.llama_model.answer = "<OBJ000></s>"
        original = core.llama_model.generate
        output = self.root / "actual_stream"
        output.mkdir()
        result = cli._generate_scene(core, natural_fixtures.dataset(rows=1), {}, output,
                                     {**self.options, "complete_host_ending": True})
        self.assertEqual(result["completed_predictions"], 2)
        self.assertEqual(result["answer_stopping"]["ending_token_ids"], [60, 47, 115, 62])
        rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["prediction"] == "<OBJ002>" for row in rows))
        self.assertEqual(core.llama_model.generate, original)


if __name__ == "__main__":
    unittest.main()
