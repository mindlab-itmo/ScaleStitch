"""Frozen-host public commands reject retired codec workflows before loading."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import test_natural_qa as natural_fixtures
import test_public_runtime_provenance as provenance_fixtures
from scale_interface import __main__ as cli
from scale_interface import natural_qa as qa


class FrozenCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_retired_actions_fail_before_output_creation_or_model_loading(self):
        for action in ("evaluate", "train-codec", "generate"):
            output = self.root / action
            with self.subTest(action=action), mock.patch.object(sys, "stderr"), \
                 mock.patch.object(cli, "load_host") as load, self.assertRaises(SystemExit):
                cli.main([action, "--config", str(self.root / "absent.json"), "--output", str(output)])
            self.assertFalse(output.exists())
            load.assert_not_called()

    def test_only_current_representations_reach_preflight(self):
        for representation in ("trained", "random", "mean", "trim"):
            with self.subTest(representation=representation), \
                 mock.patch.object(qa, "preflight_natural_qa") as prepare, \
                 self.assertRaisesRegex(ValueError, "only absent, lexical and masked"):
                cli._generate_scene(object(), object(), {}, self.root,
                                    {"conditions": [{"representation": representation}]})
            prepare.assert_not_called()
        self.assertFalse((self.root / "plan.json").exists())
        self.assertFalse((self.root / "predictions.jsonl").exists())

    def test_retired_configuration_fails_before_hashing_or_loading_inputs(self):
        cases = [{"run": {"conditions": [{"representation": value}]}}
                 for value in ("trained", "random", "mean", "trim")]
        cases += [{"run": {"conditions": [{"representation": "absent"}]}, key: value}
                  for key, value in (("codec", "missing.pt"), ("codec", None), ("random_codec_seed", 42))]
        for index, config in enumerate(cases):
            path = self.root / f"config_{index}.json"
            output = self.root / f"run_{index}"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.subTest(config=config), mock.patch.object(cli.RuntimeProvenance, "configure") as configure, \
                 mock.patch.object(cli, "SceneDataset") as data, \
                 mock.patch.object(cli, "load_host") as load, self.assertRaises(ValueError):
                cli.main(["generate-scene", "--config", str(path), "--output", str(output)])
            configure.assert_not_called()
            data.assert_not_called()
            load.assert_not_called()
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
            self.assertFalse((output / "predictions.jsonl").exists())

    def test_cpu_generation_preserves_all_conditions_and_matched_masked_lengths(self):
        core, data = natural_fixtures.NaturalHost(), natural_fixtures.dataset(rows=1)
        options = {"conditions": [{"representation": value} for value in ("absent", "lexical", "masked")]}
        result = cli._generate_scene(core, data, {}, self.root, options)
        rows = [json.loads(line) for line in (self.root / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(result["completed_predictions"], 3)
        self.assertEqual([row["condition"] for row in rows],
                         [qa.NaturalQACondition(representation=value).name for value in ("absent", "lexical", "masked")])
        for field in ("packet_tokens", "prefix_tokens"):
            self.assertEqual(rows[1][field], rows[2][field])
        self.assertEqual(json.loads((self.root / "cohort.json").read_text()), [rows[0]["qid"]])

    def test_effective_configuration_can_be_replayed_without_retired_codec_fields(self):
        fixture = provenance_fixtures.PublicRuntimeTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        first, second = fixture.root / "first", fixture.root / "second"
        fixture.run_main(first)
        effective = json.loads((first / "effective_config.json").read_text())
        self.assertNotIn("codec", effective)
        self.assertNotIn("random_codec_seed", effective)
        fixture.config = effective
        fixture.run_main(second)
        self.assertEqual(json.loads((second / "status.json").read_text())["status"], "complete")
        self.assertEqual((first / "predictions.jsonl").read_bytes(), (second / "predictions.jsonl").read_bytes())

    def test_private_dispatch_cannot_reintroduce_retired_actions(self):
        with mock.patch.object(cli, "SceneDataset") as data, \
             mock.patch.object(cli, "load_host") as load, self.assertRaisesRegex(ValueError, "only generate-scene"):
            cli._execute(SimpleNamespace(action="train-codec"), {}, mock.Mock())
        data.assert_not_called()
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
