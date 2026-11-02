"""Decoder overrides through real host assembly, plus durable CLI receipts."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_interface import __main__ as cli
from scale_interface import natural_qa as qa
from scale_interface.physical_stream_host_boundary import HostPhysicalStream
from test_natural_qa import NaturalHost, dataset


class DecoderOptionsTests(unittest.TestCase):
    def test_explicit_penalty_overrides_fixed_host_kwargs_in_actual_generate_call(self):
        core, data = NaturalHost(), dataset(rows=1)
        conditions = [qa.NaturalQACondition("absent"), qa.NaturalQACondition()]
        plan = qa.preflight_natural_qa(core, data, conditions, repetition_penalty=1.0)
        self.assertEqual(core.llama_model.calls, [])
        rows = list(qa.iter_natural_qa(core, data, plan))
        # host.py passes 3.0; the real packet boundary must override it with 1.0.
        self.assertEqual([call["repetition_penalty"] for call in core.llama_model.calls], [1.0, 1.0])
        self.assertEqual(plan.metadata()["generation"]["repetition_penalty"], 1.0)
        self.assertEqual([row["effective_generation"]["repetition_penalty"] for row in rows], [1.0, 1.0])
        self.assertTrue(all(call["num_beams"] == 1 and not call["do_sample"] for call in core.llama_model.calls))

    def test_omitted_penalty_preserves_historical_value_and_records_it(self):
        core, data = NaturalHost(), dataset(rows=1)
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition("absent")])
        list(qa.iter_natural_qa(core, data, plan))
        self.assertEqual(core.llama_model.calls[0]["repetition_penalty"], 3.0)
        self.assertEqual(plan.metadata()["generation"]["repetition_penalty"], 3.0)

    def test_nonfinite_nonpositive_or_nonnumeric_penalties_fail_before_generation(self):
        for value in [0, -1, float("nan"), float("inf"), -float("inf"), True, None, "1.0"]:
            with self.subTest(value=value):
                core, data = NaturalHost(), dataset(rows=1)
                with self.assertRaisesRegex(ValueError, "repetition_penalty"):
                    qa.preflight_natural_qa(core, data, [qa.NaturalQACondition("absent")], repetition_penalty=value)
                self.assertEqual(core.llama_model.calls, [])
                with self.assertRaisesRegex(ValueError, "repetition_penalty"):
                    HostPhysicalStream(core, generation_options={"repetition_penalty": value})

    def test_invalid_modified_plan_cannot_bypass_generation_validation(self):
        core, data = NaturalHost(), dataset(rows=1)
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition("absent")])
        invalid = replace(plan, repetition_penalty=float("nan"))
        with self.assertRaisesRegex(ValueError, "repetition_penalty"):
            list(qa.iter_natural_qa(core, data, invalid))
        self.assertEqual(core.llama_model.calls, [])

    def test_valid_post_preflight_decoder_change_invalidates_the_sealed_plan(self):
        core, data = NaturalHost(), dataset(rows=1)
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition("absent")])
        changed = replace(plan, repetition_penalty=1.0)
        with self.assertRaisesRegex(ValueError, "generation.*preflight"):
            list(qa.iter_natural_qa(core, data, changed))
        self.assertEqual(core.llama_model.calls, [])
        self.assertEqual(plan.metadata()["generation_fingerprint"], plan.generation_fingerprint)

    def test_cli_config_and_flag_record_the_effective_value_in_real_receipts(self):
        for configured, override, expected in [(None, None, 3.0), (1.0, None, 1.0), (3.0, 1.0, 1.0)]:
            with self.subTest(configured=configured, override=override), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                core, data = NaturalHost(), dataset(rows=1)
                options = {"conditions": [{"representation": "absent"}]}
                if configured is not None:
                    options["repetition_penalty"] = configured
                config = dict(device="cpu", seed=42, data={"rows": []}, host={"model": {}}, run=options)
                config_path = path / "config.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                args = ["generate-scene", "--config", str(config_path), "--output", str(path / "run")]
                if override is not None:
                    args.extend(["--repetition-penalty", str(override)])
                # Only loading external assets is substituted. Assembly, boundary,
                # streaming and RuntimeProvenance all execute their real code.
                with mock.patch.object(cli, "SceneDataset", return_value=data), \
                     mock.patch.object(cli, "load_host", return_value=core):
                    cli.main(args)
                effective = json.loads((path / "run/effective_config.json").read_text(encoding="utf-8"))
                status = json.loads((path / "run/status.json").read_text(encoding="utf-8"))
                plan = json.loads((path / "run/plan.json").read_text(encoding="utf-8"))
                self.assertEqual(effective["run"]["repetition_penalty"], expected)
                self.assertEqual(plan["generation"]["repetition_penalty"], expected)
                self.assertEqual(core.llama_model.calls[0]["repetition_penalty"], expected)
                self.assertEqual(status["status"], "complete")
                self.assertEqual(status["completed_predictions"], 1)
                self.assertIn("effective_config_sha256", status)

    def test_cli_rejects_invalid_config_before_loading_external_assets(self):
        for value in [0, float("nan"), float("inf")]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                config_path = path / "config.json"
                config_path.write_text(json.dumps({"run": {
                    "conditions": [{"representation": "absent"}], "repetition_penalty": value}}), encoding="utf-8")
                with mock.patch.object(cli, "SceneDataset", side_effect=AssertionError("assets loaded")), \
                     self.assertRaisesRegex(ValueError, "repetition_penalty"):
                    cli.main(["generate-scene", "--config", str(config_path), "--output", str(path / "run")])
                status = json.loads((path / "run/status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["status"], "failed")
                self.assertFalse((path / "run/effective_config.json").exists())


if __name__ == "__main__":
    unittest.main()
