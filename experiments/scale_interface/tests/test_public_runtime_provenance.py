"""CPU checks for public run seals, failure lifecycle and channel-hook cleanup."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface import __main__ as cli
from scale_interface import natural_qa as qa
from scale_interface.channel_controls import channel_control


class PublicRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "model.safetensors").write_bytes(b"fixed base LLM bytes")
        (self.model / "tokenizer.json").write_text("{}", encoding="utf-8")
        for name in ("features.pt", "radii.pt", "attributes.pt", "host.pt", "bridge.pt"):
            (self.root / name).write_bytes(name.encode())
        (self.root / "questions.json").write_text("[]", encoding="utf-8")
        self.config = dict(seed=73, device="cpu",
            data={key: str(self.root / name) for key, name in dict(features="features.pt", radii="radii.pt",
                  attributes="attributes.pt", rows="questions.json").items()},
            host=dict(host_checkpoint=str(self.root / "host.pt"), deployment=str(self.root / "bridge.pt"),
                      model=dict(llama_model_path=str(self.model))),
            run=dict(conditions=[{"representation": "absent"}, {}]))
        self.config_path = self.root / "config.json"
        self.core = SimpleNamespace(host_mode="full", object_img_proj=torch.nn.Linear(2, 3),
                                    edge_proj=torch.nn.Linear(2, 3))
        self.plan = SimpleNamespace(
            conditions=[qa.NaturalQACondition("absent"), qa.NaturalQACondition()],
            rows=[SimpleNamespace(qid="q1"), SimpleNamespace(qid="q2")],
            metadata=lambda: {"selected_qids": ["q1", "q2"], "excluded_qids": ["excluded"]})

    def arguments(self, output, action="generate-scene"):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        return [action, "--config", str(self.config_path), "--output", str(output)]

    def generate(self, core, data, plan, **selected):
        for qid in selected["question_ids"]:
            for condition in selected["condition_names"]:
                yield dict(qid=qid, condition=condition, prediction="chair")

    def run_main(self, output, *, data_error=None, host_error=None, produce=None):
        with mock.patch.object(cli, "seed_everything") as seed, \
             mock.patch.object(cli, "SceneDataset", side_effect=data_error, return_value=object()) as data, \
             mock.patch.object(cli, "load_host", side_effect=host_error, return_value=self.core) as host, \
             mock.patch.object(qa, "preflight_natural_qa", return_value=self.plan), \
             mock.patch.object(qa, "iter_natural_qa", side_effect=produce or self.generate):
            cli.main(self.arguments(output))
        return seed, data, host

    def test_legacy_arguments_complete_seal_hashes_and_actual_cohort(self):
        output = self.root / "complete"
        self.config["run"]["question_ids"] = ["q2"]
        seed, _, host = self.run_main(output)
        seed.assert_called_once_with(73)
        host.assert_called_once_with(self.config["host"], device="cpu")
        state = json.loads((output / "status.json").read_text())
        self.assertEqual(state["schema_version"], "scale_interface_run/v1")
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["completed_predictions"], 2)
        self.assertEqual(json.loads((output / "cohort.json").read_text()), ["q2"])
        self.assertTrue(state["finished_utc"] >= state["started_utc"])
        self.assertEqual(state["seed"], 73)
        self.assertIn("__main__.py", state["code_sha256"])
        self.assertIn("channel_controls.py", state["code_sha256"])
        for path in (self.root / "features.pt", self.root / "radii.pt", self.root / "host.pt",
                     self.root / "bridge.pt", self.model / "model.safetensors", self.model / "tokenizer.json"):
            self.assertEqual(state["inputs"][str(path)], hashlib.sha256(path.read_bytes()).hexdigest())
        for name in ("predictions.jsonl", "plan.json", "result.json", "effective_config.json", "cohort.json"):
            self.assertEqual(state["artifacts"][name], hashlib.sha256((output / name).read_bytes()).hexdigest())
        self.assertEqual(state["predictions_sha256"], state["artifacts"]["predictions.jsonl"])
        effective = json.loads((output / "effective_config.json").read_text())
        self.assertEqual(effective["run"]["channel_intervention"], "original")
        self.assertEqual(effective["run"]["maximum_length"], 2502)

    def test_data_and_host_load_failures_record_failed_terminal_status(self):
        for phase, kwargs in (("loading_data", {"data_error": RuntimeError("bad data")}),
                              ("loading_host", {"host_error": RuntimeError("bad checkpoint")})):
            output = self.root / phase
            with self.subTest(phase=phase), self.assertRaises(RuntimeError):
                self.run_main(output, **kwargs)
            state = json.loads((output / "status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["phase"], phase)
            self.assertIn("RuntimeError", state["error"])
            self.assertIn("finished_utc", state)
            self.assertFalse((output / "result.json").exists())

    def test_zero_controls_preserve_shape_dtype_rng_and_cleanup(self):
        values = torch.ones(2, 2)
        for mode, target, other in (("zero_images", self.core.object_img_proj, self.core.edge_proj),
                                    ("zero_relations", self.core.edge_proj, self.core.object_img_proj)):
            before, other_before = target(values), other(values)
            rng = torch.get_rng_state().clone()
            with channel_control(self.core, mode):
                actual = target(values)
                self.assertEqual(actual.shape, before.shape)
                self.assertEqual(actual.dtype, before.dtype)
                self.assertTrue(torch.equal(actual, torch.zeros_like(before)))
                self.assertTrue(torch.equal(other(values), other_before))
            self.assertTrue(torch.equal(before, target(values)))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(target._forward_hooks)
        with channel_control(object(), "original"):
            pass
        for core, mode in ((object(), "zero_images"), (self.core, "drop_images")):
            with self.assertRaises(ValueError), channel_control(core, mode):
                pass

    def test_preflight_failure_removes_channel_hook(self):
        output = self.root / "preflight"
        output.mkdir()
        with mock.patch.object(qa, "preflight_natural_qa", side_effect=RuntimeError("bad full input")), \
             self.assertRaisesRegex(RuntimeError, "bad full input"):
            cli._generate_scene(self.core, object(), self.config, output,
                dict(self.config["run"], channel_intervention="zero_relations"))
        self.assertFalse(self.core.edge_proj._forward_hooks)

    def test_generation_failure_seals_partial_predictions_and_removes_hook(self):
        output = self.root / "generation"
        self.config["run"]["channel_intervention"] = "zero_images"
        closed = []

        def broken(core, data, plan, **selected):
            try:
                self.assertTrue(torch.equal(core.object_img_proj(torch.ones(2, 2)), torch.zeros(2, 3)))
                yield dict(qid="q1", condition=selected["condition_names"][0], prediction="first")
                raise RuntimeError("generation failed")
            finally:
                closed.append(True)

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.run_main(output, produce=broken)
        state = json.loads((output / "status.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertIn("predictions.jsonl", state["artifacts"])
        self.assertNotIn("result.json", state["artifacts"])
        self.assertEqual(closed, [True])
        self.assertFalse(self.core.object_img_proj._forward_hooks)

    def test_existing_output_and_unknown_action_do_not_change_files(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "status.json"
        sentinel.write_text('{"status":"complete"}')
        with self.assertRaises(FileExistsError):
            cli.main(self.arguments(output))
        self.assertEqual(sentinel.read_text(), '{"status":"complete"}')
        unused = self.root / "unused"
        with mock.patch.object(sys, "stderr"), self.assertRaises(SystemExit):
            cli.main(self.arguments(unused, action="unknown"))
        self.assertFalse(unused.exists())


if __name__ == "__main__":
    unittest.main()
