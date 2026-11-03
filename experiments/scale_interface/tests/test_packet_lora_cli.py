"""CPU contracts for the portable matched continuation command."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_interface import packet_lora as pl
from test_packet_lora import DifferentiableHost, example, frozen_fingerprint, identity


class OverflowOnce:
    """A finite-loss/nonfinite-gradient AMP attempt, then an ordinary update."""
    def __init__(self, optimizer, overflow=True):
        self.optimizer, self.overflow, self.updates = optimizer, overflow, 0

    def scale(self, loss):
        return loss * (float("inf") if self.overflow else 1.)

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        if not self.overflow:
            optimizer.step()

    def update(self):
        self.overflow = False
        self.updates += 1

    def get_scale(self):
        return 1.


class PacketLoraCommandTests(unittest.TestCase):
    def runtime(self):
        spec = importlib.util.find_spec("scale_interface.packet_lora_cli")
        self.assertIsNotNone(spec, "portable matched training/evaluation CLI is missing")
        return __import__(spec.name, fromlist=["*"])

    def test_fixed_schedule_order_and_clean_group_boundaries(self):
        cli = self.runtime()
        order = cli.ordered_indices(35321, 35330)
        expected = list(range(70651))
        random.Random(20261102).shuffle(expected)
        self.assertEqual(order, expected)
        self.assertEqual(list(cli.update_groups(70640, 70651)), [(70640, 70651)])
        self.assertEqual(cli.learning_rate(0), 2e-5 / 133)
        self.assertEqual(cli.learning_rate(132), 2e-5)
        self.assertEqual(cli.learning_rate(4415), 0.)
        for position, step in ((0, 0), (3504, 219), (70651, 4416)):
            cli.validate_progress(position, step)
        for position, step in ((3, 1), (3504, 218), (70650, 4416)):
            with self.assertRaises(ValueError):
                cli.validate_progress(position, step)

    def test_labels_are_separate_and_bound_to_the_queried_source_pair(self):
        cli = self.runtime()
        row = dict(qid="r1", scene_id="scene0001_00", prompt="Which <OBJ000> or <OBJ002>?",
                   assigned_ids=[2, 0, 1])
        label = dict(qid="r1", scene_id="scene0001_00", answer="<OBJ002>")
        self.assertEqual(cli.target_source(row, label), 2)
        public = dict(question_id="r1", scene_id="scene0001_00", answers=["<OBJ002>"])
        self.assertEqual(cli.target_source(row, public), 2)
        for answers in ([], ["<OBJ000>", "<OBJ002>"]):
            with self.assertRaises(ValueError):
                cli.target_source(row, {**public, "answers": answers})
        for change in ({"answer": "<OBJ001>"}, {"scene_id": "scene0002_00"}, {"qid": "other"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                cli.target_source(row, {**label, **change})
        with self.assertRaises(ValueError):
            cli.target_source({**row, "answer": "<OBJ002>"}, label)

    def test_split_guard_uses_scene_families_and_question_ids(self):
        cli = self.runtime()
        splits = {"train": [dict(qid="train", scene_id="scene0001_00")],
                  "validation": [dict(qid="val", scene_id="scene0002_00")]}
        cli.check_splits(splits, ["scene0003_00"])
        for changed, held_out in (({**splits, "validation": [dict(qid="val", scene_id="scene0001_01")]}, []),
                                  ({**splits, "validation": [dict(qid="train", scene_id="scene0002_00")]}, []),
                                  (splits, ["scene0001_02"])):
            with self.assertRaisesRegex(ValueError, "overlap"):
                cli.check_splits(changed, held_out)

    def test_pin_checks_every_source_and_never_replaces_a_config(self):
        cli = self.runtime()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.bin"
            source.write_bytes(b"checkpoint")
            model = root / "model"
            model.mkdir()
            (model / "weights.safetensors").write_bytes(b"model")
            deployment = root / "deployment.pt"
            torch.save(dict(architecture={"adapter_kind": "identity_no_radius"}, state={"scalestitch": {}}), deployment)
            host = root / "host.json"
            host.write_text(json.dumps({"host": {"host_checkpoint": str(source), "deployment": str(deployment),
                "model": {"llama_model_path": str(model)}}}), encoding="utf-8")
            spec = {key: str(source) for key in ("features", "radii", "attributes", "height_measurement_source")}
            spec.update({prop: {key: str(source) for key in ("questions", "labels", "cohort")}
                         for prop in ("radius", "height")})
            config = root / "template.json"
            config.write_text(json.dumps(dict(host_config=str(host), splits={"train": spec, "validation": spec},
                held_out_scenes=str(source), expected_counts=cli.COUNTS, input_sha256={})), encoding="utf-8")
            pinned = root / "pinned.json"
            cli.pin_config(config, pinned)
            value = json.loads(pinned.read_text())
            files = cli.required_inputs(value)
            self.assertEqual(value["input_sha256"][str(source.resolve())], hashlib.sha256(b"checkpoint").hexdigest())
            self.assertEqual(len(files), 4)
            cli.verify_hashes(value["input_sha256"], files)
            with self.assertRaises(FileExistsError):
                cli.pin_config(config, pinned)
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                cli.verify_hashes(value["input_sha256"], files)
            with self.assertRaisesRegex(ValueError, "pin"):
                cli.verify_hashes({}, files)

    def test_accumulated_real_lora_loss_matches_one_mean_loss_update(self):
        cli = self.runtime()
        torch.manual_seed(11)
        core = DifferentiableHost()
        selected = pl.unfreeze_lora_only(core, expected_numel=16)
        prefix, native, _ = example(core)
        target = pl.short_answer_target(core, native)
        before = frozen_fingerprint(core)
        initial = {name: parameter.detach().clone() for name, parameter in core.named_parameters()}
        optimizer = torch.optim.AdamW(selected.parameters, lr=2e-5, weight_decay=.01)
        scaler = OverflowOnce(optimizer, overflow=False)
        result = cli.train_group(core, selected, optimizer, scaler, [1, 2],
            lambda factor: pl.next_token_loss(core, prefix, target, selected) * factor)
        actual = [parameter.detach().clone() for parameter in selected.parameters]
        with torch.no_grad():
            for name, parameter in core.named_parameters():
                parameter.copy_(initial[name])
        expected_optimizer = torch.optim.AdamW(selected.parameters, lr=2e-5, weight_decay=.01)
        loss = pl.next_token_loss(core, prefix, target, selected) * 1.5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selected.parameters, 1., error_if_nonfinite=True)
        expected_optimizer.step()
        expected_optimizer.zero_grad(set_to_none=True)
        self.assertTrue(all(torch.equal(got, parameter) for got, parameter in zip(actual, selected.parameters)))
        self.assertEqual(before, frozen_fingerprint(core))
        self.assertEqual((result["group_size"], result["overflow_retries"]), (2, 0))

    def test_overflow_retries_the_same_group_and_rng_without_advancing(self):
        cli = self.runtime()
        random.seed(9); np.random.seed(9); torch.manual_seed(9)
        core = DifferentiableHost()
        selected = pl.unfreeze_lora_only(core, expected_numel=16)
        prefix, native, _ = example(core)
        target = pl.short_answer_target(core, native)
        optimizer = torch.optim.AdamW(selected.parameters, lr=2e-5, weight_decay=.01)
        seen = []
        def loss_for(index):
            values = (index, random.random(), float(np.random.random()), float(torch.rand(())))
            seen.append(values)
            return pl.next_token_loss(core, prefix, target, selected) * (.5 + values[1])
        result = cli.train_group(core, selected, optimizer, OverflowOnce(optimizer), [0, 1], loss_for)
        self.assertEqual(seen[:2], seen[2:])
        self.assertEqual(result["overflow_retries"], 1)
        self.assertTrue(all(parameter.grad is None for parameter in selected.parameters))
        self.assertTrue(all(int(state["step"]) == 1 for state in optimizer.state.values()))

    def test_validation_scorer_rejects_format_and_mapping_failures(self):
        cli = self.runtime()
        valid = dict(raw_prediction="<OBJ002></s>", prediction="<OBJ000>", invalid_object_id=False)
        self.assertEqual(cli.native_hit(valid, 0, [2, 1, 0], "</s>"), 1)
        for change in ({"raw_prediction": "Object <OBJ002>"}, {"invalid_object_id": True}):
            self.assertEqual(cli.native_hit({**valid, **change}, 0, [2, 1, 0], "</s>"), 0)
        with self.assertRaisesRegex(ValueError, "mapping"):
            cli.native_hit({**valid, "prediction": "<OBJ001>"}, 0, [2, 1, 0], "</s>")

    def test_actual_loaded_checkpoint_is_bound_and_resume_is_clean(self):
        cli = self.runtime()
        torch.manual_seed(14)
        core = DifferentiableHost()
        selected = pl.unfreeze_lora_only(core, expected_numel=16)
        optimizer = torch.optim.AdamW(selected.parameters, lr=2e-5)
        with tempfile.TemporaryDirectory() as folder:
            pl.save_checkpoint(folder, core, selected, optimizer, step=219, row_position=3504,
                identity=identity(), source_checkpoint=Path(folder) / "source.pt", pending_microbatches=0)
            Path(folder, "last.pt").rename(Path(folder, "previous.pt"))
            loaded = pl.load_checkpoint(folder, core, selected, optimizer, identity=identity())
            proof = cli.checkpoint_proof(folder, loaded, "pilot")
            self.assertEqual(proof["path"], str(Path(folder, "previous.pt").resolve()))
            self.assertEqual(proof["row_position"], 3504)
            with self.assertRaises(ValueError):
                cli.checkpoint_proof(folder, {**loaded, "row_position": 3505}, "pilot")
            with self.assertRaises(ValueError):
                cli.checkpoint_proof(folder, {**loaded, "path": str(Path(folder) / "source.pt")}, "pilot")

    def test_full_validation_matrix_resumes_without_duplicate_answers(self):
        cli = self.runtime()
        from scale_interface.natural_qa import NaturalQACondition
        records = [("radius", 0, 0), ("height", 0, 0)]
        core = types.SimpleNamespace(end_sym="</s>")
        stopped, calls = [], []
        def inputs(record, arm):
            prop, index, target = record
            cpu = {"assigned_ids": torch.tensor([[2, 1, 0]])}
            condition = NaturalQACondition(representation=arm, measurement=prop if arm != "absent" else "radius")
            return prop + "0", "scene0001_00", "question", cpu, cpu, condition, None, target, prop
        def run_one(core, batch, question, packet, condition, **kwargs):
            self.assertEqual(kwargs["repetition_penalty"], 1.)
            self.assertEqual(kwargs["max_new_tokens"], 32)
            calls.append(condition.name)
            if len(calls) == 1:
                stopped.append("pause")
            return None, dict(raw_prediction="<OBJ002></s>", prediction="<OBJ000>", invalid_object_id=False)
        with tempfile.TemporaryDirectory() as folder:
            proof = dict(path="checkpoint.pt", sha256="a" * 64, row_position=3504, step=219)
            first = cli.evaluate(core, records, inputs, Path(folder), identity(), proof,
                                 stopped=stopped, run_one=run_one)
            self.assertEqual((first["status"], first["completed"]), ("paused", 1))
            stopped.clear()
            result = cli.evaluate(core, records, inputs, Path(folder), identity(), proof,
                                  stopped=stopped, run_one=run_one)
            self.assertEqual(len(calls), 4)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["counts"], {"absent": {"n": 2, "correct": 2}, "lexical": {"n": 2, "correct": 2}})
            terminal = Path(folder) / "SUCCESS.json"
            changed = {**result, "counts": {**result["counts"], "lexical": {"n": 2, "correct": 0}}}
            terminal.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed validation"):
                cli.evaluate(core, records, inputs, Path(folder), identity(), proof, run_one=run_one)
            self.assertEqual(len(calls), 4)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                cli.evaluate(core, records, inputs, Path(folder), identity(), {**proof, "sha256": "b" * 64}, run_one=run_one)

    def test_nonowner_cannot_take_the_same_arm_lock(self):
        cli = self.runtime()
        with tempfile.TemporaryDirectory() as folder:
            with cli.arm_lock(Path(folder)):
                with self.assertRaises((BlockingIOError, OSError)):
                    with cli.arm_lock(Path(folder)):
                        self.fail("second owner acquired the arm")

    def test_tampered_partial_validation_stops_before_another_forward(self):
        cli = self.runtime()
        from scale_interface.natural_qa import NaturalQACondition
        for tamper in ("score", "qid", "raw", "assignment"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as folder:
                calls, stopped = [], []
                core = types.SimpleNamespace(end_sym="</s>")
                def inputs(record, arm):
                    condition = NaturalQACondition(representation=arm)
                    cpu = {"assigned_ids": torch.tensor([[2, 1, 0]])}
                    return "r0", "scene0001_00", "question", cpu, cpu, condition, None, 0, "radius"
                def run_one(*args, **kwargs):
                    calls.append(1); stopped.append("pause")
                    return None, dict(raw_prediction="<OBJ002></s>", prediction="<OBJ000>", invalid_object_id=False)
                proof = dict(path="checkpoint.pt", sha256="a" * 64, row_position=3504, step=219)
                cli.evaluate(core, [("radius", 0, 0)], inputs, Path(folder), identity(), proof,
                             stopped=stopped, run_one=run_one)
                database = sqlite3.connect(Path(folder) / "predictions.sqlite3")
                payload = json.loads(database.execute("SELECT payload FROM answers").fetchone()[0])
                if tamper == "score":
                    database.execute("UPDATE answers SET correct=0")
                else:
                    if tamper == "qid": payload["qid"] = "foreign"
                    if tamper == "raw": payload["raw_prediction"] = "answer <OBJ002>"
                    if tamper == "assignment": payload["input_sha256"] = "b" * 64
                    database.execute("UPDATE answers SET payload=?", (json.dumps(payload),))
                database.commit(); database.close(); stopped.clear()
                with self.assertRaisesRegex(ValueError, "stored validation"):
                    cli.evaluate(core, [("radius", 0, 0)], inputs, Path(folder), identity(), proof, run_one=run_one)
                self.assertEqual(len(calls), 1)

    def test_resume_before_first_update_is_safe_only_without_any_update_log(self):
        cli = self.runtime()
        self.assertEqual(cli.resume_action("train", True, False, False), "fresh")
        self.assertEqual(cli.resume_action("train", True, True, True), "load")
        for action, checkpoint, updates in (("evaluate", False, False), ("train", False, True), ("smoke", True, False)):
            with self.assertRaises(ValueError):
                cli.resume_action(action, True, checkpoint, updates)

    def test_completed_stage_is_verified_without_resaving_accepted_checkpoint(self):
        cli = self.runtime()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            proof = dict(path="last.pt", sha256="a" * 64, row_position=3504, step=219)
            value = dict(status="complete", arm="lexical", protocol_sha256=identity()["protocol_hash"],
                         source_hashes=identity()["source_hashes"], checkpoint_sha256=proof["sha256"],
                         row_position=3504, successful_updates=219)
            (output / "SUCCESS_pilot.json").write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(cli.completed_stage(output, "pilot", identity(), proof), value)
            with self.assertRaises(ValueError):
                cli.completed_stage(output, "pilot", identity(), {**proof, "sha256": "b" * 64})

    def test_pilot_checkpoint_bits_remain_available_after_epoch_replaces_last(self):
        cli = self.runtime()
        torch.manual_seed(14)
        core = DifferentiableHost()
        selected = pl.unfreeze_lora_only(core, expected_numel=16)
        optimizer = torch.optim.AdamW(selected.parameters, lr=2e-5)
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            path = pl.save_checkpoint(work, core, selected, optimizer, step=219, row_position=3504,
                identity=identity(), source_checkpoint=work / "source.pt", pending_microbatches=0)
            proof = cli.seal_stage(work, "pilot", path, 3504, 219, identity())
            self.assertEqual(proof["sha256"], hashlib.sha256(Path(path).read_bytes()).hexdigest())
            pl.save_checkpoint(work, core, selected, optimizer, step=4416, row_position=70651,
                identity=identity(), source_checkpoint=work / "source.pt", pending_microbatches=0)
            stage = work / "stages/pilot"
            loaded = pl.load_checkpoint(stage, core, selected, optimizer, identity=identity())
            self.assertEqual(loaded["row_position"], 3504)
            self.assertEqual(cli.checkpoint_proof(stage, loaded, "pilot"), proof)
            self.assertEqual(cli.seal_stage(work, "pilot", stage / "last.pt", 3504, 219, identity()), proof)

    def test_exported_host_strict_roundtrip_matches_manual_lora_copy(self):
        cli = self.runtime()
        torch.manual_seed(21)
        core = DifferentiableHost()
        selected = pl.unfreeze_lora_only(core, expected_numel=16)
        state = core.state_dict()
        for name in selected.names:
            state[name] = state[name].to(torch.bfloat16)
        source = dict(model=state, epoch=4)
        trained = {name: state[name].float() + .01 for name in selected.names}
        merged = cli.merge_lora_state(source, trained)
        self.assertEqual(merged["epoch"], 4)
        for name, value in state.items():
            self.assertTrue(torch.equal(value, source["model"][name]))
            if name not in trained:
                self.assertIs(value, merged["model"][name])
                self.assertEqual(value.dtype, merged["model"][name].dtype)
        restored, reference = DifferentiableHost(), DifferentiableHost()
        restored.load_state_dict(merged["model"], strict=True)
        reference.load_state_dict(state, strict=True)
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if name in trained:
                    parameter.copy_(trained[name])
        inputs = torch.ones(1, 4, 4)
        options = dict(inputs_embeds=inputs, attention_mask=torch.ones(1, 4), use_cache=False,
                       return_dict=True, logits_to_keep=2)
        self.assertTrue(torch.equal(restored.llama_model(**options).logits, reference.llama_model(**options).logits))

    def test_export_rejects_unmatched_or_nonfinite_trained_lora(self):
        cli = self.runtime()
        core = DifferentiableHost()
        selected = pl.unfreeze_lora_only(core, expected_numel=16)
        state = core.state_dict()
        trained = {name: state[name].clone() for name in selected.names}
        bad_sets = [dict(list(trained.items())[1:]), {**trained, "unknown.lora_A.default.weight": torch.ones(2, 4)}]
        for kind in ("nonfinite", "dtype", "shape"):
            bad = {name: value.clone() for name, value in trained.items()}
            first = selected.names[0]
            if kind == "nonfinite": bad[first].fill_(float("nan"))
            if kind == "dtype": bad[first] = bad[first].half()
            if kind == "shape": bad[first] = torch.ones(1)
            bad_sets.append(bad)
        for bad in bad_sets:
            with self.assertRaisesRegex(ValueError, "LoRA"):
                cli.merge_lora_state({"model": state}, bad)

    def test_matched_protocol_rejects_any_radius_conditioned_deployment(self):
        cli = self.runtime()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "deployment.pt"
            valid = dict(architecture={"adapter_kind": "identity_no_radius"}, state={"scalestitch": {}})
            torch.save(valid, path)
            self.assertEqual(cli.validate_deployment(path), hashlib.sha256(path.read_bytes()).hexdigest())
            for bundle in (dict(valid, architecture={"adapter_kind": "radius_linear_shift"}),
                           dict(valid, state={"scalestitch": {"shift": torch.zeros(2)}})):
                torch.save(bundle, path)
                with self.assertRaisesRegex(ValueError, "identity_no_radius"):
                    cli.validate_deployment(path)

    def small_split(self, root):
        scene = "scene0001_00"
        attributes = root / "attributes.pt"
        torch.save({scene: {"locs": torch.tensor([[0., 0., 0., 1., 1., value] for value in (1., 2., 3.)]),
                            "obj_ids": torch.arange(3)}}, attributes)
        torch.save({f"{scene}_{index}": torch.ones(1280) for index in range(3)}, root / "features.pt")
        torch.save({"records": {f"{scene}_{index}": dict(scene_id=scene, object_id=index, rho=.1 * (index + 1), radius_valid=True)
                                for index in range(3)}}, root / "radii.pt")
        declaration = dict(measurement="full_object_axis_aligned_height", input_id=hashlib.sha256(attributes.read_bytes()).hexdigest(),
                           source_reference="synthetic aligned geometry", verified_scenes={scene: True})
        (root / "height_source.json").write_text(json.dumps(declaration), encoding="utf-8")
        spec = dict(features=str(root / "features.pt"), radii=str(root / "radii.pt"), attributes=str(attributes),
                    height_measurement_source=str(root / "height_source.json"))
        for prop in ("radius", "height"):
            row = dict(qid=prop + "0", scene_id=scene, prompt="Which <OBJ000> or <OBJ002>?", assigned_ids=[2, 0, 1])
            values = dict(questions=[row], labels=[dict(qid=row["qid"], scene_id=scene, answer="<OBJ002>")], cohort=[row["qid"]])
            spec[prop] = {}
            for key, value in values.items():
                path = root / (prop + "_" + key + ".json")
                path.write_text(json.dumps(value), encoding="utf-8")
                spec[prop][key] = str(path)
        return spec

    def test_split_rejects_bad_saved_assignment_before_loading_a_host(self):
        cli = self.runtime()
        with tempfile.TemporaryDirectory() as folder:
            spec = self.small_split(Path(folder))
            counts = dict(radius=1, height=1)
            with mock.patch.object(cli.lora, "load_packet_lora_host", side_effect=AssertionError("GPU host loaded")):
                self.assertEqual(len(cli.read_split(spec, counts)[1]), 2)
                path = Path(spec["radius"]["questions"])
                rows = json.loads(path.read_text())
                rows[0]["assigned_ids"] = [0, 0, 2]
                path.write_text(json.dumps(rows), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "permutation"):
                    cli.read_split(spec, counts)

    def test_split_rejects_missing_or_masked_candidate_before_loading_a_host(self):
        cli = self.runtime()
        for failure in ("missing", "masked"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                spec = self.small_split(Path(folder))
                path = Path(spec["radius"]["questions"])
                rows = json.loads(path.read_text())
                if failure == "missing":
                    rows[0]["prompt"] = "Which <OBJ999> or <OBJ002>?"
                else:
                    rows[0]["assigned_ids"] = [0, 1, 2]
                path.write_text(json.dumps(rows), encoding="utf-8")
                dataset_type = cli.SceneDataset
                def dataset(**kwargs):
                    value = dataset_type(**kwargs)
                    if failure == "masked":
                        value.scene_masks["scene0001_00"][0] = 0
                    return value
                with mock.patch.object(cli.lora, "load_packet_lora_host", side_effect=AssertionError("GPU host loaded")):
                    with mock.patch.object(cli, "SceneDataset", side_effect=dataset):
                        with self.assertRaisesRegex(ValueError, "candidate|queried"):
                            cli.read_split(spec, dict(radius=1, height=1))


if __name__ == "__main__":
    unittest.main()
