"""Portable baseline: fixed inputs, full denominators and interruption recovery."""
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import torch


MODULE = "experiments.scale_interface.text_baseline_runtime"
QUESTION = ("Between <OBJ000> and <OBJ001>, which object has the larger measured metric radius? "
            "Answer with exactly one object identifier and no other text.")


class PlainTokenizer:
    chat_template = "base template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "USER " + messages[0]["content"] + " ASSISTANT "

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": list(text.encode("utf-8"))}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return "".join("<|eot_id|>" if i == 999 else chr(i) for i in ids)


class TinyModel:
    """Only the external LLM is replaced; preparation, storage and scoring are real."""
    config = types.SimpleNamespace(max_position_embeddings=10000)

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def generate(self, *, input_ids, attention_mask, generation_config, logits_processor):
        if self.fail_after is not None and self.calls == self.fail_after:
            raise RuntimeError("synthetic interrupted forward")
        self.calls += 1
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        assert generation_config.eos_token_id == [999]
        assert generation_config.max_new_tokens == 32 and not generation_config.do_sample
        # Ordinary text wrapper is valid only for the separate sensitivity score.
        return torch.cat((input_ids, torch.tensor([[*b"Object C", 999]])), dim=1)


class TextBaselineRuntimeTests(unittest.TestCase):
    def runtime(self):
        self.assertIsNotNone(importlib.util.find_spec(MODULE), "portable text baseline is missing")
        from experiments.scale_interface import text_baseline_runtime
        return text_baseline_runtime

    def pair(self):
        return dict(qid="q", scene_id="scene0000_00", task="radius", original_prompt=QUESTION,
            visible_question=QUESTION.replace("<OBJ000>", "<OBJ002>"),
            batch=dict(scene_mask=torch.tensor([[True] * 3]), assigned_ids=torch.tensor([[2, 0, 1]]),
                scene_radius=torch.tensor([[.1, .2, .3]]), radius_valid=torch.tensor([[True] * 3])),
            base_tokenizer=PlainTokenizer(), adapted_tokenizer=PlainTokenizer(),
            native_object_token_ids={999999})

    def fixture(self, folder):
        root = Path(folder)
        scene = "scene0000_00"
        torch.save({f"{scene}_{i}": torch.tensor([1., 2., 3., 4.]) for i in range(3)}, root / "features.pt")
        torch.save({scene: dict(locs=torch.ones(3, 6), obj_ids=torch.arange(3))}, root / "attributes.pt")
        torch.save({"records": {f"{scene}_{i}": dict(scene_id=scene, object_id=str(i), rho=.1 * (i + 1),
            radius_valid=True) for i in range(3)}}, root / "radii.pt")
        (root / "rows.json").write_text(json.dumps([dict(qid="q", scene_id=scene, prompt=QUESTION,
            assigned_ids=[2, 0, 1], caption="must not reach model inputs", answers=["hidden"])]))
        (root / "cohort.json").write_text('["q"]')
        return dict(task="radius", seed=20260911, device="cpu", cohort=str(root / "cohort.json"),
            data=dict(features=str(root / "features.pt"), attributes=str(root / "attributes.pt"),
                radii=str(root / "radii.pt"), rows=str(root / "rows.json"), feature_dim=4))

    def test_pair_keeps_external_swapping_donor_and_ordinary_question_mapping(self):
        r = self.runtime()
        pair = self.pair()
        pair["visible_question"] = QUESTION.replace("<OBJ001>", "<OBJ002>").replace("<OBJ000>", "<OBJ001>")
        rows = r.prepare_pair(**pair)
        self.assertEqual([x["condition"] for x in rows], ["absent", "full_true", "full_constant",
            "full_swapped", "pair_true", "pair_constant", "pair_swapped"])
        focused = rows[-1]
        self.assertIn("Between Object B and Object C", focused["user_text"])
        self.assertIn("Object B = 30 cm", focused["user_text"])
        self.assertIn("Object C = 10 cm", focused["user_text"])
        self.assertNotIn("Object A =", focused["user_text"])
        self.assertEqual([x["value_source_row"] for x in focused["packet_metadata"]["records"]], [2, 0])
        self.assertNotIn("<OBJ", focused["rendered_text"])

    def test_decoder_and_separate_scores_keep_native_tokens_and_non_nested_rules(self):
        r = self.runtime()
        options = dict(label_to_source_row={"A": 2, "B": 0, "C": 1}, candidate_labels=["B", "C"])
        self.assertEqual(r.decode_new_tokens(PlainTokenizer(), [8, 9, 67, 999], [8, 9], [999])["raw_output"], "C")
        self.assertEqual(r.decode_new_tokens(PlainTokenizer(), [8, 67, 999, 66], [8], [999])["raw_output"], "C<|eot_id|>B")
        for text, expected in [("C.", (True, False)), ("Object C", (False, True)),
                               ("c", (False, True)), ("Object C.", (False, False)),
                               ("C because", (False, False))]:
            score = r.score_record(dict(options, raw_output=text), 1)
            self.assertEqual((score["primary"]["correct"], score["whole_label_sensitivity"]["correct"]), expected)
        self.assertFalse(r.score_record(dict(options, raw_output="C", error="failed"), 1)["primary"]["correct"])

    def test_durable_resume_keeps_rows_and_rejects_new_identity_or_changed_record(self):
        r = self.runtime()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.sqlite3"
            with r.DurableStore(path, {"source": "fixed"}) as store:
                store.add("q", {"answer": "C"})
                store.add_failure("pending", {"error": "interruption"})
            with r.DurableStore(path, {"source": "fixed"}) as store:
                self.assertEqual(store.count(), 1)
                self.assertEqual(store.failure_count(), 1)
                store.add("q", {"answer": "C"})
                with self.assertRaisesRegex(ValueError, "record"):
                    store.add("q", {"answer": "B"})
            with self.assertRaisesRegex(ValueError, "identity"):
                r.DurableStore(path, {"source": "changed"})

    def test_cohort_selects_saved_order_and_never_passes_reference_fields(self):
        r = self.runtime()
        with tempfile.TemporaryDirectory() as folder:
            config = self.fixture(folder)
            rows = r.source_rows(config)
            self.assertEqual(list(rows[0]), ["qid", "scene_id", "prompt", "assigned_ids"])
            self.assertEqual(rows[0]["qid"], "q")
            Path(config["cohort"]).write_text('["q","q"]')
            with self.assertRaisesRegex(ValueError, "cohort|duplicate"):
                r.source_rows(config)

    def test_interrupted_full_generation_resumes_without_dropping_any_condition(self):
        r = self.runtime()
        with tempfile.TemporaryDirectory() as folder:
            config = self.fixture(folder)
            output = Path(folder) / "run"
            identity = {"source": "fixed"}
            tokenizers = (PlainTokenizer(), PlainTokenizer(), {999999}, {"common_eos_token_ids": [999]})
            prepared = r.prepare_inputs(config, output, identity, tokenizers)
            self.assertEqual(prepared["inputs"], 7)
            with self.assertRaisesRegex(RuntimeError, "inference|technical"):
                r.generate(config, output, identity, PlainTokenizer(), TinyModel(fail_after=2), [999])
            self.assertFalse((output / "SUCCESS.json").exists())
            model = TinyModel()
            complete = r.generate(config, output, identity, PlainTokenizer(), model, [999])
            self.assertEqual(model.calls, 5)
            self.assertEqual(complete["outputs"], 7)
            references = Path(folder) / "references.json"
            references.write_text(json.dumps([dict(question_id="q", scene_id="scene0000_00", answers=["<OBJ001>"])]))
            report = r.score(output, identity, references)
            self.assertEqual(report["questions"], 1)
            self.assertEqual(set(report["conditions"]), set(r.CONDITIONS))
            for condition in report["conditions"].values():
                self.assertEqual(condition["N"], 1)
                self.assertEqual(condition["primary_correct"], 0)
                self.assertEqual(condition["whole_label_sensitivity_correct"], 1)
            self.assertEqual(set(report["scores"]), {"primary", "whole_label_sensitivity"})
            for endpoint in report["scores"].values():
                self.assertEqual([f["family_size"] for f in endpoint["families"].values()], [3, 3, 2])
                for family in endpoint["families"].values():
                    self.assertEqual(family["questions"], 1)
                    self.assertEqual(family["replicates"], 10000)
                    self.assertEqual(family["seed"], 20260911)
                    self.assertTrue(all(x["ci95"] is None for x in family["contrasts"].values()))

    def test_scoring_refuses_incomplete_outputs_and_config_rejects_unfixed_decoding(self):
        r = self.runtime()
        with tempfile.TemporaryDirectory() as folder:
            config = self.fixture(folder)
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(dict(config, schema_version="scalestitch_text_baseline/v1",
                model={"local_path": "model"}, repetition_penalty=3)))
            with self.assertRaisesRegex(ValueError, "unknown|decoding|configuration"):
                r.load_config(path)
            with self.assertRaises((FileNotFoundError, ValueError)):
                r.score(Path(folder), {"source": "fixed"}, Path(folder) / "refs.json")

    def cli(self):
        name = "experiments.scale_interface.text_baseline_cli"
        self.assertIsNotNone(importlib.util.find_spec(name), "portable text CLI is missing")
        from experiments.scale_interface import text_baseline_cli
        return text_baseline_cli

    def test_default_cli_is_inert(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(self.cli().main([]), 0)
        self.assertEqual(json.loads(stream.getvalue())["status"], "not_launched")

    def test_tokenizer_uses_source_generation_eos_list_not_single_tokenizer_eos(self):
        cli = self.cli()
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as folder:
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
                {"[UNK]": 0, "[EOS]": 1, "[END]": 2}, unk_token="[UNK]")),
                unk_token="[UNK]", eos_token="[EOS]")
            tokenizer.chat_template = "{{ messages[0]['content'] }} assistant"
            tokenizer.save_pretrained(folder)
            Path(folder, "generation_config.json").write_text('{"eos_token_id":[1,2]}')
            base, comparison, native, stamp = cli.load_tokenizers(folder)
            self.assertEqual(base.eos_token_id, 1)
            self.assertEqual(stamp["common_eos_token_ids"], [1, 2])
            self.assertEqual(len(native), 150)
            self.assertEqual(len(comparison), len(base) + 150)

    def test_identity_binds_real_source_model_and_cohort_hashes_across_invocations(self):
        cli = self.cli()
        from experiments.scale_interface.runtime_provenance import RuntimeProvenance
        with tempfile.TemporaryDirectory() as folder:
            config = self.fixture(folder)
            model = Path(folder) / "model"
            model.mkdir()
            (model / "weights.safetensors").write_bytes(b"synthetic model weights")
            (model / "generation_config.json").write_text('{"eos_token_id":[999]}')
            config.update(schema_version="scalestitch_text_baseline/v1", model={"local_path": str(model)})
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config))
            effective = self.runtime().load_config(path)

            def identity(number):
                output = Path(folder) / ("invocation" + str(number))
                output.mkdir()
                with RuntimeProvenance(output, "text-baseline-prepare", path, command=[], torch_module=torch) as provenance:
                    return cli.source_identity(effective, provenance, {"common_eos_token_ids": [999]})

            first = identity(1)
            self.assertEqual(first, identity(2))
            (model / "weights.safetensors").write_bytes(b"changed model weights")
            third = identity(3)
            self.assertNotEqual(first, third)
            Path(config["cohort"]).write_text('["different"]')
            self.assertNotEqual(third, identity(4))

    def test_scope_bootstrap_keeps_captures_together_and_weights_questions(self):
        name = "experiments.scale_interface.text_baseline_statistics"
        self.assertIsNotNone(importlib.util.find_spec(name), "paired text statistics are missing")
        from experiments.scale_interface import text_baseline_statistics as stats
        rows = [dict(qid="a", scene_id="scene0001_00", scores=dict(absent=0, full_true=1,
                    full_constant=0, full_swapped=0, pair_true=0, pair_constant=0, pair_swapped=1)),
                dict(qid="b", scene_id="scene0001_01", scores=dict(absent=1, full_true=0,
                    full_constant=0, full_swapped=0, pair_true=1, pair_constant=0, pair_swapped=0)),
                dict(qid="c", scene_id="scene0002_00", scores=dict(absent=0, full_true=0,
                    full_constant=1, full_swapped=1, pair_true=1, pair_constant=1, pair_swapped=0))]
        result = stats.bootstrap_interactions(rows, stats.FAMILIES["scope"]["weights"], family_size=2)
        self.assertEqual((result["questions"], result["scene_families"]), (3, 2))
        self.assertEqual(result["contrasts"]["true_scope_effect"]["delta"], 1 / 3)
        self.assertEqual(result["contrasts"]["true_scope_effect"]["ci_bonferroni"], [0., 1.])
        self.assertEqual(result["contrasts"]["assignment_interaction"]["delta"], 1 / 3)
        self.assertEqual(result["contrasts"]["assignment_interaction"]["ci_bonferroni"], [-.5, 2.])

    def test_output_lock_prevents_two_runners_for_one_durable_store(self):
        cli = self.cli()
        with tempfile.TemporaryDirectory() as folder:
            with cli.output_lock(folder):
                with self.assertRaisesRegex(RuntimeError, "another"):
                    with cli.output_lock(folder):
                        self.fail("a second runner acquired the active output")
            with cli.output_lock(folder):
                pass

    def test_output_matrix_requires_every_condition_and_exact_input_binding_before_seal(self):
        r = self.runtime()
        pair = self.pair()
        pair["visible_question"] = QUESTION.replace("<OBJ001>", "<OBJ002>").replace("<OBJ000>", "<OBJ001>")
        inputs = r.prepare_pair(**pair)
        outputs = [dict(row, arm="base", raw_output="C", error=None) for row in inputs]
        r.validate_outputs(outputs, inputs)
        broken = [outputs[:-1], outputs[:-1] + [outputs[0]],
                  [dict(outputs[0], input_ids_sha256="wrong"), *outputs[1:]],
                  [dict(outputs[0], arm="adapted"), *outputs[1:]]]
        for records in broken:
            with self.subTest(records=len(records)), self.assertRaisesRegex(ValueError, "matrix|binding|base"):
                r.validate_outputs(records, inputs)

    def test_generate_checks_prepared_cohort_before_loading_model(self):
        cli = self.cli()
        with tempfile.TemporaryDirectory() as folder:
            config = self.fixture(folder)
            model = Path(folder) / "model"
            model.mkdir()
            (model / "weights.safetensors").write_bytes(b"synthetic model weights")
            config.update(schema_version="scalestitch_text_baseline/v1", model={"local_path": str(model)})
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config))
            tokenizers = (PlainTokenizer(), PlainTokenizer(), {999999}, {"common_eos_token_ids": [999]})
            with mock.patch.object(cli, "load_tokenizers", return_value=tokenizers), \
                    mock.patch.object(cli, "load_model", side_effect=AssertionError("model loaded before prepared cohort")):
                with self.assertRaises(FileNotFoundError):
                    cli.main(["generate", "--config", str(path), "--output", str(Path(folder) / "run")])


if __name__ == "__main__":
    unittest.main()
