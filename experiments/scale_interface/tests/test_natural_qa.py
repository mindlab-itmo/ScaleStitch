"""CPU checks of streaming QA using the real host assembly and synthetic inputs."""
from dataclasses import replace
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import weakref

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface.data import SceneDataset
from scale_interface import natural_qa as qa
from scale_interface.scene_measurements import build_scene_radius_packet
from test_object_feature_controls import SmallHost, CaptureModel, TokenBatch


class AnswerGuard(dict):
    def __getitem__(self, key):
        if key in ("caption", "answers", "ref_captions", "answer"):
            raise AssertionError("answer field was read while preparing model inputs")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in ("caption", "answers", "ref_captions", "answer"):
            raise AssertionError("answer field was read while preparing model inputs")
        return super().get(key, default)


class CharacterTokenizer:
    pad_token_id = eos_token_id = 0

    def __call__(self, text, **kwargs):
        ids = [ord(c) for c in text]
        if kwargs.get("return_tensors") == "pt":
            values = torch.tensor([ids])
            return TokenBatch(input_ids=values, attention_mask=torch.ones_like(values))
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(chr(int(value)) for value in ids)

    def get_vocab(self):
        return {chr(i): i for i in range(256)}


class Recorder(CaptureModel):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(max_position_embeddings=4096, use_cache=True)
        self.calls = []
        self.answer = "generated!"

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[ord(c) for c in self.answer]])


class NaturalHost(SmallHost):
    def __init__(self):
        super().__init__()
        self.llama_model = Recorder()
        self.llama_tokenizer = CharacterTokenizer()

    def get_text_emb(self, text, device="cpu"):
        values = self.llama_tokenizer(text, return_tensors="pt").input_ids.to(device).float()
        return torch.stack([values + i for i in range(4)], dim=-1)


class CountingCodec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(4))
        self.calls = 0

    def forward(self, block):
        self.calls += 1
        return block.mean(0, keepdim=True).expand(2, -1) + self.offset


def dataset(*, radii=(.1, .2, .3), rows=2):
    features, records, attributes, questions = {}, {}, {}, []
    for index in range(rows):
        scene = f"scene{index:04d}_00"
        attributes[scene] = {"locs": torch.ones(len(radii), 6), "obj_ids": torch.arange(len(radii))}
        for obj, radius in enumerate(radii):
            key = f"{scene}_{obj}"
            features[key] = torch.tensor([1. + obj, 2., 3., 4.])
            records[key] = {"scene_id": scene, "object_id": str(obj), "rho": radius, "radius_valid": True}
        questions.append(AnswerGuard(qid=f"q{index}", scene_id=scene, prompt="Is the chair larger than the table?",
                                     assigned_ids=list(reversed(range(len(radii)))),
                                     caption="must never be read", ref_captions=["hidden"]))
    with mock.patch("scale_interface.data.torch.load", side_effect=[features, attributes, {"records": records}]):
        return SceneDataset(features="features.pt", radii="radii.pt", attributes="attrs.pt", rows=questions, feature_dim=4)


def principal_conditions():
    return tuple(qa.NaturalQACondition(rep, objects, mode)
                 for objects in qa.OBJECT_CONTROLS
                 for rep, mode in (("absent", "true"), ("lexical", "true"),
                                   ("lexical", "constant"), ("lexical", "swapped")))


class NaturalQATests(unittest.TestCase):
    def test_actual_prefix_preflight_reads_no_answers_and_never_generates(self):
        core, data = NaturalHost(), dataset()
        progress = []
        with mock.patch.object(data, "batches", side_effect=AssertionError("materialized dataset")), \
             mock.patch.object(data, "__getitem__", side_effect=AssertionError("label-reading access")):
            plan = qa.preflight_natural_qa(core, data, principal_conditions(), progress=progress.append)
        self.assertEqual([r.qid for r in plan.rows], ["q0", "q1"])
        self.assertEqual(plan.exclusions, ())
        self.assertEqual(core.llama_model.calls, [])
        self.assertEqual(len(progress), 2)
        self.assertEqual(plan.metadata()["selected_qids"], ["q0", "q1"])
        self.assertFalse(hasattr(core, "_rps_boundary_active"))
        self.assertEqual(core.object_feature_intervention, "full")

    def test_generator_is_streaming_greedy_and_matches_all_preflight_lengths(self):
        core, data = NaturalHost(), dataset()
        conditions = principal_conditions()
        plan = qa.preflight_natural_qa(core, data, conditions)
        iterator = qa.iter_natural_qa(core, data, plan)
        self.assertEqual(len(core.llama_model.calls), 0)
        first = next(iterator)
        self.assertEqual(len(core.llama_model.calls), 1)
        results = [first, *iterator]
        self.assertEqual(len(results), 2 * len(conditions))
        expected = {(row.qid, budget.condition): budget for row in plan.rows for budget in row.budgets}
        for row, call in zip(results, core.llama_model.calls):
            budget = expected[row["qid"], row["condition"]]
            self.assertEqual(row["prefix_tokens"], budget.prefix_tokens)
            self.assertEqual(row["packet_tokens"], budget.packet_tokens)
            self.assertEqual(row["prediction"], "generated")
            self.assertEqual(row["raw_prediction"], "generated!")
            self.assertFalse(row["invalid_object_id"])
            self.assertEqual(call["inputs_embeds"].shape[1], budget.prefix_tokens)
            self.assertTrue(torch.all(call["attention_mask"] == 1))
            self.assertEqual((call["max_new_tokens"], call["num_beams"], call["do_sample"]), (32, 1, False))
            self.assertNotIn("answers", call)
            self.assertNotIn("labels", call)

    def test_radius_ineligible_scene_is_excluded_from_absent_too(self):
        core, data = NaturalHost(), dataset()
        data.scene_radii["scene0001_00"][0] = .001
        plan = qa.preflight_natural_qa(core, data, principal_conditions())
        self.assertEqual([r.qid for r in plan.rows], ["q0"])
        self.assertEqual({e.qid for e in plan.exclusions}, {"q1"})
        self.assertEqual(len(plan.exclusions), len(principal_conditions()))
        self.assertTrue(all("rounds to zero" in e.reason for e in plan.exclusions))
        self.assertEqual({r["qid"] for r in qa.iter_natural_qa(core, data, plan)}, {"q0"})

    def test_one_failing_condition_excludes_row_from_every_condition(self):
        core, data = NaturalHost(), dataset(radii=(.2,), rows=1)
        conditions = (qa.NaturalQACondition("absent"), qa.NaturalQACondition(object_intervention="permute_object_slots"))
        plan = qa.preflight_natural_qa(core, data, conditions)
        self.assertEqual(plan.rows, ())
        self.assertEqual(len(plan.exclusions), 1)
        self.assertIn("two visible objects", plan.exclusions[0].reason)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            list(qa.iter_natural_qa(core, data, plan))

    def test_whole_prefix_budget_includes_generation_and_fixed_embeddings(self):
        core, data = NaturalHost(), dataset(rows=1)
        core.p_0_embed = torch.ones(33, 4)
        conditions = (qa.NaturalQACondition(),)
        initial = qa.preflight_natural_qa(core, data, conditions)
        exact = initial.rows[0].budgets[0].prefix_tokens + 32
        accepted = qa.preflight_natural_qa(core, data, conditions, maximum_length=exact)
        rejected = qa.preflight_natural_qa(core, data, conditions, maximum_length=exact - 1)
        self.assertEqual(len(accepted.rows), 1)
        self.assertEqual(rejected.rows, ())
        self.assertIn("assembled prefix", rejected.exclusions[0].reason)
        self.assertEqual(core.llama_model.calls, [])

    def test_host_context_limit_is_checked_even_with_absent_packet(self):
        core, data = NaturalHost(), dataset(rows=1)
        core.llama_model.config.max_position_embeddings = 40
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition("absent")])
        self.assertEqual(plan.maximum_length, 40)
        self.assertEqual(plan.rows, ())
        self.assertIn("exceeds budget 40", plan.exclusions[0].reason)

    def test_units_and_record_order_are_explicit_and_use_every_object(self):
        core, data = NaturalHost(), dataset(rows=1)
        conditions = [qa.NaturalQACondition(unit=unit, reverse=reverse)
                      for unit in ("cm", "mm", "m") for reverse in (False, True)]
        plan = qa.preflight_natural_qa(core, data, conditions)
        rows = list(qa.iter_natural_qa(core, data, plan))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["numerical_blocks"] == 3 for row in rows))
        _, _, _, batch = qa._cpu_input(data, 0)
        forward = qa._packet(batch, conditions[0], 100)
        reverse = qa._packet(batch, conditions[1], 100)
        self.assertEqual(reverse.records, forward.records[::-1])
        self.assertEqual([record.radius_cm for record in forward.records], [30, 20, 10])

    def test_every_numerical_block_is_compressed(self):
        core, codec = NaturalHost(), CountingCodec()
        packet = build_scene_radius_packet(assigned_rows=[2, 0, 1], scene_mask=[1, 1, 1],
                                           radius_m=[.1, .2, .3], radius_valid=[True] * 3)
        lexical = qa.encode_scene_packet(core, packet, "lexical")
        trained = qa.encode_scene_packet(core, packet, "trained", codec=codec)
        self.assertEqual(codec.calls, 3)
        numeric_lengths = sum(len(packet.parts[index]) for index in packet.numerical_block_indices)
        self.assertEqual(len(trained), len(lexical) - numeric_lengths + 3 * 2)
        trim = qa.encode_scene_packet(core, packet, "trim")
        self.assertEqual(len(trim), len(lexical) - 3)

    def test_all_compression_conditions_preflight_and_generate(self):
        core, data, codec, random_codec = NaturalHost(), dataset(rows=1), CountingCodec(), CountingCodec()
        conditions = [qa.NaturalQACondition(representation=rep) for rep in ("trim", "trained", "random", "mean")]
        plan = qa.preflight_natural_qa(core, data, conditions, codec=codec, random_codec=random_codec)
        rows = list(qa.iter_natural_qa(core, data, plan, codec=codec, random_codec=random_codec))
        self.assertEqual(len(rows), 4)
        self.assertEqual((codec.calls, random_codec.calls), (6, 6))

    def test_modified_inputs_are_rejected_before_first_generation(self):
        for change in ("features", "radius", "question", "assignment"):
            with self.subTest(change=change):
                core, data = NaturalHost(), dataset(rows=1)
                plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition()])
                if change == "features":
                    data.scene_feats["scene0000_00"][0, 0] += 1
                elif change == "radius":
                    data.scene_radii["scene0000_00"][0] += .1
                elif change == "question":
                    data.anno[0]["prompt"] += " changed"
                else:
                    data.anno[0]["assigned_ids"] = [0, 1, 2]
                with self.assertRaisesRegex(ValueError, "dataset differs"):
                    list(qa.iter_natural_qa(core, data, plan))
                self.assertEqual(core.llama_model.calls, [])

    def test_reference_change_is_neither_read_nor_an_input_change(self):
        core, data = NaturalHost(), dataset(rows=1)
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition()])
        data.anno[0]["caption"] = "changed reference"
        self.assertEqual(len(list(qa.iter_natural_qa(core, data, plan))), 1)

    def test_changed_host_tokenizer_or_codec_is_rejected(self):
        for change in ("host", "tokenizer", "codec"):
            with self.subTest(change=change):
                core, data, codec = NaturalHost(), dataset(rows=1), CountingCodec()
                plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition("trained")], codec=codec)
                with torch.no_grad():
                    if change == "host":
                        core.object_proj.weight.add_(1)
                    elif change == "codec":
                        codec.offset.add_(1)
                    else:
                        core.llama_tokenizer = CharacterTokenizer()
                with self.assertRaisesRegex(ValueError, "differs from the completed preflight"):
                    list(qa.iter_natural_qa(core, data, plan, codec=codec))
                self.assertEqual(core.llama_model.calls, [])

    def test_changed_actual_prefix_is_rejected_before_language_model_generation(self):
        core, data = NaturalHost(), dataset(rows=1)
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition()])
        budget = replace(plan.rows[0].budgets[0], prefix_tokens=1)
        edited = replace(plan, rows=(replace(plan.rows[0], budgets=(budget,)),))
        with self.assertRaisesRegex(RuntimeError, "generation prefix differs"):
            list(qa.iter_natural_qa(core, data, edited))
        self.assertEqual(core.llama_model.calls, [])

    def test_only_approved_question_condition_subsets_are_allowed(self):
        core, data = NaturalHost(), dataset()
        conditions = [qa.NaturalQACondition("absent"), qa.NaturalQACondition()]
        plan = qa.preflight_natural_qa(core, data, conditions)
        rows = list(qa.iter_natural_qa(core, data, plan, condition_names=[conditions[1].name], question_ids=["q1"]))
        self.assertEqual([(r["qid"], r["condition"]) for r in rows], [("q1", conditions[1].name)])
        for kwargs in ({"question_ids": ["unknown"]}, {"question_ids": ["q0", "q0"]},
                       {"condition_names": ["unknown"]}, {"condition_names": []}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "subset"):
                list(qa.iter_natural_qa(core, data, plan, **kwargs))

    def test_valid_object_ids_are_restored_without_reading_reference(self):
        core, data = NaturalHost(), dataset(rows=1)
        core.llama_model.answer = " <OBJ000> is larger than <OBJ002>.! ignored"
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition("absent")])
        result = next(qa.iter_natural_qa(core, data, plan))
        self.assertEqual(result["prediction"], "<OBJ002> is larger than <OBJ000>.")
        self.assertEqual(result["raw_prediction"], core.llama_model.answer)
        self.assertFalse(result["invalid_object_id"])

    def test_invalid_object_ids_are_retained_and_never_randomly_replaced(self):
        for answer in ("<OBJ999>", "<OBJ1000>", "<OBJ-1>", "<OBJ000", "<OBJabc>"):
            with self.subTest(answer=answer):
                core, data = NaturalHost(), dataset(rows=1)
                core.llama_model.answer = answer
                plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition("absent")])
                with mock.patch("random.randint", side_effect=AssertionError("random fallback")):
                    result = next(qa.iter_natural_qa(core, data, plan))
                self.assertEqual(result["prediction"], answer)
                self.assertTrue(result["invalid_object_id"])

    def test_one_device_batch_is_retained_at_a_time(self):
        core, data = NaturalHost(), dataset()
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition("absent")])
        transfer = qa._on_device
        previous = []

        def track(batch, device):
            self.assertTrue(all(ref() is None for ref in previous))
            result = transfer(batch, device)
            previous.append(weakref.ref(result["scene_feat"]))
            return result

        with mock.patch.object(qa, "_on_device", side_effect=track):
            results = list(qa.iter_natural_qa(core, data, plan))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(ref() is None for ref in previous))

    def test_model_failure_is_not_silently_excluded_and_state_is_restored(self):
        core, data = NaturalHost(), dataset(rows=1)
        condition = qa.NaturalQACondition(object_intervention="zero_object_slots")
        plan = qa.preflight_natural_qa(core, data, [condition])
        original = core.llama_model.generate
        core.llama_model.generate = mock.Mock(side_effect=RuntimeError("injected model failure"))
        with self.assertRaisesRegex(RuntimeError, "injected model failure"):
            list(qa.iter_natural_qa(core, data, plan))
        self.assertEqual(core.object_feature_intervention, "full")
        self.assertFalse(hasattr(core, "_rps_boundary_active"))
        self.assertTrue(core.training)
        core.llama_model.generate = original

    def test_bad_conditions_are_rejected_before_preflight(self):
        for kwargs in ({"reverse": 1}, {"representation": "unknown"},
                       {"representation": "absent", "reverse": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                qa.NaturalQACondition(**kwargs)
        core, data = NaturalHost(), dataset(rows=1)
        condition = qa.NaturalQACondition()
        for conditions in ([], [condition, condition], [qa.NaturalQACondition("trained")]):
            with self.subTest(conditions=conditions), self.assertRaises(ValueError):
                qa.preflight_natural_qa(core, data, conditions)

    def test_runtime_fingerprint_supports_torch_config_and_added_token_objects(self):
        from tokenizers import AddedToken
        core, data = NaturalHost(), dataset(rows=1)
        core.config = types.SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
        core.llama_tokenizer.init_kwargs = {"eos_token": AddedToken("</s>", special=True),
                                           "extra": {"a", "b"}}
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition("absent")])
        self.assertEqual(len(list(qa.iter_natural_qa(core, data, plan))), 1)


if __name__ == "__main__":
    unittest.main()
