from dataclasses import replace
from pathlib import Path
import re
import sys
import types
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_interface import natural_qa as qa
from test_natural_qa import NaturalHost, CharacterTokenizer, Recorder
from test_candidate_packet_integration import direct_data


class NativeTokenizer(CharacterTokenizer):
    def __call__(self, text, **kwargs):
        parts = re.split(r"(<OBJ[0-9]{3}>)", text)
        ids = [token for part in parts for token in
               ([1000 + int(part[4:-1])] if re.fullmatch(r"<OBJ[0-9]{3}>", part) else [ord(c) for c in part])]
        if kwargs.get("return_tensors") == "pt":
            from test_object_feature_controls import TokenBatch
            values = torch.tensor([ids])
            return TokenBatch(input_ids=values, attention_mask=torch.ones_like(values))
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(f"<OBJ{int(value) - 1000:03d}>" if int(value) >= 1000 else chr(int(value)) for value in ids)

    def get_vocab(self):
        return dict(super().get_vocab(), **{f"<OBJ{i:03d}>": 1000 + i for i in range(150)})


class DecisionModel(Recorder):
    def __init__(self):
        super().__init__()
        self.forwards = []

    def forward(self, **kwargs):
        self.forwards.append(kwargs)
        logits = torch.zeros(1, 1, 1150)
        logits[0, 0, 1000] = 4.
        logits[0, 0, 1001] = 8.  # A visible distractor has the largest unrestricted logit.
        logits[0, 0, 1002] = 3.
        return types.SimpleNamespace(logits=logits)


class CandidateDecisionIntegrationTests(unittest.TestCase):
    def test_restricted_candidates_must_be_visible_even_without_a_packet(self):
        core, data = NaturalHost(), direct_data()
        core.llama_tokenizer, core.llama_model = NativeTokenizer(), DecisionModel()
        data.scene_masks["scene0000_00"][0] = False
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition("absent")],
                                       repetition_penalty=1., decision_mode="candidate_logits")
        self.assertEqual(plan.rows, ())
        self.assertIn("visible", plan.exclusions[0].reason)
        self.assertEqual(core.llama_model.forwards, [])

    def test_same_prefix_uses_restricted_logits_and_maps_visible_id_once(self):
        core, data = NaturalHost(), direct_data()
        core.llama_tokenizer, core.llama_model = NativeTokenizer(), DecisionModel()
        condition = qa.NaturalQACondition(packet_scope="queried_pair")
        free = qa.preflight_natural_qa(core, data, [condition], repetition_penalty=1.)
        plan = qa.preflight_natural_qa(core, data, [condition], repetition_penalty=1., decision_mode="candidate_logits")
        self.assertEqual(free.rows[0].budgets, plan.rows[0].budgets)
        self.assertEqual(core.llama_model.forwards, [])
        self.assertEqual(core.llama_model.calls, [])
        row = list(qa.iter_natural_qa(core, data, plan))[0]
        self.assertEqual(row["raw_prediction"], "<OBJ000>")
        self.assertEqual(row["prediction"], "<OBJ002>")  # source mapping [2,1,0]
        self.assertEqual(row["candidate_decision"]["candidate_identifiers"], ["<OBJ002>", "<OBJ000>"])
        self.assertEqual(row["candidate_decision"]["unrestricted_first_token_id"], 1001)
        self.assertEqual(row["effective_generation"]["decision_mode"], "candidate_logits")
        self.assertEqual(core.llama_model.forwards[0]["inputs_embeds"].shape[1], row["prefix_tokens"])
        self.assertEqual(core.llama_model.calls, [])

    def test_mode_changes_are_sealed_and_penalty_three_is_not_silently_reinterpreted(self):
        core, data = NaturalHost(), direct_data()
        with self.assertRaisesRegex(ValueError, "penalty|repetition_penalty"):
            qa.preflight_natural_qa(core, data, [qa.NaturalQACondition()], decision_mode="candidate_logits")
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition()], repetition_penalty=1.)
        with self.assertRaisesRegex(ValueError, "generation.*preflight"):
            list(qa.iter_natural_qa(core, data, replace(plan, decision_mode="candidate_logits")))
        self.assertEqual(core.llama_model.calls, [])


if __name__ == "__main__":
    unittest.main()
