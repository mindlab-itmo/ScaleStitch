"""A two-ID next-token decision must use the actual unchanged host prefix."""
from pathlib import Path
import sys
import types
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface.candidate_decision import candidate_token_decision


class Tokenizer:
    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": {"<OBJ002>": [2], "<OBJ007>": [7]}.get(text, [1, 3])}


class Model:
    def __init__(self):
        self.calls = []
        self.scores = torch.tensor([-.5, 0., 1., .2, 4., .2, .3, 2.])

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(logits=self.scores.reshape(1, 1, -1))


class CandidateDecisionTests(unittest.TestCase):
    def test_selects_only_eligible_single_tokens_and_reports_full_vocabulary_mass(self):
        model = Model()
        prefix = torch.ones(1, 5, 4)
        mask = torch.ones(1, 5, dtype=torch.long)
        token, info = candidate_token_decision(model, Tokenizer(), inputs_embeds=prefix,
            attention_mask=mask, candidate_identifiers=("<OBJ002>", "<OBJ007>"))
        self.assertEqual(token.tolist(), [[7]])
        self.assertIs(model.calls[0]["inputs_embeds"], prefix)
        self.assertIs(model.calls[0]["attention_mask"], mask)
        self.assertEqual(info["unrestricted_first_token_id"], 4)
        expected = model.scores.softmax(0)[[2, 7]].sum().item()
        self.assertAlmostEqual(info["pair_probability"], expected, places=6)
        self.assertEqual(info["candidate_token_ids"], [2, 7])
        self.assertEqual(info["candidate_logits"], [1., 2.])
        self.assertFalse(model.calls[0]["use_cache"])

    def test_rejects_multitoken_duplicate_or_invalid_candidates_before_forward(self):
        for candidates in (("<OBJ002>", "<OBJ003>"), ("<OBJ002>", "<OBJ002>"), ("2", "7")):
            model = Model()
            with self.assertRaises(ValueError):
                candidate_token_decision(model, Tokenizer(), inputs_embeds=torch.ones(1, 5, 4),
                    attention_mask=torch.ones(1, 5), candidate_identifiers=candidates)
            self.assertEqual(model.calls, [])

    def test_ties_are_explicit_and_follow_question_candidate_order(self):
        model = Model()
        model.scores[2] = model.scores[7]
        token, info = candidate_token_decision(model, Tokenizer(), inputs_embeds=torch.ones(1, 5, 4),
            attention_mask=torch.ones(1, 5), candidate_identifiers=("<OBJ007>", "<OBJ002>"))
        self.assertTrue(info["tie"])
        self.assertEqual(token.tolist(), [[7]])


if __name__ == "__main__":
    unittest.main()
