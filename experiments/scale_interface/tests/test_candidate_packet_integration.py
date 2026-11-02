"""Candidate scope must use the same filtered scene intervention in both stages."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_interface import natural_qa as qa
from test_natural_qa import NaturalHost, dataset


def direct_data():
    data = dataset(radii=(.1, 1.2, 12.3), rows=1)
    data.anno[0]["prompt"] = "Which object, <OBJ000> or <OBJ002>, has the larger measured metric radius? Answer with exactly one object identifier and no other text."
    return data


class CandidatePacketIntegrationTests(unittest.TestCase):
    def test_candidate_scope_preflight_and_generated_receipts_use_two_source_rows(self):
        core, data = NaturalHost(), direct_data()
        conditions = [qa.NaturalQACondition("absent"), qa.NaturalQACondition(),
                      qa.NaturalQACondition(packet_scope="queried_pair"),
                      qa.NaturalQACondition(mode="swapped", packet_scope="queried_pair")]
        plan = qa.preflight_natural_qa(core, data, conditions)
        self.assertEqual(plan.exclusions, ())
        self.assertEqual(core.llama_model.calls, [])
        budgets = plan.rows[0].budgets
        self.assertEqual([b.numerical_blocks for b in budgets], [0, 3, 2, 2])
        self.assertLess(budgets[2].packet_tokens, budgets[1].packet_tokens)
        rows = list(qa.iter_natural_qa(core, data, plan))
        metadata = rows[3]["packet_metadata"]
        self.assertEqual(metadata["packet_scope"], "queried_pair")
        self.assertEqual(metadata["full_scene_objects"], 3)
        self.assertEqual({r["source_row"] for r in metadata["records"]}, {0, 2})
        # The full-scene rotation supplies a donor outside the candidate pair.
        self.assertIn(1, [r["value_source_row"] for r in metadata["records"]])
        self.assertEqual(rows[2]["prefix_tokens"], budgets[2].prefix_tokens)

    def test_focused_mask_changes_only_number_unit_blocks_at_matched_token_lengths(self):
        core, data = NaturalHost(), direct_data()
        conditions = [qa.NaturalQACondition(packet_scope="queried_pair"),
                      qa.NaturalQACondition(representation="masked", packet_scope="queried_pair")]
        plan = qa.preflight_natural_qa(core, data, conditions)
        left, right = plan.rows[0].budgets
        self.assertEqual((left.packet_tokens, left.prefix_tokens), (right.packet_tokens, right.prefix_tokens))
        _, scene, _, batch = qa._cpu_input(data, 0)
        packet = qa._packet(batch, conditions[0], 100, scene_id=scene, candidate_source_rows=(0, 2))
        lexical = qa.encode_scene_packet(core, packet, "lexical")
        masked = qa.encode_scene_packet(core, packet, "masked")
        offset = 0
        for i, part in enumerate(packet.parts):
            length = len(qa._encode(core, part))
            block = masked[offset:offset + length]
            if i in packet.numerical_block_indices:
                self.assertTrue(torch.equal(block, core.get_text_emb("?").squeeze(0).expand(length, -1)))
            else:
                self.assertTrue(torch.equal(block, lexical[offset:offset + length]))
            offset += length
        results = list(qa.iter_natural_qa(core, data, plan))
        self.assertEqual(results[1]["packet_metadata"]["display_representation"], "masked")
        self.assertEqual(results[1]["packet_metadata"]["numeric_metadata_is_displayed"], False)

    def test_source_pair_and_condition_scope_are_sealed_before_generation(self):
        core, data = NaturalHost(), direct_data()
        focused = qa.NaturalQACondition(packet_scope="queried_pair")
        plan = qa.preflight_natural_qa(core, data, [focused])
        changed = replace(plan, conditions=(qa.NaturalQACondition(),))
        with self.assertRaisesRegex(ValueError, "condition.*preflight"):
            list(qa.iter_natural_qa(core, data, changed))
        data.anno[0]["prompt"] = data.anno[0]["prompt"].replace("<OBJ002>", "<OBJ001>")
        with self.assertRaisesRegex(ValueError, "dataset|inputs"):
            list(qa.iter_natural_qa(core, data, plan))
        self.assertEqual(core.llama_model.calls, [])

    def test_implicit_named_or_ambiguous_candidates_are_not_inferred(self):
        for prompt in ("Which chair is taller?", "<OBJ000> or <OBJ000>?", "<OBJ000> <OBJ001> <OBJ002>?"):
            core, data = NaturalHost(), direct_data()
            data.anno[0]["prompt"] = prompt
            plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition(packet_scope="queried_pair")])
            self.assertEqual(plan.rows, ())
            self.assertIn("two distinct", plan.exclusions[0].reason)
            self.assertEqual(core.llama_model.calls, [])


if __name__ == "__main__":
    unittest.main()
