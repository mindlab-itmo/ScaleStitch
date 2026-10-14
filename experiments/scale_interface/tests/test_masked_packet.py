"""Matched-position control must change numerical blocks and nothing else."""
from pathlib import Path
import sys
import unittest
from unittest import mock
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface import natural_qa as qa
from test_natural_qa import NaturalHost, dataset


class MaskedPacketTests(unittest.TestCase):
    def test_exact_length_and_non_numerical_embeddings_unchanged(self):
        core, data = NaturalHost(), dataset(radii=(.1, 1.2, 12.3), rows=1)
        _, scene, _, batch = qa._cpu_input(data, 0)
        packet = qa._packet(batch, qa.NaturalQACondition(), 100)
        true = qa.encode_scene_packet(core, packet, 'lexical')
        masked = qa.encode_scene_packet(core, packet, 'masked')
        self.assertEqual(true.shape, masked.shape)
        offset = 0
        for index, part in enumerate(packet.parts):
            length = len(qa._encode(core, part))
            block = masked[offset:offset+length]
            if index in packet.numerical_block_indices:
                expected = core.get_text_emb('?').squeeze(0).expand(length, -1)
                self.assertTrue(torch.equal(block, expected))
                self.assertFalse(torch.equal(block, true[offset:offset+length]))
            else:
                self.assertTrue(torch.equal(block, true[offset:offset+length]))
            offset += length

    def test_preflight_and_generation_share_same_matched_prefix(self):
        core, data = NaturalHost(), dataset(rows=1)
        conditions = [qa.NaturalQACondition(), qa.NaturalQACondition(representation='masked')]
        prepared = qa.preflight_natural_qa(core, data, conditions)
        left, right = prepared.rows[0].budgets
        self.assertEqual(left.packet_tokens, right.packet_tokens)
        self.assertEqual(left.prefix_tokens, right.prefix_tokens)
        self.assertEqual(core.llama_model.calls, [])
        answers = list(qa.iter_natural_qa(core, data, prepared))
        self.assertEqual(len(answers), 2)
        self.assertEqual(answers[0]['prefix_tokens'], answers[1]['prefix_tokens'])

    def test_special_placeholder_and_alternative_mask_fail(self):
        core, data = NaturalHost(), dataset(rows=1)
        _, _, _, batch = qa._cpu_input(data, 0)
        packet = qa._packet(batch, qa.NaturalQACondition(), 100)
        core.llama_tokenizer.all_special_ids = [ord('?')]
        with self.assertRaisesRegex(ValueError, 'ordinary token'):
            qa.encode_scene_packet(core, packet, 'masked')
        with self.assertRaises(ValueError):
            qa.NaturalQACondition(representation='masked', mode='swapped')

    def test_missing_policy_must_match_before_preflight(self):
        core, data = NaturalHost(), dataset(rows=1)
        core.missing_channel_policy = 'zero_projected_declared'
        with self.assertRaisesRegex(ValueError, 'policies disagree'):
            qa.preflight_natural_qa(core, data, [qa.NaturalQACondition()])
        self.assertEqual(core.llama_model.calls, [])

    def test_missing_manifest_or_policy_mutation_invalidates_preflight(self):
        for change in ('manifest', 'policy'):
            core, data = NaturalHost(), dataset(rows=1)
            data.full_manifest = {'missing_channels': {'scene0000_00': {'image_rows': []}}}
            prepared = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition()])
            if change == 'manifest':
                data.full_manifest['missing_channels']['scene0000_00']['image_rows'].append(0)
            else:
                core.missing_channel_policy = data.missing_channel_policy = 'zero_projected_declared'
            with self.assertRaises(ValueError):
                list(qa.iter_natural_qa(core, data, prepared))
            self.assertEqual(core.llama_model.calls, [])


if __name__ == '__main__':
    unittest.main()
