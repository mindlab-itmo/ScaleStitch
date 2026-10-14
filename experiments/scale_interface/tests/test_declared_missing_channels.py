"""Exact declared-missing masks and post-projector masking on the real CPU path."""
import copy
import json
import unittest

import torch

from test_full_data import FullDataTests
from test_full_host import FullHost, full_batch
from scale_interface.data import SceneDataset

POLICY = "zero_projected_declared"


class DeclaredDataTests(unittest.TestCase):
    setUp = FullDataTests.setUp
    files = FullDataTests.files

    def parameters(self, image_rows=(), relation_pairs=(), *, version=2):
        params = self.files()
        path = params["full_manifest"]
        manifest = json.loads(path.read_text())
        manifest.update(schema_version=f"scalestitch_full_input/v{version}",
                        missing_channel_policy=POLICY,
                        missing_channels={self.scene: dict(image_rows=list(image_rows),
                            relation_pairs=[list(pair) for pair in relation_pairs])})
        path.write_text(json.dumps(manifest))
        return params

    def test_explicit_zeros_stay_in_exact_source_rows_before_assignment(self):
        self.values["images"][f"{self.scene}_42"].zero_()
        self.values["relations"][self.scene][0, 2].zero_()
        data = SceneDataset(**self.parameters([1], [(0, 2)]))
        self.assertEqual(data.missing_channel_policy, POLICY)
        sample = data[0]
        self.assertEqual(sample[5].tolist(), [2, 0, 1])
        self.assertFalse(sample[1][1].any())
        self.assertTrue(sample[1][0].any())
        self.assertFalse(sample[6][0, 2].any())
        self.assertTrue(sample[6][2, 0].any())
        self.assertEqual(sample[-1], "Find <OBJ002>")

    def test_all_available_v2_data_are_byte_equal_to_v1(self):
        legacy = SceneDataset(**self.files())[0]
        available = SceneDataset(**self.parameters())[0]
        for left, right in zip(legacy, available):
            self.assertTrue(torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right)

    def test_v1_cannot_opt_in_and_default_v1_missing_image_fails(self):
        with self.assertRaisesRegex(ValueError, "v1 cannot"):
            SceneDataset(**self.parameters(version=1))
        self.values["images"][f"{self.scene}_7"].zero_()
        with self.assertRaises(RuntimeError):
            SceneDataset(**self.files())

    def test_undeclared_zero_and_falsely_declared_available_slot_fail(self):
        self.values["images"][f"{self.scene}_42"].zero_()
        for declared in ([], [0], [1, 2]):
            with self.subTest(declared=declared), self.assertRaisesRegex(ValueError, "exact declared"):
                SceneDataset(**self.parameters(declared))
        self.values["images"][f"{self.scene}_42"].fill_(1)
        self.values["relations"][self.scene][0, 2].zero_()
        for declared in ([], [(2, 0)], [(0, 2), (1, 2)]):
            with self.subTest(declared=declared), self.assertRaisesRegex(ValueError, "exact declared"):
                SceneDataset(**self.parameters(relation_pairs=declared))

    def test_missing_dictionary_entry_cannot_be_filled_even_when_declared(self):
        del self.values["images"][f"{self.scene}_42"]
        with self.assertRaises(KeyError):
            SceneDataset(**self.parameters([1]))

    def test_nonfinite_rejected_in_declared_slot_and_diagonal(self):
        self.values["images"][f"{self.scene}_42"].fill_(float("nan"))
        with self.assertRaises(RuntimeError):
            SceneDataset(**self.parameters([1]))
        self.values["images"][f"{self.scene}_42"].fill_(1)
        for pair in ((0, 1), (1, 1)):
            self.values["relations"][self.scene][pair[0], pair[1], 0] = float("inf")
            with self.assertRaisesRegex(ValueError, "finite"):
                SceneDataset(**self.parameters())
            self.values["relations"][self.scene][pair[0], pair[1], 0] = 1

    def test_masks_reject_duplicates_booleans_range_diagonal_or_omitted_scene(self):
        for image, pair in (([1, 1], []), ([True], []), ([3], []), ([], [(0, 0)]),
                            ([], [(0, 1), (0, 1)]), ([], [(0, 3)])):
            with self.subTest(image=image, pair=pair), self.assertRaises(ValueError):
                SceneDataset(**self.parameters(image, pair))
        params = self.parameters()
        manifest = json.loads(params["full_manifest"].read_text())
        manifest["missing_channels"] = {}
        params["full_manifest"].write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "every scene"):
            SceneDataset(**params)

    def test_diagonal_zeros_are_ignored_without_erasing_nonself_coverage(self):
        for row in range(3):
            self.values["relations"][self.scene][row, row].zero_()
        SceneDataset(**self.parameters())


class DeclaredHostTests(unittest.TestCase):
    def prefix(self, core, source, train=False):
        if train:
            core.forward_train(**source, questions=["Which object?"], answers=["A"])
        else:
            core.evaluate(**source, custom_prompt=["Which object?"])
        return core.llama_model.captured

    def test_available_host_path_byte_parity_train_and_evaluate(self):
        for train in (False, True):
            legacy, source = FullHost(), full_batch()
            legacy.add_pos_emb = True
            opted = copy.deepcopy(legacy)
            opted.missing_channel_policy = POLICY
            before, after = self.prefix(legacy, source, train), self.prefix(opted, source, train)
            for field in ("inputs_embeds", "attention_mask") + (("labels",) if train else ()):
                self.assertTrue(torch.equal(before[field], after[field]), (train, field))

    def test_only_missing_slots_zero_after_bias_and_position_with_assignment(self):
        for train in (False, True):
            core, source = FullHost(), full_batch()
            core.missing_channel_policy = POLICY
            core.add_pos_emb = True
            with torch.no_grad():
                core.object_img_proj.bias.fill_(17)
            baseline = self.prefix(core, source, train)["inputs_embeds"].clone()
            source["scene_img_feat"][0, 1].zero_()
            source["scene_gnn_feats"][0, 0, 1].zero_()
            actual = self.prefix(core, source, train)["inputs_embeds"]
            expected = baseline.clone()
            # Prefix: two fixed tokens, then [ID,image,(obj,edge,neighbor)*2].
            order = source["assigned_ids"][0].tolist()
            expected[0, 2 + order.index(1)*8 + 1] = 0
            expected[0, 2 + order.index(0)*8 + 3] = 0
            self.assertEqual(actual.shape, baseline.shape)
            self.assertTrue(torch.equal(actual, expected), train)
            self.assertTrue(baseline[0, 2 + order.index(1)*8 + 1].any())
            self.assertTrue(baseline[0, 2 + order.index(0)*8 + 3].any())

    def test_nonfinite_remains_error_and_policy_must_match_full_host(self):
        core, source = FullHost(), full_batch()
        core.missing_channel_policy = POLICY
        source["scene_img_feat"][0, 1, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.prefix(core, source)
        core.host_mode = "object_only"
        with self.assertRaisesRegex(ValueError, "host_mode=full"):
            core._declared_missing_channels()
        core.host_mode, core.missing_channel_policy = "full", "implicit"
        with self.assertRaisesRegex(ValueError, "invalid"):
            core._declared_missing_channels()


if __name__ == "__main__":
    unittest.main()
