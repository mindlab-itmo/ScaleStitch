"""Synthetic CPU checks for feature, radius and source-object identity joins."""
import unittest
from unittest import mock

import numpy as np
import torch

from scale_interface.data import SceneDataset
from scale_interface.scene_data import BaseDataset


class SceneIdentityTests(unittest.TestCase):
    def inputs(self, *, scene="scene0000_00", obj_ids=None):
        obj_ids = torch.tensor([0, 1]) if obj_ids is None else obj_ids
        values = [int(value) for value in obj_ids]
        lookup_ids = [value + 31 if scene == "scene0217_00" else value for value in values]
        features = {f"{scene}_{oid}": torch.tensor([1. + i, 2., 3., 4.])
                    for i, oid in enumerate(lookup_ids)}
        radii = {"records": {f"{scene}_{oid}": {
            "scene_id": scene, "object_id": str(oid),
            "rho": .25 + i, "radius_valid": True,
        } for i, oid in enumerate(lookup_ids)}}
        attributes = {scene: {"locs": torch.ones(len(values), 6), "obj_ids": obj_ids}}
        rows = [{"qid": "q1", "scene_id": scene, "prompt": "A is <OBJ000>; B is <OBJ001>.",
                 "caption": "<OBJ001>", "assigned_ids": [1, 0]}]
        return features, radii, attributes, rows

    def dataset(self, inputs):
        features, radii, attributes, rows = inputs
        with mock.patch("scale_interface.data.torch.load", side_effect=[features, attributes, radii]):
            return SceneDataset(features="features.pt", radii="radii.pt", attributes="attributes.pt",
                                rows=rows, feature_dim=4)

    def test_valid_inputs_preserve_values_and_permutation(self):
        inputs = self.inputs()
        data = self.dataset(inputs)
        batch = data.batches()[0]
        self.assertTrue(torch.equal(batch["scene_feat"][0], torch.stack(list(inputs[0].values()))))
        self.assertTrue(torch.equal(batch["scene_radius"][0], torch.tensor([.25, 1.25])))
        self.assertEqual(batch["assigned_ids"].tolist(), [[1, 0]])
        self.assertEqual(batch["questions"], ("A is <OBJ001>; B is <OBJ000>.",))
        self.assertEqual(batch["answers"], ("<OBJ000>",))
        self.assertEqual(batch["scene_gnn_feats"].shape, (1, 0, 512))

    def test_offset_changes_lookup_not_source_tensor(self):
        inputs = self.inputs(scene="scene0217_00")
        source_ids = inputs[2]["scene0217_00"]["obj_ids"]
        original = source_ids.clone()
        data = self.dataset(inputs)
        first = data.scene_feats["scene0217_00"].clone()
        second = data.prepare_scene_features()[0]["scene0217_00"]
        self.assertTrue(torch.equal(source_ids, original))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(data.scene_foreground_ids["scene0217_00"].tolist(), [0, 1])

    def test_source_ids_need_not_equal_row_indices(self):
        data = self.dataset(self.inputs(obj_ids=torch.tensor([7, 42])))
        self.assertEqual(data.scene_foreground_ids["scene0000_00"].tolist(), [7, 42])
        self.assertEqual(data.batches()[0]["assigned_ids"].tolist(), [[1, 0]])

    def test_padded_source_ids_remain_usable(self):
        data = self.dataset(self.inputs(obj_ids=["00", "01"]))
        self.assertEqual(data.scene_foreground_ids["scene0000_00"].tolist(), [0, 1])

    def test_duplicate_source_ids_fail_including_padded_alias(self):
        for ids in ([1, 1], ["1", "01"]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "Duplicate source"):
                self.dataset(self.inputs(obj_ids=ids))

    def test_noninteger_source_ids_fail_without_truncation(self):
        for ids in ([0.5, 1.5], [False, True], [[0], [1]], [-1, 1]):
            inputs = self.inputs()
            inputs[2]["scene0000_00"]["obj_ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.dataset(inputs)

    def test_source_id_count_must_match_locations(self):
        inputs = self.inputs()
        inputs[2]["scene0000_00"]["obj_ids"] = [0]
        with self.assertRaisesRegex(ValueError, "aligned"):
            self.dataset(inputs)

    def test_empty_scene_fails(self):
        inputs = self.inputs()
        inputs[2]["scene0000_00"] = {"locs": torch.empty(0, 6), "obj_ids": []}
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.dataset(inputs)

    def test_wholly_missing_scene_fails_during_construction(self):
        inputs = self.inputs()
        inputs[0].clear()
        with self.assertRaisesRegex(KeyError, "Missing object feature"):
            self.dataset(inputs)

    def test_partially_missing_scene_fails(self):
        inputs = self.inputs()
        del inputs[0]["scene0000_00_1"]
        with self.assertRaisesRegex(KeyError, "Missing object feature"):
            self.dataset(inputs)

    def test_missing_attribute_scene_fails(self):
        inputs = self.inputs()
        inputs[2].clear()
        with self.assertRaisesRegex(ValueError, "missing from attributes"):
            self.dataset(inputs)

    def test_bad_required_feature_shape_dtype_or_values_fail(self):
        for value in (torch.ones(1, 4), torch.ones(2, 2), torch.ones(4, dtype=torch.long),
                      torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.complex64),
                      torch.zeros(4), torch.tensor([1., 2., float("nan"), 4.]),
                      torch.tensor([1., 2., float("inf"), 4.])):
            inputs = self.inputs()
            inputs[0]["scene0000_00_0"] = value
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "Invalid object feature"):
                self.dataset(inputs)

    def test_required_aliases_fail_even_if_values_identical(self):
        inputs = self.inputs()
        inputs[0]["scene0000_00_01"] = inputs[0]["scene0000_00_1"].clone()
        with self.assertRaisesRegex(ValueError, "Ambiguous object feature aliases"):
            self.dataset(inputs)

    def test_invalid_first_alias_cannot_fall_back_to_valid_alias(self):
        inputs = self.inputs()
        inputs[0]["scene0000_00_01"] = inputs[0]["scene0000_00_1"].clone()
        inputs[0]["scene0000_00_1"] = torch.zeros(4)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.dataset(inputs)

    def test_single_padded_alias_uses_its_exact_radius_key(self):
        inputs = self.inputs()
        inputs[0]["scene0000_00_01"] = inputs[0].pop("scene0000_00_1")
        records = inputs[1]["records"]
        records["scene0000_00_01"] = records.pop("scene0000_00_1")
        data = self.dataset(inputs)
        self.assertEqual(data.scene_radii["scene0000_00"].tolist(), [.25, 1.25])

    def test_radius_does_not_fall_back_to_another_alias(self):
        inputs = self.inputs()
        inputs[0]["scene0000_00_01"] = inputs[0].pop("scene0000_00_1")
        with self.assertRaisesRegex(KeyError, "Missing radius"):
            self.dataset(inputs)

    def test_radius_identity_checks_scene_and_object(self):
        for field, value in (("scene_id", "scene9999_00"), ("object_id", "2"),
                             ("object_id", 1.9), ("object_id", True)):
            inputs = self.inputs()
            inputs[1]["records"]["scene0000_00_1"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.dataset(inputs)

    def test_radius_valid_requires_actual_scalar_boolean(self):
        for value in ("False", 0, 1, [], [True], torch.tensor([True])):
            inputs = self.inputs()
            inputs[1]["records"]["scene0000_00_1"]["radius_valid"] = value
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.dataset(inputs)

    def test_rho_requires_finite_nonnegative_scalar_and_positive_when_valid(self):
        for value in (float("nan"), float("inf"), -1., 0., True, [.5], 1 + 2j):
            inputs = self.inputs()
            inputs[1]["records"]["scene0000_00_1"]["rho"] = value
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.dataset(inputs)

    def test_zero_radius_may_be_explicitly_invalid(self):
        inputs = self.inputs()
        inputs[1]["records"]["scene0000_00_1"].update(rho=0., radius_valid=False)
        data = self.dataset(inputs)
        self.assertEqual(data.scene_radius_valid["scene0000_00"].tolist(), [True, False])

    def test_v3_radius_threshold_checks_both_sides(self):
        for rho, valid, accepted in ((1e-7, False, True), (1e-7, True, False),
                                     (.1, False, False), (.1, True, True),
                                     (0., False, True)):
            inputs = self.inputs()
            inputs[1]["schema_version"] = "scalestitch_object_radius/v3"
            inputs[1]["records"]["scene0000_00_1"].update(rho=rho, radius_valid=valid)
            with self.subTest(rho=rho, valid=valid):
                if accepted:
                    self.dataset(inputs)
                else:
                    with self.assertRaisesRegex(ValueError, "radius_valid disagrees"):
                        self.dataset(inputs)

    def test_invalid_permutations_fail_before_get_anno(self):
        for ids in ([0, 0], [0, 2], [0], [[0, 1]], [.1, 1.9], [False, True], [0., 1.]):
            inputs = self.inputs()
            inputs[3][0]["assigned_ids"] = ids
            data = self.dataset(inputs)
            with self.subTest(ids=ids), mock.patch.object(data, "get_anno") as get_anno:
                with self.assertRaisesRegex(ValueError, "permutation"):
                    data[0]
                get_anno.assert_not_called()

    def test_valid_permutation_is_copied(self):
        original = torch.tensor([1, 0])
        actual = BaseDataset._validate_assigned_ids(original, 2)
        actual[0] = 0
        self.assertEqual(original.tolist(), [1, 0])

    def test_mixed_boolean_permutations_fail_before_tensor_coercion(self):
        values = ([True, 0], [1, False], (np.bool_(True), 0),
                  [1, np.bool_(False)], [torch.tensor(True), 0],
                  [torch.tensor(1), torch.tensor(False)],
                  [np.array(True), 0], np.array([True, False]))
        for ids in values:
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "integer permutation"):
                BaseDataset._validate_assigned_ids(ids, 2)

    def test_mixed_integer_scalar_permutation_remains_valid(self):
        for ids in ([np.int64(1), 0], [torch.tensor(1), 0],
                    np.array([1, 0], dtype=np.int32)):
            with self.subTest(ids=ids):
                self.assertEqual(BaseDataset._validate_assigned_ids(ids, 2).tolist(), [1, 0])

    def test_duplicate_and_ambiguous_question_ids_fail(self):
        for ids in (("q1", "q1"), (1, "1"), (True, "q2"), (None, "q2"), ("", "q2")):
            inputs = self.inputs()
            inputs[3][:] = [dict(inputs[3][0], qid=qid) for qid in ids]
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "qid"):
                self.dataset(inputs)

    def test_unused_feature_aliases_do_not_reject_selected_cohort(self):
        inputs = self.inputs()
        inputs[0].update({"scene9999_00_1": torch.ones(4), "scene9999_00_01": torch.zeros(4)})
        self.dataset(inputs)

    def test_optional_legacy_lookup_keeps_original_key_priority(self):
        features = {"scene0000_00_1": torch.ones(4), "scene0000_00_01": torch.full((4,), 2.)}
        feature, key = BaseDataset._lookup_object_feature_with_key(features, "scene0000_00", 1, 4)
        self.assertEqual(key, "scene0000_00_1")
        self.assertTrue(torch.equal(feature, torch.ones(4)))


if __name__ == "__main__":
    unittest.main()
