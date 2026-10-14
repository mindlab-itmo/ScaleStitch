"""CPU identity, integrity and batching checks for the full image/graph input."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface.data import SceneDataset
from scale_interface.scene_data import train_collate_fn
from scale_interface import natural_qa as qa


class FullDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scene = "scene0000_00"
        self.ids = [7, 42, 91]
        self.values = {
            "features": {f"{self.scene}_{oid}": torch.full((4,), float(i + 1))
                         for i, oid in enumerate(self.ids)},
            "radii": {"records": {f"{self.scene}_{oid}": {
                "scene_id": self.scene, "object_id": oid, "rho": .2 + i,
                "radius_valid": True,
            } for i, oid in enumerate(self.ids)}},
            "attributes": {self.scene: {"locs": torch.ones(3, 6), "obj_ids": self.ids}},
            "images": {f"{self.scene}_{oid}": torch.full((1024,), float(i + 5))
                       for i, oid in enumerate(self.ids)},
            "relations": {self.scene: torch.arange(3 * 3 * 512).reshape(3, 3, 512).float()},
        }
        self.rows = [{"qid": "q1", "scene_id": self.scene, "prompt": "Find <OBJ001>",
                      "assigned_ids": [2, 0, 1]}]

    def files(self):
        paths = {name: self.root / f"{name}.pt" for name in self.values}
        for name, path in paths.items():
            torch.save(self.values[name], path)
        manifest = {"schema_version": "scalestitch_full_input/v1",
                    "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in paths.items()},
                    "scene_object_ids": {self.scene: self.ids}}
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return {**paths, "full_manifest": manifest_path, "rows": self.rows, "feature_dim": 4}

    def test_full_preserves_attribute_row_order_and_dense_pairs(self):
        data = SceneDataset(**self.files())
        sample = data[0]
        self.assertTrue(data.full_input)
        self.assertEqual(sample[1][:, 0].tolist(), [5., 6., 7.])
        self.assertTrue(torch.equal(sample[6], self.values["relations"][self.scene]))
        self.assertEqual(sample[5].tolist(), [2, 0, 1])
        self.assertEqual(sample[-1], "Find <OBJ002>")
        self.assertEqual(data.batches()[0]["scene_gnn_feats"].shape, (1, 3, 3, 512))

    def test_full_never_pads_pair_axes_or_mixes_banks(self):
        sample = SceneDataset(**self.files())[0]
        with self.assertRaisesRegex(ValueError, "batch size one"):
            train_collate_fn([sample, sample])

    def test_all_partial_full_configurations_fail(self):
        params = self.files()
        for missing in ("images", "relations", "full_manifest"):
            invalid = dict(params)
            del invalid[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "together"):
                SceneDataset(**invalid)

    def test_object_only_retains_zero_placeholders(self):
        params = self.files()
        for key in ("images", "relations", "full_manifest"):
            del params[key]
        data = SceneDataset(**params)
        self.assertFalse(data.full_input)
        self.assertFalse(data[0][1].any())
        self.assertEqual(data.batches()[0]["scene_gnn_feats"].shape, (1, 0, 512))

    def test_each_input_is_hash_bound(self):
        for changed in self.values:
            params = self.files()
            with params[changed].open("ab") as handle:
                handle.write(b"changed")
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, f"SHA256 mismatch for {changed}"):
                SceneDataset(**params)

    def test_manifest_cannot_silently_reorder_source_ids(self):
        params = self.files()
        manifest = json.loads(params["full_manifest"].read_text())
        manifest["scene_object_ids"][self.scene] = list(reversed(self.ids))
        params["full_manifest"].write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "row identities"):
            SceneDataset(**params)

    def test_invalid_or_missing_image_cannot_be_substituted(self):
        key = f"{self.scene}_7"
        for bad in (None, torch.zeros(1024), torch.ones(1, 1024), torch.ones(1024).long(),
                    torch.full((1024,), float("nan"))):
            if bad is None:
                self.values["images"].pop(key, None)
            else:
                self.values["images"][key] = bad
            with self.subTest(bad=bad), self.assertRaises((KeyError, RuntimeError)):
                SceneDataset(**self.files())

    def test_image_alias_ambiguity_fails(self):
        self.values["images"][f"{self.scene}_07"] = torch.ones(1024)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            SceneDataset(**self.files())

    def test_relation_pair_dimensions_values_and_scene_are_checked(self):
        for bank in ({}, {self.scene: torch.ones(6, 512)},
                     {self.scene: torch.ones(3, 2, 512)},
                     {self.scene: torch.ones(3, 3, 512).long()},
                     {self.scene: torch.full((3, 3, 512), float("inf"))}):
            self.values["relations"] = bank
            with self.subTest(shape=list(bank)), self.assertRaisesRegex(ValueError, "relation bank"):
                SceneDataset(**self.files())

    def test_absent_relation_pairs_survive_until_actual_host_selection(self):
        self.values["relations"][self.scene].zero_()
        data = SceneDataset(**self.files())
        self.assertFalse(data[0][6].any())

    def test_full_natural_qa_requires_validated_dataset_and_causal_host(self):
        core = types.SimpleNamespace(host_mode="full", knn=2, add_img_token=True,
                                     add_scene_token=False, bbox_embed=False, bidirection=False,
                                     llama_model=types.SimpleNamespace(config=None))
        condition = qa.NaturalQACondition(representation="absent")
        qa._check_request(core, [condition], None, None, 32, 2502, "question_tail", 100)
        with self.assertRaisesRegex(ValueError, "manifest-validated"):
            qa.preflight_natural_qa(core, types.SimpleNamespace(full_input=False), [condition])
        core.bidirection = True
        with self.assertRaisesRegex(ValueError, "causal"):
            qa._check_request(core, [condition], None, None, 32, 2502, "question_tail", 100)

    def test_full_preflight_streams_real_assembly_and_excludes_missing_selected_edges(self):
        from test_full_host import FullHost
        from test_natural_qa import NaturalHost, Recorder, CharacterTokenizer

        class FullNaturalHost(FullHost):
            get_text_emb = NaturalHost.get_text_emb

        self.values["attributes"][self.scene]["locs"][:, 0] = torch.tensor([0., 1., 4.])
        conditions = tuple(qa.NaturalQACondition(representation, mode=mode)
                           for representation, mode in (("absent", "true"), ("lexical", "true"),
                                                        ("lexical", "constant"), ("lexical", "swapped")))
        for missing in (False, True):
            if missing:
                self.values["relations"][self.scene][0, 1] = 0
            data = SceneDataset(**self.files())
            core = FullNaturalHost()
            core.llama_model, core.llama_tokenizer = Recorder(), CharacterTokenizer()
            plan = qa.preflight_natural_qa(core, data, conditions)
            self.assertEqual(core.llama_model.calls, [])
            if missing:
                self.assertFalse(plan.rows)
                self.assertEqual(len(plan.exclusions), 4)
                self.assertTrue(all("missing" in item.reason for item in plan.exclusions))
            else:
                self.assertFalse(plan.exclusions)
                self.assertEqual(len(plan.rows), 1)
                results = list(qa.iter_natural_qa(core, data, plan))
                self.assertEqual(len(results), 4)
                self.assertTrue(all(item["prediction"] == "generated" for item in results))


if __name__ == "__main__":
    unittest.main()
