"""Small CPU checks for packing existing GT image and directed relation features."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface.prepare_full_inputs import prepare_full_inputs, sha256


class PrepareFullInputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scene = "scene0000_00"
        self.ids = [7, 42, 91]
        attrs = {self.scene: {"locs": torch.arange(18).reshape(3, 6).float(), "obj_ids": self.ids}}
        self.banks = {"attributes": attrs, "source_attributes": attrs,
            "features": {f"{self.scene}_{oid}": torch.ones(4) for oid in self.ids},
            "radii": {"records": {f"{self.scene}_{oid}": {"scene_id": self.scene,
                "object_id": oid, "rho": 1., "radius_valid": True} for oid in self.ids}},
            "image_features": {f"{self.scene}_{oid}": torch.ones(1024) * oid for oid in self.ids},
            "raw_relations": {f"{self.scene}_7_42": torch.ones(512) * 3,
                              f"{self.scene}_42_7": torch.ones(512) * 5}}
        self.rows = [{"qid": "kept", "scene_id": self.scene, "prompt": "Find <OBJ001>",
                      "assigned_ids": [2, 0, 1]},
                     {"qid": "excluded", "scene_id": "scene0001_00", "prompt": "Other?", "assigned_ids": [0]}]
        self.base = {"seed": 42, "host": {"host_checkpoint": "host.pt", "deployment": "bridge.pt",
                    "model": {"llama_model_path": "llama", "host_mode": "object_only", "gt_pretrain": False}},
                    "data": {key: key + ".pt" for key in ("features", "radii", "attributes")},
                    "run": {"conditions": [{"representation": "absent"}, {"representation": "lexical"}]}}
        self.base["data"].update(rows="rows.json", feature_dim=4)
        self.spec = {key: key + (".json" if key in {"base_config", "allowed_scenes"} else ".pt")
                     for key in ("base_config", "allowed_scenes", "source_attributes", "image_features", "raw_relations")}
        self.allowed = [self.scene]

    def prepare(self, name="out"):
        for key, value in self.banks.items():
            torch.save(value, self.root / (key + ".pt"))
        for key, value in (("rows", self.rows), ("base_config", self.base),
                           ("allowed_scenes", self.allowed), ("prep", self.spec)):
            (self.root / (key + ".json")).write_text(json.dumps(value), encoding="utf-8")
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            return prepare_full_inputs(self.root / "prep.json", self.root / name)
        finally:
            os.chdir(previous)

    def test_preserves_rows_and_directed_identity_and_emits_loadable_config(self):
        summary = self.prepare()
        out = self.root / "out"
        config = json.loads((out / "config.json").read_text())
        self.assertEqual(summary["questions"], 1)
        self.assertEqual(json.loads((out / "questions.json").read_text()), self.rows[:1])
        self.assertTrue(config["host"]["model"]["gt_pretrain"])
        self.assertEqual(config["host"]["model"]["knn"], 2)
        self.assertEqual(config["run"], self.base["run"])
        bank = torch.load(out / "relations.pt", weights_only=True)[self.scene]
        self.assertEqual(bank[0, 1, 0].item(), 3)
        self.assertEqual(bank[1, 0, 0].item(), 5)
        self.assertFalse(bank[0, 2].any())
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["scene_object_ids"][self.scene], self.ids)
        self.assertEqual(manifest["sha256"]["images"], sha256(out / "images.pt"))
        self.assertEqual(manifest["provenance"]["allowed_scenes"]["sha256"], sha256(self.root / "allowed_scenes.json"))
        self.assertEqual(json.loads((out / "exclusions.json").read_text())["excluded_qids"], ["excluded"])

    def test_existing_output_is_never_overwritten(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_official_newline_relation_keys_preserve_direction_and_missing_pairs(self):
        self.banks["raw_relations"] = {
            f"{self.scene}\n_7_42": torch.ones(512, dtype=torch.float16) * 3,
            f"{self.scene}\n_42_7": torch.ones(512, dtype=torch.float16) * 5,
        }
        self.prepare()
        bank = torch.load(self.root / "out/relations.pt", weights_only=True)[self.scene]
        self.assertEqual(bank.shape, (3, 3, 512))
        self.assertEqual(bank.dtype, torch.float32)
        self.assertTrue(torch.equal(bank[0, 1], torch.full((512,), 3.)))
        self.assertTrue(torch.equal(bank[1, 0], torch.full((512,), 5.)))
        self.assertFalse(bank[0, 2].any())
        manifest = json.loads((self.root / "out/manifest.json").read_text())
        self.assertEqual(manifest["missing_directed_pairs"][self.scene], 4)

    def test_duplicate_relation_aliases_fail_even_when_equal(self):
        for value in (3., 9.):
            self.banks["raw_relations"][f"{self.scene}\n_7_42"] = torch.full((512,), value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "ambiguous directed relation"):
                self.prepare()
            self.assertFalse((self.root / "out").exists())

    def test_wholly_missing_scene_relations_fail_before_any_output(self):
        for values in ({}, {f"{self.scene}\n_7_7": torch.ones(512)},
                       {f"{self.scene} _7_42": torch.ones(512)}):
            self.banks["raw_relations"] = values
            with self.subTest(keys=list(values)), self.assertRaisesRegex(ValueError, "no non-self directed relations"):
                self.prepare()
            self.assertFalse((self.root / "out").exists())

    def test_fixed_coverage_does_not_depend_on_answer_fields(self):
        self.rows[0]["answers"] = ["different answer"]
        self.rows[1]["answers"] = ["another answer"]
        self.assertEqual(self.prepare()["questions"], 1)

    def test_gt_coordinate_and_row_identity_mismatches_fail(self):
        for field, bad in (("locs", torch.zeros(3, 6)), ("obj_ids", list(reversed(self.ids)))):
            original = self.banks["source_attributes"]
            self.banks["source_attributes"] = {self.scene: dict(original[self.scene], **{field: bad})}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.prepare()
            self.banks["source_attributes"] = original

    def test_missing_ambiguous_and_invalid_images_fail(self):
        key = f"{self.scene}_7"
        for bad in (None, torch.zeros(1024), torch.ones(512), torch.full((1024,), float("nan"))):
            if bad is None:
                self.banks["image_features"].pop(key, None)
            else:
                self.banks["image_features"][key] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "image"):
                self.prepare()
        self.banks["image_features"][key] = torch.ones(1024)
        self.banks["image_features"][f"{self.scene}_07"] = torch.ones(1024)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.prepare()

    def test_present_malformed_relation_is_not_treated_as_missing(self):
        for bad in (torch.ones(1024), torch.zeros(512), torch.ones(512).long(), torch.full((512,), float("inf"))):
            self.banks["raw_relations"][f"{self.scene}_7_42"] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "directed relation"):
                self.prepare()

    def test_duplicate_qids_and_empty_coverage_fail(self):
        self.rows[1]["qid"] = "kept"
        with self.assertRaisesRegex(ValueError, "unique"):
            self.prepare()
        self.rows[1]["qid"] = "excluded"
        self.allowed = ["scene9999_00"]
        with self.assertRaisesRegex(ValueError, "no questions"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
