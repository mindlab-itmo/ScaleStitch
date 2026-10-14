"""Public v2 preparation and CLI policy guards; small CPU source fixtures."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import test_prepare_full_inputs as fixtures
from scale_interface import __main__ as cli
from scale_interface.data import SceneDataset
from scale_interface.prepare_full_inputs import sha256

POLICY = "zero_projected_declared"


class DeclaredPreparationTests(unittest.TestCase):
    def setUp(self):
        fixtures.PrepareFullInputsTests.setUp(self)
        self.spec.update(missing_channel_policy=POLICY, missing_channels="missing.json")
        self.masks = {self.scene: {"image_rows": [],
            "relation_pairs": [[0, 2], [1, 2], [2, 0], [2, 1]]}}

    def prepare(self, name="out"):
        (self.root / "missing.json").write_text(json.dumps(self.masks), encoding="utf-8")
        return fixtures.PrepareFullInputsTests.prepare(self, name)

    def test_fixed_mask_packs_exact_missing_rows_and_directed_pairs(self):
        del self.banks["image_features"][f"{self.scene}_42"]
        self.masks[self.scene]["image_rows"] = [1]
        result = self.prepare()
        config = json.loads(Path(result["config"]).read_text())
        data = SceneDataset(**config["data"])
        manifest = data.full_manifest
        self.assertEqual(data.missing_channel_policy, POLICY)
        self.assertEqual(config["host"]["model"]["missing_channel_policy"], POLICY)
        self.assertEqual(manifest["schema_version"], "scalestitch_full_input/v2")
        self.assertEqual(manifest["missing_channels"], self.masks)
        self.assertEqual(manifest["provenance"]["missing_channels"]["sha256"],
                         sha256(self.root / "missing.json"))
        self.assertEqual(data[0][5].tolist(), [2, 0, 1])
        self.assertFalse(data[0][1][1].any())
        self.assertTrue(torch.equal(data[0][1][0], self.banks["image_features"][f"{self.scene}_7"]))
        self.assertTrue(torch.equal(data[0][6][0, 1], self.banks["raw_relations"][f"{self.scene}_7_42"]))
        self.assertFalse(data[0][6][0, 2].any())

    def test_all_available_v2_and_v1_samples_are_identical(self):
        self.banks["raw_relations"] = {f"{self.scene}_{left}_{right}": torch.full((512,), float(i*3+j+1))
            for i, left in enumerate(self.ids) for j, right in enumerate(self.ids) if i != j}
        self.masks[self.scene]["relation_pairs"] = []
        v2 = self.prepare("v2")
        del self.spec["missing_channel_policy"], self.spec["missing_channels"]
        v1 = self.prepare("v1")
        before = SceneDataset(**json.loads(Path(v1["config"]).read_text())["data"])[0]
        after = SceneDataset(**json.loads(Path(v2["config"]).read_text())["data"])[0]
        for left, right in zip(before, after):
            self.assertTrue(torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right)

    def test_declared_source_entries_must_be_absent_not_zero_or_available(self):
        for kind in ("image", "relation"):
            for zero in (False, True):
                key = f"{self.scene}_42" if kind == "image" else f"{self.scene}_7_42"
                bank = self.banks["image_features" if kind == "image" else "raw_relations"]
                saved = bank[key].clone()
                self.masks[self.scene]["image_rows"] = [1] if kind == "image" else []
                self.masks[self.scene]["relation_pairs"] = [[0, 2], [1, 2], [2, 0], [2, 1]] + ([[0, 1]] if kind == "relation" else [])
                if zero:
                    bank[key].zero_()
                with self.subTest(kind=kind, zero=zero), self.assertRaisesRegex(ValueError, "declared missing.*is present"):
                    self.prepare()
                self.assertFalse((self.root / "out").exists())
                bank[key] = saved

    def test_unexpected_source_missingness_is_never_added_to_fixed_declaration(self):
        for key in (f"{self.scene}_7_42", f"{self.scene}_42_7"):
            saved = self.banks["raw_relations"].pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "undeclared missing directed relation"):
                self.prepare()
            self.banks["raw_relations"][key] = saved
        del self.banks["image_features"][f"{self.scene}_42"]
        with self.assertRaisesRegex(ValueError, "undeclared image"):
            self.prepare()
        self.assertFalse((self.root / "out").exists())

    def test_invalid_masks_and_missing_scene_rejected_before_output(self):
        valid = copy.deepcopy(self.masks)
        for mask in ({"image_rows": [True], "relation_pairs": []},
                     {"image_rows": [3], "relation_pairs": []},
                     {"image_rows": [], "relation_pairs": [[1, 1]]},
                     {"image_rows": [], "relation_pairs": [[0, 2], [0, 2]]}):
            self.masks[self.scene] = mask
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "invalid missing"):
                self.prepare()
        self.masks = {}
        with self.assertRaisesRegex(ValueError, "both masks"):
            self.prepare()
        self.masks = valid
        self.assertFalse((self.root / "out").exists())

    def test_retired_codec_base_config_is_rejected_before_output(self):
        for key, value in (("codec", "missing.pt"), ("codec", None), ("random_codec_seed", 42)):
            self.base[key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "codec settings are not supported"):
                self.prepare()
            self.assertFalse((self.root / "out").exists())
            del self.base[key]

    def test_policy_and_declaration_are_required_together(self):
        for spec in ({"missing_channel_policy": POLICY}, {"missing_channels": "missing.json"},
                     {"missing_channel_policy": "implicit", "missing_channels": "missing.json"}):
            self.spec.pop("missing_channels", None)
            self.spec.pop("missing_channel_policy", None)
            self.spec.update(spec)
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, "together|invalid missing_channel_policy"):
                self.prepare()
        self.assertFalse((self.root / "out").exists())

    def test_completely_missing_relations_remain_a_source_error_even_if_declared(self):
        self.banks["raw_relations"] = {}
        self.masks[self.scene]["relation_pairs"] = [[i, j] for i in range(3) for j in range(3) if i != j]
        with self.assertRaisesRegex(ValueError, "no non-self directed relations"):
            self.prepare()
        self.assertFalse((self.root / "out").exists())


class PolicyCLITests(unittest.TestCase):
    def test_incompatible_policy_is_rejected_before_loading_model(self):
        for model, data_policy in (({"host_mode": "full"}, POLICY),
                ({"host_mode": "full", "missing_channel_policy": POLICY}, "reject"),
                ({"host_mode": "object_only", "missing_channel_policy": POLICY}, POLICY),
                ({"host_mode": "full", "missing_channel_policy": "implicit"}, "reject")):
            config = {"data": {}, "host": {"model": model}}
            with self.subTest(model=model, data_policy=data_policy), \
                 mock.patch.object(cli, "seed_everything"), \
                 mock.patch.object(cli, "SceneDataset", return_value=SimpleNamespace(missing_channel_policy=data_policy)), \
                 mock.patch.object(cli, "load_host") as load, self.assertRaises(ValueError):
                cli._execute(SimpleNamespace(action="generate-scene"), config, mock.Mock())
            load.assert_not_called()

    def test_matching_declared_policy_reaches_unchanged_generation_config(self):
        config = {"device": "cpu", "data": {}, "host": {"model": {
            "host_mode": "full", "missing_channel_policy": POLICY}}, "run": {"conditions": [{"representation": "masked"}]}}
        with mock.patch.object(cli, "seed_everything"), \
             mock.patch.object(cli, "SceneDataset", return_value=SimpleNamespace(missing_channel_policy=POLICY)), \
             mock.patch.object(cli, "load_host") as load, \
             mock.patch.object(cli, "_generate_scene", return_value={"status": "complete"}) as generate:
            result = cli._execute(SimpleNamespace(action="generate-scene", output=Path("unused")), config, mock.Mock())
        self.assertEqual(result["status"], "complete")
        load.assert_called_once_with(config["host"], device="cpu")
        self.assertEqual(generate.call_args.args[-1], config["run"])


if __name__ == "__main__":
    unittest.main()
