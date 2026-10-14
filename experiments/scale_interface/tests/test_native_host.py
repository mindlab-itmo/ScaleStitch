"""CPU checks of explicit native host inputs and unchanged deployment loading."""
import copy
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface.loading import load_host
from test_object_feature_controls import Graph3DLLM


class SmallNativeHost(Graph3DLLM):
    def __init__(self, config=None):
        torch.nn.Module.__init__(self)
        self.config = config
        self.object_proj = torch.nn.Linear(1024, 4)
        self.encoder_layer = torch.nn.Linear(2, 2)
        self.p_0_embed = torch.ones(1, 4)
        self.p_1_embed = torch.ones(1, 4)
        self.scalestitch = None
        self._scalestitch_deployment_bundle = False
        self._scalestitch_freeze_host = False


class NativeHostTests(unittest.TestCase):
    def config(self):
        return {"feature_mode": "native", "model": {"host_mode": "object_only", "bidirection": False,
                "source_feature_dim": 1024, "input_dim": 1024}, "lora": {}, "host_checkpoint": "host.pt"}

    def load(self, config, change=None):
        source = SmallNativeHost()
        with torch.no_grad():
            source.object_proj.weight.fill_(.125)
            source.object_proj.bias.copy_(torch.arange(4).float())
        state = {key: value for key, value in source.state_dict().items() if not key.startswith("encoder_layer.")}
        if change:
            change(state)
        host_module = types.ModuleType("scale_interface.host")
        host_module.Graph3DLLM = SmallNativeHost
        with mock.patch.dict("sys.modules", {"scale_interface.host": host_module}), \
             mock.patch("scale_interface.loading.torch.load", side_effect=[{"model": state}, {}]) as read, \
             mock.patch("scale_interface.loading.ScaleStitchDeployment", return_value=torch.nn.Identity()) as bridge:
            core = load_host(config, device="cpu")
        return core, source, read, bridge

    def test_native_loads_only_host_checkpoint_and_creates_no_bridge(self):
        config = self.config()
        original = copy.deepcopy(config)
        core, source, read, bridge = self.load(config)
        self.assertEqual(config, original)
        self.assertEqual(read.call_count, 1)
        read.assert_called_once_with("host.pt", map_location="cpu", weights_only=True)
        bridge.assert_not_called()
        self.assertIsNone(core.scalestitch)
        self.assertFalse(core._scalestitch_deployment_bundle)
        self.assertEqual(set(core.state_dict()), set(source.state_dict()))
        self.assertTrue(torch.equal(core.object_proj.weight, source.object_proj.weight))
        self.assertTrue(torch.equal(core.object_proj.bias, source.object_proj.bias))
        self.assertFalse(any(parameter.requires_grad for parameter in core.parameters()))
        self.assertFalse(core.training)

    def test_native_encoding_is_exact_original_l2_without_radius_conditioning(self):
        core, _, _, _ = self.load(self.config())
        features = torch.arange(2048).reshape(1, 2, 1024).float()
        images = torch.zeros(1, 2, 4)
        locations = torch.zeros(1, 2, 6)
        actual, _ = core.encode_object_feat(features, images, locations)
        with_radius, _ = core.encode_object_feat(features, images, locations,
                                                object_radius=torch.tensor([[.1, 9.]]),
                                                radius_valid=torch.tensor([[True, True]]),
                                                scene_mask=torch.tensor([[True, True]]))
        self.assertTrue(torch.equal(actual, F.normalize(features, dim=-1)))
        self.assertTrue(torch.equal(actual, with_radius))
        expected = F.linear(F.normalize(features, dim=-1), core.object_proj.weight, core.object_proj.bias)
        self.assertTrue(torch.equal(core.object_proj(actual), expected))

    def test_default_and_explicit_deployment_keep_previous_loading_path(self):
        for mode in (None, "deployment"):
            config = self.config()
            if mode is None:
                del config["feature_mode"]
            else:
                config["feature_mode"] = mode
            config["deployment"] = "deployment.pt"
            config["model"]["source_feature_dim"] = 1280
            with self.subTest(mode=mode):
                core, _, read, bridge = self.load(config)
                self.assertEqual(read.call_count, 2)
                self.assertEqual(read.call_args.args[0], "deployment.pt")
                bridge.assert_called_once_with({}, execution_mode="inference")
                self.assertTrue(core._scalestitch_deployment_bundle)

    def test_native_rejects_conflicting_and_ambiguous_config_before_construction(self):
        changes = [{"feature_mode": "unknown"}, {"deployment": None}, {"deployment": "deployment.pt"},
                   {"aqt": {}}, {"aqt": None}]
        configs = [{**self.config(), **change} for change in changes]
        for key in ("source_feature_dim", "input_dim"):
            for value in (1280, 512, 1024., True, None):
                config = self.config()
                config["model"][key] = value
                configs.append(config)
            config = self.config()
            del config["model"][key]
            configs.append(config)
        for config in configs:
            with self.subTest(config=config):
                host_module = types.ModuleType("scale_interface.host")
                host_module.Graph3DLLM = mock.Mock(side_effect=AssertionError("model constructed"))
                with mock.patch.dict("sys.modules", {"scale_interface.host": host_module}), \
                     mock.patch("scale_interface.loading.torch.load") as read, self.assertRaises(ValueError):
                    load_host(config, device="cpu")
                host_module.Graph3DLLM.assert_not_called()
                read.assert_not_called()

    def test_native_keeps_required_checkpoint_validation(self):
        changes = (lambda state: state.pop("object_proj.weight"),
                   lambda state: state.update({"object_proj.weight": torch.zeros(4, 1280)}),
                   lambda state: state.update({"object_proj.weight": torch.full((4, 1024), float("nan"))}),
                   lambda state: state.update({"unexpected": torch.zeros(1)}))
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(self.config(), change)


if __name__ == "__main__":
    unittest.main()
