"""CPU object-slot controls through the real train and generation assemblers."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


PACKAGE = Path(__file__).resolve().parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))


def load_host_class():
    # No language-model construction or download is needed for boundary checks.
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = transformers.AutoModelForCausalLM = object
    transformers.StoppingCriteria = object
    transformers.StoppingCriteriaList = list
    peft = types.ModuleType("peft")
    peft.LoraConfig = peft.get_peft_model = object
    spec = importlib.util.spec_from_file_location(
        "scale_interface._object_control_test_host", PACKAGE / "scale_interface/host.py"
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {
        "transformers": transformers, "peft": peft, "einops": types.ModuleType("einops")
    }):
        spec.loader.exec_module(module)
    return module.Graph3DLLM


Graph3DLLM = load_host_class()


class TokenBatch(types.SimpleNamespace):
    def to(self, device):
        return TokenBatch(**{k: v.to(device) for k, v in vars(self).items()})


class Tokenizer:
    pad_token_id = eos_token_id = 0

    def __call__(self, text, **kwargs):
        ids = torch.tensor([[ord(c) for c in text]])
        return TokenBatch(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, ids):
        return "answer"


class CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **kwargs):
        self.captured = kwargs
        return types.SimpleNamespace(loss=kwargs["inputs_embeds"].sum())

    def generate(self, **kwargs):
        self.captured = kwargs
        return torch.tensor([[1]])


class Positions(torch.nn.Module):
    def forward(self, xyz, input_range):
        return torch.cat((xyz, torch.ones_like(xyz[..., :1])), dim=-1)


class SmallHost(Graph3DLLM):
    def __init__(self, mode="full"):
        torch.nn.Module.__init__(self)
        self.host_mode = "object_only"
        self.object_feature_intervention = mode
        self.knn = 0
        self.add_img_token = self.add_scene_token = self.bbox_embed = False
        self.gt_pretrain = self.bidirection = self.low_resource = False
        self.add_pos_emb = True
        self.no_obj = False
        self.max_txt_len = 4
        self.end_sym = "!"
        self.role = ["USER", "ASSISTANT"]
        self.llama_tokenizer = Tokenizer()
        self.llama_model = CaptureModel()
        self.object_proj = torch.nn.Linear(4, 4)
        self.object_img_proj = torch.nn.Linear(4, 4)
        self.edge_proj = torch.nn.Linear(4, 4)
        self.pos_proj = torch.nn.Linear(4, 4)
        self.pos_embedding = Positions()
        with torch.no_grad():
            self.object_proj.weight.copy_(2 * torch.eye(4))
            self.object_proj.bias.fill_(11)
            self.pos_proj.weight.copy_(3 * torch.eye(4))
            self.pos_proj.bias.fill_(7)
        self.register_buffer("ids", torch.arange(20).reshape(5, 4).float() + 100)
        self.register_buffer("p_0_embed", torch.full((2, 4), 201.))
        self.register_buffer("p_1_embed", torch.full((1, 4), 301.))
        self.metric_inputs = []

    def get_objid_embeds(self):
        return self.ids

    def encode_object_feat(self, feat, img_feat, locs, **kwargs):
        return feat, img_feat

    def get_text_emb(self, text, device="cpu"):
        values = self.llama_tokenizer(text).input_ids.to(device).float()
        return torch.stack([values + i for i in range(4)], dim=-1)

    def _metric_quotient_tokens_for_sample(self, **kwargs):
        self.metric_inputs.append(kwargs)
        return kwargs["scene_radius"][:, None, None].expand(-1, 2, 4) + 400


def batch():
    return dict(
        scene_feat=torch.arange(16).reshape(1, 4, 4).float(),
        scene_img_feat=torch.zeros(1, 4, 4),
        scene_locs=torch.arange(24).reshape(1, 4, 6).float(),
        scene_mask=torch.tensor([[True, False, True, True]]),
        obj_ids=torch.tensor([0]), assigned_ids=torch.tensor([[3, 1, 2, 0]]),
        scene_gnn_feats=torch.zeros(1, 0, 4), foreground_ids=[torch.arange(4)],
        scene_radius=torch.tensor([[.3, .6, .9, 1.2]]),
        radius_valid=torch.tensor([[True, False, True, True]]),
    )


def old_assembler(core, embed_obj, embed_img, embed_scene, mask, obj_id,
                  assignment, edges, locs, foreground, feat, metric=None):
    valid = torch.where(mask.bool())[0]
    blocks = []
    for row, assigned in zip(valid, assignment[valid]):
        tokens = [core.ids[row], embed_obj[assigned]]
        if metric is not None:
            tokens.extend(metric[assigned].unbind(0))
        blocks.append(torch.stack(tokens))
    return torch.cat(blocks).to(core.ids.dtype)


class ObjectFeatureControls(unittest.TestCase):
    def run_host(self, mode, *, train=False, packet=False, original=False):
        core = SmallHost(mode)
        if original:
            core.get_object_list_embed = types.MethodType(old_assembler, core)
        source = batch()
        before = {k: v.clone() for k, v in source.items() if isinstance(v, torch.Tensor)}
        question = "Which object is larger, <OBJ000> or <OBJ002>?"
        if packet:
            from scale_interface.physical_stream_host_boundary import HostPhysicalStream
            payload = torch.tensor([[601., 602., 603., 604.], [611., 612., 613., 614.]])
            with HostPhysicalStream(core) as boundary:
                offset = len(question) + (0 if train else 1)
                boundary.arm(payload, offset, mode="train" if train else "generate", placement=packet)
                if train:
                    core.forward_train(**source, questions=[question], answers=["A"])
                else:
                    core.evaluate(**source, custom_prompt=[question])
                audit = boundary.finish()
                self.assertEqual(audit["packet_tokens"], 2)
                self.assertEqual(audit["boundary_calls"], 1)
        elif train:
            core.forward_train(**source, questions=[question], answers=["A"])
        else:
            core.evaluate(**source, custom_prompt=[question])
        for key, value in before.items():
            self.assertTrue(torch.equal(source[key], value), key)
        return core

    def test_full_exactly_matches_previous_train_and_evaluation(self):
        for train in (False, True):
            for packet in (False, "object_end", "question_tail"):
                with self.subTest(train=train, packet=packet):
                    new = self.run_host("full", train=train, packet=packet).llama_model.captured
                    old = self.run_host("full", train=train, packet=packet, original=True).llama_model.captured
                    for field in ("inputs_embeds", "attention_mask") + (("labels",) if train else ()):
                        self.assertTrue(torch.equal(new[field], old[field]), field)

    def test_only_object_slots_change_in_both_paths_and_packet_placements(self):
        for train in (False, True):
            for packet in (False, "object_end", "question_tail"):
                reference_core = self.run_host("full", train=train, packet=packet)
                reference = reference_core.llama_model.captured
                # Packet boundary disables inline metric tokens, but retains the packet.
                block_size = 2 if packet else 4
                slots = torch.tensor([3 + row * block_size for row in range(3)])
                for mode in ("zero_object_slots", "drop_object_features", "permute_object_slots"):
                    with self.subTest(train=train, packet=packet, mode=mode):
                        core = self.run_host(mode, train=train, packet=packet)
                        actual = core.llama_model.captured
                        expected = reference["inputs_embeds"].clone()
                        if mode == "drop_object_features":
                            keep = torch.ones(expected.shape[1], dtype=torch.bool)
                            keep[slots] = False
                            expected = expected[:, keep]
                        elif mode == "zero_object_slots":
                            expected[:, slots] = 0
                        else:
                            # Active source assignment [3, 2, 0] -> [2, 0, 3].
                            expected[:, slots] = expected[:, slots[[1, 2, 0]]]
                        self.assertTrue(torch.equal(actual["inputs_embeds"], expected))
                        for field in ("attention_mask",) + (("labels",) if train else ()):
                            target = reference[field][:, keep] if mode == "drop_object_features" else reference[field]
                            self.assertTrue(torch.equal(actual[field], target), field)
                        if not packet:
                            for field in ("scene_radius", "radius_valid", "scene_mask", "assigned_ids", "projected_objects"):
                                self.assertTrue(torch.equal(core.metric_inputs[0][field], reference_core.metric_inputs[0][field]))
                            self.assertEqual(core.metric_inputs[0]["question"], reference_core.metric_inputs[0]["question"])

    def test_zeroing_is_after_nonzero_projection_bias_and_position(self):
        full = self.run_host("full").llama_model.captured["inputs_embeds"]
        zero = self.run_host("zero_object_slots").llama_model.captured["inputs_embeds"]
        self.assertTrue(torch.all(full[:, [3, 7, 11]] != 0))
        self.assertEqual(torch.count_nonzero(zero[:, [3, 7, 11]]).item(), 0)

    def test_nontrivial_permutation_is_deterministic_and_id_order_independent(self):
        core = SmallHost("permute_object_slots")
        objects = torch.arange(20).reshape(5, 4).float()
        order = torch.tensor([4, 0, 2])
        first = core._object_feature_slots(objects, order)
        self.assertTrue(torch.equal(first, objects[torch.tensor([2, 4, 0])]))
        self.assertTrue(torch.equal(first, core._object_feature_slots(objects, order)))
        reordered = core._object_feature_slots(objects, order[[2, 0, 1]])
        self.assertTrue(torch.equal(reordered, first[[2, 0, 1]]))
        self.assertFalse(torch.any(torch.all(first == objects[order], dim=1)))

    def test_permutation_rejects_empty_singleton_and_duplicate_assignments(self):
        core = SmallHost("permute_object_slots")
        for values in ([], [1], [1, 1]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "distinct active objects"):
                core._object_feature_slots(torch.ones(4, 4), torch.tensor(values, dtype=torch.long))

    def test_gradients_reach_projection_only_when_object_slots_present(self):
        for mode in ("full", "permute_object_slots", "zero_object_slots", "drop_object_features"):
            with self.subTest(mode=mode):
                core = self.run_host(mode, train=True)
                value = core.llama_model.captured["inputs_embeds"].sum()
                if value.requires_grad:
                    value.backward()
                for module in (core.object_proj, core.pos_proj):
                    if mode in ("full", "permute_object_slots"):
                        self.assertGreater(module.weight.grad.abs().sum().item(), 0)
                    else:
                        self.assertIsNone(module.weight.grad)

    def test_default_and_old_no_obj_flag_preserve_full(self):
        core = SmallHost()
        objects, ids = torch.ones(3, 4), torch.tensor([2, 0])
        del core.object_feature_intervention
        core.no_obj = True
        self.assertTrue(torch.equal(core._object_feature_slots(objects, ids), objects[ids]))

    def test_controls_reject_visible_slots_bound_to_masked_source_rows(self):
        for mode in ("zero_object_slots", "drop_object_features", "permute_object_slots"):
            for train in (False, True):
                with self.subTest(mode=mode, train=train):
                    core = SmallHost(mode)
                    source = batch()
                    source["assigned_ids"] = torch.tensor([[1, 3, 2, 0]])
                    with self.assertRaisesRegex(ValueError, "active source row"):
                        if train:
                            core.forward_train(**source, questions=["Question?"], answers=["A"])
                        else:
                            core.evaluate(**source, custom_prompt=["Question?"])
                    self.assertFalse(hasattr(core.llama_model, "captured"))

    def test_full_keeps_previous_behavior_when_assignment_crosses_mask(self):
        for train in (False, True):
            captured = []
            for original in (False, True):
                core = SmallHost("full")
                if original:
                    core.get_object_list_embed = types.MethodType(old_assembler, core)
                source = batch()
                source["assigned_ids"] = torch.tensor([[1, 3, 2, 0]])
                if train:
                    core.forward_train(**source, questions=["Question?"], answers=["A"])
                else:
                    core.evaluate(**source, custom_prompt=["Question?"])
                captured.append(core.llama_model.captured)
            for field in ("inputs_embeds", "attention_mask") + (("labels",) if train else ()):
                self.assertTrue(torch.equal(captured[0][field], captured[1][field]), field)

    def test_controls_accept_sparse_masks_with_active_source_assignments(self):
        for mode in ("zero_object_slots", "drop_object_features", "permute_object_slots"):
            for train in (False, True):
                with self.subTest(mode=mode, train=train):
                    # The shared batch masks source row 1 and maps [0,2,3] to [3,2,0].
                    core = self.run_host(mode, train=train)
                    self.assertTrue(hasattr(core.llama_model, "captured"))

    def test_unknown_and_non_object_only_controls_rejected_before_model_loading(self):
        config = json.loads((PACKAGE / "scale_interface/configs/object_only.json").read_text())["host"]
        from scale_interface.loading import namespace
        for change in ({"object_feature_intervention": "unknown"},
                       {"object_feature_intervention": "zero_object_slots", "host_mode": "legacy"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Graph3DLLM(namespace({"model": {**config["model"], **change},
                                     "lora": config["lora"], "debug": False}))


if __name__ == "__main__":
    unittest.main()
