"""CPU full-host identity, projector loading, and causal-boundary checks."""
import copy
import sys
import types
import unittest
from unittest import mock

import torch

from test_object_feature_controls import SmallHost
from scale_interface.loading import load_host
from scale_interface.physical_stream_host_boundary import HostPhysicalStream


class RecordingProjection(torch.nn.Linear):
    def __init__(self):
        super().__init__(512, 4)
        self.shapes = []
        with torch.no_grad():
            self.weight.copy_(torch.arange(2048).reshape(4, 512))
            self.bias.fill_(7)

    def forward(self, value):
        self.shapes.append(tuple(value.shape))
        return super().forward(value)


class FullHost(SmallHost):
    def __init__(self):
        super().__init__()
        self.host_mode = "full"
        self.knn, self.max_knn = 2, 5
        self.add_img_token = self.gt_pretrain = True
        self.nms = self.nn_distance = False
        self.add_pos_emb = False
        self.object_img_proj = torch.nn.Linear(1024, 4)
        self.edge_proj = RecordingProjection()

    def _metric_quotient_tokens_for_sample(self, **kwargs):
        return None


def full_batch():
    edges = torch.zeros(1, 4, 4, 512)
    for source in range(4):
        for target in range(4):
            if source != target:
                edges[0, source, target, source * 4 + target] = 1
    locations = torch.zeros(1, 4, 6)
    locations[0, :, 0] = torch.tensor([0., 1., 4., 10.])
    return dict(scene_feat=torch.arange(16).reshape(1, 4, 4).float(),
                scene_img_feat=torch.ones(1, 4, 1024), scene_locs=locations,
                scene_mask=torch.ones(1, 4, dtype=torch.bool), obj_ids=torch.tensor([0]),
                assigned_ids=torch.tensor([[3, 1, 0, 2]]), scene_gnn_feats=edges,
                foreground_ids=[torch.arange(4)], scene_radius=torch.ones(1, 4),
                radius_valid=torch.ones(1, 4, dtype=torch.bool))


class FullHostTests(unittest.TestCase):
    def assemble(self, core, source):
        core._apply_host_mode(source["scene_img_feat"], source["scene_gnn_feats"], source["scene_mask"])
        return core.get_object_list_embed(
            source["scene_feat"][0], torch.arange(16).reshape(4, 4).float() + 50, None,
            source["scene_mask"][0], 0, source["assigned_ids"][0], source["scene_gnn_feats"][0],
            source["scene_locs"][0], torch.arange(4), source["scene_feat"][0])

    def test_source_target_lookup_follows_runtime_neighbors_after_reordering(self):
        core, source = FullHost(), full_batch()
        blocks = self.assemble(core, source).reshape(4, 8, 4)
        neighbors = {0: [1, 2], 1: [0, 2], 2: [1, 0], 3: [2, 1]}
        for visible, raw in enumerate(source["assigned_ids"][0].tolist()):
            self.assertTrue(torch.equal(blocks[visible, 0], core.ids[visible]))
            for slot, target in enumerate(neighbors[raw]):
                self.assertTrue(torch.equal(blocks[visible, 2 + 3 * slot], source["scene_feat"][0, raw]))
                self.assertTrue(torch.equal(blocks[visible, 4 + 3 * slot], source["scene_feat"][0, target]))
                expected_edge = core.edge_proj.weight[:, raw * 4 + target] + core.edge_proj.bias
                self.assertTrue(torch.equal(blocks[visible, 3 + 3 * slot], expected_edge))
        self.assertEqual(core.edge_proj.shapes, [(4, 2, 512)])

    def test_unselected_missing_edges_allowed_but_selected_missing_never_projected(self):
        source = full_batch()
        source["scene_gnn_feats"][0, 0, 3] = 0  # Not a selected neighbor.
        self.assemble(FullHost(), source)
        source["scene_gnn_feats"][0, 0, 1] = 0
        core = FullHost()
        with self.assertRaisesRegex(ValueError, "missing.*zero imputation"):
            self.assemble(core, source)
        self.assertEqual(core.edge_proj.shapes, [])

    def test_small_scenes_missing_images_and_old_slot_caches_are_rejected(self):
        for kind in ("small", "image", "slots"):
            source = full_batch()
            if kind == "small":
                source["scene_mask"][0, 2:] = False
            elif kind == "image":
                source["scene_img_feat"][0, 1] = 0
            else:
                source["scene_gnn_feats"] = torch.ones(1, 20, 512)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.assemble(FullHost(), source)

    def test_geometry_stays_fp32_even_with_half_projectors(self):
        core, source = FullHost().half(), full_batch()
        source["scene_locs"][0, 1, 0] = 4.001
        source["scene_locs"][0, 2, 0] = 4.002
        objects, images, edges, locations = core._cast_scene_inputs(
            source["scene_feat"], source["scene_img_feat"], source["scene_gnn_feats"], source["scene_locs"])
        self.assertEqual(objects.dtype, torch.float16)
        self.assertEqual(images.dtype, torch.float16)
        self.assertEqual(edges.dtype, torch.float32)
        self.assertEqual(locations.dtype, torch.float32)
        self.assertTrue(torch.equal(locations, source["scene_locs"]))
        self.assertNotEqual(float(locations[0, 1, 0]), float(locations[0, 2, 0]))

    def test_equal_distance_neighbor_ties_do_not_depend_on_visible_order(self):
        core, source = FullHost(), full_batch()
        source["scene_locs"][0, :, 0] = torch.tensor([-1., 0., 1., 3.])
        first = self.assemble(core, source).reshape(4, 8, 4)
        old_ids = source["assigned_ids"][0].tolist()
        source["assigned_ids"][0] = torch.tensor([0, 2, 1, 3])
        second = self.assemble(core, source).reshape(4, 8, 4)
        for row, raw in enumerate(source["assigned_ids"][0].tolist()):
            self.assertTrue(torch.equal(second[row, 1:], first[old_ids.index(raw), 1:]))

    def test_full_train_and_generation_packet_splice_preserves_native_prefix(self):
        for train in (False, True):
            for placement in ("object_end", "question_tail"):
                core, source = FullHost(), full_batch()
                question = "Which object?"
                if train:
                    core.forward_train(**source, questions=[question], answers=["A"])
                else:
                    core.evaluate(**source, custom_prompt=[question])
                native = core.llama_model.captured
                packet = torch.full((2, 4), 12345.)
                with HostPhysicalStream(core) as boundary:
                    offset = len(question) + (0 if train else 1)
                    boundary.arm(packet, offset, mode="train" if train else "generate", placement=placement)
                    if train:
                        core.forward_train(**source, questions=[question], answers=["A"])
                    else:
                        core.evaluate(**source, custom_prompt=[question])
                    audit = boundary.finish()
                    index = boundary.index
                actual = core.llama_model.captured
                self.assertEqual(index, 2 + 4 * 8 + (1 + offset if placement == "question_tail" else 0))
                self.assertEqual(audit["packet_tokens"], 2)
                self.assertEqual(audit["native_parity_assertions"], 3 if train else 2)
                for field in ("inputs_embeds", "attention_mask") + (("labels",) if train else ()):
                    recovered = torch.cat((actual[field][:, :index], actual[field][:, index + 2:]), dim=1)
                    self.assertTrue(torch.equal(recovered, native[field]), (train, placement, field))
                self.assertEqual(core.edge_proj.shapes, [(4, 2, 512), (4, 2, 512)])


class FrozenLoadHost(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.object_proj = torch.nn.Linear(1024, 4)
        self.object_img_proj = torch.nn.Linear(1024, 4)
        self.edge_proj = torch.nn.Linear(512, 4)
        self.pos_proj = torch.nn.Linear(4, 4)
        self.p_0_embed = self.p_1_embed = torch.zeros(1, 4)
        self.requires_grad_(False)


class FullCheckpointTests(unittest.TestCase):
    def config(self):
        return {"feature_mode": "native", "host_checkpoint": "host.pt", "lora": {}, "model": {
            "host_mode": "full", "bidirection": False, "source_feature_dim": 1024, "input_dim": 1024,
            "add_img_token": True, "img_input_dim": 1024, "gt_pretrain": True, "knn": 2, "max_knn": 5,
            "add_scene_token": False, "bbox_embed": False, "add_pos_emb": True}}

    def load(self, state):
        module = types.ModuleType("scale_interface.host")
        module.Graph3DLLM = FrozenLoadHost
        with mock.patch.dict(sys.modules, {"scale_interface.host": module}), \
             mock.patch("scale_interface.loading.torch.load", return_value={"model": state}):
            return load_host(self.config(), device="cpu")

    def test_frozen_image_edge_and_position_projectors_are_mandatory(self):
        complete = FrozenLoadHost().state_dict()
        self.load(complete)
        for name in ("object_img_proj.weight", "object_img_proj.bias", "edge_proj.weight", "edge_proj.bias", "pos_proj.weight"):
            state = copy.copy(complete)
            del state[name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "missing host tensors"):
                self.load(state)


if __name__ == "__main__":
    unittest.main()
