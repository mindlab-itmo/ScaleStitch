"""CPU contracts for the paired continuation core; no checkpoint or GPU is loaded."""
from dataclasses import replace
from pathlib import Path
import random
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface import natural_qa as qa
from scale_interface import packet_lora as pl
from scale_interface.scene_measurements import serialize_scene_radius_records
from test_natural_qa import CharacterTokenizer, NaturalHost, TokenBatch, dataset


class NativeTokenizer(CharacterTokenizer):
    def __call__(self, text, **kwargs):
        pieces = re.findall(r"<OBJ[0-9]{3}>|.", text, flags=re.S)
        ids = [1000 + int(x[4:7]) if x.startswith("<OBJ") else ord(x) for x in pieces]
        if kwargs.get("return_tensors") == "pt":
            values = torch.tensor([ids], dtype=torch.long)
            return TokenBatch(input_ids=values, attention_mask=torch.ones_like(values))
        return {"input_ids": ids}


class PeftLinear(torch.nn.Module):
    """The PEFT LoRA collection contract, with actual differentiable A/B weights."""
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")

    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Linear(4, 4, bias=False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False)})
        self.lora_embedding_A = torch.nn.ParameterDict()
        self.lora_embedding_B = torch.nn.ParameterDict()
        self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(0.0)})
        self.active_adapters = ["default"]
        self.merged = self.disable_adapters = False
        self.use_dora = {"default": False}
        self.forward_modes = []

    def forward(self, values):
        self.forward_modes.append((self.training, self.lora_dropout["default"].training))
        return self.base_layer(values) + self.lora_B["default"](
            self.lora_A["default"](self.lora_dropout["default"](values)))


class TinyPeftModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(1100, 4)
        self.layer = PeftLinear()
        self.lm_head = torch.nn.Linear(4, 1100, bias=False)
        self.config = types.SimpleNamespace(max_position_embeddings=4096)
        self.forward_modes = []
        self.gradient_checkpointing = False

    def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict, logits_to_keep):
        assert use_cache is False and return_dict is True and 1 <= logits_to_keep <= 32
        self.forward_modes.append((self.training, self.layer.lora_dropout["default"].training))
        self.last_inputs = inputs_embeds.detach().clone()
        self.last_logit_count = logits_to_keep
        values = torch.stack([inputs_embeds[:, :position + 1].mean(1)
                              for position in range(inputs_embeds.shape[1] - logits_to_keep, inputs_embeds.shape[1])], 1)
        hidden = checkpoint(self.layer, values, use_reentrant=False) if self.gradient_checkpointing else self.layer(values)
        return types.SimpleNamespace(logits=self.lm_head(hidden))

    def generate(self, **kwargs):
        raise AssertionError("continuation must never generate answers")


class DifferentiableHost(NaturalHost):
    def __init__(self):
        super().__init__()
        self.llama_model = TinyPeftModel()
        self.llama_tokenizer = NativeTokenizer()
        # A misleading name must never make an unrelated parameter trainable.
        self.lora_A_decoy = torch.nn.Linear(4, 4)

    def get_text_emb(self, text, device="cpu"):
        ids = self.llama_tokenizer(text, return_tensors="pt").input_ids.to(device)
        return self.llama_model.embed_tokens(ids)


def example(core, representation="lexical", **options):
    data = dataset(rows=1)
    data.anno[0]["prompt"] = "Which is larger, <OBJ000> or <OBJ002>?"
    _, _, question, batch = qa._cpu_input(data, 0)
    condition = qa.NaturalQACondition(representation)
    packet = qa._packet(batch, condition, 100)
    prefix = pl.capture_prefix(core, batch, question, packet, condition, **options)
    target = pl.native_target(core.llama_tokenizer, 0, assigned_ids=batch["assigned_ids"],
                              scene_mask=batch["scene_mask"], eligible_source_ids=(0, 2))
    return prefix, target, (batch, question, packet, condition)


def identity(arm="lexical"):
    return {"arm": arm, "source_hashes": {"host": "a" * 64, "bridge": "b" * 64},
            "input_hashes": {"questions": "c" * 64, "scene_rows": "d" * 64},
            "protocol_hash": "e" * 64}


def frozen_fingerprint(core):
    return qa._fingerprint({name: p for name, p in core.named_parameters() if not p.requires_grad})


class PacketLoraTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_unfreezes_only_adapter_collection_identities_without_casting(self):
        core = DifferentiableHost().requires_grad_(False)
        dtypes = {name: p.dtype for name, p in core.named_parameters()}
        selected = pl.unfreeze_lora_only(core, expected_numel=16)
        expected = {id(p) for collection in (core.llama_model.layer.lora_A,
                                            core.llama_model.layer.lora_B) for p in collection.parameters()}
        self.assertEqual({id(p) for p in selected.parameters}, expected)
        self.assertEqual({id(p) for p in core.parameters() if p.requires_grad}, expected)
        self.assertEqual(selected.numel, 16)
        self.assertEqual({name: p.dtype for name, p in core.named_parameters()}, dtypes)
        self.assertFalse(core.lora_A_decoy.weight.requires_grad)

    def test_rejects_missing_or_ambiguous_adapter_collections_before_unfreeze(self):
        for change in ("bad_count", "missing_declaration", "inactive", "embedding", "merged", "base_alias", "storage_alias"):
            with self.subTest(change=change):
                core = DifferentiableHost().requires_grad_(False)
                layer = core.llama_model.layer
                if change == "missing_declaration":
                    layer.adapter_layer_names = ()
                elif change == "inactive":
                    layer.active_adapters = ["other"]
                elif change == "embedding":
                    layer.lora_embedding_A["default"] = torch.nn.Parameter(torch.ones(2, 4), requires_grad=False)
                elif change == "merged":
                    layer.merged = True
                elif change == "base_alias":
                    layer.base_layer.weight = layer.lora_A["default"].weight
                elif change == "storage_alias":
                    layer.base_layer.weight = torch.nn.Parameter(layer.lora_A["default"].weight.detach(), requires_grad=False)
                with self.assertRaises(ValueError):
                    pl.unfreeze_lora_only(core, expected_numel=17 if change == "bad_count" else 16)
                self.assertFalse(any(p.requires_grad for p in core.parameters()))

    def test_loader_reuses_frozen_host_and_rejects_other_feature_protocols(self):
        config = {"model": {"host_mode": "object_only", "bidirection": False, "knn": 0,
                            "add_img_token": False, "add_scene_token": False, "bbox_embed": False,
                            "source_feature_dim": 1280, "input_dim": 1024, "use_lora": True},
                  "feature_mode": "deployment", "deployment": "bridge.pt", "host_checkpoint": "host.pt"}
        core = DifferentiableHost().requires_grad_(False)
        with mock.patch("scale_interface.loading.load_host", return_value=core) as loader:
            actual, selected = pl.load_packet_lora_host(config, device="cpu", expected_numel=16)
        self.assertIs(actual, core)
        self.assertEqual(selected.numel, 16)
        self.assertEqual(loader.call_args.kwargs, {"device": "cpu"})
        with self.assertRaises(ValueError):
            pl.load_packet_lora_host({**config, "feature_mode": "native"}, device="cpu")

    def test_prefix_uses_actual_host_boundary_without_answers_or_generation(self):
        core = DifferentiableHost()
        core.train()
        core.llama_model.layer.lora_dropout.eval()
        modes = [m.training for m in core.modules()]
        prefix, target, (batch, question, packet, condition) = example(core)
        expected, _ = qa._run_one(core, batch, question, packet, condition, codec=None, random_codec=None,
                                  max_new_tokens=32, maximum_length=2502, placement="question_tail",
                                  capture_only=True)
        self.assertEqual(prefix.budget, expected)
        self.assertEqual(target.visible_id, 2)
        self.assertEqual([m.training for m in core.modules()], modes)
        self.assertFalse(prefix.inputs_embeds.requires_grad)
        self.assertIsNone(prefix.inputs_embeds.grad_fn)
        self.assertTrue(torch.all(prefix.attention_mask == 1))
        self.assertEqual(core.llama_model.forward_modes, [])
        self.assertFalse(hasattr(core, "_rps_boundary_active"))
        with self.assertRaises(ValueError):
            pl.capture_prefix(core, {**batch, "answers": "secret"}, question, packet, condition)

    def test_target_mapping_is_stable_eligible_visible_and_single_native_token(self):
        core = DifferentiableHost()
        prefix, target, (batch, _, _, _) = example(core)
        self.assertEqual((target.source_id, target.visible_id, target.token_id), (0, 2, 1002))
        self.assertEqual(target.eligible_token_ids, (1002, 1000))
        before = prefix.inputs_embeds.clone()
        other = pl.native_target(core.llama_tokenizer, 2, assigned_ids=batch["assigned_ids"],
                                 scene_mask=batch["scene_mask"], eligible_source_ids=(0, 2))
        self.assertEqual((other.visible_id, other.token_id), (0, 1000))
        self.assertTrue(torch.equal(prefix.inputs_embeds, before))
        for options in ({"source_id": 1}, {"scene_mask": torch.tensor([[False, True, True]])},
                        {"assigned_ids": torch.tensor([[2, 2, 0]])}, {"eligible_source_ids": (0, 0)}):
            args = dict(source_id=0, assigned_ids=batch["assigned_ids"], scene_mask=batch["scene_mask"],
                        eligible_source_ids=(0, 2))
            with self.subTest(options=options), self.assertRaises(ValueError):
                pl.native_target(core.llama_tokenizer, **{**args, **options})
        with self.assertRaises(ValueError):
            pl.native_target(CharacterTokenizer(), 0, assigned_ids=batch["assigned_ids"],
                             scene_mask=batch["scene_mask"], eligible_source_ids=(0, 2))

    def test_actual_backward_and_update_touch_only_lora_and_restore_modes(self):
        core = DifferentiableHost().requires_grad_(False)
        selected = pl.unfreeze_lora_only(core)
        optimizer = torch.optim.SGD(selected.parameters, lr=0.001, momentum=0.9)
        prefix, target, _ = example(core)
        core.eval()
        modes = [m.training for m in core.modules()]
        frozen = frozen_fingerprint(core)
        before = [p.detach().clone() for p in selected.parameters]
        loss = pl.next_token_step(core, prefix, pl.short_answer_target(core, target), selected, optimizer)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(core.llama_model.forward_modes, [(True, True)])
        self.assertEqual([m.training for m in core.modules()], modes)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected.parameters))
        self.assertTrue(any(torch.count_nonzero(p.grad).item() for p in selected.parameters))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, selected.parameters)))
        self.assertEqual(frozen_fingerprint(core), frozen)
        self.assertTrue(all(p.grad is None for p in core.parameters() if not p.requires_grad))

    def test_loss_is_full_vocabulary_native_next_token_ce(self):
        core = DifferentiableHost().requires_grad_(False)
        selected = pl.unfreeze_lora_only(core)
        prefix, target, _ = example(core, "absent")
        loss = pl.native_id_loss(core, prefix, target, selected)
        with torch.no_grad():
            logits = core.llama_model.lm_head(core.llama_model.layer(prefix.inputs_embeds.mean(1)))
            expected = torch.logsumexp(logits.float(), -1)[0] - logits[0, 1002].float()
        self.assertAlmostEqual(float(loss.detach()), float(expected), places=5)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(prefix.budget.packet_tokens, 0)

    def test_checkpoint_recomputation_keeps_lora_dropout_in_train_mode(self):
        core = DifferentiableHost().requires_grad_(False)
        selected = pl.unfreeze_lora_only(core)
        prefix, target, _ = example(core)
        core._scalestitch_freeze_host, core.scalestitch = True, None
        core.eval()
        core.llama_model.gradient_checkpointing = True
        core.llama_model.layer.lora_dropout["default"].p = 0.4
        optimizer = torch.optim.SGD(selected.parameters, lr=0.001)
        pl.next_token_step(core, prefix, pl.short_answer_target(core, target), selected, optimizer)
        self.assertGreaterEqual(len(core.llama_model.layer.forward_modes), 2)
        self.assertTrue(all(mode == (True, True) for mode in core.llama_model.layer.forward_modes))
        self.assertFalse(core.llama_model.training)

    def test_short_target_supervises_host_ending_after_correct_id_without_prompt_labels(self):
        core = DifferentiableHost().requires_grad_(False)
        selected = pl.unfreeze_lora_only(core)
        prefix, target, _ = example(core)
        with torch.no_grad():
            core.llama_model.layer.base_layer.weight.copy_(torch.eye(4))
            core.llama_model.layer.lora_A["default"].weight.fill_(0.1)
            core.llama_model.layer.lora_B["default"].weight.fill_(0.1)
            core.llama_model.lm_head.weight.zero_()
            core.llama_model.lm_head.weight[1002].fill_(1)
            core.llama_model.lm_head.weight[33].fill_(-1)
        short = pl.short_answer_target(core, target)
        self.assertEqual(short.token_ids, (1002, 33))
        prefix_before = prefix.inputs_embeds.clone()
        id_loss = pl.native_id_loss(core, prefix, target, selected)
        self.assertLess(float(id_loss.detach()), 0.001)
        with pl.lora_training_mode(core):
            loss = pl.next_token_loss(core, prefix, short, selected)
            loss.backward()
        self.assertGreater(float(loss.detach()), 1)
        self.assertEqual(core.llama_model.last_logit_count, 2)
        self.assertTrue(torch.equal(core.llama_model.last_inputs[:, :prefix.budget.prefix_tokens], prefix_before))
        self.assertTrue(torch.equal(core.llama_model.last_inputs[:, -1], core.llama_model.embed_tokens.weight[1002][None]))
        self.assertTrue(all(p.grad is None for p in core.parameters() if not p.requires_grad))
        self.assertTrue(any(torch.count_nonzero(p.grad).item() for p in selected.parameters))
        core.end_sym = "!!"
        self.assertEqual(pl.short_answer_target(core, target).token_ids, (1002, 33, 33))
        with self.assertRaises(ValueError):
            pl.next_token_loss(core, prefix, short, selected)
        with self.assertRaises(ValueError):
            pl.next_token_step(core, prefix, target, selected, torch.optim.SGD(selected.parameters, lr=0.001))

    def test_accumulation_context_covers_checkpointed_backward_and_restores_caller_modes(self):
        core = DifferentiableHost().requires_grad_(False)
        selected = pl.unfreeze_lora_only(core)
        prefix, target, _ = example(core)
        short = pl.short_answer_target(core, target)
        core.eval()
        core.llama_model.gradient_checkpointing = True
        core.llama_model.layer.lora_dropout["default"].p = 0.4
        with pl.lora_training_mode(core):
            for _ in range(2):
                (pl.next_token_loss(core, prefix, short, selected) / 2).backward()
        self.assertTrue(all(mode == (True, True) for mode in core.llama_model.layer.forward_modes))
        self.assertFalse(core.llama_model.training)
        self.assertTrue(any(torch.count_nonzero(p.grad).item() for p in selected.parameters))

    def test_bad_optimizer_trainables_or_attached_prefix_are_rejected_before_update(self):
        core = DifferentiableHost().requires_grad_(False)
        selected = pl.unfreeze_lora_only(core)
        prefix, target, _ = example(core)
        short = pl.short_answer_target(core, target)
        bad = torch.optim.SGD([*selected.parameters, core.object_proj.weight], lr=0.001)
        with self.assertRaises(ValueError):
            pl.next_token_step(core, prefix, short, selected, bad)
        with self.assertRaises(ValueError):
            pl.next_token_loss(core, replace(prefix, inputs_embeds=prefix.inputs_embeds.requires_grad_()), short, selected)
        prefix.inputs_embeds.requires_grad_(False)
        core.object_proj.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            pl.next_token_loss(core, prefix, short, selected)
        self.assertEqual(core.llama_model.forward_modes, [])

    def test_budget_and_condition_failures_restore_boundary(self):
        core = DifferentiableHost()
        prefix, _, args = example(core)
        batch, question, packet, condition = args
        with self.assertRaisesRegex(ValueError, "exceeds budget"):
            pl.capture_prefix(core, batch, question, packet, condition,
                              maximum_length=prefix.budget.prefix_tokens + 31)
        for invalid in (replace(condition, mode="swapped"), replace(condition, packet_scope="queried_pair"),
                        replace(condition, representation="masked")):
            with self.assertRaises(ValueError):
                pl.capture_prefix(core, batch, question, packet, invalid)
        self.assertFalse(hasattr(core, "_rps_boundary_active"))
        self.assertEqual(core.llama_model.forward_modes, [])

    def test_incomplete_or_incorrect_packet_is_rejected_in_both_arms(self):
        core = DifferentiableHost()
        _, _, (batch, question, packet, condition) = example(core)
        truncated = serialize_scene_radius_records(packet.records[:2])
        altered = serialize_scene_radius_records((replace(packet.records[0], radius_cm=999), *packet.records[1:]))
        for arm in (condition, replace(condition, representation="absent")):
            for invalid in (truncated, altered):
                with self.subTest(arm=arm.representation), self.assertRaises(ValueError):
                    pl.capture_prefix(core, batch, question, invalid, arm)
        self.assertEqual(core.llama_model.forward_modes, [])

    def test_height_packet_provenance_capture_target_and_real_lora_gradient(self):
        core = DifferentiableHost().requires_grad_(False)
        selected = pl.unfreeze_lora_only(core)
        data = dataset(rows=1)
        data.anno[0]["prompt"] = "Which is taller, <OBJ000> or <OBJ002>?"
        data.attributes["scene0000_00"]["locs"][:, 5] = torch.tensor([1.23, .7, .9])
        _, scene, question, batch = qa._cpu_input(data, 0)
        source = {"measurement": "full_object_axis_aligned_height", "input_id": "synthetic-heights",
                  "source_reference": "CPU fixture Z extents", "verified_scenes": {scene: True}}
        condition = qa.NaturalQACondition(measurement="height")
        packet = qa._packet(batch, condition, 100, measurement_source=source, scene_id=scene)
        frozen = frozen_fingerprint(core)
        target = pl.native_target(core.llama_tokenizer, 0, assigned_ids=batch["assigned_ids"],
                                  scene_mask=batch["scene_mask"], eligible_source_ids=(0, 2))
        self.assertEqual((target.visible_id, target.token_id), (2, 1002))
        for arm in (condition, qa.NaturalQACondition("absent")):
            prefix = pl.capture_prefix(core, batch, question, packet, arm,
                                       measurement_source=source, scene_id=scene)
            self.assertEqual(prefix.budget.packet_tokens > 0, arm.representation == "lexical")
            with pl.lora_training_mode(core):
                loss = pl.next_token_loss(core, prefix, pl.short_answer_target(core, target), selected)
                loss.backward()
            self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad).item()
                                for p in selected.parameters))
            self.assertTrue(all(p.grad is None for p in core.parameters() if not p.requires_grad))
            with self.assertRaises(ValueError):
                pl.capture_prefix(core, batch, question, packet, arm, scene_id=scene)
            with self.assertRaises(ValueError):
                pl.capture_prefix(core, batch, question, packet, arm, measurement_source={
                    **source, "verified_scenes": {scene: False}}, scene_id=scene)
        self.assertEqual(frozen_fingerprint(core), frozen)
        self.assertEqual(core.llama_model.forward_modes, [(True, True), (True, True)])

    def test_compact_atomic_checkpoint_resume_restores_lora_optimizer_and_rng(self):
        core = DifferentiableHost().requires_grad_(False)
        selected = pl.unfreeze_lora_only(core)
        optimizer = torch.optim.SGD(selected.parameters, lr=0.001, momentum=0.9)
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=1)
        prefix, target, _ = example(core)
        short = pl.short_answer_target(core, target)
        pl.next_token_step(core, prefix, short, selected, optimizer, scaler=scaler)
        optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.pt"
            source.write_bytes(b"immutable source")
            destination = root / "continuation"
            saved = [p.detach().clone() for p in selected.parameters]
            frozen = frozen_fingerprint(core)
            pl.save_checkpoint(destination, core, selected, optimizer, step=1, row_position=7,
                               identity=identity(), source_checkpoint=source, scaler=scaler, pending_microbatches=0)
            saved_scaler = scaler.state_dict()
            draws = (random.random(), np.random.rand(), torch.rand(3))
            pl.next_token_step(core, prefix, short, selected, optimizer, scaler=scaler)
            optimizer.zero_grad(set_to_none=True)
            continued = [p.detach().clone() for p in selected.parameters]
            state = pl.load_checkpoint(destination, core, selected, optimizer, identity=identity(), scaler=scaler)
            self.assertEqual(state, {"step": 1, "row_position": 7, "path": str(destination / "last.pt")})
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(saved, selected.parameters)))
            self.assertEqual((random.random(), np.random.rand()), draws[:2])
            self.assertTrue(torch.equal(torch.rand(3), draws[2]))
            self.assertEqual(scaler.state_dict(), saved_scaler)
            pl.next_token_step(core, prefix, short, selected, optimizer, scaler=scaler)
            optimizer.zero_grad(set_to_none=True)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(continued, selected.parameters)))
            self.assertEqual(frozen_fingerprint(core), frozen)
            payload = torch.load(destination / "last.pt", weights_only=True)
            self.assertEqual(set(payload["lora"]), set(selected.names))
            self.assertNotIn("model", payload)
            self.assertTrue(optimizer.state)
            pl.save_checkpoint(destination, core, selected, optimizer, step=2, row_position=8,
                               identity=identity(), source_checkpoint=source, scaler=scaler, pending_microbatches=0)
            self.assertEqual(sorted(p.name for p in destination.iterdir()), ["last.pt", "previous.pt"])
            self.assertEqual(torch.load(destination / "previous.pt", weights_only=True)["step"], 1)
            self.assertEqual(source.read_bytes(), b"immutable source")
            (destination / "last.pt").unlink()
            self.assertEqual(pl.load_checkpoint(destination, core, selected, optimizer,
                                                 identity=identity(), scaler=scaler)["step"], 1)

    def test_checkpoint_rejects_pending_accumulation_without_touching_valid_checkpoint(self):
        core = DifferentiableHost().requires_grad_(False)
        selected = pl.unfreeze_lora_only(core)
        optimizer = torch.optim.SGD(selected.parameters, lr=0.001, momentum=0.9)
        prefix, native, _ = example(core)
        target = pl.short_answer_target(core, native)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.pt"
            source.write_bytes(b"source")
            pl.save_checkpoint(root, core, selected, optimizer, step=0, row_position=0,
                               identity=identity(), source_checkpoint=source, pending_microbatches=0)
            valid = (root / "last.pt").read_bytes()
            with pl.lora_training_mode(core):
                (pl.next_token_loss(core, prefix, target, selected) / 16).backward()
            for pending in (0, 1):
                with self.subTest(pending=pending), self.assertRaises(ValueError):
                    pl.save_checkpoint(root, core, selected, optimizer, step=0, row_position=1,
                                       identity=identity(), source_checkpoint=source, pending_microbatches=pending)
                self.assertEqual((root / "last.pt").read_bytes(), valid)
                self.assertFalse((root / "previous.pt").exists())
                self.assertTrue(all(p.grad is not None for p in selected.parameters))
            with self.assertRaises(ValueError):
                pl.load_checkpoint(root, core, selected, optimizer, identity=identity())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pl.save_checkpoint(root, core, selected, optimizer, step=1, row_position=1,
                               identity=identity(), source_checkpoint=source, pending_microbatches=0)
            payload = torch.load(root / "last.pt", weights_only=True)
            self.assertEqual(payload["checkpoint_boundary"], "optimizer_boundary")
            self.assertEqual(payload["pending_microbatches"], 0)
            self.assertEqual(pl.load_checkpoint(root, core, selected, optimizer, identity=identity())["step"], 1)
            for pending in (False, -1, 1):
                with self.subTest(pending=pending), self.assertRaises(ValueError):
                    pl.save_checkpoint(root, core, selected, optimizer, step=1, row_position=1,
                                       identity=identity(), source_checkpoint=source, pending_microbatches=pending)
            with self.assertRaises(TypeError):
                pl.save_checkpoint(root, core, selected, optimizer, step=1, row_position=1,
                                   identity=identity(), source_checkpoint=source)
            payload["pending_microbatches"] = 1
            torch.save(payload, root / "last.pt")
            with self.assertRaises(ValueError):
                pl.load_checkpoint(root, core, selected, optimizer, identity=identity())

    def test_nonfinite_scaled_step_backs_off_without_update_and_allows_retry(self):
        core = DifferentiableHost().requires_grad_(False)
        selected = pl.unfreeze_lora_only(core)
        optimizer = torch.optim.SGD(selected.parameters, lr=0.001)
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
        prefix, native, _ = example(core)
        target = pl.short_answer_target(core, native)
        before = [p.detach().clone() for p in selected.parameters]
        hook = selected.parameters[0].register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        try:
            with self.assertRaisesRegex(RuntimeError, "gradient"):
                pl.next_token_step(core, prefix, target, selected, optimizer, scaler=scaler)
        finally:
            hook.remove()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, selected.parameters)))
        self.assertLess(scaler.get_scale(), 8.0)
        self.assertTrue(all(p.grad is None for p in selected.parameters))
        pl.next_token_step(core, prefix, target, selected, optimizer, scaler=scaler)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, selected.parameters)))

    def test_resume_mismatches_and_source_overwrite_are_rejected_without_weight_mutation(self):
        core = DifferentiableHost().requires_grad_(False)
        selected = pl.unfreeze_lora_only(core)
        optimizer = torch.optim.AdamW(selected.parameters, lr=0.001)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.pt"
            source.write_bytes(b"source")
            pl.save_checkpoint(root, core, selected, optimizer, step=0, row_position=0,
                               identity=identity(), source_checkpoint=source, pending_microbatches=0)
            before = qa._fingerprint(dict(core.named_parameters()))
            for changed in ({"arm": "absent"}, {"protocol_hash": "f" * 64},
                            {"source_hashes": {"host": "f" * 64}},
                            {"input_hashes": {"questions": "f" * 64}}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    pl.load_checkpoint(root, core, selected, optimizer, identity={**identity(), **changed})
                self.assertEqual(qa._fingerprint(dict(core.named_parameters())), before)
            payload = torch.load(root / "last.pt", weights_only=True)
            payload["lora"][selected.names[0]] = torch.zeros(1)
            torch.save(payload, root / "last.pt")
            with self.assertRaises(ValueError):
                pl.load_checkpoint(root, core, selected, optimizer, identity=identity())
            self.assertEqual(qa._fingerprint(dict(core.named_parameters())), before)
            with self.assertRaises(ValueError):
                pl.save_checkpoint(root, core, selected, optimizer, step=1, row_position=1,
                                   identity=identity(), source_checkpoint=root / "last.pt", pending_microbatches=0)


if __name__ == "__main__":
    unittest.main()
