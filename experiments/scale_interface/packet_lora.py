"""Narrow LoRA-only continuation primitives for the paired packet/absent control.

This module neither selects data nor chooses an optimizer or training schedule.
Capture the unchanged natural-QA prefix first; only then supply its native target
to the differentiable next-token endpoint. Real host backward/VRAM validation is
a separate runtime gate from the CPU contracts.
"""
from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from .candidate_decision import native_candidate_tokens
from .natural_qa import (NaturalQACondition, PrefixBudget, _check_request, _encode,
                         _evaluation_state, _packet, _prefix_size, encode_scene_packet)
from .physical_stream_host_boundary import HostPhysicalStream, generation_question_boundary
from .scene_extents import SceneHeightPacket
from .scene_measurements import SceneRadiusPacket


@dataclass(frozen=True)
class LoraParameters:
    names: tuple
    parameters: tuple

    @property
    def numel(self):
        return sum(p.numel() for p in self.parameters)


@dataclass(frozen=True)
class NativeTarget:
    source_id: int
    visible_id: int
    token_id: int
    eligible_token_ids: tuple


@dataclass(frozen=True)
class ShortAnswerTarget:
    native: NativeTarget
    token_ids: tuple
    end_sym: str

    @property
    def text(self):
        return f"<OBJ{self.native.visible_id:03d}>{self.end_sym}"


@dataclass(frozen=True)
class CapturedPrefix:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    budget: PrefixBudget


def _lora_parameters(core):
    """Resolve A/B by PEFT's declared collections and parameter identity."""
    parameters, owners = {}, set()
    for module in core.llama_model.modules():
        declared = getattr(module, "adapter_layer_names", ())
        if "lora_A" not in declared or "lora_B" not in declared:
            continue
        if (getattr(module, "merged", False) or getattr(module, "disable_adapters", False)
                or list(getattr(module, "active_adapters", ())) != ["default"]):
            raise ValueError("requires one active, unmerged default LoRA adapter")
        for name in declared:
            collection = getattr(module, name, None)
            if name not in ("lora_A", "lora_B") and collection is not None and len(collection):
                raise ValueError("embedding and additional adapter collections are unsupported")
        if any(getattr(module, "use_dora", {}).values()):
            raise ValueError("DoRA is outside the A/B-only continuation")
        for name in ("lora_A", "lora_B"):
            collection = getattr(module, name, None)
            if not isinstance(collection, torch.nn.ModuleDict) or set(collection) != {"default"}:
                raise ValueError("requires explicit PEFT default A/B ModuleDict collections")
            layer = collection["default"]
            if not isinstance(layer, torch.nn.Linear) or layer.bias is not None:
                raise ValueError("requires ordinary bias-free linear LoRA A/B weights")
            owners.add(id(layer))
            for parameter in layer.parameters():
                if parameter.dtype != torch.float32 or not torch.isfinite(parameter).all():
                    raise ValueError("existing LoRA storage must be finite FP32; no blanket cast is allowed")
                if id(parameter) in parameters:
                    raise ValueError("aliased LoRA parameters are unsupported")
                parameters[id(parameter)] = parameter
    if not parameters:
        raise ValueError("no declared PEFT LoRA A/B collections found")
    if any(id(p) in parameters and id(module) not in owners for module in core.modules()
           for p in module.parameters(recurse=False)):
        raise ValueError("LoRA parameters must not alias a frozen base, projection, embedding or head")
    storage = {(p.device, p.untyped_storage().data_ptr()) for p in parameters.values()}
    if any((p.device, p.untyped_storage().data_ptr()) in storage for p in core.parameters() if id(p) not in parameters):
        raise ValueError("LoRA storage must not be shared with any frozen host parameter")
    named = [(name, p) for name, p in core.named_parameters() if id(p) in parameters]
    if len(named) != len(parameters):
        raise ValueError("adapter parameters must belong to the actual host")
    return LoraParameters(tuple(name for name, _ in named), tuple(p for _, p in named))


def unfreeze_lora_only(core, *, expected_numel=None):
    """Freeze the host, enabling only its already-loaded PEFT A/B parameters."""
    selected = _lora_parameters(core)
    if expected_numel is not None and (type(expected_numel) is not int or selected.numel != expected_numel):
        raise ValueError("loaded LoRA parameter count differs from the reviewed count")
    core.requires_grad_(False)
    for parameter in core.parameters():
        parameter.grad = None
    for parameter in selected.parameters:
        parameter.requires_grad_(True)
    return selected


def load_packet_lora_host(config, *, device="cuda", expected_numel=None):
    """Load the existing object-only ULIP2 bridge host, then narrowly unfreeze A/B."""
    from .loading import load_host
    model = config["model"]
    expected = {"host_mode": "object_only", "bidirection": False, "knn": 0,
                "add_img_token": False, "add_scene_token": False, "bbox_embed": False,
                "source_feature_dim": 1280, "input_dim": 1024, "use_lora": True}
    if (any(model.get(k) != v or type(model.get(k)) is not type(v) for k, v in expected.items())
            or config.get("feature_mode", "deployment") != "deployment"
            or not config.get("deployment") or not config.get("host_checkpoint") or "aqt" in config):
        raise ValueError("requires the existing causal object-only ULIP2 bridge and host checkpoint")
    core = load_host(config, device=device)
    return core, unfreeze_lora_only(core, expected_numel=expected_numel)


def native_target(tokenizer, source_id, *, assigned_ids, scene_mask, eligible_source_ids):
    """Validate once and map a source answer to its visible native identifier."""
    eligible = tuple(eligible_source_ids)
    if (len(eligible) != 2 or len(set(eligible)) != 2
            or any(type(x) is not int or not 0 <= x < 1000 for x in (*eligible, source_id))
            or source_id not in eligible):
        raise ValueError("target must be one of two explicit distinct eligible source IDs")
    assignment, mask = torch.as_tensor(assigned_ids), torch.as_tensor(scene_mask)
    if assignment.ndim == 2 and assignment.shape[0] == 1:
        assignment = assignment[0]
    if mask.ndim == 2 and mask.shape[0] == 1:
        mask = mask[0]
    if (assignment.ndim != 1 or mask.shape != assignment.shape or not len(assignment)
            or assignment.dtype not in (torch.int32, torch.int64)
            or sorted(assignment.tolist()) != list(range(len(assignment)))
            or not torch.all((mask == 0) | (mask == 1))):
        raise ValueError("requires one exact source-row permutation and visible mask")
    active = set(torch.where(mask.bool())[0].tolist())
    if {int(assignment[row]) for row in active} != active:
        raise ValueError("source-to-visible assignment must preserve the active source set")
    visible = tuple(assignment.tolist().index(source) if source < len(assignment) else -1 for source in eligible)
    if any(row < 0 or row >= 1000 or not bool(mask[row]) for row in visible):
        raise ValueError("each eligible source must map exactly once to a visible object slot")
    _, tokens = native_candidate_tokens(tokenizer, tuple(f"<OBJ{row:03d}>" for row in visible))
    index = eligible.index(source_id)
    return NativeTarget(source_id, visible[index], tokens[index], tuple(tokens))


def _check_native(target):
    if (not isinstance(target, NativeTarget) or type(target.token_id) is not int or target.token_id < 0
            or len(target.eligible_token_ids) != 2 or len(set(target.eligible_token_ids)) != 2
            or any(type(x) is not int or x < 0 for x in target.eligible_token_ids)
            or target.token_id not in target.eligible_token_ids):
        raise ValueError("requires the validated eligible native target")


def short_answer_target(core, native):
    """Tokenize native ID + the exact host ending, as in host.forward_train."""
    _check_native(native)
    ending = core.end_sym
    if not isinstance(ending, str) or not ending:
        raise ValueError("the host must declare its nonempty answer ending")
    ids = tuple(_encode(core, f"<OBJ{native.visible_id:03d}>{ending}"))
    if (not 2 <= len(ids) <= 32 or ids[0] != native.token_id
            or any(type(x) is not int or x < 0 for x in ids)):
        raise ValueError("native ID plus actual host ending must fit the 32-token answer budget")
    return ShortAnswerTarget(native, ids, ending)


class _Captured(Exception):
    pass


def capture_prefix(core, batch, question, packet, condition, *, maximum_length=2502,
                   placement="question_tail", measurement_source=None, scene_id=None):
    """Assemble evaluate's exact prefix and interrupt before model generation.

    The signature intentionally has no target. ``batch`` is the label-free
    natural_qa._cpu_input batch, moved to the host device by the caller. Reserve
    the same 32 generation tokens and full 2502-token budget used by natural QA.
    """
    allowed = {"scene_feat", "scene_img_feat", "scene_locs", "scene_mask", "obj_ids",
               "assigned_ids", "scene_gnn_feats", "foreground_ids", "scene_radius", "radius_valid"}
    if set(batch) != allowed:
        raise ValueError("prefix assembly requires only the declared label-free scene inputs")
    if (not isinstance(condition, NaturalQACondition) or condition.representation not in ("lexical", "absent")
            or condition.object_intervention != "full" or condition.mode != "true" or condition.unit != "cm"
            or condition.reverse or condition.packet_scope != "all_objects"):
        raise ValueError("continuation requires the correct complete scene packet or the shared absent arm")
    if core.host_mode != "object_only":
        raise ValueError("continuation is limited to the object-only host")
    if type(packet) not in (SceneRadiusPacket, SceneHeightPacket):
        raise ValueError("requires a typed complete radius or height packet")
    measurement = "height" if isinstance(packet, SceneHeightPacket) else "radius"
    if condition.representation != "absent" and condition.measurement != measurement:
        raise ValueError("lexical condition and measurement packet disagree")
    # The absent arm still validates the same eligible full measurement packet.
    # Its historic default radius condition does not determine the task family.
    correct = replace(condition, representation="lexical", measurement=measurement)
    expected_packet = _packet(batch, correct, 100, measurement_source=measurement_source, scene_id=scene_id)
    if packet != expected_packet:
        raise ValueError("requires every visible object's correct measurement and exact source provenance")
    _, maximum_length = _check_request(core, (condition,), None, None, 32, maximum_length, placement, 100)
    original = core.llama_model.generate
    observed = []

    def capture(**kwargs):
        length = _prefix_size(kwargs, maximum_length, 32)
        if kwargs.get("labels") is not None or kwargs.get("customized_mask") is not None:
            raise ValueError("requires a label-free ordinary causal prefix")
        observed.append((kwargs["inputs_embeds"].detach(), kwargs["attention_mask"].detach(), length))
        raise _Captured()

    options = {"max_new_tokens": 32, "num_beams": 1, "do_sample": False,
               "use_cache": True, "repetition_penalty": 3.0}
    with _evaluation_state(core, None, None, "full"), torch.no_grad():
        payload = encode_scene_packet(core, packet, condition.representation)
        offset = generation_question_boundary(question, core.role[1], lambda text: _encode(core, text))
        core.llama_model.generate = capture
        try:
            with HostPhysicalStream(core, generation_options=options) as boundary:
                boundary.arm(payload, offset, mode="generate", placement=placement)
                try:
                    core.evaluate(**batch, custom_prompt=(question,))
                except _Captured:
                    pass
                audit = boundary.finish()
        finally:
            core.llama_model.generate = original
    if len(observed) != 1 or audit["total_tokens"] != observed[0][2]:
        raise RuntimeError("prefix capture must cross exactly one complete native boundary")
    budget = PrefixBudget(condition.name, audit["native_tokens"], audit["packet_tokens"], audit["total_tokens"],
                          0 if payload is None else len(packet.numerical_block_indices))
    return CapturedPrefix(observed[0][0], observed[0][1], budget)


def _validate_trainables(core, selected, optimizer=None):
    actual = dict(core.named_parameters())
    if (not isinstance(selected, LoraParameters) or not selected.parameters
            or len(selected.names) != len(selected.parameters)
            or any(actual.get(name) is not p for name, p in zip(selected.names, selected.parameters))
            or {id(p) for p in core.parameters() if p.requires_grad} != {id(p) for p in selected.parameters}
            or any(p.grad is not None for p in core.parameters() if not p.requires_grad)):
        raise ValueError("trainable identities differ from the validated LoRA-only host")
    if optimizer is not None:
        owned = [p for group in optimizer.param_groups for p in group["params"]]
        if len(owned) != len(selected.parameters) or {id(p) for p in owned} != {id(p) for p in selected.parameters}:
            raise ValueError("optimizer must own every selected LoRA parameter exactly once")


@contextmanager
def lora_training_mode(core):
    """Keep the LLM in train mode across forward AND accumulated backward."""
    model = core.llama_model
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.train()
        yield
    finally:
        for module, training in modes:
            module.training = training


def _answer_loss(core, prefix, native, selected, token_ids, text=None):
    _validate_trainables(core, selected)
    _check_native(native)
    if prefix.inputs_embeds.requires_grad or prefix.inputs_embeds.grad_fn is not None:
        raise ValueError("requires the frozen detached input prefix")
    _prefix_size({"inputs_embeds": prefix.inputs_embeds, "attention_mask": prefix.attention_mask}, 2502, 32)
    inputs, mask = prefix.inputs_embeds, prefix.attention_mask
    count = len(token_ids)
    if text is not None:
        with torch.no_grad():
            embedded = core.get_text_emb(text, device=inputs.device).detach()
        if embedded.shape != (1, count, inputs.shape[-1]) or not torch.isfinite(embedded).all():
            raise ValueError("answer embeddings differ from the exact host target tokenization")
        inputs = torch.cat((inputs, embedded[:, :-1]), 1)
        mask = torch.cat((mask, mask.new_ones((1, count - 1))), 1)
    with lora_training_mode(core), core.maybe_autocast():
        result = core.llama_model(inputs_embeds=inputs, attention_mask=mask,
                                  use_cache=False, return_dict=True, logits_to_keep=count)
        logits = result.logits
        if (logits.ndim != 3 or logits.shape[:2] != (1, count)
                or max((*native.eligible_token_ids, *token_ids)) >= logits.shape[-1]
                or not torch.isfinite(logits).all()):
            raise ValueError("invalid native next-token vocabulary or nonfinite logits")
        # Only answer positions have a target: the complete input prefix is masked.
        loss = F.cross_entropy(logits.reshape(count, -1).float(), logits.new_tensor(token_ids, dtype=torch.long))
    if not loss.requires_grad:
        raise RuntimeError("native CE is disconnected from the loaded LoRA parameters")
    return loss


def native_id_loss(core, prefix, target, selected):
    """Diagnostic first-ID CE only; this is not the continuation objective."""
    _check_native(target)
    return _answer_loss(core, prefix, target, selected, (target.token_id,))


def next_token_loss(core, prefix, target, selected):
    """Teacher-forced short-answer CE; the complete prompt is unsupervised.

    For accumulation, call loss and backward inside lora_training_mode(core),
    dividing each example loss by the actual microbatch-group size. The context
    must cover checkpoint recomputation. next_token_step does this for one row.
    """
    if (not isinstance(target, ShortAnswerTarget)
            or target != short_answer_target(core, target.native)):
        raise ValueError("training requires native ID plus the current host ending")
    return _answer_loss(core, prefix, target.native, selected, target.token_ids, target.text)


def next_token_step(core, prefix, target, selected, optimizer, *, scaler=None):
    """One explicit short-answer update, including checkpointed backward."""
    _validate_trainables(core, selected, optimizer)
    optimizer.zero_grad(set_to_none=True)
    with lora_training_mode(core):
        loss = next_token_loss(core, prefix, target, selected)
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        missing = any(p.grad is None for p in selected.parameters)
        nonfinite = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in selected.parameters)
        if missing or nonfinite:
            if scaler is not None:
                if nonfinite and scaler.is_enabled():
                    # unscale_ recorded the overflow: step skips the optimizer,
                    # update backs off and resets its per-optimizer stage.
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    scaler.update(new_scale=scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("missing or nonfinite LoRA gradients; update rejected")
        _validate_trainables(core, selected, optimizer)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
    return float(loss.detach())


def _identity(value):
    if (not isinstance(value, dict) or set(value) != {"arm", "source_hashes", "input_hashes", "protocol_hash"}
            or value["arm"] not in ("lexical", "absent")
            or not isinstance(value["protocol_hash"], str) or not value["protocol_hash"]):
        raise ValueError("checkpoint identity must pin arm, source/input hashes and reviewed protocol")
    for field in ("source_hashes", "input_hashes"):
        if (not isinstance(value[field], dict) or not value[field]
                or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                       for k, v in value[field].items())):
            raise ValueError("source and input hashes must be explicit nonempty mappings")
    return value


def _optimizer_identity(optimizer, selected):
    names = {id(p): name for name, p in zip(selected.names, selected.parameters)}
    return {"class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "groups": [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]}


def _rng_state():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def _optimizer_boundary(selected, pending_microbatches):
    if (type(pending_microbatches) is not int or pending_microbatches != 0
            or any(p.grad is not None for p in selected.parameters)):
        raise ValueError("checkpoint requires pending_microbatches=0 and zero_grad(set_to_none=True)")


def save_checkpoint(directory, core, selected, optimizer, *, step, row_position, identity,
                    source_checkpoint, pending_microbatches, scaler=None):
    """Save at a clean optimizer boundary, retaining atomic last + previous.

    The caller must finish the update and zero_grad(set_to_none=True), then
    explicitly attest pending_microbatches=0. Partial accumulation is rejected;
    this checkpoint format does not save pending gradients.
    """
    _validate_trainables(core, selected, optimizer)
    _optimizer_boundary(selected, pending_microbatches)
    _identity(identity)
    if any(type(x) is not int or x < 0 for x in (step, row_position)):
        raise ValueError("step and next row position must be nonnegative integers")
    directory = Path(directory)
    last, previous = directory / "last.pt", directory / "previous.pt"
    source = Path(source_checkpoint).resolve()
    if any(target.resolve() == source or (target.exists() and source.exists() and os.path.samefile(target, source))
           for target in (last, previous)):
        raise ValueError("continuation must never overwrite the source checkpoint")
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"format": 2, "checkpoint_boundary": "optimizer_boundary", "pending_microbatches": 0,
               "identity": identity, "lora": {name: p.detach().cpu()
               for name, p in zip(selected.names, selected.parameters)}, "optimizer": optimizer.state_dict(),
               "optimizer_identity": _optimizer_identity(optimizer, selected),
               "scaler": None if scaler is None else scaler.state_dict(), "rng": _rng_state(),
               "step": step, "row_position": row_position}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="packet-lora-", suffix=".tmp", dir=directory, delete=False) as output:
            temporary = Path(output.name)
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        if last.exists():
            os.replace(last, previous)
        os.replace(temporary, last)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(last)


def load_checkpoint(directory, core, selected, optimizer, *, identity, scaler=None):
    """Resume only an identical arm/source/input/protocol, validating before copy."""
    _validate_trainables(core, selected, optimizer)
    _optimizer_boundary(selected, 0)
    _identity(identity)
    directory = Path(directory)
    path = directory / "last.pt"
    if not path.exists():
        path = directory / "previous.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != 2 or payload.get("checkpoint_boundary") != "optimizer_boundary"
            or type(payload.get("pending_microbatches")) is not int or payload["pending_microbatches"] != 0
            or payload.get("identity") != identity
            or payload.get("optimizer_identity") != _optimizer_identity(optimizer, selected)
            or (payload.get("scaler") is None) != (scaler is None)
            or any(type(payload.get(key)) is not int or payload[key] < 0 for key in ("step", "row_position"))):
        raise ValueError("checkpoint source, input, protocol, optimizer or progress mismatch")
    lora = payload.get("lora", {})
    if set(lora) != set(selected.names):
        raise ValueError("checkpoint must contain exactly the selected LoRA tensors")
    for name, parameter in zip(selected.names, selected.parameters):
        value = lora[name]
        if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype
                or not torch.isfinite(value).all()):
            raise ValueError("checkpoint LoRA tensor shape/dtype/value mismatch")
    rng = payload["rng"]
    if rng["cuda"] and (not torch.cuda.is_initialized() or len(rng["cuda"]) != torch.cuda.device_count()):
        raise ValueError("CUDA RNG topology differs from the continuation checkpoint")
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    with torch.no_grad():
        for name, parameter in zip(selected.names, selected.parameters):
            parameter.copy_(lora[name])
    random.setstate(rng["python"])
    numpy = rng["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return {"step": payload["step"], "row_position": payload["row_position"], "path": str(path)}
