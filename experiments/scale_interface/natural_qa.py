"""Streaming natural-question controls with one shared, preflighted cohort.

Inputs use SceneDataset's CPU scene cache. Questions and saved object-row
assignments are read explicitly; captions, reference answers and answer-based
object selection are excluded from input construction. Preflight assembles the
actual host prefix but intercepts generation before the language-model forward.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import time
from typing import Tuple

import torch

from .conditions import compress
from .physical_stream_host_boundary import HostPhysicalStream, generation_question_boundary
from .scene_data import update_caption
from .scene_measurements import build_scene_radius_packet
from .scene_extents import build_scene_height_packet, normalize_measurement_source


OBJECT_CONTROLS = ("full", "zero_object_slots", "drop_object_features", "permute_object_slots")
REPRESENTATIONS = ("absent", "lexical", "trim", "trained", "random", "mean", "masked")


@dataclass(frozen=True)
class NaturalQACondition:
    representation: str = "lexical"
    object_intervention: str = "full"
    mode: str = "true"
    unit: str = "cm"
    reverse: bool = False
    measurement: str = "radius"

    def __post_init__(self):
        if self.representation not in REPRESENTATIONS or self.object_intervention not in OBJECT_CONTROLS:
            raise ValueError("unknown representation or object intervention")
        if self.mode not in ("true", "constant", "swapped") or self.unit not in ("cm", "mm", "m"):
            raise ValueError("unknown measurement condition or unit")
        if not isinstance(self.reverse, bool):
            raise ValueError("reverse must be boolean")
        if self.measurement not in ("radius", "height"):
            raise ValueError("measurement must be radius or height")
        if self.representation == "absent" and (self.mode != "true" or self.unit != "cm" or self.reverse):
            raise ValueError("absent uses true/cm/forward to avoid duplicate identical conditions")
        if self.representation == "absent" and self.measurement != "radius":
            raise ValueError("use the unchanged absent baseline; height absent would duplicate it")
        if self.representation == "masked" and (self.mode != "true" or self.unit != "cm" or self.reverse):
            raise ValueError("masked uses true/cm/forward lengths; no alternative masks")

    @property
    def name(self):
        order = "reverse" if self.reverse else "forward"
        name = f"{self.object_intervention}_{self.representation}_{self.mode}_{self.unit}_{order}"
        return name if self.measurement == "radius" else name + "_height"


@dataclass(frozen=True)
class PrefixBudget:
    condition: str
    native_tokens: int
    packet_tokens: int
    prefix_tokens: int
    numerical_blocks: int


@dataclass(frozen=True)
class PreparedNaturalQuestion:
    index: int
    qid: str
    scene_id: str
    input_fingerprint: str
    budgets: Tuple[PrefixBudget, ...]


@dataclass(frozen=True)
class NaturalQAExclusion:
    index: int
    qid: str
    scene_id: str
    condition: str
    reason: str


@dataclass(frozen=True)
class NaturalQAPlan:
    conditions: Tuple[NaturalQACondition, ...]
    rows: Tuple[PreparedNaturalQuestion, ...]
    exclusions: Tuple[NaturalQAExclusion, ...]
    dataset_fingerprint: str
    runtime_fingerprint: str
    candidate_count: int
    max_new_tokens: int
    maximum_length: int
    placement: str
    constant_cm: int
    measurement_source: object = None
    measurement_source_fingerprint: str = ""

    def metadata(self):
        metadata = {
            "conditions": [{"name": c.name, **asdict(c)} for c in self.conditions],
            "candidate_count": self.candidate_count,
            "selected_qids": [r.qid for r in self.rows],
            "selected_scene_ids": sorted({r.scene_id for r in self.rows}),
            "excluded_qids": list(dict.fromkeys(e.qid for e in self.exclusions)),
            "exclusions": [asdict(e) for e in self.exclusions],
            "rows": [asdict(r) for r in self.rows],
            "dataset_fingerprint": self.dataset_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "max_new_tokens": self.max_new_tokens,
            "maximum_length": self.maximum_length,
            "placement": self.placement,
            "constant_cm": self.constant_cm,
            "generation": {"num_beams": 1, "do_sample": False, "use_cache": True},
        }
        if self.measurement_source is not None:
            metadata["measurement_source"] = json.loads(json.dumps(self.measurement_source))
            metadata["measurement_source_fingerprint"] = self.measurement_source_fingerprint
        return metadata


def _fingerprint(value):
    digest = hashlib.blake2b(digest_size=20)

    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item):
                add(key)
                add(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(b"[")
            for child in item:
                add(child)
            digest.update(b"]")
        elif isinstance(item, (set, frozenset)):
            add({"type": type(item).__name__, "values": sorted(item, key=repr)})
        elif hasattr(item, "to_dict"):
            add(item.to_dict())
        elif hasattr(item, "__dict__"):
            add({"type": f"{type(item).__module__}.{type(item).__qualname__}", "fields": vars(item)})
        else:
            try:
                encoded = json.dumps(item, ensure_ascii=False, allow_nan=False)
            except TypeError:
                encoded = json.dumps({"type": f"{type(item).__module__}.{type(item).__qualname__}",
                                      "value": repr(item)}, ensure_ascii=False)
            digest.update(encoded.encode())
            digest.update(b";")

    add(value)
    return digest.hexdigest()


def _dataset_fingerprint(dataset):
    # Restrict the row fields: even fingerprinting must not inspect answers.
    rows = [{key: row.get(key) for key in ("qid", "scene_id", "prompt", "assigned_ids")}
            for row in dataset.anno]
    scenes = sorted({row["scene_id"] for row in rows})
    caches = ("scene_feats", "scene_img_feats", "scene_masks", "scene_gnn_feats",
              "scene_foreground_ids", "scene_radii", "scene_radius_valid")
    inputs = {name: {scene: getattr(dataset, name)[scene] for scene in scenes} for name in caches}
    inputs["locations"] = {scene: dataset.attributes[scene]["locs"] for scene in scenes}
    return _fingerprint({"rows": rows, "inputs": inputs,
                         "full_manifest": getattr(dataset, "full_manifest", None),
                         "missing_channel_policy": getattr(dataset, "missing_channel_policy", "reject")})


def _module_stamp(module):
    if module is None:
        return None
    # Keep the same loaded model instance; never copy LLM weights to the CPU.
    return {"instance": id(module), "tensors": [
        (name, value.data_ptr(), value._version, str(value.dtype), str(value.device), tuple(value.shape))
        for name, value in (*module.named_parameters(), *module.named_buffers())
    ]}


def _runtime_fingerprint(core, codec, random_codec):
    attributes = ("host_mode", "knn", "bidirection", "add_img_token", "add_scene_token",
                  "bbox_embed", "add_pos_emb", "role", "end_sym", "max_txt_len")
    return _fingerprint({
        "host": _module_stamp(core), "codec": _module_stamp(codec), "random_codec": _module_stamp(random_codec),
        "attributes": {key: getattr(core, key) for key in attributes},
        "missing_channel_policy": getattr(core, "missing_channel_policy", "reject"),
        "config": getattr(core, "config", None), "llm_config": getattr(core.llama_model, "config", None),
        "tokenizer_vocab": core.llama_tokenizer.get_vocab(),
        "tokenizer_instance": id(core.llama_tokenizer),
        "tokenizer_options": {key: getattr(core.llama_tokenizer, key, None) for key in
                              ("padding_side", "truncation_side", "model_max_length", "init_kwargs")},
        "tokenizer_special": getattr(core.llama_tokenizer, "special_tokens_map", {}),
        "p_0": core.p_0_embed, "p_1": core.p_1_embed,
    })


def _cpu_input(dataset, index):
    row = dataset.anno[index]
    prompt = row["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("a nonempty question prompt is required")
    scene, features, images, mask, locations, assignment, edges, foreground, radii, valid = dataset.get_anno(
        index, row["assigned_ids"]
    )
    tensors = (features, images, mask, locations, assignment, edges, foreground, radii, valid)
    if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in tensors):
        raise ValueError("SceneDataset must retain all scene inputs on the CPU")
    question = update_caption(prompt, assignment)
    batch = {"scene_feat": features.unsqueeze(0), "scene_img_feat": images.unsqueeze(0),
             "scene_mask": mask.bool().unsqueeze(0), "scene_locs": locations.unsqueeze(0),
             "assigned_ids": assignment.unsqueeze(0), "scene_gnn_feats": edges.unsqueeze(0),
             "foreground_ids": (foreground,), "scene_radius": radii.unsqueeze(0),
             "radius_valid": valid.unsqueeze(0), "obj_ids": torch.zeros(1, dtype=torch.long)}
    return str(row["qid"]), str(scene), question, batch


def _on_device(batch, device):
    return {key: value.detach().to(device) if isinstance(value, torch.Tensor) else
            tuple(item.detach().to(device) for item in value)
            for key, value in batch.items()}


def _packet(batch, condition, constant_cm, *, measurement_source=None, scene_id=None):
    mask = batch["scene_mask"][0].tolist()
    visible = [index for index, active in enumerate(mask) if active]
    if condition.object_intervention == "permute_object_slots" and len(visible) < 2:
        raise ValueError("object permutation requires at least two visible objects")
    if condition.measurement == "height":
        heights = batch["scene_locs"][0, :, 5]
        valid = torch.isfinite(heights) & (heights > 0)
        return build_scene_height_packet(
            scene_id=scene_id, measurement_source=measurement_source,
            assigned_rows=batch["assigned_ids"][0].tolist(), scene_mask=mask,
            height_m=heights.tolist(), height_valid=valid.tolist(),
            mode=condition.mode, unit=condition.unit, constant_cm=constant_cm,
            record_order=list(reversed(visible)) if condition.reverse else None,
        )
    return build_scene_radius_packet(
        assigned_rows=batch["assigned_ids"][0].tolist(), scene_mask=mask,
        radius_m=batch["scene_radius"][0].tolist(), radius_valid=batch["radius_valid"][0].tolist(),
        mode=condition.mode, unit=condition.unit, constant_cm=constant_cm,
        record_order=list(reversed(visible)) if condition.reverse else None,
    )


def _encode(core, text):
    return list(core.llama_tokenizer(text, add_special_tokens=False)["input_ids"])


def encode_scene_packet(core, packet, representation, *, codec=None, random_codec=None):
    """Compress every explicit numerical block, never a fixed pair of objects."""
    if representation == "absent":
        return None
    indices = set(packet.numerical_block_indices)
    ids = [_encode(core, part) for part in packet.parts]
    if sum(ids, []) != _encode(core, packet.text):
        raise ValueError("scene packet segmentation changes native tokenization")
    device = next(core.llama_model.parameters()).device
    blocks = []
    for index, part in enumerate(packet.parts):
        block = core.get_text_emb(part, device=device).squeeze(0).detach()
        if block.ndim != 2 or len(block) != len(ids[index]) or not torch.isfinite(block).all():
            raise ValueError("packet embeddings differ from the tokenized numerical blocks")
        if index in indices and representation == "masked":
            placeholder_ids = _encode(core, "?")
            special_ids = set(getattr(core.llama_tokenizer, "all_special_ids", ()))
            if len(placeholder_ids) != 1 or placeholder_ids[0] in special_ids:
                raise ValueError("fixed masked placeholder must be exactly one ordinary token")
            placeholder = core.get_text_emb("?", device=device).squeeze(0).detach()
            if placeholder.shape != (1, block.shape[-1]) or not torch.isfinite(placeholder).all():
                raise ValueError("invalid fixed placeholder embedding")
            block = placeholder.expand(len(block), -1).clone()
        elif index in indices:
            block = compress(block, ids[index], representation=representation, trained=codec,
                             random=random_codec, decode=core.llama_tokenizer.decode)
        if block.ndim != 2 or block.shape[-1] != core.p_0_embed.shape[-1] or not torch.isfinite(block).all():
            raise ValueError("compressed numerical block is malformed")
        blocks.append(block)
    return torch.cat(blocks)


@contextmanager
def _evaluation_state(core, codec, random_codec, object_intervention):
    modules = list(core.modules())
    for module in (codec, random_codec):
        if module is not None:
            modules.extend(module.modules())
    states = [(module, module.training) for module in dict.fromkeys(modules)]
    had_control = hasattr(core, "object_feature_intervention")
    previous = getattr(core, "object_feature_intervention", "full")
    try:
        core.eval()
        for module in (codec, random_codec):
            if module is not None:
                module.eval()
        core.object_feature_intervention = object_intervention
        yield
    finally:
        for module, training in states:
            module.training = training
        if had_control:
            core.object_feature_intervention = previous
        else:
            del core.object_feature_intervention


class _PrefixCaptured(Exception):
    pass


class _PredictionCaptured(Exception):
    pass


def _decoded_prediction(core, token_ids, batch):
    raw = core.llama_tokenizer.decode(token_ids)
    text = raw.split(core.end_sym)[0].replace("  ", " ").replace(" .", ".").strip()
    assignment = batch["assigned_ids"][0].tolist()
    visible = set(torch.where(batch["scene_mask"][0].bool())[0].tolist())
    matches = list(re.finditer(r"<OBJ([^>]*)>", text))
    invalid = text.count("<OBJ") != len(matches) or any(
        not re.fullmatch(r"[0-9]{3}", match.group(1)) or int(match.group(1)) not in visible
        for match in matches)
    if not invalid:
        text = re.sub(r"<OBJ([0-9]{3})>", lambda match: f"<OBJ{assignment[int(match.group(1))]:03d}>", text)
    return {"prediction": text, "raw_prediction": raw,
            "output_token_ids": token_ids.tolist(), "invalid_object_id": invalid}


def _prefix_size(kwargs, maximum_length, max_new_tokens):
    values, mask = kwargs["inputs_embeds"], kwargs["attention_mask"]
    if (values.ndim != 3 or values.shape[0] != 1 or mask.shape != values.shape[:2]
            or not torch.all(mask == 1) or not torch.isfinite(values).all()):
        raise ValueError("generation requires one finite, complete, unpadded prefix")
    length = values.shape[1]
    if length + max_new_tokens > maximum_length:
        raise ValueError(f"assembled prefix {length} + generation {max_new_tokens} exceeds budget {maximum_length}")
    return length


def _run_one(core, batch, question, packet, condition, *, codec, random_codec,
             max_new_tokens, maximum_length, placement, capture_only, expected=None):
    original_generate = core.llama_model.generate
    observed, decoded = [], []

    def capture_or_generate(**kwargs):
        observed.append(_prefix_size(kwargs, maximum_length, max_new_tokens))
        packet_length = 0 if payload is None else len(payload)
        budget = PrefixBudget(condition.name, observed[-1] - packet_length, packet_length,
                              observed[-1], 0 if payload is None else len(packet.numerical_block_indices))
        if expected is not None and budget != expected:
            raise RuntimeError("generation prefix differs from its completed preflight")
        if capture_only:
            raise _PrefixCaptured()
        generated = original_generate(**kwargs)
        if not isinstance(generated, torch.Tensor) or generated.ndim != 2 or generated.shape[0] != 1:
            raise RuntimeError("one tensor of generated token IDs is required")
        decoded.append(_decoded_prediction(core, generated[0].detach().cpu(), batch))
        # Bypass the retained host's random fallback for invalid object IDs.
        raise _PredictionCaptured()

    options = {"max_new_tokens": max_new_tokens, "num_beams": 1,
               "do_sample": False, "use_cache": True}
    with _evaluation_state(core, codec, random_codec, condition.object_intervention), torch.no_grad():
        payload = encode_scene_packet(core, packet, condition.representation, codec=codec, random_codec=random_codec)
        offset = generation_question_boundary(question, core.role[1], lambda text: _encode(core, text))
        core.llama_model.generate = capture_or_generate
        try:
            with HostPhysicalStream(core, generation_options=options) as boundary:
                boundary.arm(payload, offset, mode="generate", placement=placement)
                try:
                    core.evaluate(**batch, custom_prompt=(question,))
                except _PrefixCaptured:
                    if not capture_only:
                        raise
                except _PredictionCaptured:
                    if capture_only:
                        raise
                audit = boundary.finish()
        finally:
            core.llama_model.generate = original_generate
    if len(observed) != 1 or audit["total_tokens"] != observed[0]:
        raise RuntimeError("generation prefix differs from the assembled boundary")
    budget = PrefixBudget(condition.name, audit["native_tokens"], audit["packet_tokens"],
                          audit["total_tokens"], 0 if payload is None else len(packet.numerical_block_indices))
    if not capture_only and len(decoded) != 1:
        raise RuntimeError("one generated answer is required per question")
    return budget, None if capture_only else decoded[0]


def _check_request(core, conditions, codec, random_codec, max_new_tokens, maximum_length, placement, constant_cm):
    conditions = tuple(conditions)
    if not conditions or any(not isinstance(c, NaturalQACondition) for c in conditions):
        raise ValueError("provide a nonempty sequence of NaturalQACondition values")
    if len({c.name for c in conditions}) != len(conditions):
        raise ValueError("duplicate natural-question conditions")
    for name, value in (("max_new_tokens", max_new_tokens), ("maximum_length", maximum_length), ("constant_cm", constant_cm)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if maximum_length > 2502 or max_new_tokens >= maximum_length:
        raise ValueError("the complete sequence budget must be at most 2502 and include generation")
    if placement not in ("question_tail", "object_end"):
        raise ValueError("unknown packet placement")
    object_only = core.host_mode == "object_only" and core.knn == 0 and not core.add_img_token
    full = core.host_mode == "full" and core.knn == 2 and core.add_img_token
    if (not (object_only or full) or core.bidirection or core.add_scene_token or core.bbox_embed):
        raise ValueError("natural-question controls require a causal object-only or full image/graph host")
    for representation, module in (("trained", codec), ("random", random_codec)):
        if any(c.representation == representation for c in conditions) and module is None:
            raise ValueError(f"{representation} requires explicit compressor weights")
    context = getattr(getattr(core.llama_model, "config", None), "max_position_embeddings", maximum_length)
    return conditions, min(maximum_length, int(context))


def preflight_natural_qa(core, dataset, conditions, *, codec=None, random_codec=None,
                         max_new_tokens=32, maximum_length=2502, placement="question_tail",
                         constant_cm=100, progress=None, measurement_source=None):
    """Check every condition before generation and exclude a row from all together.

    Actual embeddings, attention masks and inserted-packet lengths are checked.
    No language-model prediction is computed. Model projection/embedding work
    runs on the host device, retaining only one scene batch there at a time.
    """
    conditions, maximum_length = _check_request(
        core, conditions, codec, random_codec, max_new_tokens, maximum_length, placement, constant_cm)
    if core.host_mode == "full" and not getattr(dataset, "full_input", False):
        raise ValueError("full host requires manifest-validated image and relation inputs")
    if getattr(core, "missing_channel_policy", "reject") != getattr(dataset, "missing_channel_policy", "reject"):
        raise ValueError("host and data missing-channel policies disagree")
    if any(condition.measurement == "height" for condition in conditions):
        measurement_source = normalize_measurement_source(measurement_source)
        scenes = {row["scene_id"] for row in dataset.anno}
        uncovered = [scene for scene in scenes if measurement_source["verified_scenes"].get(scene) is not True]
        if uncovered:
            raise ValueError(f"height source does not cover candidate scenes: {sorted(uncovered)}")
    elif measurement_source is not None:
        raise ValueError("measurement_source is only used by explicit height conditions")
    dataset_identity = _dataset_fingerprint(dataset)
    runtime_identity = _runtime_fingerprint(core, codec, random_codec)
    rows, exclusions = [], []
    device = next(core.llama_model.parameters()).device
    for index in range(len(dataset.anno)):
        qid, scene = str(dataset.anno[index]["qid"]), str(dataset.anno[index]["scene_id"])
        try:
            _, _, question, cpu_batch = _cpu_input(dataset, index)
            input_identity = _fingerprint({"question": question, "batch": cpu_batch})
        except (ValueError, KeyError, IndexError) as error:
            exclusions.append(NaturalQAExclusion(index, qid, scene, "input", str(error)))
            if progress:
                progress({"phase": "preflight", "index": index, "qid": qid, "status": "excluded", "reason": str(error)})
            continue
        device_batch = _on_device(cpu_batch, device)
        budgets, failures = [], []
        try:
            for condition in conditions:
                try:
                    packet = _packet(cpu_batch, condition, constant_cm,
                                     measurement_source=measurement_source, scene_id=scene)
                    budget, _ = _run_one(core, device_batch, question, packet, condition,
                                        codec=codec, random_codec=random_codec, max_new_tokens=max_new_tokens,
                                        maximum_length=maximum_length, placement=placement, capture_only=True)
                    budgets.append(budget)
                except (ValueError, KeyError, IndexError) as error:
                    failures.append(NaturalQAExclusion(index, qid, scene, condition.name, str(error)))
            if failures:
                exclusions.extend(failures)
            else:
                rows.append(PreparedNaturalQuestion(index, qid, scene, input_identity, tuple(budgets)))
            if progress:
                progress({"phase": "preflight", "index": index, "qid": qid,
                          "status": "excluded" if failures else "eligible", "checked_conditions": len(conditions)})
        finally:
            del device_batch
    if _dataset_fingerprint(dataset) != dataset_identity or _runtime_fingerprint(core, codec, random_codec) != runtime_identity:
        raise RuntimeError("inputs or runtime changed during preflight")
    return NaturalQAPlan(conditions, tuple(rows), tuple(exclusions), dataset_identity, runtime_identity,
                         len(dataset.anno), max_new_tokens, maximum_length, placement, constant_cm,
                         measurement_source, "" if measurement_source is None else _fingerprint(measurement_source))


def iter_natural_qa(core, dataset, plan, *, codec=None, random_codec=None,
                    condition_names=None, question_ids=None):
    """Yield one answer per question/condition; callers persist each record.

    Consume or close this iterator before reusing the host. The plan is tied to
    the same loaded host/codec instances and unchanged CPU inputs. References
    should be joined by qid only later, in an independent metric calculation.
    """
    if not isinstance(plan, NaturalQAPlan) or not plan.rows:
        raise ValueError("generation requires a nonempty completed NaturalQAPlan")
    if getattr(core, "missing_channel_policy", "reject") != getattr(dataset, "missing_channel_policy", "reject"):
        raise ValueError("host and data missing-channel policies disagree")
    if (plan.measurement_source is not None
            and _fingerprint(plan.measurement_source) != plan.measurement_source_fingerprint):
        raise ValueError("height measurement source changed after preflight")
    def selected(values, allowed, name):
        if values is None:
            return set(allowed)
        values = list(values)
        if (not values or any(not isinstance(value, str) for value in values)
                or len(set(values)) != len(values) or not set(values) <= set(allowed)):
            raise ValueError(f"{name} must be a nonempty unique subset of the completed plan")
        return set(values)
    selected_conditions = selected(condition_names, [c.name for c in plan.conditions], "condition_names")
    selected_questions = selected(question_ids, [r.qid for r in plan.rows], "question_ids")
    if _dataset_fingerprint(dataset) != plan.dataset_fingerprint:
        raise ValueError("dataset differs from the completed preflight")
    if _runtime_fingerprint(core, codec, random_codec) != plan.runtime_fingerprint:
        raise ValueError("host, tokenizer or compressor differs from the completed preflight")
    device = next(core.llama_model.parameters()).device
    for prepared in plan.rows:
        if prepared.qid not in selected_questions:
            continue
        qid, scene, question, cpu_batch = _cpu_input(dataset, prepared.index)
        if (qid != prepared.qid or scene != prepared.scene_id
                or _fingerprint({"question": question, "batch": cpu_batch}) != prepared.input_fingerprint):
            raise ValueError("question inputs changed after preflight")
        device_batch = _on_device(cpu_batch, device)
        try:
            for condition, expected in zip(plan.conditions, prepared.budgets):
                if condition.name not in selected_conditions:
                    continue
                start = time.monotonic()
                packet = _packet(cpu_batch, condition, plan.constant_cm,
                                 measurement_source=plan.measurement_source, scene_id=scene)
                observed, prediction = _run_one(
                    core, device_batch, question, packet, condition, codec=codec, random_codec=random_codec,
                    max_new_tokens=plan.max_new_tokens, maximum_length=plan.maximum_length,
                    placement=plan.placement, capture_only=False, expected=expected)
                if observed != expected:
                    raise RuntimeError("generated prefix differs from its completed preflight")
                yield {"qid": qid, "scene_id": scene, "condition": condition.name, **prediction,
                       "packet_tokens": observed.packet_tokens, "prefix_tokens": observed.prefix_tokens,
                       "native_tokens": observed.native_tokens, "numerical_blocks": observed.numerical_blocks,
                       "elapsed_seconds": time.monotonic() - start}
        finally:
            del device_batch
