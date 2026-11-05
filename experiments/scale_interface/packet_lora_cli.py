"""Matched packet/absent LoRA continuation on fixed, disjoint scene cohorts."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sqlite3
import sys
import tempfile
import time

import numpy as np
import torch

from . import packet_lora as lora
from .data import SceneDataset
from .loading import seed_everything
from .natural_qa import NaturalQACondition, _candidate_source_rows, _cpu_input, _on_device, _packet, _run_one
from .prediction_scoring import NATIVE
from .runtime_provenance import sha256
from .scene_extents import normalize_measurement_source


SEED, LR, WARMUP = 20261102, 2e-5, 133
TRAIN_ROWS, PILOT_ROWS, UPDATES, ACCUMULATION = 70651, 3504, 4416, 16
EXPECTED_LORA, EXPECTED_TENSORS = 41943040, 448
COUNTS = {"train": {"radius": 35321, "height": 35330},
          "validation": {"radius": 2187, "height": 2183}}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_bytes(value)); stream.flush(); os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ordered_indices(radius, height, *, seed=SEED):
    order = list(range(radius + height))
    random.Random(seed).shuffle(order)
    return order


def update_groups(start, end):
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= TRAIN_ROWS or start % ACCUMULATION:
        raise ValueError("groups require a clean accumulation boundary")
    for first in range(start, end, ACCUMULATION):
        yield first, min(first + ACCUMULATION, end)


def validate_progress(position, step):
    if (type(position) is not int or type(step) is not int or not 0 <= position <= TRAIN_ROWS
            or (position != TRAIN_ROWS and position % ACCUMULATION) or step != math.ceil(position / ACCUMULATION)):
        raise ValueError("checkpoint progress must be an exact successful-update boundary")


def learning_rate(completed):
    if type(completed) is not int or not 0 <= completed < UPDATES:
        raise ValueError("invalid successful update")
    update = completed + 1
    return LR * (update / WARMUP if update <= WARMUP else (UPDATES - update) / (UPDATES - WARMUP))


def _canonical_label(label):
    if {"qid", "answer"} <= label.keys() and not {"question_id", "answers"} & label.keys():
        return label
    if {"question_id", "answers"} <= label.keys() and not {"qid", "answer"} & label.keys():
        answers = label["answers"]
        if (not isinstance(answers, list) or not answers or any(not isinstance(value, str) for value in answers)
                or len(set(answers)) != 1):
            raise ValueError("a controlled reference requires exactly one native target")
        return dict(qid=label["question_id"], scene_id=label["scene_id"], answer=answers[0])
    raise ValueError("label schema must be qid/answer or question_id/answers")


def target_source(row, label):
    label = _canonical_label(label)
    if set(row) & {"answer", "answers", "target", "references", "caption", "ref_captions"}:
        raise ValueError("input rows must stay label-free")
    ids = re.findall(r"<OBJ([0-9]{3})>", row["prompt"])
    answer = label.get("answer", "")
    if (len(ids) != 2 or len(set(ids)) != 2 or row["prompt"].count("<OBJ") != 2
            or label.get("qid") != row.get("qid") or label.get("scene_id") != row.get("scene_id")
            or not re.fullmatch(r"<OBJ[0-9]{3}>", answer) or answer[4:7] not in ids):
        raise ValueError("label must be exactly one queried source-row identifier")
    return int(answer[4:7])


def native_hit(prediction, target, assignment, end_sym):
    invalid, raw, mapped = (prediction[key] for key in ("invalid_object_id", "raw_prediction", "prediction"))
    if type(invalid) is not bool or not isinstance(raw, str) or not isinstance(mapped, str):
        raise ValueError("malformed raw/mapped native prediction")
    raw_id = NATIVE.fullmatch(raw.split(end_sym, 1)[0].strip())
    mapped_id = NATIVE.fullmatch(mapped.strip())
    if invalid or raw_id is None or mapped_id is None:
        return 0
    assignment = assignment.tolist() if hasattr(assignment, "tolist") else list(assignment)
    visible, source = int(raw_id[1]), int(mapped_id[1])
    if visible >= len(assignment) or source not in assignment:
        return 0
    if assignment[visible] != source:
        raise ValueError("raw visible/source mapping mismatch")
    return int(source == target)


def check_splits(splits, held_out_scenes):
    families, qids = {}, {}
    for role, rows in splits.items():
        families[role] = {row["scene_id"].rsplit("_", 1)[0] for row in rows}
        qids[role] = {row["qid"] for row in rows}
        if len(qids[role]) != len(rows):
            raise ValueError("question identity overlap within split")
    if families["train"] & families["validation"] or qids["train"] & qids["validation"]:
        raise ValueError("train/validation overlap")
    held_out = {scene.rsplit("_", 1)[0] for scene in held_out_scenes}
    if held_out & (families["train"] | families["validation"]):
        raise ValueError("held-out scene-family overlap")


def validate_deployment(path):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if (bundle.get("architecture", {}).get("adapter_kind") != "identity_no_radius"
            or bundle.get("state", {}).get("scalestitch") != {}):
        raise ValueError("matched training requires identity_no_radius with an empty scalestitch state")
    return sha256(path)


def required_inputs(config):
    if set(config) != {"host_config", "splits", "held_out_scenes", "expected_counts", "input_sha256"}:
        raise ValueError("config requires host_config, splits, held_out_scenes, expected_counts, input_sha256")
    if config["expected_counts"] != COUNTS or set(config["splits"]) != set(COUNTS):
        raise ValueError("the matched protocol requires its complete train/validation cohorts")
    paths = {Path(config["host_config"]).resolve(), Path(config["held_out_scenes"]).resolve()}
    host = _read(config["host_config"])["host"]
    validate_deployment(host["deployment"])
    paths.update(Path(host[key]).resolve() for key in ("host_checkpoint", "deployment"))
    model = Path(host["model"]["llama_model_path"]).resolve()
    model_files = {path.resolve() for path in model.rglob("*") if path.is_file()
                   and not {".cache", ".git"} & set(path.relative_to(model).parts)}
    if not any(path.suffix == ".safetensors" for path in model_files):
        raise ValueError("base model directory must contain its safetensors weights")
    paths.update(model_files)
    for spec in config["splits"].values():
        if set(spec) != {"features", "radii", "attributes", "height_measurement_source", "radius", "height"}:
            raise ValueError("each split requires feature banks, height provenance and both properties")
        paths.update(Path(spec[key]).resolve() for key in ("features", "radii", "attributes", "height_measurement_source"))
        for prop in ("radius", "height"):
            if set(spec[prop]) != {"questions", "labels", "cohort"}:
                raise ValueError("each property requires separate questions, labels and cohort files")
            paths.update(Path(value).resolve() for value in spec[prop].values())
    if any(not path.is_file() for path in paths):
        raise ValueError("a required model or cohort input is missing")
    return sorted(paths)


def verify_hashes(expected, files):
    actual = {str(Path(path).resolve()): sha256(path) for path in files}
    if not expected or set(expected) != set(actual):
        raise ValueError("pin the complete input set before running")
    if actual != expected:
        raise ValueError("pinned input hash changed")
    return actual


def pin_config(source, output):
    config = _read(source)
    files = required_inputs(config)
    config["input_sha256"] = {str(Path(path).resolve()): sha256(path) for path in files}
    _write(output, config)
    return config


def read_split(spec, counts):
    declaration = normalize_measurement_source(_read(spec["height_measurement_source"]))
    if declaration["input_id"] != sha256(spec["attributes"]):
        raise ValueError("height declaration differs from the split's actual attributes")
    datasets, records, all_rows = {}, [], []
    for prop in ("radius", "height"):
        rows, labels, cohort = (_read(spec[prop][key]) for key in ("questions", "labels", "cohort"))
        if (not isinstance(rows, list) or len(rows) != counts[prop] or not isinstance(labels, list)
                or cohort != [row["qid"] for row in rows]):
            raise ValueError("complete fixed cohort identities/counts changed")
        canonical = [_canonical_label(label) for label in labels]
        labelmap = {row["qid"]: row for row in canonical}
        if len(labelmap) != len(labels) or len(set(cohort)) != len(rows):
            raise ValueError("duplicate question or label identity")
        if not set(cohort) <= labelmap.keys():
            raise ValueError("missing label from the fixed cohort")
        datasets[prop] = SceneDataset(features=spec["features"], radii=spec["radii"],
            attributes=spec["attributes"], rows=rows, feature_dim=1280, point_cloud_type="gt")
        dataset, checked = datasets[prop], {}
        for index, row in enumerate(rows):
            assignment = dataset._validate_assigned_ids(row["assigned_ids"],
                dataset.attributes[row["scene_id"]]["locs"].shape[0])
            key = (row["scene_id"], tuple(assignment.tolist()))
            if key not in checked:
                try:
                    _, scene, _, cpu = _cpu_input(dataset, index)
                except KeyError as error:
                    raise ValueError("queried candidate is missing from the saved scene assignment") from error
                packet = _packet(cpu, NaturalQACondition(measurement=prop), 100,
                    measurement_source=declaration if prop == "height" else None, scene_id=scene)
                checked[key] = {record.source_row for record in packet.records}
            if not set(_candidate_source_rows(row["prompt"])) <= checked[key]:
                raise ValueError("both queried candidates must be active in the saved scene assignment")
            records.append((prop, index, target_source(row, labelmap[row["qid"]])))
        all_rows.extend(rows)
    if not {row["scene_id"] for row in all_rows} <= {
            scene for scene, verified in declaration["verified_scenes"].items() if verified}:
        raise ValueError("height declaration must cover every split scene")
    return datasets, records, declaration, all_rows


def load_protocol(path):
    config = _read(path)
    hashes = verify_hashes(config["input_sha256"], required_inputs(config))
    loaded = {role: read_split(spec, COUNTS[role]) for role, spec in config["splits"].items()}
    held_out = _read(config["held_out_scenes"])
    if (not isinstance(held_out, list) or not held_out or len(set(held_out)) != len(held_out)
            or any(not isinstance(scene, str) or not scene for scene in held_out)):
        raise ValueError("held_out_scenes must be a nonempty unique scene list without labels")
    check_splits({role: value[3] for role, value in loaded.items()}, held_out)
    host = _read(config["host_config"])["host"]
    source_hashes = {file.name: sha256(file) for file in sorted(Path(__file__).parent.glob("*.py"))}
    return host, loaded, dict(source_hashes=source_hashes, input_hashes=hashes, protocol_hash=sha256(path))


def _restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def train_group(core, selected, optimizer, scaler, examples, loss_for):
    initial_rng = lora._rng_state()
    for attempt in range(9):
        optimizer.zero_grad(set_to_none=True)
        if attempt:
            _restore_rng(initial_rng)
        losses = []
        with lora.lora_training_mode(core):
            for example in examples:
                loss = loss_for(example)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite forward loss; group incomplete")
                losses.append(float(loss.detach()))
                scaler.scale(loss / len(examples)).backward()
            scaler.unscale_(optimizer)
            if any(parameter.grad is None for parameter in selected.parameters):
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("missing LoRA gradient")
            if any(not torch.isfinite(parameter.grad).all() for parameter in selected.parameters):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                if attempt == 8:
                    raise RuntimeError("overflow persisted after 8 same-group retries")
                continue
            norm = torch.nn.utils.clip_grad_norm_(selected.parameters, 1., error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        return dict(loss=sum(losses) / len(losses), gradient_norm=float(norm),
                    overflow_retries=attempt, scale=scaler.get_scale(), group_size=len(examples))
    raise AssertionError("unreachable")


@contextmanager
def arm_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".worker.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0); lock.write(b"\0"); lock.flush(); lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def checkpoint_proof(directory, loaded, stage):
    path = Path(loaded["path"]).resolve()
    if path not in {(Path(directory) / name).resolve() for name in ("last.pt", "previous.pt")}:
        raise ValueError("validation requires the actual loaded last/previous checkpoint")
    expected = {"pilot": PILOT_ROWS, "epoch": TRAIN_ROWS}[stage]
    validate_progress(loaded["row_position"], loaded["step"])
    if loaded["row_position"] != expected:
        raise ValueError("loaded checkpoint does not match the requested completed stage")
    return dict(path=str(path), sha256=sha256(path), row_position=loaded["row_position"], step=loaded["step"])


def seal_stage(directory, stage, checkpoint, position, step, identity):
    directory = Path(directory)
    folder = directory / "stages" / stage
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "last.pt"
    source = Path(checkpoint).resolve()
    if source not in {(directory / name).resolve() for name in ("last.pt", "previous.pt")} | {destination.resolve()}:
        raise ValueError("seal only the actual loaded/saved checkpoint of this arm")
    proof = checkpoint_proof(source.parent, dict(path=str(source), row_position=position, step=step), stage)
    proof["path"] = str(destination.resolve())
    manifest = dict(identity=identity, checkpoint=proof)
    if destination.exists():
        if sha256(destination) != proof["sha256"]:
            raise ValueError("sealed stage checkpoint changed")
    else:
        os.link(source, destination)
    if (folder / "manifest.json").exists():
        if _read(folder / "manifest.json") != manifest:
            raise ValueError("sealed stage identity changed")
    else:
        _write(folder / "manifest.json", manifest)
    return proof


def sealed_stage(directory, stage, identity):
    folder = Path(directory) / "stages" / stage
    manifest = _read(folder / "manifest.json")
    proof = manifest["checkpoint"]
    if (manifest["identity"] != identity or proof["path"] != str((folder / "last.pt").resolve())
            or proof["sha256"] != sha256(folder / "last.pt")):
        raise ValueError("sealed stage checkpoint or source identity changed")
    checkpoint_proof(folder, dict(path=proof["path"], row_position=proof["row_position"], step=proof["step"]), stage)
    return folder


def merge_lora_state(source, trained):
    """Replace exactly the source host's A/B tensors, preserving every other value."""
    state = source.get("model", source)
    if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("source host must contain a tensor state dictionary")
    pattern = re.compile(r".*\.lora_[AB]\.default\.weight")
    names = {name for name in state if pattern.fullmatch(name)}
    if (not names or set(trained) != names
            or any(".lora_" in name and name not in names for name in state)):
        raise ValueError("trained LoRA names must exactly cover the source host's A/B tensors")
    for name, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("nonfinite source host tensor")
        if name in names:
            replacement = trained[name]
            if (not isinstance(replacement, torch.Tensor) or replacement.dtype != torch.float32
                    or replacement.device.type != "cpu" or replacement.shape != value.shape
                    or not torch.isfinite(replacement).all()):
                raise ValueError("trained LoRA shape, dtype or value differs")
    merged = dict(state)
    merged.update(trained)
    return dict(source, model=merged) if "model" in source else merged


def export_host(host, identity, training_output, output):
    source = Path(host["host_checkpoint"]).resolve()
    if identity["input_hashes"].get(str(source)) != sha256(source):
        raise ValueError("source host checkpoint hash changed")
    work = Path(training_output) / identity["arm"] / "training"
    stage = sealed_stage(work, "epoch", identity)
    payload = torch.load(stage / "last.pt", map_location="cpu", weights_only=True)
    loaded = dict(path=str(stage / "last.pt"), row_position=payload.get("row_position"), step=payload.get("step"))
    proof = checkpoint_proof(stage, loaded, "epoch")
    if (payload.get("format") != 2 or payload.get("checkpoint_boundary") != "optimizer_boundary"
            or payload.get("pending_microbatches") != 0 or payload.get("identity") != identity):
        raise ValueError("epoch checkpoint source or optimizer boundary differs")
    if completed_stage(Path(training_output) / identity["arm"], "epoch", identity, proof) is None:
        raise ValueError("export requires a completed epoch")
    trained = payload.get("lora", {})
    if len(trained) != EXPECTED_TENSORS or sum(value.numel() for value in trained.values()) != EXPECTED_LORA:
        raise ValueError("epoch LoRA tensor count differs")
    original = torch.load(source, map_location="cpu", weights_only=True)
    merged = merge_lora_state(original, trained)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "host.pt"
    with destination.open("xb") as stream:
        torch.save(merged, stream); stream.flush(); os.fsync(stream.fileno())
    manifest = dict(status="complete", identity=identity, source_host=str(source), source_host_sha256=sha256(source),
        epoch_checkpoint=proof, changed_parameters=sorted(trained), changed_numel=EXPECTED_LORA,
        unchanged_tensors=len(original.get("model", original)) - len(trained),
        host_checkpoint=str(destination.resolve()), host_checkpoint_sha256=sha256(destination),
        host=dict(host, host_checkpoint=str(destination.resolve())))
    _write(output / "manifest.json", manifest)
    return manifest


def resume_action(action, resume, checkpoint_exists, updates_exist):
    if not resume:
        return "fresh"
    if action == "smoke":
        raise ValueError("smoke uses a fresh output root")
    if checkpoint_exists:
        return "load"
    if action == "train" and not updates_exist:
        return "fresh"
    raise ValueError("no clean checkpoint exists for this continuation")


def completed_stage(output, stage, identity, proof):
    path = Path(output) / ("SUCCESS_" + stage + ".json")
    if not path.exists():
        return None
    result = _read(path)
    if (result.get("status") != "complete" or result.get("arm") != identity["arm"]
            or result.get("protocol_sha256") != identity["protocol_hash"]
            or result.get("source_hashes") != identity["source_hashes"]
            or result.get("checkpoint_sha256") != proof["sha256"]
            or result.get("row_position") != proof["row_position"]
            or result.get("successful_updates") != proof["step"]):
        raise ValueError("completed stage checkpoint or invocation identity changed")
    return result


def _question_identity(qid, scene, question, cpu, condition, prop):
    binding = dict(qid=qid, scene_id=scene, task=prop, condition=condition.name)
    digest = hashlib.sha256(_bytes(dict(binding, question=question,
        assigned_ids=cpu["assigned_ids"].tolist()))).hexdigest()
    return dict(binding, input_sha256=digest)


def evaluate(core, records, inputs, output, identity, proof, *, stopped=(), run_one=None):
    with lora.complete_answer_stopping(core) as answer_stopping:
        return _evaluate(core, records, inputs, output, identity, proof,
                         stopped=stopped, run_one=run_one, answer_stopping=answer_stopping)


def _evaluate(core, records, inputs, output, identity, proof, *, stopped, run_one, answer_stopping):
    run_one = _run_one if run_one is None else run_one
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(checkpoint=proof, identity=identity, questions=len(records), answer_stopping=answer_stopping,
                    conditions=["absent", "lexical"], repetition_penalty=1., maximum_length=2502, max_new_tokens=32)
    path = output / "manifest.json"
    if path.exists():
        if _read(path) != manifest:
            raise ValueError("validation checkpoint or input identity changed")
    else:
        _write(path, manifest)
    terminal = output / "SUCCESS.json"
    if terminal.exists():
        result = _read(terminal)
        if result["checkpoint"] != proof or result["predictions_sha256"] != sha256(output / "predictions.sqlite3"):
            raise ValueError("completed validation checkpoint or predictions changed")
        database = sqlite3.connect((output / "predictions.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
        try:
            counts = {condition: dict(n=count, correct=correct) for condition, count, correct in
                      database.execute("SELECT condition,COUNT(*),SUM(correct) FROM answers GROUP BY condition")}
        finally:
            database.close()
        if (result.get("status") != "complete" or result.get("protocol_sha256") != identity["protocol_hash"]
                or result.get("questions") != manifest["questions"] or result.get("counts") != counts
                or set(counts) != set(manifest["conditions"])
                or any(value["n"] != manifest["questions"] for value in counts.values())):
            raise ValueError("completed validation counts or protocol changed")
        return result
    database = sqlite3.connect(output / "predictions.sqlite3")
    database.execute("PRAGMA journal_mode=WAL"); database.execute("PRAGMA synchronous=FULL")
    database.execute("CREATE TABLE IF NOT EXISTS answers (idx INTEGER, condition TEXT, correct INTEGER, payload TEXT, PRIMARY KEY(idx,condition))")
    try:
        stored = list(database.execute("SELECT idx,condition,correct,payload FROM answers"))
        known = {(index, condition) for index, condition, _, _ in stored}
        expected = {(index, condition) for index in range(len(records)) for condition in ("absent", "lexical")}
        if not known <= expected or len(known) != len(stored):
            raise ValueError("validation matrix differs from the fixed cohort")
        for index, representation, correct, encoded in stored:
            qid, scene, question, cpu, device, condition, packet, target, prop = inputs(records[index], representation)
            payload = json.loads(encoded)
            binding = _question_identity(qid, scene, question, cpu, condition, prop)
            if (any(payload.get(key) != value for key, value in binding.items())
                    or type(correct) is not int or correct not in (0, 1)
                    or native_hit(payload, target, cpu["assigned_ids"][0], core.end_sym) != correct):
                raise ValueError("stored validation payload, assignment or score changed")
        for index, example in enumerate(records):
            for representation in ("absent", "lexical"):
                if (index, representation) in known:
                    continue
                if stopped:
                    return dict(status="paused", completed=len(known), expected=len(expected))
                qid, scene, question, cpu, device, condition, packet, target, prop = inputs(example, representation)
                _, prediction = run_one(core, device, question, packet, condition, codec=None, random_codec=None,
                    max_new_tokens=32, maximum_length=2502, placement="question_tail", capture_only=False, repetition_penalty=1.)
                correct = native_hit(prediction, target, cpu["assigned_ids"][0], core.end_sym)
                payload = dict(prediction, **_question_identity(qid, scene, question, cpu, condition, prop))
                database.execute("INSERT INTO answers VALUES (?,?,?,?)",
                                 (index, representation, correct, json.dumps(payload, allow_nan=False)))
                database.commit(); known.add((index, representation))
        counts = {condition: dict(n=count, correct=correct) for condition, count, correct in
                  database.execute("SELECT condition,COUNT(*),SUM(correct) FROM answers GROUP BY condition")}
        if any(counts.get(condition, {}).get("n") != len(records) for condition in ("absent", "lexical")):
            raise ValueError("validation incomplete")
    finally:
        database.close()
    result = dict(status="complete", checkpoint=proof, protocol_sha256=identity["protocol_hash"], answer_stopping=answer_stopping,
                  counts=counts, questions=len(records), predictions_sha256=sha256(output / "predictions.sqlite3"),
                  selection="fixed pilot/epoch; no hyperparameter or test selection")
    _write(terminal, result)
    return result


def _audit_host(core, selected):
    special = {id(parameter) for module in (core.llama_model.get_input_embeddings(),
               core.llama_model.get_output_embeddings()) for parameter in module.parameters()}
    special.update(id(parameter) for parameter in selected.parameters)
    counts = {"float16": 0, "float32": 0}
    for parameter in core.llama_model.parameters():
        expected = torch.float32 if id(parameter) in special else torch.float16
        if parameter.dtype != expected or parameter.device.type != "cuda":
            raise ValueError("host requires mixed FP16 base and existing FP32 LoRA/embeddings/head on CUDA")
        counts[str(parameter.dtype).split(".")[-1]] += parameter.numel()
    if selected.numel != EXPECTED_LORA or len(selected.parameters) != EXPECTED_TENSORS:
        raise ValueError("existing LoRA tensor count differs")
    return counts


def _inputs(core, datasets, declaration):
    def inputs(example, representation):
        prop, index, target = example
        qid, scene, question, cpu = _cpu_input(datasets[prop], index)
        condition = NaturalQACondition(representation=representation, measurement=prop if representation != "absent" else "radius")
        packet = _packet(cpu, NaturalQACondition(measurement=prop), 100,
                         measurement_source=declaration if prop == "height" else None, scene_id=scene)
        return qid, scene, question, cpu, _on_device(cpu, next(core.llama_model.parameters()).device), condition, packet, target, prop
    return inputs


def _run(args, host, loaded, identity):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("assign exactly one visible CUDA device before starting")
    output = Path(args.output) / args.arm
    work = output / ("smoke" if args.action == "smoke" else "training")
    work.mkdir(parents=True, exist_ok=True)
    identity_path = work / "identity.json"
    if identity_path.exists():
        if (not args.resume and args.action != "evaluate") or _read(identity_path) != identity:
            raise ValueError("existing invocation requires identical --resume identity")
    else:
        if args.resume or args.action == "evaluate" or any(work.iterdir()):
            raise ValueError("no matching clean invocation to resume")
        _write(identity_path, identity)
    status_path = output / (args.action + "_status.json")
    progress = dict(status="loading", action=args.action, arm=args.arm, pid=os.getpid(), started=time.time(),
                    protocol_sha256=identity["protocol_hash"], source_hashes=identity["source_hashes"])
    def report(**values):
        progress.update(values); _write(status_path, progress, replace=True)
    report()
    stopped = []
    previous_handlers = {}
    for number in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[number] = signal.signal(number, lambda *_: stopped.append("signal"))
    try:
        seed_everything(SEED)
        core, selected = lora.load_packet_lora_host(host, device="cuda", expected_numel=EXPECTED_LORA)
        report(actual_dtype=_audit_host(core, selected), gpu=torch.cuda.get_device_name(),
               torch_version=str(torch.__version__), cuda_version=torch.version.cuda, python_version=sys.version,
               transformers_version=importlib.metadata.version("transformers"), peft_version=importlib.metadata.version("peft"), seed=SEED)
        optimizer = torch.optim.AdamW(selected.parameters, lr=LR, weight_decay=.01)
        scaler = torch.amp.GradScaler("cuda")
        optimizer.zero_grad(set_to_none=True)
        position = step = 0
        checkpoint = None
        checkpoint_directory = sealed_stage(work, args.checkpoint_stage, identity) if args.action == "evaluate" else work
        action = resume_action(args.action, args.resume or args.action == "evaluate",
            any((checkpoint_directory / name).exists() for name in ("last.pt", "previous.pt")), any(work.glob("updates_*.jsonl")))
        if action == "load":
            checkpoint = lora.load_checkpoint(checkpoint_directory, core, selected, optimizer, identity=identity, scaler=scaler)
            position, step = checkpoint["row_position"], checkpoint["step"]
            validate_progress(position, step)
        frozen = {name: (parameter.data_ptr(), parameter._version) for name, parameter in core.named_parameters() if not parameter.requires_grad}
        role = "validation" if args.action == "evaluate" else "train"
        datasets, records, declaration, _ = loaded[role]
        inputs = _inputs(core, datasets, declaration)
        def save():
            if any((parameter.data_ptr(), parameter._version) != frozen[name]
                   for name, parameter in core.named_parameters() if name in frozen):
                raise RuntimeError("frozen host parameter changed")
            return lora.save_checkpoint(work, core, selected, optimizer, step=step, row_position=position,
                identity=identity, source_checkpoint=host["host_checkpoint"], pending_microbatches=0, scaler=scaler)
        def loss_for(example):
            _, scene, question, cpu, device, condition, packet, target, prop = inputs(example, args.arm)
            prefix = lora.capture_prefix(core, device, question, packet, condition,
                measurement_source=declaration if prop == "height" else None, scene_id=scene)
            source_pair = tuple(int(value) for value in re.findall(r"<OBJ([0-9]{3})>", datasets[prop].anno[example[1]]["prompt"]))
            native = lora.native_target(core.llama_tokenizer, target, assigned_ids=cpu["assigned_ids"],
                scene_mask=cpu["scene_mask"], eligible_source_ids=source_pair)
            answer = lora.short_answer_target(core, native)
            return lora.next_token_loss(core, prefix, answer, selected)
        if args.action == "evaluate":
            proof = checkpoint_proof(checkpoint_directory, checkpoint, args.checkpoint_stage)
            result = evaluate(core, records, inputs, output / ("validation_" + args.checkpoint_stage), identity, proof, stopped=stopped)
            report(**result)
            return result
        if args.action == "train" and checkpoint is not None:
            if position == (PILOT_ROWS if args.stop_after == "pilot" else TRAIN_ROWS):
                proof = seal_stage(work, args.stop_after, checkpoint["path"], position, step, identity)
                terminal = completed_stage(output, args.stop_after, identity, proof)
                if terminal is not None:
                    report(**terminal)
                    return terminal
                report(status="complete", row_position=position, successful_updates=step, checkpoint_sha256=proof["sha256"])
                _write(output / ("SUCCESS_" + args.stop_after + ".json"), progress)
                return progress
        if args.action == "smoke":
            metrics = train_group(core, selected, optimizer, scaler, [records[0], records[COUNTS["train"]["radius"]]], loss_for)
            position, step = 2, 1
            path = save()
            before = [parameter.detach().cpu().clone() for parameter in selected.parameters]
            roundtrip = lora.load_checkpoint(work, core, selected, optimizer, identity=identity, scaler=scaler)
            if (roundtrip["row_position"] != 2 or roundtrip["step"] != 1
                    or any(not torch.equal(parameter.detach().cpu(), value) for parameter, value in zip(selected.parameters, before))):
                raise RuntimeError("CUDA checkpoint round-trip failed")
            report(status="complete", checkpoint_sha256=sha256(path), metrics=metrics,
                   peak_memory_allocated=torch.cuda.max_memory_allocated())
            _write(output / "SUCCESS_smoke.json", progress)
            return progress
        smoke = _read(output / "SUCCESS_smoke.json")
        if smoke.get("status") != "complete" or smoke.get("protocol_sha256") != identity["protocol_hash"]:
            raise ValueError("this arm needs its own finite backward/round-trip smoke")
        end = PILOT_ROWS if args.stop_after == "pilot" else TRAIN_ROWS
        if args.stop_after == "epoch":
            pilot = _read(output / "SUCCESS_pilot.json")
            validation = _read(output / "validation_pilot/SUCCESS.json")
            sealed = _read(sealed_stage(work, "pilot", identity) / "manifest.json")
            if (not args.resume or pilot.get("status") != "complete" or validation.get("status") != "complete"
                    or pilot.get("protocol_sha256") != identity["protocol_hash"]
                    or validation.get("protocol_sha256") != identity["protocol_hash"]
                    or validation["checkpoint"]["sha256"] != pilot["checkpoint_sha256"]
                    or sealed["checkpoint"]["sha256"] != pilot["checkpoint_sha256"]):
                raise ValueError("continue the same completed pilot only after its full validation")
        if position > end:
            raise ValueError("checkpoint is beyond requested stage")
        if checkpoint is None:
            save()
        order = ordered_indices(COUNTS["train"]["radius"], COUNTS["train"]["height"])
        report(status="training", permutation_sha256=hashlib.sha256(_bytes(order)).hexdigest(),
               row_position=position, successful_updates=step)
        log_path = work / ("updates_" + str(time.time_ns()) + ".jsonl")
        for first, last in update_groups(position, end):
            if stopped:
                save(); report(status="paused", row_position=position, successful_updates=step)
                return progress
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step)
            metrics = train_group(core, selected, optimizer, scaler, [records[order[index]] for index in range(first, last)], loss_for)
            position = last; step += 1; validate_progress(position, step)
            with log_path.open("ab") as log:
                log.write(json.dumps(dict(step=step, next_row=position, lr=optimizer.param_groups[0]["lr"], **metrics), allow_nan=False).encode() + b"\n")
                log.flush(); os.fsync(log.fileno())
            if step % 16 == 0:
                save()
            report(row_position=position, successful_updates=step, last_metrics=metrics)
        path = save()
        proof = seal_stage(work, args.stop_after, path, position, step, identity)
        report(status="complete", row_position=position, successful_updates=step, checkpoint_sha256=proof["sha256"])
        terminal = output / ("SUCCESS_" + args.stop_after + ".json")
        if not terminal.exists():
            _write(terminal, progress)
        elif _read(terminal)["checkpoint_sha256"] != progress["checkpoint_sha256"]:
            raise ValueError("a completed stage checkpoint changed")
        return progress
    except BaseException as error:
        report(status="failed", error=repr(error))
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pin", "validate", "smoke", "train", "evaluate", "export-host"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-output", type=Path)
    parser.add_argument("--arm", choices=("absent", "lexical"))
    parser.add_argument("--stop-after", choices=("pilot", "epoch"), default="pilot")
    parser.add_argument("--checkpoint-stage", choices=("pilot", "epoch"), default="pilot")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "pin":
        if args.output is None:
            parser.error("pin requires a new output config file")
        pin_config(args.config, args.output)
        print(json.dumps(dict(status="pinned", config=str(args.output))))
        return
    host, loaded, common = load_protocol(args.config)
    if args.action == "validate":
        print(json.dumps(dict(status="validated", counts=COUNTS, input_files=len(common["input_hashes"]), protocol_sha256=common["protocol_hash"])))
        return
    if args.action == "export-host":
        if args.arm is None or args.output is None or args.training_output is None:
            parser.error("export-host requires arm, training-output and a new output directory")
        with arm_lock(args.training_output / args.arm):
            manifest = export_host(host, dict(common, arm=args.arm), args.training_output, args.output)
        print(json.dumps(dict(status=manifest["status"], host_checkpoint=manifest["host_checkpoint"])))
        return
    if args.arm is None or args.output is None or (args.action == "smoke" and args.resume):
        parser.error("GPU actions require an arm/output; smoke cannot resume")
    identity = dict(common, arm=args.arm)
    with arm_lock(args.output / args.arm):
        print(json.dumps(_run(args, host, loaded, identity), sort_keys=True))


if __name__ == "__main__":
    main()
