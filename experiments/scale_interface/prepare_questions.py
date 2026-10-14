"""Prepare ScanQA/SQA3D questions with separate references and explicit cohorts."""
import argparse
import hashlib
import json
from pathlib import Path

from . import scanqa_protocol, sqa3d_protocol


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path, artifacts, name):
    path = Path(path).resolve()
    raw = path.read_bytes()
    artifacts[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    return json.loads(raw.decode("utf-8-sig"))


def _path(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("input paths must be nonempty strings")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _keys(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    missing, unknown = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"configuration keys: missing={sorted(missing)}, unknown={sorted(unknown)}")


def _ids(values, name="question IDs"):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty JSON list")
    result = [scanqa_protocol._question_key(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {name}")
    return result


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _finish(output, manifest, artifacts, code_files):
    # Refuse a source that changed while exports or metrics were being built.
    for artifact in artifacts.values():
        if _sha(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"input changed during command: {artifact['path']}")
    manifest["inputs"] = artifacts
    manifest["code_sha256"] = {path.name: _sha(path) for path in code_files}
    manifest["outputs_sha256"] = {
        str(path.relative_to(output)).replace("\\", "/"): _sha(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    _write(output / "manifest.json", manifest)
    return manifest


def _records(value, key):
    if isinstance(value, dict) and key in value:
        value = value[key]
    if not isinstance(value, list):
        raise ValueError(f"official {key} must be a JSON list or an object containing {key}")
    return value


def prepare(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"output must be a new directory: {output}")
    artifacts = {}
    config = _read(config_path, artifacts, "config")
    _keys(config, {"dataset", "split", "seed", "questions", "scene_object_ids", "cohort"}, {"annotations", "instruction"})
    dataset = config["dataset"]
    if dataset not in ("scanqa", "sqa3d") or config["split"] not in ("train", "val", "test"):
        raise ValueError("dataset must be scanqa/sqa3d and split must be train/val/test")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if (dataset == "sqa3d") != ("annotations" in config):
        raise ValueError("annotations is required only for SQA3D")
    base = Path.cwd()
    records = _records(_read(_path(base, config["questions"]), artifacts, "questions"), "questions")
    identities = _read(_path(base, config["scene_object_ids"]), artifacts, "scene_object_ids")
    if not isinstance(identities, dict) or not identities:
        raise ValueError("scene_object_ids must be a nonempty scene-to-row-IDs mapping")
    counts = {}
    for scene, ids in identities.items():
        scanqa_protocol._scene_id(scene)
        if (not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids)
                or ids != list(range(len(ids)))):
            raise ValueError("scene_object_ids must contain complete sequential input row IDs starting at zero")
        counts[scene] = len(ids)
    protocol = scanqa_protocol if dataset == "scanqa" else sqa3d_protocol
    rows = protocol.build_questions(records, scene_object_counts=counts)
    coverage = None
    if dataset == "sqa3d":
        from .controlled_protocol import build_full_sqa3d_inputs
        coverage = build_full_sqa3d_inputs(records, scene_object_counts=counts)
    selection = config["cohort"]
    _keys(selection, {"mode"}, {"path", "per_scene"})
    mode = selection["mode"]
    if mode == "all_available":
        _keys(selection, {"mode"})
        qids = _ids([row["qid"] for row in rows])
    elif mode == "question_ids":
        _keys(selection, {"mode", "path"})
        qids = _ids(_read(_path(base, selection["path"]), artifacts, "fixed_question_ids"))
    elif mode == "sqa3d_per_scene" and dataset == "sqa3d":
        _keys(selection, {"mode", "per_scene"})
        qids = _ids(sqa3d_protocol.fixed_question_ids(rows, per_scene=selection["per_scene"]))
    else:
        raise ValueError("unknown cohort mode for this dataset")
    selected = set(qids)
    available = {str(row["qid"]) for row in rows}
    if not selected <= available:
        raise ValueError("fixed cohort includes questions outside available input scenes")
    rows = [row for row in rows if str(row["qid"]) in selected]
    qids = [str(row["qid"]) for row in rows]  # Preserve official source order.
    if dataset == "scanqa":
        references = scanqa_protocol.build_references(records, question_ids=qids)
    else:
        annotations = _records(_read(_path(base, config["annotations"]), artifacts, "annotations"), "annotations")
        references = sqa3d_protocol.build_references(annotations, questions=rows)
    if "instruction" in config:
        from .controlled_protocol import append_instruction
        if config["instruction"] != "natural":
            raise ValueError("natural-QA preparation supports instruction=natural")
        rows = append_instruction(rows, "natural")
    output.mkdir(parents=True, exist_ok=False)
    (output / "scoring").mkdir()
    _write(output / "questions.json", rows)
    _write(output / "cohort.json", qids)
    _write(output / "scoring/references.json", references)
    if coverage is not None:
        _write(output / "input_exclusions.json", coverage["exclusions"])
    return _finish(output, {
        "status": "complete", "command": "prepare_questions", "schema_version": 1,
        "dataset": dataset, "declared_split": config["split"], "seed": config["seed"],
        "selection": selection, "selection_uses_answers": False,
        "instruction": config.get("instruction"),
        "source_questions": coverage["source_questions"] if coverage else len(records),
        "complete_input_coverage": coverage["complete_input_coverage"] if coverage else None,
        "seed_usage": "recorded only; input coverage, explicit IDs, or existing deterministic SQA3D hash protocol",
        "questions": len(rows), "scenes": len({row["scene_id"] for row in rows}),
        "question_summary": scanqa_protocol.summarize_questions(rows),
        "inference_rows": "questions.json", "scoring_references": "scoring/references.json",
    }, artifacts, [Path(__file__), Path(scanqa_protocol.__file__), Path(sqa3d_protocol.__file__),
                   Path(__file__).with_name("controlled_protocol.py")])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.config, args.output)


if __name__ == "__main__":
    main()
