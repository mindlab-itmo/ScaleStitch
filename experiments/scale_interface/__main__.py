import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from .data import SceneDataset
from .loading import load_host, seed_everything
from .channel_controls import channel_control
from .runtime_provenance import RuntimeProvenance


def _public_conditions(config, options):
    from .natural_qa import NaturalQACondition

    if not isinstance(config, dict) or not isinstance(options, dict):
        raise ValueError("configuration and run options must be objects")
    if any(key in config for key in ("codec", "random_codec_seed")):
        raise ValueError("codec settings are not supported by the frozen-host CLI")
    specifications = options.get("conditions")
    if (not isinstance(specifications, list) or not specifications
            or any(not isinstance(value, dict) for value in specifications)):
        raise ValueError("generate-scene requires run.conditions as a nonempty list of objects")
    requested = [NaturalQACondition(**value) for value in specifications]
    if any(value.representation not in {"absent", "lexical", "masked"} for value in requested):
        raise ValueError("the frozen-host CLI supports only absent, lexical and masked representations")
    return requested


def _generate_scene(core, data, config, output, options):
    options = dict(options)
    intervention = options.pop("channel_intervention", "original")
    with channel_control(core, intervention):
        return _generate_scene_controlled(core, data, config, output, options, intervention)


def _generate_scene_controlled(core, data, config, output, options, intervention):
    from .natural_qa import preflight_natural_qa, iter_natural_qa
    from .scene_extents import normalize_measurement_source

    requested = _public_conditions(config, options)
    options = dict(options)
    options.pop("conditions")
    filters = {key: options.pop(key, None) for key in ("condition_names", "question_ids")}
    measurement_options = {}
    if "measurement_source" in options:
        source_path = options.pop("measurement_source")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("run.measurement_source must be a nonempty JSON file path")
        source = normalize_measurement_source(json.loads(Path(source_path).read_text(encoding="utf-8")))
        digest = hashlib.sha256()
        with Path(config["data"]["attributes"]).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if source["input_id"] != digest.hexdigest():
            raise ValueError("height declaration differs from actual attributes")
        measurement_options["measurement_source"] = source
    allowed_options = {"max_new_tokens", "maximum_length", "placement", "constant_cm"}
    if set(options) - allowed_options:
        raise ValueError(f"unknown generate-scene options: {sorted(set(options) - allowed_options)}")
    options.setdefault("max_new_tokens", 32)
    plan = preflight_natural_qa(core, data, requested, **measurement_options, **options)
    metadata = plan.metadata()
    metadata["channel_intervention"] = intervention
    metadata["requested_generation"] = filters
    (output / "plan.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    allowed = {"condition_names": [value.name for value in plan.conditions],
               "question_ids": [row.qid for row in plan.rows]}
    if not allowed["question_ids"]:
        raise ValueError("no questions passed all-condition preflight; see plan.json exclusions")
    selected = {}
    for key, values in filters.items():
        if values is None:
            selected[key] = allowed[key]
            continue
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"run.{key} must be a nonempty list of unique strings")
        missing = set(values) - set(allowed[key])
        if missing:
            raise ValueError(f"run.{key} contains unknown or excluded values: {sorted(missing)}; see plan.json")
        selected[key] = values
    with (output / "cohort.json").open("x", encoding="utf-8") as stream:
        json.dump(selected["question_ids"], stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    count = 0
    iterator = iter_natural_qa(core, data, plan, **selected)
    try:
        with (output / "predictions.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            for row in iterator:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                count += 1
    finally:
        iterator.close()
    if count != len(selected["question_ids"]) * len(selected["condition_names"]):
        raise RuntimeError("incomplete question/condition matrix")
    return {"status": "complete", "completed_predictions": count, **selected}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Object-size interface experiments")
    parser.add_argument("action", choices=["generate-scene"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", __package__, *(sys.argv[1:] if argv is None else argv)]
    with RuntimeProvenance(args.output, args.action, args.config, command=command, torch_module=torch) as provenance:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        _public_conditions(config, config.get("run", {}))
        provenance.configure(config)
        result = _execute(args, config, provenance)
        with (args.output / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        provenance.result = result


def _execute(args, config, provenance):
    if args.action != "generate-scene":
        raise ValueError("the frozen-host CLI supports only generate-scene")
    seed_everything(config.get("seed", 42))
    provenance.phase("loading_data")
    data = SceneDataset(**config["data"])
    declared_policy = config["host"]["model"].get("missing_channel_policy", "reject")
    if declared_policy not in {"reject", "zero_projected_declared"}:
        raise ValueError("invalid model.missing_channel_policy")
    if declared_policy == "zero_projected_declared" and config["host"]["model"].get("host_mode") != "full":
        raise ValueError("declared missing channels require host_mode=full")
    if declared_policy != getattr(data, "missing_channel_policy", "reject"):
        raise ValueError("model and input missing-channel policies disagree")
    provenance.phase("loading_host")
    core = load_host(config["host"], device=config.get("device", "cuda"))
    provenance.record_gpu()
    provenance.phase(args.action)
    options = dict(config.get("run", {}))
    return _generate_scene(core, data, config, args.output, options)



if __name__ == "__main__":
    main()
