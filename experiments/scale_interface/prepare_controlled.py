"""Export every eligible direct/named object pair from ordered scene geometry.

Run from the repository root. Config input paths resolve relative to the working
directory. The scene_objects JSON maps scene IDs to feature-row-ordered objects
with source_id, category and rho/radius_valid or height_m/height_valid. Heights
require a separately checked measurement_source declaration. No model is loaded.
"""
import argparse
from pathlib import Path

from . import controlled_protocol, controlled_semantic_probe, metric_pair_numeric_prompt
from . import prepare_questions, scanqa_protocol, scene_extents, scene_measurements, sqa3d_protocol
from .prepare_questions import _finish, _keys, _path, _read, _write


def prepare(config_path, output):
    """Write inference-only rows, separate scoring data, cohorts and hash manifest."""
    config_path, output = Path(config_path).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"output must be a new directory: {output}")
    artifacts = {}
    config = _read(config_path, artifacts, "config")
    _keys(config, {"property", "scene_objects"}, {"measurement_source", "instruction"})
    measurement = config["property"]
    if measurement not in controlled_protocol.TARGETS:
        raise ValueError("property must be radius or height")
    instruction = config.get("instruction")
    if instruction not in (None, "candidate"):
        raise ValueError("controlled instruction must be candidate or null")
    if (measurement == "height") != ("measurement_source" in config):
        raise ValueError("measurement_source is required only for height")
    base = Path.cwd()
    scene_objects = _read(_path(base, config["scene_objects"]), artifacts, "scene_objects")
    if not isinstance(scene_objects, dict) or not scene_objects:
        raise ValueError("scene_objects must be a nonempty scene-to-ordered-objects mapping")
    source = (_read(_path(base, config["measurement_source"]), artifacts, "measurement_source")
              if measurement == "height" else None)
    built = controlled_protocol.build_native_id_pairs(
        scene_objects, measurement=measurement, measurement_source=source)
    direct = controlled_protocol.append_instruction(built["direct_rows"], instruction)
    named = built["named_rows"]
    cohorts = {"direct": [row["qid"] for row in direct],
               "named": [row["qid"] for row in named]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "scoring").mkdir()
    _write(output / "direct.json", direct)
    _write(output / "named.json", named)
    for task, qids in cohorts.items():
        _write(output / f"cohort_{task}.json", qids)
    _write(output / "scene_measurements.json", built["scene_measurements"])
    _write(output / "scoring/references.json", built["references"])
    _write(output / "scoring/pair_metadata.json", built["pair_metadata"])
    _write(output / "scoring/selection_summary.json", built["summary"])
    return _finish(output, {
        "status": "complete", "command": "prepare_controlled", "schema_version": 1,
        "protocol": built["protocol"], "property": measurement,
        "measurement_source": built["measurement_source"],
        "selection": "all_eligible_unique_category_pairs", "pair_cap": None,
        "question_cap": None, "selection_uses_geometry": True,
        "prompt_uses_targets": False,
        "geometry_verification": "caller-supplied; declaration validation does not certify geometry",
        "runtime_preflight": "not performed; pair eligibility does not certify whole-scene inputs",
        "instruction": {"direct": instruction, "named": None},
        "instruction_text": controlled_protocol.CANDIDATE_INSTRUCTION if instruction else None,
        "pairs": len(built["pair_metadata"]), "questions": len(direct) + len(named),
        "scenes": built["summary"]["scenes"],
        "cohorts": {task: {"path": f"cohort_{task}.json", "questions": len(qids),
                            "qids_sha256": controlled_protocol._qid_hash(qids)}
                    for task, qids in cohorts.items()},
        "inference_rows": {"direct": "direct.json", "named": "named.json"},
        "scene_measurements": "scene_measurements.json",
        "scoring_references": "scoring/references.json",
        "scoring_pair_metadata": "scoring/pair_metadata.json",
        "selection_summary": "scoring/selection_summary.json",
    }, artifacts, [Path(__file__), *[Path(module.__file__) for module in (
        controlled_protocol, controlled_semantic_probe, metric_pair_numeric_prompt,
        prepare_questions, scanqa_protocol, scene_extents, scene_measurements, sqa3d_protocol)]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.config, args.output)


if __name__ == "__main__":
    main()
