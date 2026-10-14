"""Score a completed run on a fixed cohort and optionally export diagnostic outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

from . import natural_qa_metrics, prediction_scoring
from .prepare_questions import _finish, _ids, _keys, _path, _read, _write


def score(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"output must be a new directory: {output}")
    artifacts = {}
    config = _read(config_path, artifacts, "config")
    _keys(config, {"run", "references", "cohort", "conditions", "contrasts", "seed", "family_size"},
          {"replicates", "exploratory_linear_contrasts", "scoring", "questions",
           "pair_metadata", "export_scored_rows"})
    scoring = config.get("scoring", "natural_qa")
    export = config.get("export_scored_rows", False)
    if scoring not in ("natural_qa", "native_id") or type(export) is not bool:
        raise ValueError("invalid scoring mode or export_scored_rows flag")
    if (scoring == "native_id") != ("questions" in config and "pair_metadata" in config):
        raise ValueError("native_id scoring requires questions and pair_metadata")
    if scoring == "natural_qa" and ({"questions", "pair_metadata"} & config.keys()):
        raise ValueError("controlled metadata requires native_id scoring")
    base = Path.cwd()
    run = _path(base, config["run"])
    status = _read(run / "status.json", artifacts, "run_status")
    if (not isinstance(status, dict) or status.get("status") != "complete"
            or status.get("schema_version") != "scale_interface_run/v1"
            or status.get("action") != "generate-scene"
            or not isinstance(status.get("artifacts"), dict)):
        raise ValueError("run requires a complete generate-scene status with artifact hashes")
    result = _read(run / "result.json", artifacts, "run_result")
    plan = _read(run / "plan.json", artifacts, "run_plan")
    if not isinstance(result, dict) or result.get("status") != "complete":
        raise ValueError("run result must declare status complete")
    if not isinstance(plan, dict):
        raise ValueError("run plan must be a JSON object")
    run_qids = _ids(result.get("question_ids"), "completed run question IDs")
    run_conditions = _ids(result.get("condition_names"), "completed run conditions")
    conditions = _ids(config["conditions"], "scoring conditions")
    if any(not isinstance(value, str) or not value.strip() for value in config["conditions"]):
        raise ValueError("condition names must be nonempty strings")
    if set(conditions) != set(run_conditions):
        raise ValueError("scoring conditions must match the complete run conditions")
    eligible = _ids(plan.get("selected_qids"), "preflight question IDs")
    declared = plan.get("conditions")
    if not isinstance(declared, list) or any(not isinstance(item, dict) for item in declared):
        raise ValueError("preflight conditions must be a list of condition specifications")
    plan_conditions = _ids([item.get("name") for item in declared], "preflight conditions")
    if not set(run_qids) <= set(eligible) or not set(run_conditions) <= set(plan_conditions):
        raise ValueError("completed run contains questions or conditions outside preflight")
    requested = plan.get("requested_generation", {})
    if not isinstance(requested, dict):
        raise ValueError("requested_generation must be an object")
    for field, actual in (("question_ids", run_qids), ("condition_names", run_conditions)):
        if requested.get(field) is not None and set(_ids(requested[field], field)) != set(actual):
            raise ValueError("completion differs from requested generation")
    references = _read(_path(base, config["references"]), artifacts, "references")
    cohort = _ids(_read(_path(base, config["cohort"]), artifacts, "cohort"))
    if not set(cohort) <= set(run_qids):
        raise ValueError("fixed cohort includes questions absent from the completed run")
    if Path(artifacts["cohort"]["path"]) == (run / "cohort.json").resolve():
        if (status["artifacts"].get("cohort.json") != artifacts["cohort"]["sha256"]
                or set(cohort) != set(run_qids)):
            raise ValueError("generated cohort differs from the sealed completed run")
    path = (run / "predictions.jsonl").resolve()
    predictions, digest, pairs = [], hashlib.sha256(), set()
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                raise ValueError(f"blank prediction line {number}")
            row = json.loads(raw.decode("utf-8-sig" if number == 1 else "utf-8"))
            if not isinstance(row, dict):
                raise ValueError("predictions must be JSON objects")
            pair = (_ids([row.get("condition")], "prediction condition")[0],
                    _ids([row.get("qid")], "prediction question ID")[0])
            if pair in pairs:
                raise ValueError("duplicate prediction condition/question ID")
            pairs.add(pair)
            predictions.append(row)
    artifacts["predictions"] = {"path": str(path), "sha256": digest.hexdigest()}
    for filename, key in (("result.json", "run_result"), ("plan.json", "run_plan"),
                          ("predictions.jsonl", "predictions")):
        if status["artifacts"].get(filename) != artifacts[key]["sha256"]:
            raise ValueError(f"completed run artifact hash mismatch: {filename}")
    for alias, key in (("predictions_sha256", "predictions"), ("result_sha256", "run_result")):
        if alias in status and status[alias] != artifacts[key]["sha256"]:
            raise ValueError(f"completed run hash alias mismatch: {alias}")
    expected = {(condition, qid) for condition in run_conditions for qid in run_qids}
    if pairs != expected:
        raise ValueError("incomplete pairing or unexpected predictions in completed run")
    if type(result.get("completed_predictions")) is not int or result["completed_predictions"] != len(predictions):
        raise ValueError("completion count differs from actual predictions")
    evaluated, outcomes = predictions, None
    if export or scoring == "native_id":
        effective = _read(run / "effective_config.json", artifacts, "effective_config")
        if status["artifacts"].get("effective_config.json") != artifacts["effective_config"]["sha256"]:
            raise ValueError("completed run artifact hash mismatch: effective_config.json")
        end_sym = effective.get("host", {}).get("model", {}).get("end_sym")
        questions, metadata = None, None
        if scoring == "native_id":
            questions = _read(_path(base, config["questions"]), artifacts, "questions")
            metadata = _read(_path(base, config["pair_metadata"]), artifacts, "pair_metadata")
            recorded_path = status.get("input_roles", {}).get("data.rows")
            recorded_hash = status.get("inputs", {}).get(recorded_path)
            if not recorded_hash or recorded_hash != artifacts["questions"]["sha256"]:
                raise ValueError("controlled questions differ from recorded inference input")
        evaluated, outcomes = prediction_scoring.score_predictions(
            predictions, references, cohort=cohort, end_sym=end_sym, scoring=scoring,
            questions=questions, pair_metadata=metadata)
    metrics = natural_qa_metrics.evaluate_natural_qa(
        evaluated, references, cohort_qids=cohort, conditions=conditions,
        contrasts=config["contrasts"], seed=config["seed"], family_size=config["family_size"],
        replicates=config.get("replicates", 10000),
        exploratory_linear_contrasts=config.get("exploratory_linear_contrasts"))
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "metrics.json", metrics)
    if export:
        _write(output / "scored_rows.json", outcomes)
    return _finish(output, {
        "status": "complete", "command": "score_natural_qa", "schema_version": 1,
        "fixed_cohort_questions": len(cohort), "complete_run_questions": len(run_qids),
        "completed_predictions": len(predictions), "conditions": conditions,
        "cohort_selected_from_predictions": False,
        "metric_scope": "experiment exact-match protocol, not the official ScanQA evaluator",
        "scoring": scoring, "scored_rows": "scored_rows.json" if export else None,
        "metrics": "metrics.json",
    }, artifacts, [Path(__file__), Path(__file__).with_name("prepare_questions.py"),
                   Path(natural_qa_metrics.__file__), Path(prediction_scoring.__file__),
                   Path(__file__).with_name("scanqa_protocol.py")])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    score(args.config, args.output)


if __name__ == "__main__":
    main()
