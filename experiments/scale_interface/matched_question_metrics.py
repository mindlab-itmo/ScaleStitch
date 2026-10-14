"""Match direct and semantic questions before using the shared QA scorer."""
from collections.abc import Mapping

from .natural_qa_metrics import (
    _cohort, _condition, _qid, _references, _rows, _scene, evaluate_natural_qa,
)


_TASKS = ("direct", "semantic")


def _endpoint(value, conditions):
    if isinstance(value, str):
        value = value.split("|", 1)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("contrast endpoints must be (task, condition) or task|condition")
    task, condition = value
    if task not in _TASKS or condition not in conditions:
        raise ValueError("contrast endpoint names an undeclared task or condition")
    return task + "|" + condition


def _contrasts(contrasts, linear, conditions):
    if not isinstance(contrasts, Mapping) or not contrasts:
        raise ValueError("provide nonempty matched pairwise contrasts")
    pairs = {}
    for name, pair in contrasts.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each contrast needs treatment and control endpoints")
        pairs[name] = tuple(_endpoint(value, conditions) for value in pair)
    converted_linear = None
    if linear is not None:
        if not isinstance(linear, Mapping):
            raise ValueError("exploratory_linear_contrasts must be a mapping")
        converted_linear = {}
        for name, weights in linear.items():
            if not isinstance(weights, Mapping):
                raise ValueError("linear contrast weights must be a mapping")
            converted = {}
            for endpoint, weight in weights.items():
                key = _endpoint(endpoint, conditions)
                if key in converted:
                    raise ValueError("duplicate aliases for one linear contrast endpoint")
                converted[key] = weight
            converted_linear[name] = converted
    return pairs, converted_linear


def evaluate_matched_questions(
    predictions, references, *, rows_by_task, conditions, cohort_pair_ids, contrasts,
    exploratory_linear_contrasts=None, replicates=10000, seed=20260909, family_size=3,
):
    """Score a fixed pair cohort with one direct and one semantic row per pair.

    A pair must share its scene and exact set of accepted reference strings.
    Conditions become ``direct|condition`` and ``semantic|condition``; pair IDs
    replace question IDs only for the existing scene-family bootstrap scorer.
    Contrast endpoints accept those strings or ``(task, condition)`` tuples.

    Reference supersets may include other tasks, whose records are validated.
    Predictions must belong to the declared rows and conditions. Valid pairs
    outside the explicit cohort are validated and counted as ignored; blank,
    invalid or incorrect predictions never remove a selected pair. This does
    not select pairs using recognition performance or any other model output.
    """
    if not isinstance(rows_by_task, Mapping) or set(rows_by_task) != set(_TASKS):
        raise ValueError("rows_by_task must contain exactly direct and semantic")
    if isinstance(conditions, (str, bytes, Mapping)):
        raise ValueError("conditions must be a sequence of names")
    conditions = tuple(_condition(value) for value in conditions)
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be nonempty and unique")
    if cohort_pair_ids is None:
        raise ValueError("an explicit fixed cohort_pair_ids is required")
    paired_rows, question_map = {}, {}
    for task in _TASKS:
        task_pairs = {}
        for row in _rows(rows_by_task[task], task + " rows"):
            qid, pair_id, scene = _qid(row["qid"]), _qid(row["pair_id"]), _scene(row["scene_id"])
            if "question_id" in row and _qid(row["question_id"]) != qid:
                raise ValueError("row qid and question_id disagree")
            if qid in question_map or pair_id in task_pairs:
                raise ValueError("duplicate question or task/pair identity")
            task_pairs[pair_id] = (qid, scene)
            question_map[qid] = (task, pair_id, scene)
        paired_rows[task] = task_pairs
    if not paired_rows["direct"] or paired_rows["direct"].keys() != paired_rows["semantic"].keys():
        raise ValueError("every pair must have one direct and one semantic question")

    reference_rows = list(_rows(references, "references"))
    validated_refs = _references(reference_rows)
    raw_refs = {_qid(row["question_id"]): row for row in reference_rows}
    pair_refs = []
    for pair_id in sorted(paired_rows["direct"]):
        direct_qid, direct_scene = paired_rows["direct"][pair_id]
        semantic_qid, semantic_scene = paired_rows["semantic"][pair_id]
        if direct_scene != semantic_scene:
            raise ValueError("matched questions must have the same scene")
        for qid in (direct_qid, semantic_qid):
            if qid not in validated_refs or validated_refs[qid]["scene_id"] != direct_scene:
                raise ValueError("missing paired reference or reference/row scene mismatch")
        if validated_refs[direct_qid]["answers"] != validated_refs[semantic_qid]["answers"]:
            raise ValueError("matched questions must have the same exact accepted reference strings")
        pair_refs.append({"question_id": pair_id, "scene_id": direct_scene,
                          "answers": list(raw_refs[direct_qid]["answers"])})
    selected = _cohort(cohort_pair_ids, paired_rows["direct"])
    converted_predictions = []
    for row in _rows(predictions, "predictions"):
        qid = _qid(row["qid"])
        if qid not in question_map:
            raise ValueError("prediction question ID is absent from the declared matched tasks")
        task, pair_id, scene = question_map[qid]
        condition = _condition(row["condition"])
        if condition not in conditions or _scene(row["scene_id"]) != scene:
            raise ValueError("prediction condition or matched-row scene mismatch")
        if "question_id" in row and _qid(row["question_id"]) != qid:
            raise ValueError("prediction qid and question_id disagree")
        converted = dict(row)
        converted.update(qid=pair_id, condition=task + "|" + condition)
        if "question_id" in converted:
            converted["question_id"] = pair_id
        converted_predictions.append(converted)
    pairwise, linear = _contrasts(contrasts, exploratory_linear_contrasts, conditions)
    combined_conditions = [task + "|" + condition for task in _TASKS for condition in conditions]
    result = evaluate_natural_qa(
        converted_predictions, pair_refs, conditions=combined_conditions,
        contrasts=pairwise, cohort_qids=selected, exploratory_linear_contrasts=linear,
        replicates=replicates, seed=seed, family_size=family_size,
    )
    result["cohort"]["unit"] = "matched_pair"
    result["matched_questions"] = {
        "tasks": list(_TASKS), "selected_pair_ids": selected,
        "pairs": len(selected), "original_questions": 2 * len(selected),
        "ignored_unmapped_reference_questions": len(validated_refs) - len(question_map),
        "condition_mapping": {task + "|" + condition: {"task": task, "condition": condition}
                              for task in _TASKS for condition in conditions},
    }
    result["interpretation"] += (" Cross-task differences also reflect question wording and category "
                                 "information; they do not isolate object recognition.")
    return result
