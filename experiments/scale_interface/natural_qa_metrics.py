"""Strict paired natural-QA scoring with scene-family cluster intervals.

References are separate ``question_id/scene_id/answers`` records. Predictions
use ``qid/scene_id/condition/prediction``. An explicit cohort permits a larger
reference/prediction population, but duplicates and scene conflicts still fail.
Only the requested conditions may be supplied. No substring matching, answer
selection, or model-output-dependent cohort selection is performed here.

The primary normalization is specific to this experiment, not an implementation
of an official ScanQA evaluator. All accuracies and intervals are fractions.
"""
from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import math
from numbers import Integral, Real
import random
import re


METRICS = ("raw_exact", "lower_whitespace_exact", "protocol_normalized_exact", "punctuation_only_exact")


def normalize_answer(text, *, metric="protocol_normalized_exact"):
    """Preserve internal punctuation, numbers, decimal points and minus signs.

    Protocol normalization lowercases, collapses whitespace, removes terminal
    '.', '!' and '?' characters, and removes one leading a/an/the plus space.
    It does not equate number words, repair spelling, or discard punctuation
    inside an answer. Unicode letters are retained and use Python str.lower().
    """
    if not isinstance(text, str):
        raise ValueError("answer text must be a string")
    if metric == "raw_exact":
        return text
    normalized = " ".join(text.lower().split())
    if metric == "lower_whitespace_exact":
        return normalized
    if metric not in ("protocol_normalized_exact", "punctuation_only_exact"):
        raise ValueError("unknown exact-match normalization")
    normalized = normalized.rstrip(".!?").rstrip()
    if metric == "punctuation_only_exact":
        return normalized
    return re.sub(r"^(?:a|an|the)\s+", "", normalized)


def _qid(value):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)) or not str(value).strip():
        raise ValueError("question ID must be a nonempty string or integer")
    return str(value)


def _scene(value):
    if not isinstance(value, str) or re.fullmatch(r"scene[0-9]{4}_[0-9]{2}", value) is None:
        raise ValueError("scene_id must be a ScanNet scene identifier")
    return value


def _condition(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("condition names must be nonempty strings")
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _rows(values, name):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an iterable of records")
    for row in values:
        if not isinstance(row, Mapping):
            raise ValueError(f"{name} records must be mappings")
        yield row


def _references(rows):
    result = {}
    for row in _rows(rows, "references"):
        qid, scene = _qid(row["question_id"]), _scene(row["scene_id"])
        if qid in result:
            raise ValueError(f"duplicate reference question ID: {qid}")
        answers = row["answers"]
        if (not isinstance(answers, (list, tuple)) or not answers
                or any(not isinstance(answer, str) or not answer.strip() for answer in answers)):
            raise ValueError("reference answers must be a nonempty list of nonblank strings")
        normalized = {metric: {normalize_answer(answer, metric=metric) for answer in answers}
                      for metric in METRICS}
        if any(not (values - {""}) for values in normalized.values()):
            raise ValueError(f"reference answers normalize to empty text: {qid}")
        result[qid] = {"scene_id": scene, "answers": normalized}
    if not result:
        raise ValueError("references cannot be empty")
    return result


def _cohort(qids, references):
    if qids is None:
        return sorted(references)
    if isinstance(qids, (str, bytes, Mapping)):
        raise ValueError("cohort_qids must be an iterable of question IDs")
    selected = [_qid(value) for value in qids]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("cohort_qids must be nonempty and unique")
    if not set(selected) <= references.keys():
        raise ValueError("cohort contains question IDs absent from references")
    return sorted(selected)


def _predictions(rows, references, selected, conditions, allow_outside_cohort):
    selected = set(selected)
    result = {condition: {} for condition in conditions}
    ignored, seen = Counter(), set()
    for row in _rows(rows, "predictions"):
        qid, scene = _qid(row["qid"]), _scene(row["scene_id"])
        condition = _condition(row["condition"])
        if condition not in result:
            raise ValueError(f"unexpected prediction condition: {condition}")
        if (condition, qid) in seen:
            raise ValueError(f"duplicate prediction condition/question ID: {condition}/{qid}")
        seen.add((condition, qid))
        if qid not in references or references[qid]["scene_id"] != scene:
            raise ValueError(f"prediction/reference question or scene mismatch: {qid}")
        prediction = row["prediction"]
        if not isinstance(prediction, str):
            raise ValueError("prediction must be a string, including empty failed answers")
        invalid = row.get("invalid_object_id", False)
        if type(invalid) is not bool:
            raise ValueError("invalid_object_id must be boolean when supplied")
        if qid not in selected:
            if not allow_outside_cohort:
                raise ValueError(f"prediction outside selected cohort: {qid}")
            ignored[condition] += 1
            continue
        texts = {metric: normalize_answer(prediction, metric=metric) for metric in METRICS}
        hits = {metric: int(not invalid and bool(text.strip())
                            and text in references[qid]["answers"][metric])
                for metric, text in texts.items()}
        result[condition][qid] = {"scene_id": scene, "texts": texts, "hits": hits,
                                  "blank": not prediction.strip(), "invalid": invalid}
    for condition, records in result.items():
        if records.keys() != selected:
            missing = sorted(selected - records.keys())
            raise ValueError(f"incomplete pairing for {condition}: missing {len(missing)} question IDs")
    return result, {condition: ignored[condition] for condition in conditions}


def _percentile(ordered, probability):
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _interval(values, confidence):
    ordered = sorted(values)
    tail = (1 - confidence) / 2
    return [_percentile(ordered, tail), _percentile(ordered, 1 - tail)]


def _contrast_specs(contrasts, interactions, conditions, family_size):
    if not isinstance(contrasts, Mapping) or not contrasts:
        raise ValueError("contrasts must map names to (treatment, control) pairs")
    if family_size is not None:
        family_size = _positive_int(family_size, "family_size")
        if len(contrasts) > family_size:
            raise ValueError("family_size cannot be smaller than the declared pairwise family")
    result = {}
    for name, pair in contrasts.items():
        _condition(name)
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each pairwise contrast needs treatment and control")
        treatment, control = map(_condition, pair)
        if treatment == control or treatment not in conditions or control not in conditions:
            raise ValueError("contrasts require two different selected conditions")
        result[name] = {"weights": {treatment: 1, control: -1}, "pair": (treatment, control),
                        "family_size": family_size}
    if interactions is not None:
        if not isinstance(interactions, Mapping):
            raise ValueError("exploratory_linear_contrasts must be a mapping")
        for name, weights in interactions.items():
            _condition(name)
            if name in result or not isinstance(weights, Mapping) or len(weights) < 2:
                raise ValueError("linear contrast needs a unique name and at least two weights")
            if any(condition not in conditions for condition in weights):
                raise ValueError("linear contrast names an unselected condition")
            if any(isinstance(weight, bool) or not isinstance(weight, Real)
                   or not math.isfinite(float(weight)) or weight == 0 for weight in weights.values()):
                raise ValueError("linear contrast weights must be nonzero finite numbers")
            if not math.isclose(sum(weights.values()), 0, rel_tol=0, abs_tol=1e-12):
                raise ValueError("linear contrast weights must sum to zero")
            result[name] = {"weights": dict(weights), "pair": None, "family_size": None}
    return result


def evaluate_natural_qa(
    predictions, references, *, conditions, contrasts, cohort_qids=None,
    replicates=10000, seed=20260909, family_size=3, exploratory_linear_contrasts=None,
):
    """Score complete paired conditions on an explicit, externally fixed cohort.

    ``contrasts`` maps names to (treatment, control). ``family_size=3`` gives
    98.333...% Bonferroni intervals for the primary normalization in addition
    to pointwise 95% intervals. The other normalizations are supplemental.
    Use None for explicitly exploratory pairwise analyses. Optional exploratory
    linear contrasts map names to condition coefficients (e.g. +1/-1/-1/+1).

    Sampling units are sceneNNNN families, retaining all captures and questions.
    Primary deltas weight questions; supplemental deltas weight scenes equally.
    Identical family resamples are used for all metrics and contrasts. References
    and predictions outside an explicit cohort are validated, then counted as
    ignored. They never choose the cohort. Empty/invalid-object-ID predictions
    count as errors, never exclusions. No per-question McNemar test is used.
    """
    replicates = _positive_int(replicates, "replicates")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(conditions, (str, bytes, Mapping)):
        raise ValueError("conditions must be a sequence of condition names")
    conditions = tuple(_condition(value) for value in conditions)
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be nonempty and unique")
    refs = _references(references)
    selected = _cohort(cohort_qids, refs)
    scored, ignored = _predictions(predictions, refs, selected, conditions, cohort_qids is not None)
    specs = _contrast_specs(contrasts, exploratory_linear_contrasts, conditions, family_size)
    by_scene = defaultdict(list)
    for qid in selected:
        by_scene[refs[qid]["scene_id"]].append(qid)
    by_family = defaultdict(list)
    for scene in sorted(by_scene):
        by_family[scene.split("_")[0]].append(scene)
    families = sorted(by_family)
    family_questions = [sum(len(by_scene[s]) for s in by_family[f]) for f in families]
    family_scenes = [len(by_family[f]) for f in families]
    summaries = {}
    for condition in conditions:
        records = scored[condition]
        summaries[condition] = {
            "questions": len(selected), "blank_predictions": sum(r["blank"] for r in records.values()),
            "invalid_object_id_predictions": sum(r["invalid"] for r in records.values()),
            "metrics": {metric: {
                "correct": sum(records[qid]["hits"][metric] for qid in selected),
                "accuracy": sum(records[qid]["hits"][metric] for qid in selected) / len(selected),
                "scene_equal_accuracy": sum(sum(records[qid]["hits"][metric] for qid in qids)
                                            / len(qids) for qids in by_scene.values()) / len(by_scene),
            } for metric in METRICS},
        }

    # Preaggregate counts once; bootstrap iterations never revisit question text.
    prepared = {}
    for name, spec in specs.items():
        prepared[name] = {}
        for metric in METRICS:
            scene_delta = {
                scene: sum(sum(weight * scored[condition][qid]["hits"][metric]
                               for condition, weight in spec["weights"].items()) for qid in qids)
                for scene, qids in by_scene.items()
            }
            counts = [sum(scene_delta[s] for s in by_family[f]) for f in families]
            equal = [sum(scene_delta[s] / len(by_scene[s]) for s in by_family[f]) for f in families]
            prepared[name][metric] = {"counts": counts, "equal": equal, "samples": [], "scene_samples": []}
    rng_seed = int.from_bytes(hashlib.blake2b(f"natural-qa-bootstrap\0{int(seed)}".encode(),
                                             digest_size=16).digest(), "big")
    rng = random.Random(rng_seed)
    if len(families) >= 2:
        for _ in range(replicates):
            draws = Counter(rng.randrange(len(families)) for _ in families)
            questions = sum(weight * family_questions[index] for index, weight in draws.items())
            scenes = sum(weight * family_scenes[index] for index, weight in draws.items())
            for metrics in prepared.values():
                for values in metrics.values():
                    values["samples"].append(sum(weight * values["counts"][index]
                                                  for index, weight in draws.items()) / questions)
                    values["scene_samples"].append(sum(weight * values["equal"][index]
                                                        for index, weight in draws.items()) / scenes)
    comparisons = {}
    exploratory = {}
    for name, spec in specs.items():
        result = {"weights": spec["weights"], "metrics": {}}
        if spec["pair"]:
            treatment, control = spec["pair"]
            result["answer_changes"] = {
                metric: sum(scored[treatment][qid]["texts"][metric] != scored[control][qid]["texts"][metric]
                            for qid in selected) for metric in METRICS}
        for metric, values in prepared[name].items():
            samples, scene_samples = values["samples"], values["scene_samples"]
            confidence = (1 - .05 / spec["family_size"]
                          if spec["family_size"] is not None and metric == "protocol_normalized_exact" else None)
            result["metrics"][metric] = {
                "delta": sum(values["counts"]) / len(selected),
                "ci95": _interval(samples, .95) if samples else None,
                "ci_bonferroni": _interval(samples, confidence) if samples and confidence else None,
                "bonferroni_confidence": confidence,
                "scene_equal_delta": sum(values["equal"]) / len(by_scene),
                "scene_equal_ci95": _interval(scene_samples, .95) if scene_samples else None,
                "bootstrap_degenerate": bool(samples) and min(samples) == max(samples),
            }
            if spec["pair"]:
                treatment, control = spec["pair"]
                wins = sum(scored[treatment][qid]["hits"][metric] > scored[control][qid]["hits"][metric]
                           for qid in selected)
                losses = sum(scored[treatment][qid]["hits"][metric] < scored[control][qid]["hits"][metric]
                             for qid in selected)
                both_zero = all(summaries[c]["metrics"][metric]["correct"] == 0 for c in spec["pair"])
                result["metrics"][metric].update(wins=wins, losses=losses, ties=len(selected) - wins - losses,
                                                  both_conditions_zero_accuracy=both_zero)
        (comparisons if spec["pair"] else exploratory)[name] = result
    return {
        "accuracy_units": "fraction", "primary_metric": "protocol_normalized_exact",
        "normalization": "lowercase; collapse whitespace; strip trailing .!?; remove one leading a/an/the",
        "cohort": {"questions": len(selected), "scenes": len(by_scene), "scene_families": len(families),
                   "explicit_question_ids": cohort_qids is not None,
                   "ignored_reference_questions": len(refs) - len(selected),
                   "ignored_prediction_questions": ignored},
        "bootstrap": {"replicates": replicates, "seed": int(seed), "rng": "Random with BLAKE2b-derived seed",
                      "cluster": "sceneNNNN family", "retains_all_questions_and_captures": True,
                      "intervals_available": len(families) >= 2,
                      "bonferroni_primary_metric_only": True, "pairwise_family_size": family_size},
        "conditions": summaries, "comparisons": comparisons,
        "exploratory_linear_contrasts": exploratory,
        "interpretation": "Intervals are approximate cluster-bootstrap intervals. Null, floor, or degenerate "
                          "results do not establish equivalence or lack of representation use.",
    }
