"""Matched radius questions with explicit IDs or uniquely named scene objects.

This builds questions from scene annotations, not natural ScanQA questions.
Names appear only in the named-object questions, never in a scene feature input
or measurement table. Radius means the recorded metric point-sample radius;
no height, volume, or general physical-size equivalence is assumed.

The evaluator must give direct and named variants the same scene-wide ID/radius
packet. Selecting two packet entries using hidden pair references would reveal
the recognition target. ``locs`` metadata is deliberately unused here.
"""
from collections import Counter
from collections.abc import Mapping
import hashlib
from itertools import combinations
import math
from numbers import Integral, Real
import re

from .metric_pair_numeric_prompt import quantize_radius_cm


_GENERIC_CATEGORIES = {
    "object", "objects", "other", "others", "unknown", "unlabeled", "unlabelled",
    "otherfurniture", "other furniture", "misc", "miscellaneous",
}
_RADIUS_QUESTION = (
    "Which object has the larger measured metric radius? Answer exactly A or B."
)


def _source_id(value):
    if isinstance(value, bool):
        raise ValueError("source_id must be a nonnegative integer")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        result = int(value)
    else:
        raise ValueError("source_id must be a nonnegative integer")
    if result < 0:
        raise ValueError("source_id must be a nonnegative integer")
    return result


def _category(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("category must be a string or null")
    normalized = " ".join(value.casefold().replace("_", " ").split())
    if not normalized or normalized in _GENERIC_CATEGORIES:
        return None
    if re.fullmatch(r"[a-z][a-z0-9 -]*", normalized) is None:
        raise ValueError("category must contain only plain category words")
    return normalized


def _checked_objects(objects):
    if not isinstance(objects, (list, tuple)) or not 0 < len(objects) <= 1000:
        raise ValueError("scene objects must be a nonempty ordered list of at most 1000 rows")
    checked, seen = [], set()
    for row_id, record in enumerate(objects):
        if not isinstance(record, Mapping):
            raise ValueError("object metadata must be a mapping")
        source_id = _source_id(record["source_id"])
        if source_id in seen:
            raise ValueError("duplicate source object ID")
        seen.add(source_id)
        valid = record["radius_valid"]
        rho = record["rho"]
        if not isinstance(valid, bool):
            raise ValueError("radius_valid must be an actual boolean")
        if (isinstance(rho, bool) or not isinstance(rho, Real)
                or not math.isfinite(float(rho)) or rho < 0 or (valid and rho <= 0)):
            raise ValueError("rho must be finite, nonnegative, and positive when valid")
        radius_cm = quantize_radius_cm(float(rho)) if valid else None
        checked.append({"source_id": source_id, "row_id": row_id,
                        "category": _category(record.get("category")),
                        "radius_valid": valid, "radius_cm": radius_cm})
    return checked


def _role_order(scene_id, first, second):
    # A/B orientation depends on scene/object identity, not on measured values.
    identity = f"{scene_id}|{first['source_id']}|{second['source_id']}".encode("ascii")
    reverse = bool(hashlib.blake2b(identity, digest_size=8).digest()[0] & 1)
    return (second, first) if reverse else (first, second)


def _input_row(question_id, scene_id, question, object_count, *, pair_id=None):
    row = {"question_id": question_id, "qid": question_id, "scene_id": scene_id,
           "question": question, "prompt": question,
           "assigned_ids": list(range(object_count))}
    if pair_id is not None:
        row["pair_id"] = pair_id
    return row


def build_controlled_pairs(scene_objects, *, max_pairs_per_scene=4):
    """Build fixed matched pairs, separate answers, and whole-scene input records.

    ``scene_objects`` maps a ScanNet scene ID to metadata in actual feature-row
    order: ``{source_id, category, rho, radius_valid}``. Logical source IDs may
    differ from row indices. All rows count when determining category uniqueness,
    even rows with an invalid radius. Eligible pairs are ordered by source IDs;
    centimeter-quantization ties are counted/excluded before taking at most four.
    A/B orientation uses a stable identity hash, never answers or predictions.

    Recognition candidates cover every scene object, not just the selected pair.
    Score those candidates or strictly validate raw output against them; never
    replace an invalid generated ID with a random object.
    """
    if not isinstance(scene_objects, Mapping):
        raise ValueError("scene_objects must map scene IDs to ordered object metadata")
    if (isinstance(max_pairs_per_scene, bool) or not isinstance(max_pairs_per_scene, int)
            or not 1 <= max_pairs_per_scene <= 4):
        raise ValueError("max_pairs_per_scene must be an integer from 1 to 4")
    for scene_id in scene_objects:
        if not isinstance(scene_id, str) or re.fullmatch(r"scene[0-9]{4}_[0-9]{2}", scene_id) is None:
            raise ValueError("scene_id must be a ScanNet scene identifier")

    direct_rows, semantic_rows, recognition_rows = [], [], []
    references, pair_metadata = [], []
    scene_measurements, recognition_candidates, per_scene = {}, {}, {}
    for scene_id in sorted(scene_objects):
        objects = _checked_objects(scene_objects[scene_id])
        object_count = len(objects)
        categories = Counter(obj["category"] for obj in objects if obj["category"] is not None)
        excluded = {"missing_or_generic_category": 0, "non_unique_category": 0,
                    "invalid_radius": 0, "nonpositive_centimeter_radius": 0}
        eligible = []
        for obj in objects:
            if obj["category"] is None:
                excluded["missing_or_generic_category"] += 1
            elif categories[obj["category"]] != 1:
                excluded["non_unique_category"] += 1
            elif not obj["radius_valid"]:
                excluded["invalid_radius"] += 1
            elif obj["radius_cm"] <= 0:
                excluded["nonpositive_centimeter_radius"] += 1
            else:
                eligible.append(obj)
        eligible.sort(key=lambda obj: obj["source_id"])
        candidates = list(combinations(eligible, 2))
        nontied = [(first, second) for first, second in candidates
                   if first["radius_cm"] != second["radius_cm"]]
        selected = nontied[:max_pairs_per_scene]

        # No category names or target-pair selection enter these input records.
        scene_measurements[scene_id] = [
            {"object_id": obj["row_id"], "radius_cm": obj["radius_cm"],
             "available": obj["radius_valid"] and obj["radius_cm"] > 0}
            for obj in objects
        ]
        recognition_candidates[scene_id] = [f"<OBJ{row:03d}>" for row in range(object_count)]
        selected_sources = {obj["source_id"] for pair in selected for obj in pair}
        recognition_ids = {}
        for obj in eligible:
            if obj["source_id"] not in selected_sources:
                continue
            qid = f"{scene_id}/recognition{len(recognition_ids):03d}"
            recognition_ids[obj["source_id"]] = qid
            prompt = (f"Which object is the {obj['category']}? "
                      "Answer with exactly one object identifier and no other text.")
            recognition_rows.append(_input_row(qid, scene_id, prompt, object_count))
            references.append({"question_id": qid, "scene_id": scene_id,
                               "answers": [f"<OBJ{obj['row_id']:03d}>"]})
        for pair_index, (first, second) in enumerate(selected):
            role_a, role_b = _role_order(scene_id, first, second)
            pair_id = f"{scene_id}/pair{pair_index:03d}"
            direct_id, semantic_id = pair_id + "/direct", pair_id + "/semantic"
            direct = (f"Object A is <OBJ{role_a['row_id']:03d}>. "
                      f"Object B is <OBJ{role_b['row_id']:03d}>. " + _RADIUS_QUESTION)
            semantic = (f"Object A is the {role_a['category']}. "
                        f"Object B is the {role_b['category']}. " + _RADIUS_QUESTION)
            direct_rows.append(_input_row(direct_id, scene_id, direct, object_count, pair_id=pair_id))
            semantic_rows.append(_input_row(semantic_id, scene_id, semantic, object_count, pair_id=pair_id))
            answer = "A" if role_a["radius_cm"] > role_b["radius_cm"] else "B"
            for qid in (direct_id, semantic_id):
                references.append({"question_id": qid, "scene_id": scene_id, "answers": [answer]})
            pair_metadata.append({
                "pair_id": pair_id, "scene_id": scene_id,
                "direct_question_id": direct_id, "semantic_question_id": semantic_id,
                "roles": {label: {"source_id": obj["source_id"], "row_id": obj["row_id"],
                                  "category": obj["category"], "radius_cm": obj["radius_cm"],
                                  "recognition_question_id": recognition_ids[obj["source_id"]]}
                          for label, obj in (("A", role_a), ("B", role_b))},
                "answer": answer,
            })
        per_scene[scene_id] = {
            "objects": object_count, "eligible_objects": len(eligible),
            "excluded_objects_by_first_reason": excluded,
            "candidate_pairs": len(candidates), "centimeter_ties": len(candidates) - len(nontied),
            "non_tied_pairs": len(nontied), "selected_pairs": len(selected),
            "unselected_by_pair_limit": len(nontied) - len(selected),
        }
    return {
        "direct_rows": direct_rows, "semantic_rows": semantic_rows,
        "recognition_rows": recognition_rows, "references": references,
        "pair_metadata": pair_metadata, "scene_measurements": scene_measurements,
        "recognition_candidates": recognition_candidates,
        "summary": {
            "scenes": len(scene_objects),
            "scenes_with_pairs": sum(item["selected_pairs"] > 0 for item in per_scene.values()),
            "selected_pairs": len(pair_metadata), "recognition_questions": len(recognition_rows),
            "direct_questions": len(direct_rows), "semantic_questions": len(semantic_rows),
            "centimeter_ties": sum(item["centimeter_ties"] for item in per_scene.values()),
            "per_scene": per_scene,
        },
    }
