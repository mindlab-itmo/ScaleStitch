"""Strict object-ID scoring and compact per-question diagnostic outcomes."""
import re

from . import natural_qa_metrics as metrics

NATIVE = re.compile(r"<OBJ([0-9]{3})>[.!?]*")


def _unique(rows, key):
    result = {}
    for row in rows:
        qid = metrics._qid(row[key])
        if qid in result:
            raise ValueError("duplicate question identity")
        result[qid] = row
    return result


def _pair_index(metadata):
    result = {}
    for pair in metadata:
        roles = pair["roles"]
        a, b = roles["A"], roles["B"]
        if (set(roles) != {"A", "B"} or a["row_id"] == b["row_id"]
                or a["source_id"] == b["source_id"] or a["value_cm"] == b["value_cm"]
                or pair["target_source_row"] != max((a, b), key=lambda row: row["value_cm"])["row_id"]):
            raise ValueError("invalid controlled pair/target")
        for qid in pair["question_ids"].values():
            qid = metrics._qid(qid)
            if qid in result:
                raise ValueError("duplicate controlled question identity")
            result[qid] = pair
    return result


def score_predictions(predictions, references, *, cohort, end_sym, scoring="natural_qa",
                      questions=None, pair_metadata=None):
    """Return guarded predictions and paired outcome vectors for a fixed cohort."""
    if not isinstance(end_sym, str) or not end_sym:
        raise ValueError("nonempty recorded decoder end_sym required")
    if scoring not in ("natural_qa", "native_id"):
        raise ValueError("scoring must be natural_qa or native_id")
    refs = metrics._references(references)
    selected = set(metrics._cohort(cohort, refs))
    rows = _unique(questions, "qid") if questions is not None else {}
    pairs = _pair_index(pair_metadata) if pair_metadata is not None else {}
    if scoring == "native_id" and not selected <= rows.keys() & pairs.keys():
        raise ValueError("controlled questions/metadata missing from fixed cohort")
    guarded, banks = [], {}
    for prediction in predictions:
        qid = metrics._qid(prediction["qid"])
        if qid not in selected:
            guarded.append(prediction)
            continue
        scene, condition = metrics._scene(prediction["scene_id"]), prediction["condition"]
        if refs[qid]["scene_id"] != scene:
            raise ValueError("prediction/reference scene mismatch")
        invalid = prediction["invalid_object_id"]
        raw, mapped = prediction["raw_prediction"], prediction["prediction"]
        if type(invalid) is not bool or not isinstance(raw, str) or not isinstance(mapped, str):
            raise ValueError("malformed raw/mapped prediction")
        raw_id = NATIVE.fullmatch(raw.split(end_sym, 1)[0].strip())
        mapped_id = NATIVE.fullmatch(mapped.strip())
        native = not invalid and raw_id is not None and mapped_id is not None
        normalized = metrics.normalize_answer(mapped)
        unrestricted = int(not invalid and bool(normalized)
                           and normalized in refs[qid]["answers"]["protocol_normalized_exact"])
        hit, in_pair = unrestricted, None
        evaluated = prediction
        if scoring == "native_id":
            row, pair = rows[qid], pairs[qid]
            assignment = row["assigned_ids"]
            if (row["scene_id"] != scene or pair["scene_id"] != scene
                    or row["pair_id"] != pair["pair_id"]
                    or not isinstance(assignment, list) or not assignment
                    or any(type(value) is not int for value in assignment)
                    or set(assignment) != set(range(len(assignment)))
                    or any(type(obj["row_id"]) is not int or obj["row_id"] not in assignment
                           for obj in pair["roles"].values())):
                raise ValueError("controlled identity mapping mismatch")
            expected = "<OBJ%03d>" % pair["target_source_row"]
            if refs[qid]["answers"]["raw_exact"] != {expected}:
                raise ValueError("reference and controlled target disagree")
            source = None
            if native:
                visible, mapped_source = int(raw_id[1]), int(mapped_id[1])
                if visible < len(assignment) and mapped_source in assignment:
                    if assignment[visible] != mapped_source:
                        raise ValueError("raw visible/source mapping mismatch")
                    source = mapped_source
            in_pair = int(source is not None and source in [obj["row_id"] for obj in pair["roles"].values()])
            hit = int(source is not None and source == pair["target_source_row"])
            if hit != int(unrestricted and native):
                raise ValueError("native-ID and reference scorer disagree")
            evaluated = {**prediction, "prediction": mapped if native else ""}
        guarded.append(evaluated)
        bank = banks.setdefault(condition, {})
        if qid in bank:
            raise ValueError("duplicate scored condition/question")
        bank[qid] = {"scene_id": scene, "correct": hit, "candidate_pair_compliant": in_pair,
                     "native_id_only": int(native), "normalized_without_native_guard": unrestricted}
    if not banks or any(set(bank) != selected for bank in banks.values()):
        raise ValueError("incomplete scored condition matrix")
    return guarded, {"schema_version": "scalestitch_scored_rows/v1", "conditions": banks}
