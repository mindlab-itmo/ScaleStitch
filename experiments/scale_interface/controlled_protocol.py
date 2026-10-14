"""Complete matched native-ID comparisons and strict CPU-only scoring helpers."""
from collections import Counter, defaultdict
from collections.abc import Mapping
import copy
import hashlib
from itertools import combinations
import json
import math
from numbers import Real
import re

from .controlled_semantic_probe import _category, _role_order, _source_id
from .metric_pair_numeric_prompt import quantize_radius_cm
from .scanqa_protocol import _question_key, _scene_id
from .scene_extents import normalize_measurement_source
from .sqa3d_protocol import _fields, build_questions


PROTOCOL = "native_id_full_pairs/v1"
TARGETS = {"radius": "larger measured metric radius",
           "height": "larger measured axis-aligned height"}
ANSWER_INSTRUCTION = "Answer with exactly one object identifier and no other text."


CANDIDATE_INSTRUCTION = (
    "Your answer must be one of the two object identifiers in the question. "
    "Return exactly that identifier and no other text."
)
NATURAL_INSTRUCTION = (
    "Answer the question with a short natural-language answer. "
    "Do not output object identifiers."
)


def append_instruction(rows, kind=None):
    """Copy inference rows and append the fixed candidate/natural instruction.

    References and targets are never accessed. Candidate mode requires direct
    questions with exactly two distinct visible IDs. A separate natural-language
    question field is preserved; a duplicate of prompt is updated with it.
    With kind=None, rows are copied without changing their content.
    """
    instructions = {None: None, "candidate": CANDIDATE_INSTRUCTION,
                    "natural": NATURAL_INSTRUCTION}
    if kind not in instructions or not isinstance(rows, list):
        raise ValueError("provide inference rows and instruction candidate/natural or None")
    instruction = instructions[kind]
    qids, pairs, result = set(), set(), []
    for original in rows:
        if not isinstance(original, Mapping):
            raise ValueError("inference rows must be mappings")
        if any(key in original for key in ("answers", "answer", "references", "target",
                                           "target_source_row", "target_object_id")):
            raise ValueError("inference rows must not contain reference answers or targets")
        qid = _question_key(original.get("qid"))
        if qid in qids:
            raise ValueError("duplicate question identity")
        if "question_id" in original and _question_key(original["question_id"]) != qid:
            raise ValueError("question identity fields disagree")
        _scene_id(original.get("scene_id"))
        prompt = original.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("missing inference prompt")
        if kind == "candidate":
            ids = re.findall(r"<OBJ([0-9]{3})>", prompt)
            if len(ids) != 2 or len(set(ids)) != 2:
                raise ValueError("candidate instruction requires exactly two distinct object IDs")
            pair = original.get("pair_id")
            if not isinstance(pair, str) or not pair or pair in pairs:
                raise ValueError("missing or duplicate direct pair identity")
            pairs.add(pair)
        if instruction and instruction in prompt:
            raise ValueError("instruction would be appended twice")
        row = copy.deepcopy(original)
        if instruction:
            row["prompt"] = prompt + " " + instruction
            if row.get("question") == prompt:
                row["question"] = row["prompt"]
        qids.add(qid)
        result.append(row)
    return result


def _qid_hash(values):
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def four_conditions(measurement="radius"):
    if measurement not in TARGETS:
        raise ValueError("measurement must be radius or height")
    # Runtime deliberately has only one absent baseline, labelled radius.
    return [{"representation": "absent", "measurement": "radius"},
            *[{"representation": "lexical", "measurement": measurement, "mode": mode}
              for mode in ("true", "constant", "swapped")]]


def build_native_id_pairs(scene_objects, *, measurement="radius", measurement_source=None):
    """Every eligible unique-category pair, with direct/named matched questions.

    Input is feature-row-ordered scene metadata: source_id, category, and either
    rho/radius_valid or height_m/height_valid. Height is independent of radius.
    All scene rows remain visible; pair eligibility does not certify that the
    entire scene packet will pass runtime preflight. No scene/pair/question cap.
    Height callers must provide the separately geometry-verified source contract.
    """
    if measurement not in TARGETS or not isinstance(scene_objects, Mapping):
        raise ValueError("provide scene metadata and an explicit valid measurement")
    source = normalize_measurement_source(measurement_source) if measurement == "height" else None
    if measurement == "radius" and measurement_source is not None:
        raise ValueError("height source supplied for radius")
    direct, named, refs, metadata, ledgers, packets = [], [], [], [], {}, {}
    for scene in sorted(scene_objects):
        _scene_id(scene)
        if source and source["verified_scenes"].get(scene) is not True:
            raise ValueError("height source does not cover " + scene)
        raw = scene_objects[scene]
        if not isinstance(raw, (list, tuple)) or not 0 < len(raw) <= 1000:
            raise ValueError("scene must contain 1..1000 ordered object rows")
        objects, seen = [], set()
        for row_id, record in enumerate(raw):
            sid = _source_id(record["source_id"])
            if sid in seen:
                raise ValueError("duplicate source object ID")
            seen.add(sid)
            value = record["rho" if measurement == "radius" else "height_m"]
            valid = record[measurement + "_valid"]
            if (type(valid) is not bool or isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or value < 0 or (valid and value <= 0)):
                raise ValueError("invalid measurement value/validity contract")
            cm = quantize_radius_cm(float(value)) if valid else None
            objects.append(dict(source_id=sid, row_id=row_id, category=_category(record.get("category")),
                                value_cm=cm, available=valid and cm > 0))
        frequencies = Counter(o["category"] for o in objects if o["category"] is not None)
        excluded, eligible = Counter(), []
        for obj in objects:
            reason = ("missing_or_generic_category" if obj["category"] is None else
                      "non_unique_category" if frequencies[obj["category"]] != 1 else
                      "invalid_measurement" if obj["value_cm"] is None else
                      "nonpositive_centimeter_measurement" if obj["value_cm"] <= 0 else None)
            if reason:
                excluded[reason] += 1
            else:
                eligible.append(obj)
        eligible.sort(key=lambda o: o["source_id"])
        pairs, ties = 0, []
        for first, second in combinations(eligible, 2):
            # Source identities keep pair IDs stable when unrelated objects change.
            pair_id = f"{scene}/native_id/{measurement}/{first['source_id']}-{second['source_id']}"
            if first["value_cm"] == second["value_cm"]:
                ties.append(dict(pair_id=pair_id, source_ids=[first["source_id"], second["source_id"]],
                                 reason="equal_after_centimeter_quantization", value_cm=first["value_cm"]))
                continue
            a, b = _role_order(scene, first, second)
            direct_prompt = (f"Which object, <OBJ{a['row_id']:03d}> or <OBJ{b['row_id']:03d}>, "
                             f"has the {TARGETS[measurement]}? {ANSWER_INSTRUCTION}")
            named_prompt = (f"Which object, the {a['category']} or the {b['category']}, "
                            f"has the {TARGETS[measurement]}? {ANSWER_INSTRUCTION}")
            target = a if a["value_cm"] > b["value_cm"] else b
            qids = {}
            for task, prompt, output in (("direct", direct_prompt, direct), ("named", named_prompt, named)):
                qid = pair_id + "/" + task
                qids[task] = qid
                output.append(dict(question_id=qid, qid=qid, scene_id=scene, pair_id=pair_id,
                                   question=prompt, prompt=prompt, assigned_ids=list(range(len(objects)))))
                refs.append(dict(question_id=qid, scene_id=scene, answers=[f"<OBJ{target['row_id']:03d}>"]))
            metadata.append(dict(pair_id=pair_id, scene_id=scene, measurement=measurement,
                                 question_ids=qids, roles={"A": dict(a), "B": dict(b)},
                                 target_source_row=target["row_id"]))
            pairs += 1
        packets[scene] = [{"object_id": o["row_id"], measurement + "_cm": o["value_cm"],
                           "available": o["available"]} for o in objects]
        ledgers[scene] = dict(objects=len(objects), eligible_objects=len(eligible),
                             excluded_objects_by_first_reason=dict(excluded),
                             candidate_pairs=len(eligible) * (len(eligible) - 1) // 2,
                             centimeter_ties=len(ties), excluded_ties=ties, selected_pairs=pairs,
                             unselected_by_pair_limit=0,
                             whole_scene_measurements_available=all(o["available"] for o in objects))
    return dict(protocol=PROTOCOL, measurement=measurement, measurement_source=source,
                direct_rows=direct, named_rows=named, references=refs, pair_metadata=metadata,
                scene_measurements=packets, summary=dict(scenes=len(ledgers), selected_pairs=len(metadata),
                questions=len(direct)+len(named), pair_cap=None, question_cap=None, per_scene=ledgers))


def build_full_sqa3d_inputs(records, *, scene_object_counts):
    """Keep all official input records; expose missing-scene exclusions explicitly.

    The caller must verify the official test file path/hash/split. This function
    cannot infer official membership from records. No answer fields are read.
    Missing assets prevent a full-test-complete claim, even if eligible metrics run.
    """
    fields = _fields(records)
    if not isinstance(scene_object_counts, Mapping):
        raise ValueError("scene_object_counts must be a mapping")
    rows = build_questions(fields, scene_object_counts=scene_object_counts)
    selected = {str(r["qid"]) for r in rows}
    exclusions = [dict(qid=_question_key(r["question_id"]), scene_id=r["scene_id"],
                       stage="input_assets", reason="scene_not_in_input_assets")
                  for r in fields if _question_key(r["question_id"]) not in selected]
    return dict(protocol="sqa3d_full_official_inputs/v1", rows=rows, exclusions=exclusions,
                source_questions=len(fields), source_qids_sha256=_qid_hash([_question_key(r["question_id"]) for r in fields]),
                eligible_questions=len(rows), question_cap=None, complete_input_coverage=not exclusions)


def score_native_id(raw_prediction, *, assigned_ids, scene_mask, end_sym, pair_source_rows, target_source_row):
    """Strictly score raw decoder output, mapping visible ID to source row ONCE.

    Do not pass the runtime's already-remapped ``prediction`` field. Only outer
    whitespace and the runtime's exact end symbol are removed. Terminal
    sentence punctuation is accepted, as in the frozen post-hoc protocol.
    A/B, extra prose, malformed IDs and IDs outside the queried pair score zero.
    No random repair, extraction of a convenient token, or answer-conditioned fix.
    """
    if not isinstance(raw_prediction, str) or not isinstance(end_sym, str) or not end_sym:
        raise ValueError("raw text and validated nonempty runtime end_sym required")
    if (not isinstance(assigned_ids, (list, tuple)) or not 0 < len(assigned_ids) <= 1000
            or len(assigned_ids) != len(scene_mask) or any(type(x) is not int for x in assigned_ids)
            or set(assigned_ids) != set(range(len(assigned_ids))) or any(type(x) is not bool for x in scene_mask)):
        raise ValueError("invalid native-ID mapping/mask")
    active = {i for i, flag in enumerate(scene_mask) if flag}
    if {assigned_ids[i] for i in active} != active:
        raise ValueError("assignment must preserve the active source set")
    if (len(pair_source_rows) != 2 or any(type(x) is not int for x in pair_source_rows)
            or len(set(pair_source_rows)) != 2 or not set(pair_source_rows) <= active
            or type(target_source_row) is not int or target_source_row not in pair_source_rows):
        raise ValueError("reference must identify a visible pair and its target")
    text = raw_prediction.split(end_sym, 1)[0].strip()
    match = re.fullmatch(r"<OBJ([0-9]{3})>[.!?]*", text)
    result = dict(exact_match=0, status="invalid_format", source_row=None,
                  scorer_prediction="__native_id_error__", decoded_fragment=text)
    if match:
        visible_id = int(match[1])
        if visible_id not in active:
            result["status"] = "outside_visible_scene"
        else:
            source = assigned_ids[visible_id]
            result["source_row"] = source
            if source not in pair_source_rows:
                result["status"] = "outside_queried_pair"
            else:
                result.update(status="correct_target" if source == target_source_row else "other_pair_object",
                              exact_match=int(source == target_source_row), scorer_prediction=f"<OBJ{source:03d}>")
    return result


def audit_coverage(candidate_rows, eligible_qids, exclusions, *, conditions, predictions=None, matched_pairs=False):
    """Require an exact candidate partition and, if supplied, a complete matrix.

    Exclusions may contain several explicit condition failures for one qid.
    Matched-pair closure must be done by the driver and recorded as exclusions.
    This helper refuses silent omissions, duplicate predictions and partial pairs.
    """
    candidates = {}
    for row in candidate_rows:
        qid = _question_key(row["qid"])
        if qid in candidates:
            raise ValueError("duplicate candidate qid")
        candidates[qid] = row
    eligible = [_question_key(qid) for qid in eligible_qids]
    if len(set(eligible)) != len(eligible):
        raise ValueError("duplicate eligible qid")
    rejected = defaultdict(list)
    for row in exclusions:
        qid = _question_key(row["qid"])
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError("exclusions require a reason")
        if qid not in candidates or row.get("scene_id", candidates[qid]["scene_id"]) != candidates[qid]["scene_id"]:
            raise ValueError("unknown or inconsistent exclusion")
        rejected[qid].append(row["reason"])
    if set(eligible) & rejected.keys() or set(eligible) | rejected.keys() != candidates.keys():
        raise ValueError("candidate coverage is not an exact eligible/excluded partition")
    if not conditions or len(set(conditions)) != len(conditions) or any(not isinstance(c, str) or not c for c in conditions):
        raise ValueError("condition names must be nonempty and unique")
    if matched_pairs:
        pairs = defaultdict(set)
        for qid, row in candidates.items():
            if not row.get("pair_id"):
                raise ValueError("matched candidate missing pair_id")
            pairs[row["pair_id"]].add(qid)
        if any(len(qids) != 2 or len(qids & set(eligible)) not in (0, 2) for qids in pairs.values()):
            raise ValueError("partial matched pair")
    expected = {(qid, condition) for qid in eligible for condition in conditions}
    if predictions is not None:
        observed = set()
        for row in predictions:
            key = (_question_key(row["qid"]), row["condition"])
            if key in observed or key not in expected or row["scene_id"] != candidates[key[0]]["scene_id"]:
                raise ValueError("duplicate, unexpected or scene-mismatched prediction")
            observed.add(key)
        if observed != expected:
            raise ValueError("incomplete prediction matrix")
    return dict(candidate_questions=len(candidates), eligible_questions=len(eligible), excluded_questions=len(rejected),
                eligible_qids_sha256=_qid_hash(eligible), expected_predictions=len(expected),
                matrix_verified=predictions is not None, full_candidate_partition=True,
                exclusions_by_qid=dict(rejected), all_candidates_evaluated=not rejected)
