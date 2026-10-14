"""Original SQA3D situation/question inputs and fixed question-only sampling."""
from collections import defaultdict
from collections.abc import Mapping
import hashlib

from .scanqa_protocol import build_questions as _scanqa_questions
from .scanqa_protocol import _question_key, _scene_id, lexical_stratum


def _fields(records):
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be an iterable of original SQA3D questions")
    result, seen = [], set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("SQA3D questions must be mappings")
        question_id = record["question_id"]
        key = _question_key(question_id)
        if key in seen:
            raise ValueError("duplicate question_id")
        seen.add(key)
        scene = _scene_id(record["scene_id"])
        situation, question = record["situation"], record["question"]
        if not isinstance(situation, str) or not situation.strip():
            raise ValueError("the original situation must be a nonempty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("the original question must be a nonempty string")
        # Only the four original input fields are accessed. Alternatives, answers,
        # target objects, positions and orientations never enter the input export.
        result.append(dict(question_id=question_id, scene_id=scene,
                           situation=situation, question=question))
    return result


def build_questions(records, *, scene_object_counts, scene_whitelist=None):
    """Use the literal primary situation; preserve question order and whitespace.

    ``alternative_situation`` is never substituted or sampled. The host receives
    ``Situation:\n{original situation}\n\nQuestion:\n{original question}``.
    No additional physical position or facing direction is inferred.
    """
    fields = _fields(records)
    situations = {_question_key(row["question_id"]): row["situation"] for row in fields}
    rows = _scanqa_questions(fields, scene_object_counts=scene_object_counts,
                            scene_whitelist=scene_whitelist)
    for row in rows:
        situation = situations[_question_key(row["question_id"])]
        row["situation"] = situation
        row["prompt"] = "Situation:\n" + situation + "\n\nQuestion:\n" + row["question"]
        row["lexical"] = lexical_stratum(row["question"])
    return rows


def fixed_question_ids(rows, *, per_scene=8):
    """Select up to per_scene hash-ordered questions plus every lexical-required row.

    Classification reads the question alone using the ScanQA rules. It is a
    wording-based sample, not an assessment of situation-dependent necessity.
    Return IDs in original source order; answers never affect selection.
    """
    if type(per_scene) is not int or per_scene < 1:
        raise ValueError("per_scene must be a positive integer")
    fields = _fields(rows)
    by_scene, selected = defaultdict(list), set()
    for row in fields:
        key = _question_key(row["question_id"])
        by_scene[row["scene_id"]].append(row)
        if lexical_stratum(row["question"])["lexical_need"] == "required":
            selected.add(key)

    def order(row):
        key = _question_key(row["question_id"])
        return hashlib.blake2b((row["scene_id"] + "\0" + key).encode("utf-8")).hexdigest(), key

    for group in by_scene.values():
        selected.update(_question_key(row["question_id"]) for row in sorted(group, key=order)[:per_scene])
    return [_question_key(row["question_id"]) for row in fields
            if _question_key(row["question_id"]) in selected]


def build_references(annotations, *, questions):
    """Export selected canonical answer strings separately from frozen inputs.

    Original SQA3D annotations store ``answers`` as mappings with an ``answer``
    string. Preserve every selected string and its order. Position, rotation,
    answer confidence, answer type and question type are never accessed.
    """
    fields = _fields(questions)
    if isinstance(annotations, (str, bytes, Mapping)):
        raise ValueError("annotations must be an iterable of SQA3D annotation records")
    by_id = {}
    for record in annotations:
        if not isinstance(record, Mapping):
            raise ValueError("SQA3D annotations must be mappings")
        key = _question_key(record["question_id"])
        scene = _scene_id(record["scene_id"])
        if key in by_id:
            raise ValueError("duplicate annotation question_id")
        by_id[key] = record, scene
    result = []
    for question in fields:
        key = _question_key(question["question_id"])
        if key not in by_id:
            raise ValueError("selected reference question is missing")
        record, scene = by_id[key]
        if scene != question["scene_id"]:
            raise ValueError("question and reference scenes disagree")
        source_answers = record["answers"]
        if not isinstance(source_answers, list) or not source_answers:
            raise ValueError("answers must be a nonempty list of answer mappings")
        answers = []
        for entry in source_answers:
            if not isinstance(entry, Mapping):
                raise ValueError("each SQA3D answer must be an answer mapping")
            answer = entry["answer"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("reference answers must be nonempty strings")
            answers.append(answer)
        result.append(dict(question_id=question["question_id"], scene_id=scene, answers=answers))
    return result
