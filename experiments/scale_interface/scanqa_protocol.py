"""Answer-separated ScanQA inputs and conservative question-only lexical strata.

``lexical_need`` describes matched wording rather than a human judgment that a physical
quantity is necessary or sufficient. In particular, height, distance, and volume
are not interchangeable with the radius of the point sample. Unmatched wording
stays ambiguous. Blind semantic annotation is a separate step.
"""
from collections import Counter
from collections.abc import Mapping
from numbers import Integral
import re


LEXICAL_NEEDS = ("required", "not_required", "ambiguous")
METRIC_PATTERNS = {
    "radius": r"\b(?:radius|radii)\b",
    "diameter": r"\bdiameters?\b",
    "height": r"\b(?:heights?|tall|taller|tallest)\b",
    "width": r"\b(?:widths?|wide|wider|widest|narrow|narrower|narrowest)\b",
    "length": r"\b(?:lengths?|long|longer|longest)\b",
    "depth": r"\b(?:depths?|deep|deeper|deepest)\b",
    "area": r"\b(?:areas?|square (?:meters?|metres?|feet))\b",
    "volume": r"\b(?:volumes?|cubic (?:meters?|metres?|feet))\b",
    "distance": r"\b(?:distances?|how far|how close|closer|closest|nearest|farther|farthest|further|furthest)\b",
    "position": r"\b(?:high|higher|highest|low|lower|lowest|above|below|under|over|left|right|behind|between|next to|in front of|top|bottom)\b",
    "size_unspecified": r"\b(?:sizes?|big|bigger|biggest|large|larger|largest|small|smaller|smallest|short|shorter|shortest|fit|fits|fitting)\b",
}

_DIRECT_DIMENSION = re.compile(
    r"^(?:what(?:'s| is| are)|how much is) (?:the )?"
    r"(?:radius|radii|diameter|height|width|length|depth|area|volume|distance|size)\b"
)
_DIRECT_HOW = re.compile(r"^how (?:tall|wide|long|deep|big|large|small|far|close)\b")
_COMPARATIVE = (
    r"(?:taller|tallest|wider|widest|narrower|narrowest|longer|longest|deeper|deepest|"
    r"bigger|biggest|larger|largest|smaller|smallest|shorter|shortest|closer|closest|"
    r"nearest|farther|farthest|furthest)"
)
_CHOICE_COMPARISON = re.compile(
    r"^(?:which|what)\b.*?\b(?:is|are)\s+(?:the\s+)?" + _COMPARATIVE + r"\b"
)
_YES_NO_COMPARISON = re.compile(r"^(?:is|are)\b.*\b" + _COMPARATIVE + r"\s+than\b")
_NONMETRIC_TARGETS = {
    "color_target": re.compile(r"^(?:(?:what|which) colou?r\b|what is the colou?r\b)"),
    "material_target": re.compile(r"^(?:(?:what|which) material\b|what (?:is|are)\b.*\bmade (?:of|from)\b)"),
    "count_target": re.compile(r"^how many\b"),
}
_TEMPORAL = re.compile(r"^how long (?:has|have|had|ago|until|before|after)\b|\b(?:seconds?|minutes?|hours?|days?|weeks?|years?)\b")
_AUDIO_VOLUME = re.compile(r"\b(?:loud|louder|loudest|sound|audio|music|decibels?)\b")


def _question_key(value):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError("question_id must be a nonempty string or integer")
    if not str(value).strip():
        raise ValueError("question_id must be a nonempty string or integer")
    return str(value)


def _scene_id(value):
    if not isinstance(value, str) or re.fullmatch(r"scene[0-9]{4}_[0-9]{2}", value) is None:
        raise ValueError("scene_id must be a ScanNet scene identifier")
    return value


def _question_fields(record):
    if not isinstance(record, Mapping):
        raise ValueError("ScanQA records must be mappings")
    # Intentionally access only original input fields. Do not copy the record:
    # it may also contain answers, target objects, or object names.
    question_id = record["question_id"]
    _question_key(question_id)
    scene_id = _scene_id(record["scene_id"])
    question = record["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    return {"question_id": question_id, "scene_id": scene_id, "question": question}


def _checked_records(records):
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be an iterable of ScanQA records")
    result, seen = [], set()
    for record in records:
        fields = _question_fields(record)
        key = _question_key(fields["question_id"])
        if key in seen:
            raise ValueError(f"duplicate question_id: {key}")
        seen.add(key)
        result.append((record, fields))
    return result


def build_questions(records, *, scene_object_counts, scene_whitelist=None):
    """Keep source order/text and select scenes using only precomputed input coverage.

    Counts describe complete input object rows, not annotated target objects.
    Every scene object is available and assigned its row index. No answer or
    ``object_ids`` field is read, and no caption/reference enters the output.
    Duplicate source question IDs fail even outside the selected scene cohort.
    """
    checked = _checked_records(records)
    if not isinstance(scene_object_counts, Mapping):
        raise ValueError("scene_object_counts must map scene IDs to input row counts")
    if scene_whitelist is None:
        selected = list(scene_object_counts)
    else:
        if isinstance(scene_whitelist, (str, bytes, Mapping)):
            raise ValueError("scene_whitelist must be an iterable of scene IDs")
        selected = list(scene_whitelist)
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate scene in whitelist")
    counts = {}
    for scene in selected:
        _scene_id(scene)
        if scene not in scene_object_counts:
            raise ValueError(f"missing input object count for {scene}")
        count = scene_object_counts[scene]
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError(f"input object count must be a positive integer for {scene}")
        counts[scene] = int(count)
    return [{**fields, "qid": fields["question_id"], "prompt": fields["question"],
             "assigned_ids": list(range(counts[fields["scene_id"]]))}
            for _, fields in checked if fields["scene_id"] in counts]


def build_references(records, *, question_ids):
    """Explicit scoring-only export; preserve all source answer strings and order.

    Call separately from ``build_questions`` and store outside inference inputs.
    Only selected records' answer fields are read. Target-object annotations
    are never needed for either export.
    """
    checked = _checked_records(records)
    if isinstance(question_ids, (str, bytes, Mapping)):
        raise ValueError("question_ids must be an iterable of question IDs")
    keys = [_question_key(value) for value in question_ids]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate reference question_id")
    by_id = {_question_key(fields["question_id"]): (record, fields)
             for record, fields in checked}
    missing = set(keys) - by_id.keys()
    if missing:
        raise ValueError(f"reference question IDs are missing: {sorted(missing)}")
    references = []
    for key in keys:
        record, fields = by_id[key]
        answers = record["answers"]
        if (not isinstance(answers, list) or not answers
                or any(not isinstance(answer, str) or not answer.strip() for answer in answers)):
            raise ValueError(f"answers must be a nonempty list of strings for {key}")
        references.append({"question_id": fields["question_id"],
                           "scene_id": fields["scene_id"], "answers": list(answers)})
    return references


def lexical_stratum(question):
    """Return transparent wording-based strata, never semantic necessity labels.

    ``required`` means an explicit measurement/comparison wording matched;
    ``not_required`` means a narrow color/material/count target without known
    metric or position cues; other questions remain ``ambiguous``. Neither label
    certifies sufficient input, target-object coverage, or benefit from radii.
    Bare adjectives and high/low position wording are deliberately ambiguous.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    text = " ".join(question.lower().replace("\u2019", "'").split())
    cues = [{"metric_type": metric, "text": match.group(0)}
            for metric, pattern in METRIC_PATTERNS.items()
            for match in re.finditer(pattern, text)]
    metrics = [metric for metric in METRIC_PATTERNS
               if any(cue["metric_type"] == metric for cue in cues)]
    label, rule = "ambiguous", "unmatched_question"
    nonmetric = next((name for name, pattern in _NONMETRIC_TARGETS.items()
                      if pattern.search(text)), None)
    if _TEMPORAL.search(text) and "length" in metrics:
        rule = "possible_temporal_length"
    elif _AUDIO_VOLUME.search(text) and "volume" in metrics:
        rule = "possible_audio_volume"
    elif nonmetric:
        label = "ambiguous" if metrics else "not_required"
        rule = nonmetric + ("_with_metric_or_position_cue" if metrics else "_without_metric_cue")
    elif _DIRECT_DIMENSION.search(text) or _DIRECT_HOW.search(text):
        label, rule = "required", "explicit_measurement_query"
    elif _CHOICE_COMPARISON.search(text) or _YES_NO_COMPARISON.search(text):
        label, rule = "required", "explicit_metric_comparison"
    elif metrics:
        rule = "metric_or_position_cue_without_explicit_query"
    return {"annotation_method": "lexical_rules", "lexical_need": label,
            "metric_types": metrics, "rule": rule, "cues": cues}


def select_lexical_rows(rows, *, labels=None, metric_types=None):
    """Explicit optional filter: label AND any requested metric type; no ranking."""
    def checked_filter(values, choices, name):
        if values is None:
            return None
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be an iterable of categories")
        selected = set(values)
        if not selected <= set(choices):
            raise ValueError(f"unknown {name}: {sorted(selected - set(choices))}")
        return selected

    labels = checked_filter(labels, LEXICAL_NEEDS, "lexical labels")
    metric_types = checked_filter(metric_types, METRIC_PATTERNS, "metric types")
    result = []
    for row, fields in _checked_records(rows):
        annotation = lexical_stratum(fields["question"])
        if labels is not None and annotation["lexical_need"] not in labels:
            continue
        if metric_types is not None and not metric_types.intersection(annotation["metric_types"]):
            continue
        result.append(row)
    return result


def summarize_questions(rows):
    """Compact aggregate only: no answers, predictions, or raw question text."""
    checked = _checked_records(rows)
    annotations = [lexical_stratum(fields["question"]) for _, fields in checked]
    labels = Counter(annotation["lexical_need"] for annotation in annotations)
    metrics = Counter(metric for annotation in annotations for metric in annotation["metric_types"])
    rules = Counter(annotation["rule"] for annotation in annotations)
    return {"annotation_method": "lexical_rules", "questions": len(checked),
            "scenes": len({fields["scene_id"] for _, fields in checked}),
            "lexical_need": {label: labels[label] for label in LEXICAL_NEEDS},
            "metric_types": {metric: metrics[metric] for metric in METRIC_PATTERNS},
            "rules": dict(sorted(rules.items()))}
