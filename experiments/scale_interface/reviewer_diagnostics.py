"""Offline, descriptive native-ID diagnostics for a complete paired cohort.

Predictions retain the frozen ``natural_qa.iter_predictions`` schema: qid,
scene_id, condition, raw_prediction, prediction, invalid_object_id. Raw OBJ IDs
are visible positions; mapped prediction IDs are source *rows*, not ScanNet
source instance IDs. Mapping is verified exactly once.

Metadata is an explicit list of question records with qid, scene_id,
scene_family, objects=[{visible_id, source_row}], candidate_source_rows=[a, b],
target_source_row, direction (max/min), and packets={condition: rows-or-null}.
Packet rows are ordered exactly as serialized and contain visible_id,
source_row, value_cm. Values must be the actual serialized centimetres (prefer
integers or decimal strings), not unrounded source measurements. Null means
no numerical packet; it does not borrow a different condition's packet. Both
candidates must be present in every non-null packet. Subset/candidate-only
packets are supported independently of all visible scene objects.

This module does not reconstruct missing metadata, run a model, score natural
language QA, estimate confidence intervals, or turn compliance-conditioned
accuracy into a causal effect. Explicit question rows are the full denominator;
missing, duplicate, unknown-condition, and extra-question predictions fail closed.
"""
import argparse
from collections import Counter
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys


NATIVE_ID = re.compile(r"<OBJ([0-9]{3})>[.!?]*")
OUTCOMES = ("correct", "wrong_candidate", "out_of_pair", "invalid")
FEATURES = ("maximum", "minimum", "direction_extreme", "adjacent_to_candidate",
            "first_row", "last_row")
RANKINGS = ("preserved", "flipped", "tied", "reference_tied", "unavailable")


def _identity(value, field):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"{field} must be a nonempty identity")
    return str(value)


def _integer(value, field, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"metadata {field} must be a nonnegative integer")
    return value


def _value(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError("metadata value_cm must be an exact positive serialized value")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("metadata value_cm is not a decimal value") from error
    if not result.is_finite() or result <= 0:
        raise ValueError("metadata value_cm must be finite and positive")
    return result


def _metadata(rows, conditions):
    indexed = {}
    families = {}
    for row in rows:
        required = {"qid", "scene_id", "scene_family", "objects", "candidate_source_rows",
                    "target_source_row", "direction", "packets"}
        if not isinstance(row, Mapping) or not required <= row.keys():
            raise ValueError("metadata requires identities, mapping, candidates, direction and packets")
        qid = _identity(row["qid"], "metadata qid")
        if qid in indexed:
            raise ValueError("duplicate metadata question")
        scene = _identity(row["scene_id"], "metadata scene_id")
        family = _identity(row["scene_family"], "metadata scene_family")
        if families.setdefault(scene, family) != family:
            raise ValueError("metadata scene has inconsistent scene family labels")
        objects = row["objects"]
        if not isinstance(objects, list) or len(objects) < 2:
            raise ValueError("metadata objects must explicitly map all visible objects")
        mapping = {}
        for obj in objects:
            if not isinstance(obj, Mapping) or not {"visible_id", "source_row"} <= obj.keys():
                raise ValueError("metadata objects require visible_id/source_row mapping")
            visible = _integer(obj["visible_id"], "visible_id", 999)
            source = _integer(obj["source_row"], "source_row", 999)
            if visible in mapping or source in mapping.values():
                raise ValueError("metadata duplicate visible/source mapping")
            mapping[visible] = source
        candidates = row["candidate_source_rows"]
        if (not isinstance(candidates, list) or len(candidates) != 2
                or any(type(value) is not int or value not in mapping.values() for value in candidates)
                or candidates[0] == candidates[1]):
            raise ValueError("metadata requires two distinct visible candidate source rows")
        target = _integer(row["target_source_row"], "target_source_row")
        if target not in candidates or row["direction"] not in ("max", "min"):
            raise ValueError("metadata target/direction is invalid")
        if not isinstance(row["packets"], Mapping) or set(row["packets"]) != set(conditions):
            raise ValueError("metadata must explicitly provide each condition's packet or null")
        packets = {}
        for condition, packet in row["packets"].items():
            if packet is None:
                packets[condition] = None
                continue
            if not isinstance(packet, list) or not packet:
                raise ValueError("metadata packet must contain ordered serialized rows or null")
            parsed, seen = [], set()
            for obj in packet:
                if not isinstance(obj, Mapping) or not {"visible_id", "source_row", "value_cm"} <= obj.keys():
                    raise ValueError("metadata packet rows require visible_id/source_row/value_cm")
                visible = _integer(obj["visible_id"], "packet visible_id", 999)
                source = _integer(obj["source_row"], "packet source_row", 999)
                if mapping.get(visible) != source or source in seen:
                    raise ValueError("metadata packet mapping mismatch or duplicate row")
                seen.add(source)
                parsed.append((source, _value(obj["value_cm"])))
            if not set(candidates) <= seen:
                raise ValueError("metadata packet omits a candidate")
            packets[condition] = parsed
        indexed[qid] = dict(scene=scene, family=family, mapping=mapping,
                            candidates=candidates, target=target, direction=row["direction"],
                            packets=packets)
    if not indexed:
        raise ValueError("metadata must declare a nonempty fixed cohort")
    return indexed


def _selection(prediction, meta, end_sym):
    if (not {"raw_prediction", "prediction", "invalid_object_id"} <= prediction.keys()
            or not isinstance(prediction["raw_prediction"], str)
            or not isinstance(prediction["prediction"], str)
            or type(prediction["invalid_object_id"]) is not bool):
        raise ValueError("malformed frozen raw/mapped prediction schema")
    raw = NATIVE_ID.fullmatch(prediction["raw_prediction"].split(end_sym, 1)[0].strip())
    mapped = NATIVE_ID.fullmatch(prediction["prediction"].strip())
    syntax = raw is not None and mapped is not None and not prediction["invalid_object_id"]
    source = None
    if syntax:
        visible, mapped_source = int(raw[1]), int(mapped[1])
        if visible in meta["mapping"]:
            if meta["mapping"][visible] != mapped_source:
                raise ValueError("raw visible/source prediction mapping mismatch")
            source = mapped_source
        elif mapped_source in meta["mapping"].values():
            raise ValueError("nonvisible raw ID has a mapped visible source: mapping mismatch")
    outcome = ("invalid" if source is None else "correct" if source == meta["target"]
               else "wrong_candidate" if source in meta["candidates"] else "out_of_pair")
    return dict(source=source, native_syntax=bool(syntax), outcome=outcome)


def _winner(packet, candidates, direction):
    if packet is None:
        return None
    values = dict(packet)
    a, b = candidates
    if values[a] == values[b]:
        return None
    return max(candidates, key=values.get) if direction == "max" else min(candidates, key=values.get)


def _packet_diagnostics(packet, candidates, direction):
    if packet is None:
        return None
    values = dict(packet)
    order = [source for source, _ in packet]
    low, high = min(values.values()), max(values.values())
    maximum = {source for source, value in packet if value == high}
    minimum = {source for source, value in packet if value == low}
    candidate_positions = [order.index(source) for source in candidates]
    adjacent = {source for position, source in enumerate(order)
                if any(abs(position - candidate) == 1 for candidate in candidate_positions)}
    return dict(maximum=maximum, minimum=minimum,
                direction_extreme=maximum if direction == "max" else minimum,
                adjacent_to_candidate=adjacent, first_row={order[0]}, last_row={order[-1]},
                sources=set(order), distractors=set(order) - set(candidates), spread=float(high - low))


def _rate(count, denominator):
    return dict(count=count, denominator=denominator,
                fraction=count / denominator if denominator else None)


def _summarize(rows):
    n = len(rows)
    counts = Counter(row["outcome"] for row in rows)
    in_pair = counts["correct"] + counts["wrong_candidate"]
    packet_rows = [row for row in rows if row["packet_diagnostics"] is not None]
    outside = [row for row in packet_rows if row["outcome"] == "out_of_pair"]
    features = {}
    for name in FEATURES:
        count = sum(row["source"] in row["packet_diagnostics"][name] for row in outside)
        references = [len(row["packet_diagnostics"][name] & row["packet_diagnostics"]["distractors"])
                      / len(row["packet_diagnostics"]["distractors"])
                      for row in outside if row["packet_diagnostics"]["distractors"]]
        features[name] = dict(count=count,
                              fraction_all=count / n,
                              fraction_out_of_pair=count / len(outside) if outside else None,
                              random_distractor_rate=sum(references) / len(references) if references else None,
                              random_distractor_questions=len(references),
                              expected_random_count_on_outside=sum(references),
                              expected_random_fraction_all=sum(references) / n if references else None)
    rankings = Counter(row["ranking"] for row in rows)
    unique = [row for row in rows if row["packet_winner"] is not None]
    flipped = [row for row in rows if row["ranking"] == "flipped"]
    agreeing = sum(row["source"] == row["packet_winner"] for row in unique)
    agreement_all = _rate(agreeing, n)
    if not unique:
        agreement_all["fraction"] = None
    spreads = [row["packet_diagnostics"]["spread"] for row in packet_rows]
    spread_summary = dict(count=len(spreads), minimum=min(spreads) if spreads else None,
                          maximum=max(spreads) if spreads else None,
                          mean=sum(spreads) / len(spreads) if spreads else None)
    return dict(questions=n, scenes=len({row["scene_id"] for row in rows}),
                scene_families=len({row["scene_family"] for row in rows}),
                outcomes={name: counts[name] for name in OUTCOMES},
                accuracy=_rate(counts["correct"], n),
                native_id_syntax=_rate(sum(row["native_syntax"] for row in rows), n),
                visible_id=_rate(n - counts["invalid"], n),
                candidate_compliance=_rate(in_pair, n),
                in_pair_accuracy_descriptive=_rate(counts["correct"], in_pair),
                packet_following=dict(ranking_counts={name: rankings[name] for name in RANKINGS},
                    unique_winner_questions=len(unique), agreement_all=agreement_all,
                    agreement_unique=_rate(agreeing, len(unique)),
                    agreement_flipped=_rate(sum(row["source"] == row["packet_winner"] for row in flipped), len(flipped))),
                out_of_pair_diagnostics=dict(eligible_questions=len(packet_rows),
                    out_of_pair_answers=len(outside),
                    selected_not_in_packet=sum(row["source"] not in row["packet_diagnostics"]["sources"] for row in outside),
                    value_spread_cm=spread_summary, **features))


def _transitions(banks, conditions, qids):
    result = {}
    for before, after in combinations(conditions, 2):
        counts = Counter(f"{banks[before][qid]['outcome']}->{banks[after][qid]['outcome']}" for qid in qids)
        result[f"{before}->{after}"] = dict(questions=len(qids), outcome_counts=dict(sorted(counts.items())))
    return result


def analyze_predictions(predictions, metadata, *, configuration, conditions,
                        reference_condition, end_sym):
    """Analyze one configuration on an explicit complete native-ID cohort.

    Use the correct-value condition as reference_condition; this never guesses
    condition roles from their names. Ranking agreement is distinct from target
    accuracy. All invalid and out-of-pair answers stay in primary denominators.
    """
    configuration = _identity(configuration, "configuration")
    if (not isinstance(conditions, (list, tuple)) or not conditions
            or any(not isinstance(value, str) or not value for value in conditions)
            or len(set(conditions)) != len(conditions) or reference_condition not in conditions):
        raise ValueError("declare unique conditions and their reference condition")
    if not isinstance(end_sym, str) or not end_sym:
        raise ValueError("recorded nonempty end_sym is required")
    questions = _metadata(metadata, conditions)
    for meta in questions.values():
        winner = _winner(meta["packets"][reference_condition], meta["candidates"], meta["direction"])
        if winner is not None and winner != meta["target"]:
            raise ValueError("metadata target disagrees with the unique correct-packet winner")
    banks = {condition: {} for condition in conditions}
    for prediction in predictions:
        if not isinstance(prediction, Mapping) or not {"qid", "scene_id", "condition"} <= prediction.keys():
            raise ValueError("prediction identities are missing")
        qid = _identity(prediction["qid"], "prediction qid")
        condition = prediction["condition"]
        if qid not in questions or not isinstance(condition, str) or condition not in banks:
            raise ValueError("extra prediction question or condition outside declared cohort")
        meta = questions[qid]
        if _identity(prediction["scene_id"], "prediction scene_id") != meta["scene"]:
            raise ValueError("prediction/metadata scene mismatch")
        if qid in banks[condition]:
            raise ValueError("duplicate prediction condition/question")
        selected = _selection(prediction, meta, end_sym)
        packet = meta["packets"][condition]
        reference = meta["packets"][reference_condition]
        winner = _winner(packet, meta["candidates"], meta["direction"])
        reference_winner = _winner(reference, meta["candidates"], meta["direction"])
        ranking = ("unavailable" if packet is None or reference is None else "tied" if winner is None
                   else "reference_tied" if reference_winner is None
                   else "preserved" if winner == reference_winner else "flipped")
        banks[condition][qid] = dict(qid=qid, scene_id=meta["scene"], scene_family=meta["family"],
            **selected, packet_winner=winner, ranking=ranking,
            packet_diagnostics=_packet_diagnostics(packet, meta["candidates"], meta["direction"]))
    if any(set(bank) != set(questions) for bank in banks.values()):
        raise ValueError("incomplete paired condition/question matrix")
    by_family = {}
    for family in sorted({meta["family"] for meta in questions.values()}):
        qids = [qid for qid, meta in questions.items() if meta["family"] == family]
        by_family[family] = dict(questions=len(qids),
            conditions={condition: _summarize([bank[qid] for qid in qids]) for condition, bank in banks.items()},
            transitions=_transitions(banks, conditions, qids))
    return dict(schema_version="scalestitch_reviewer_diagnostics/v1", configuration=configuration,
                reference_condition=reference_condition,
                conditions={condition: _summarize(list(bank.values())) for condition, bank in banks.items()},
                transitions=_transitions(banks, conditions, questions), by_scene_family=by_family,
                interpretation="Descriptive fixed-cohort outcomes; in-pair accuracy is descriptive only, not a causal benefit. "
                    "Extremum and position categories overlap. Null packets provide no packet-following evidence. "
                    "Random-distractor rates condition on the same observed out-of-pair questions and uniform noncandidate "
                    "rows actually present in each packet; this is a descriptive selected-subset reference. Family-level "
                    "counts are paired sufficient aggregates for subsequent scene-family resampling, not confidence intervals.")


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True, help="Original server JSONL, read only")
    parser.add_argument("--metadata", type=Path, required=True, help="Explicit fixed-cohort metadata JSON list")
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--reference-condition", required=True)
    parser.add_argument("--end-sym", required=True)
    args = parser.parse_args(argv)
    try:
        before = {"predictions": _file_hash(args.predictions), "metadata": _file_hash(args.metadata)}
        with args.predictions.open(encoding="utf-8-sig") as stream:
            predictions = (json.loads(line) for line in stream if line.strip())
            metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
            result = analyze_predictions(predictions, metadata, configuration=args.configuration,
                conditions=args.conditions, reference_condition=args.reference_condition, end_sym=args.end_sym)
        after = {"predictions": _file_hash(args.predictions), "metadata": _file_hash(args.metadata)}
        if before != after:
            raise ValueError("input changed during offline analysis")
        result["input_sha256"] = before
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, TypeError) as error:
        print(f"reviewer diagnostics: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
