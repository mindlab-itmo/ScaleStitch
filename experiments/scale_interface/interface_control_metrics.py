"""Paired offline analysis of fourteen fixed packet/answer-rule controls."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from . import reviewer_diagnostics as diagnostics
from .text_baseline_statistics import bootstrap_interactions


PROPERTIES = ("radius", "height")
CELLS = tuple(["af." + mode for mode in "ATCS"] + ["pf." + mode for mode in "TCS"]
              + ["ar." + mode for mode in "ATCS"] + ["pr." + mode for mode in "TCS"])
PRIMARY = {
    "scope_true_free": {"pf.T": 1, "af.T": -1},
    "decision_utility_all": {"ar.T": 1, "ar.A": -1, "af.T": -1, "af.A": 1},
    "scope_assignment_restricted": {"pr.T": 1, "pr.S": -1, "ar.T": -1, "ar.S": 1},
    "reversed_ranking_assignment_pair_restricted": {"pr.T": 1, "pr.S": -1},
}
SEED, PRIMARY_REPLICATES, SECONDARY_REPLICATES = 20260911, 100000, 10000
OUTCOMES = diagnostics.OUTCOMES


def _cell(group, mode):
    return ("af.A" if group == "pf" else "ar.A") if mode == "A" and group in ("pf", "pr") else group + "." + mode


def secondary_families():
    packet = {group: {"T_minus_" + mode: {_cell(group, "T"): 1, _cell(group, mode): -1}
                       for mode in "ACS"} for group in ("af", "pf", "ar", "pr")}
    scope = {rule: {"true_scope_effect": {"p" + rule + ".T": 1, "a" + rule + ".T": -1},
        "assignment_interaction": {"p" + rule + ".T": 1, "p" + rule + ".S": -1,
                                   "a" + rule + ".T": -1, "a" + rule + ".S": 1}} for rule in ("f", "r")}
    readout = {"R_minus_F_A": {"ar.A": 1, "af.A": -1},
               "R_minus_F_allT": {"ar.T": 1, "af.T": -1},
               "R_minus_F_pairT": {"pr.T": 1, "pf.T": -1}}
    for scope_name, free, restricted in (("all", "af", "ar"), ("pair", "pf", "pr")):
        for control in "ACS":
            readout[scope_name + "_readout_by_" + control] = {
                _cell(restricted, "T"): 1, _cell(restricted, control): -1,
                _cell(free, "T"): -1, _cell(free, control): 1}
    return packet, scope, readout


def prepare_metadata(question, reference, full_packets, focused_packets):
    candidates = [int(value) for value in re.findall(r"<OBJ([0-9]{3})>", question["prompt"])]
    if len(candidates) != 2 or len(set(candidates)) != 2 or question["prompt"].count("<OBJ") != 2:
        raise ValueError("exactly two original source-row candidates are required")
    answers = reference["answers"]
    targets = {int(match[1]) for answer in answers for match in [diagnostics.NATIVE_ID.fullmatch(answer.strip())] if match}
    if (str(reference["question_id"]) != str(question["qid"]) or reference["scene_id"] != question["scene_id"]
            or len(targets) != 1 or len(targets) != len(set(answers))):
        raise ValueError("one source-row reference bound to the question is required")
    objects = [{key: row[key] for key in ("visible_id", "source_row")} for row in full_packets["T"]["records"]]
    assignment = question["assigned_ids"]
    if (any(type(value) is not int for value in assignment) or sorted(assignment) != list(range(len(assignment)))
            or any(obj["visible_id"] >= len(assignment) or assignment[obj["visible_id"]] != obj["source_row"] for obj in objects)):
        raise ValueError("saved source-to-visible assignment differs from the packet")
    result = dict(qid=str(question["qid"]), scene_id=question["scene_id"],
        scene_family=question["scene_id"].split("_")[0], objects=objects,
        candidate_source_rows=candidates, target_source_row=next(iter(targets)), direction="max",
        packets={"A": None, **{mode: copy.deepcopy(full_packets[mode]["records"]) for mode in "TCS"}},
        focused_packets={mode: copy.deepcopy(focused_packets[mode]["records"]) for mode in "TCS"},
        full_packet_sha256={mode: full_packets[mode]["text_sha256"] for mode in "TCS"},
        focused_parent_sha256={mode: focused_packets[mode]["full_packet_sha256"] for mode in "TCS"})
    _validate_inputs([result])
    return result


def _validate_inputs(metadata):
    parsed = diagnostics._metadata(metadata, ("A", "T", "C", "S"))
    for original in metadata:
        item = parsed[str(original["qid"])]
        if item["family"] != item["scene"].split("_")[0] or item["direction"] != "max":
            raise ValueError("the fixed larger-than cohort uses sceneNNNN families")
        if set(item["mapping"]) != set(item["mapping"].values()):
            raise ValueError("active visible/source assignment must preserve the active source set")
        sources = sorted(item["mapping"].values())
        donors = {source: sources[index - 1] for index, source in enumerate(sources)}
        truth = {row["source_row"]: row["value_cm"] for row in original["packets"]["T"]}
        for mode in "TCS":
            records = original["packets"][mode]
            if ([{key: row[key] for key in ("visible_id", "source_row")} for row in records] != original["objects"]
                    or any(type(row["value_cm"]) is not int or row["value_cm"] <= 0 for row in records)):
                raise ValueError("complete ordered centimeter packets must retain the object mapping")
            for row in records:
                source = row["source_row"]
                expected_donor = None if mode == "C" else source if mode == "T" else donors[source]
                expected_value = 100 if mode == "C" else truth[expected_donor]
                if ("value_source_row" not in row or row["value_source_row"] != expected_donor
                        or row["value_cm"] != expected_value):
                    raise ValueError("full-scene measurement donor or value differs")
            focused = [row for row in records if row["source_row"] in item["candidates"]]
            if original["focused_packets"][mode] != focused:
                raise ValueError("focused packet must be the ordered full-scene subset with unchanged donors")
            digest = original["full_packet_sha256"][mode]
            if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    or original["focused_parent_sha256"][mode] != digest):
                raise ValueError("focused packet parent hash differs")
        if diagnostics._winner(item["packets"]["T"], item["candidates"], "max") != item["target"]:
            raise ValueError("the correct input packet must have its unique physical reference winner")
        swapped = diagnostics._winner(item["packets"]["S"], item["candidates"], "max")
        item["s_ranking"] = "tied" if swapped is None else "preserved" if swapped == item["target"] else "flipped"
        item["packet_winners"] = {mode: diagnostics._winner(item["packets"][mode], item["candidates"], "max") for mode in "ATCS"}
    return parsed


def validate_metadata(metadata, *, expected_count, expected_rankings):
    parsed = _validate_inputs(metadata)
    counts = {name: sum(item["s_ranking"] == name for item in parsed.values()) for name in ("flipped", "preserved", "tied")}
    if len(parsed) != expected_count or counts != expected_rankings:
        raise ValueError("input-defined cohort or S ranking counts differ; outcomes must not be opened")
    return parsed, counts


def _restricted(row, item, selection, mode, start, count):
    info = row.get("candidate_decision", {})
    inverse = {source: visible for visible, source in item["mapping"].items()}
    visible = [inverse[source] for source in item["candidates"]]
    names = [f"<OBJ{value:03d}>" for value in visible]
    tokens = [start + value for value in visible]
    saved_tokens = info.get("candidate_token_ids")
    if (any(value >= count for value in visible) or info.get("candidate_identifiers") != names
            or not isinstance(saved_tokens, list) or any(type(value) is not int for value in saved_tokens)
            or saved_tokens != tokens or selection["outcome"] not in ("correct", "wrong_candidate")
            or info.get("score_space") != "raw model logits before generation processors"
            or info.get("decision_rule") != "maximum first-token logit; exact ties use question order"):
        raise ValueError("restricted native candidates, source mapping or decision rule differ")
    logits = info.get("candidate_logits")
    probabilities = info.get("candidate_conditional_probabilities")
    if (not isinstance(logits, list) or len(logits) != 2 or not isinstance(probabilities, list) or len(probabilities) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in logits + probabilities)):
        raise ValueError("candidate logits and probabilities must be two finite values")
    maximum = max(logits)
    exponents = [math.exp(value - maximum) for value in logits]
    expected = [value / sum(exponents) for value in exponents]
    mass = info.get("pair_probability")
    if (any(not 0 <= value <= 1 for value in probabilities) or not math.isclose(sum(probabilities), 1., abs_tol=2e-6)
            or any(not math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-6) for a, b in zip(expected, probabilities))
            or type(mass) not in (int, float) or not math.isfinite(mass) or not 0 <= mass <= 1 + 2e-6):
        raise ValueError("candidate or full-vocabulary pair probabilities are invalid")
    tie = logits[0] == logits[1]
    winner = 0 if logits[0] >= logits[1] else 1
    if (type(info.get("tie")) is not bool or info["tie"] != tie
            or selection["source"] != item["candidates"][winner]):
        raise ValueError("restricted logit winner or exact question-order tie break differs")
    if ("output_token_ids" in row and (not isinstance(row["output_token_ids"], list)
            or any(type(value) is not int for value in row["output_token_ids"])
            or row["output_token_ids"] != [tokens[winner]])):
        raise ValueError("restricted generated native token differs from its selected candidate")
    top = info.get("unrestricted_first_token_id")
    if type(top) is not int or not 0 <= top < start + count:
        raise ValueError("invalid unrestricted first-token ID")
    target_index = item["candidates"].index(item["target"])
    packet = item["packet_winners"][mode]
    return dict(pair_mass=mass, conditional_correct=probabilities[target_index], tie=tie,
        conditional_packet_correct=None if packet is None else probabilities[item["candidates"].index(packet)],
        top_category="in_pair" if top in tokens else "native_outside_pair" if start <= top < start + count else "other_vocabulary")


def score_bank(bank, metadata, *, end_sym, native_token_start, native_token_count):
    if (set(bank) != set(CELLS) or not isinstance(end_sym, str) or not end_sym
            or type(native_token_start) is not int or native_token_start < 0
            or type(native_token_count) is not int or native_token_count < 2):
        raise ValueError("declare the fourteen unique cells and native decoder interface")
    scored = {qid: {} for qid in metadata}
    for cell in CELLS:
        rows = bank[cell]
        keys = [str(row["qid"]) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != set(metadata):
            raise ValueError("prediction matrix has a missing, duplicate or extra cohort row")
        for row in rows:
            qid = str(row["qid"]); item = metadata[qid]
            if row["scene_id"] != item["scene"] or row.get("error"):
                raise ValueError("prediction scene or technical error violates the fixed cohort")
            selected = diagnostics._selection(row, item, end_sym)
            selected["probability"] = (_restricted(row, item, selected, cell[-1], native_token_start, native_token_count)
                                        if cell.startswith(("ar.", "pr.")) else None)
            if selected["probability"] is None and "candidate_decision" in row:
                raise ValueError("a free-answer cell unexpectedly contains a restricted decision")
            scored[qid][cell] = selected
    vectors = [dict(qid=qid, scene_id=metadata[qid]["scene"], scores={cell: int(item["outcome"] == "correct")
               for cell, item in cells.items()}) for qid, cells in scored.items()]
    return scored, vectors


def _family(vectors, weights, size, replicates):
    if vectors:
        result = bootstrap_interactions(vectors, weights, replicates=replicates, seed=SEED, family_size=size)
    else:
        confidence = 1 - .05 / size
        result = dict(questions=0, scene_families=0, replicates=replicates, seed=SEED, family_size=size,
            contrasts={name: dict(weights=value, delta=None, ci95=None, ci_bonferroni=None,
                bonferroni_confidence=confidence, bootstrap_degenerate=False) for name, value in weights.items()})
    if result["scene_families"] < 2:
        result["unavailable_reason"] = "empty_input_defined_population" if not vectors else "fewer_than_two_scene_families"
    return result


def primary_intervals(vectors, *, flipped_qids):
    names = list(PRIMARY)
    complete = _family(vectors, {name: PRIMARY[name] for name in names[:3]}, 8, PRIMARY_REPLICATES)
    flipped = _family([row for row in vectors if row["qid"] in flipped_qids], {names[3]: PRIMARY[names[3]]}, 8, PRIMARY_REPLICATES)
    result = {}
    for family in (complete, flipped):
        common = {key: value for key, value in family.items() if key != "contrasts"}
        for name, contrast in family["contrasts"].items():
            result[name] = dict(common, **contrast)
    return result


def _quantiles(values):
    if not values:
        return dict(count=0, quantiles=None)
    if len(values) == 1:
        quantiles = [values[0]] * 99
    else:
        quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return dict(count=len(values), quantiles=dict(q00=min(values), q05=quantiles[4], q25=quantiles[24],
        q50=quantiles[49], q75=quantiles[74], q95=quantiles[94], q100=max(values)))


def _condition_summary(cell, scored, metadata):
    n = len(metadata); outcomes = Counter(row["outcome"] for row in scored.values())
    rate = diagnostics._rate
    positions = Counter(dict.fromkeys(("first", "second", "outside", "invalid"), 0))
    visible_ids = Counter(); agreements = 0; defined = 0
    order_groups = {position: dict(questions=0, correct=0) for position in ("first", "second")}
    for qid, row in scored.items():
        meta = metadata[qid]; source = row["source"]
        group = "first" if meta["target"] == meta["candidates"][0] else "second"
        order_groups[group]["questions"] += 1; order_groups[group]["correct"] += row["outcome"] == "correct"
        positions["first" if source == meta["candidates"][0] else "second" if source == meta["candidates"][1]
                  else "outside" if source is not None else "invalid"] += 1
        if source is not None:
            inverse = {s: v for v, s in meta["mapping"].items()}; visible_ids[str(inverse[source])] += 1
        winner = meta["packet_winners"][cell[-1]]
        if winner is not None:
            defined += 1; agreements += source == winner
    result = dict(questions=n, accuracy=rate(outcomes["correct"], n), visible_ID_validity=rate(n - outcomes["invalid"], n),
        candidate_compliance=rate(outcomes["correct"] + outcomes["wrong_candidate"], n),
        outcomes={name: outcomes[name] for name in OUTCOMES}, question_position_counts=dict(positions),
        accuracy_by_input_target_order=order_groups, packet_agreement=rate(agreements, defined),
        packet_agreement_undefined_questions=n - defined)
    if cell[-1] in "AC":
        result["identifier_preference_descriptive"] = dict(visible_ID_counts=dict(sorted(visible_ids.items())),
            question_position_counts=dict(positions), denominator=n, interpretation="Descriptive choice counts; no causal ID-bias attribution")
    probabilities = [row["probability"] for row in scored.values() if row["probability"] is not None]
    if probabilities:
        result["probability"] = dict(questions=n, pair_mass=_quantiles([p["pair_mass"] for p in probabilities]),
            pair_mass_zero=sum(p["pair_mass"] == 0 for p in probabilities),
            conditional_physical_correct=_quantiles([p["conditional_correct"] for p in probabilities]),
            conditional_packet_correct=_quantiles([p["conditional_packet_correct"] for p in probabilities if p["conditional_packet_correct"] is not None]),
            exact_ties=sum(p["tie"] for p in probabilities),
            unrestricted_top_categories=dict(Counter(p["top_category"] for p in probabilities)))
    return result


def _summarize(task, metadata, ranking_counts, scored, vectors):
    packet, scope, readout = secondary_families()
    banks = {cell: {qid: rows[cell] for qid, rows in scored.items()} for cell in CELLS}
    transitions = {}
    for scope_name, free, restricted in (("all", "af", "ar"), ("pair", "pf", "pr")):
        for mode in "ATCS":
            a, b = _cell(free, mode), _cell(restricted, mode)
            raw = diagnostics._transitions(banks, [a, b], metadata)[a + "->" + b]["outcome_counts"]
            transitions[scope_name + "." + mode] = dict(questions=len(metadata), shared_absent=mode == "A",
                counts={before: {after: raw.get(before + "->" + after, 0) for after in OUTCOMES} for before in OUTCOMES})
    agreement = {}
    for stratum in ("preserved", "flipped", "tied"):
        selected = {qid for qid, meta in metadata.items() if meta["s_ranking"] == stratum}
        agreement[stratum] = {cell: diagnostics._rate(sum(scored[qid][cell]["source"] == metadata[qid]["packet_winners"]["S"]
            for qid in selected), len(selected)) if stratum != "tied" else diagnostics._rate(0, 0)
            for cell in CELLS if cell.endswith(".S")}
        if stratum == "flipped":
            for cell in ("ar.S", "pr.S"):
                correct = sum(scored[qid][cell]["outcome"] == "correct" for qid in selected)
                if correct + agreement[stratum][cell]["count"] != len(selected):
                    raise ValueError("flipped restricted packet agreement must complement physical accuracy")
    return dict(task=task, questions=len(metadata), outputs=14 * len(metadata), input_ranking_counts=ranking_counts,
        input_target_order_counts={position: sum((meta["target"] == meta["candidates"][0]) == (position == "first")
            for meta in metadata.values()) for position in ("first", "second")},
        primary=primary_intervals(vectors, flipped_qids={qid for qid, meta in metadata.items() if meta["s_ranking"] == "flipped"}),
        secondary=dict(packet={name: _family(vectors, weights, 3, SECONDARY_REPLICATES) for name, weights in packet.items()},
            scope={name: _family(vectors, weights, 2, SECONDARY_REPLICATES) for name, weights in scope.items()},
            readout=_family(vectors, readout, 9, SECONDARY_REPLICATES)),
        conditions={cell: _condition_summary(cell, banks[cell], metadata) for cell in CELLS},
        packet_agreement_by_input_S_ranking=agreement, transitions=transitions)


def analyze(metadata, banks, *, expected_counts, expected_s_ranking_counts, end_sym,
            native_token_start=128256, native_token_count=150):
    if any(set(value) != set(PROPERTIES) for value in (metadata, banks, expected_counts, expected_s_ranking_counts)):
        raise ValueError("both complete declared properties are required for the global eight-contrast family")
    # Every input-defined population is validated before any answer is scored.
    prepared = {task: validate_metadata(metadata[task], expected_count=expected_counts[task],
        expected_rankings=expected_s_ranking_counts[task]) for task in PROPERTIES}
    scored = {task: score_bank(banks[task], prepared[task][0], end_sym=end_sym,
        native_token_start=native_token_start, native_token_count=native_token_count) for task in PROPERTIES}
    return dict(schema_version="scalestitch_interface_control_analysis/v1", status="complete_declared_matrix",
        primary_global_family_size=8, primary_replicates=PRIMARY_REPLICATES, secondary_replicates=SECONDARY_REPLICATES,
        seed=SEED, outputs=sum(expected_counts.values()) * 14, raw_predictions_exported=False,
        properties={task: _summarize(task, prepared[task][0], prepared[task][1], *scored[task]) for task in PROPERTIES},
        interpretation=["Primary intervals use one global m8 Bonferroni family across both properties.",
            "Secondary families are locally adjusted descriptive follow-ups, not independent confirmation.",
            "Every answer remains in its fixed input-defined population; invalid or outside-pair answers are errors.",
            "Scope changes length, distractors and positions; restriction changes the whole answer decision rule.",
            "Uncertainty covers scene-family resampling, not host or training-seed variation."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.config is None:
        print(json.dumps(dict(status="not_executed", cells=list(CELLS), primary=PRIMARY,
            global_family_size=8, primary_replicates=PRIMARY_REPLICATES, secondary_replicates=SECONDARY_REPLICATES, seed=SEED)))
        return 0
    if args.output is None:
        parser.error("--output is required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected = {"metadata", "banks", "expected_counts", "expected_s_ranking_counts", "end_sym", "native_token_start", "native_token_count"}
    if set(config) != expected:
        raise ValueError("unknown or missing analysis configuration; resampling settings are fixed")
    if any(set(config[key]) != set(PROPERTIES) for key in
           ("metadata", "banks", "expected_counts", "expected_s_ranking_counts")):
        raise ValueError("both input properties are required before opening answer banks")
    sources = {}
    loaded = {}
    for role in ("metadata", "banks"):
        loaded[role] = {}
        for task, filename in config[role].items():
            path = (args.config.parent / filename).resolve()
            data = path.read_bytes(); sources[role + "." + task] = hashlib.sha256(data).hexdigest()
            loaded[role][task] = json.loads(data)
        if role == "metadata":
            for task in PROPERTIES:
                validate_metadata(loaded[role][task], expected_count=config["expected_counts"][task],
                    expected_rankings=config["expected_s_ranking_counts"][task])
    result = analyze(**loaded, **{key: config[key] for key in expected - {"metadata", "banks"}})
    result["source_sha256"] = sources
    result["analysis_config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    result["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], outputs=result["outputs"], primary_global_family_size=8)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
