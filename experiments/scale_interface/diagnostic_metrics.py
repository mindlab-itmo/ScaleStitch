"""CPU-only paired diagnostics over compact, already-scored condition banks.

Each bank maps condition -> question ID -> scene_id and four binary fields:
correct, candidate_pair_compliant (nullable for natural QA), native_id_only,
and normalized_without_native_guard. No answer text or model is loaded.
CLI input paths resolve relative to the working directory. Effects are fractions.
"""
import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import random

from .natural_qa_metrics import _scene


METRIC = "protocol_normalized_exact"
REPLICATES = 10000
FIELDS = ("correct", "candidate_pair_compliant", "native_id_only",
          "normalized_without_native_guard")
PACKET_INTERACTION = {"new_T": 1, "new_A": -1, "old_T": -1, "old_A": 1}
ASSIGNMENT_INTERACTION = {"new_T": 1, "new_S": -1, "old_T": -1, "old_S": 1}
PROPERTY_INTERACTION = {"R|R": 1, "H|H": 1, "R|H": -1, "H|R": -1}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _name(value):
    _require(isinstance(value, str) and value.strip(), "Nonempty string identity required")
    return value


def _validate_bank(bank):
    _require(isinstance(bank, Mapping) and bank, "Nonempty condition bank required")
    for condition, rows in bank.items():
        _name(condition)
        _require(isinstance(rows, Mapping) and rows, "Empty diagnostic condition")
        for qid, row in rows.items():
            _name(qid)
            _require(isinstance(row, Mapping), "Scored rows must be objects")
            _scene(row.get("scene_id"))
            for field in FIELDS:
                value = row.get(field)
                _require(field in row and (
                    type(value) is int and value in (0, 1)
                    or field == "candidate_pair_compliant" and value is None),
                    "Malformed binary diagnostic field: " + field)


def summaries(bank):
    result = {}
    for name, rows in bank.items():
        result[name] = {"questions": len(rows)}
        for field in FIELDS:
            values = [row[field] for row in rows.values()]
            if all(value is None for value in values):
                result[name][field] = None
            else:
                _require(all(type(v) is int and v in (0, 1) for v in values),
                         "Mixed null/binary diagnostic field")
                count = sum(values)
                result[name][field] = {"count": count, "fraction": count / len(rows)}
    return result


def paired_family(bank, contrasts, family_size, seed, replicates=REPLICATES):
    """Resample sceneNNNN families, retaining question weighting and pairing.

    The BLAKE2b-derived RNG seed, draw order, arithmetic and percentile
    interpolation preserve the executed diagnostic analysis. Every listed
    linear contrast receives the declared Bonferroni adjustment, including DiD.
    """
    _validate_bank(bank)
    _require(isinstance(contrasts, Mapping) and contrasts
             and type(family_size) is int and family_size >= len(contrasts),
             "Invalid diagnostic contrast family")
    _require(type(seed) is int, "Integer seed required")
    _require(type(replicates) is int and replicates > 0, "Positive replicate count required")
    first = next(iter(bank.values()))
    qids = sorted(first)
    _require(all(set(values) == set(qids) for values in bank.values()),
             "Unpaired diagnostic conditions")
    _require(all(all(values[q]["scene_id"] == first[q]["scene_id"] for q in qids)
                 for values in bank.values()), "Paired scenes differ")
    family_qids, scenes = defaultdict(list), set()
    for q in qids:
        scene = first[q]["scene_id"]
        scenes.add(scene)
        family_qids[scene.split("_")[0]].append(q)
    families = sorted(family_qids)
    denominators = [len(family_qids[f]) for f in families]
    prepared = {}
    for name, weights in contrasts.items():
        _name(name)
        _require(isinstance(weights, Mapping) and set(weights) <= bank.keys()
                 and len(weights) >= 2
                 and all(type(v) in (int, float) and math.isfinite(v) and v != 0
                         for v in weights.values())
                 and math.isclose(sum(weights.values()), 0, abs_tol=1e-12),
                 "Invalid linear contrast")
        prepared[name] = [sum(sum(weight * bank[c][q]["correct"]
                                  for c, weight in weights.items())
                              for q in family_qids[f]) for f in families]
    samples = {name: [] for name in prepared}
    rng_seed = int.from_bytes(hashlib.blake2b(
        ("natural-qa-bootstrap\0" + str(seed)).encode(), digest_size=16).digest(), "big")
    rng = random.Random(rng_seed)
    if len(families) >= 2:
        for _ in range(replicates):
            draws = Counter(rng.randrange(len(families)) for _ in families)
            denominator = sum(n * denominators[i] for i, n in draws.items())
            for name, counts in prepared.items():
                samples[name].append(sum(n * counts[i] for i, n in draws.items()) / denominator)

    def interval(values, confidence):
        if not values:
            return None
        ordered = sorted(values)

        def percentile(p):
            x = p * (len(ordered) - 1)
            lo, hi = math.floor(x), math.ceil(x)
            return ordered[lo] * (1 - (x - lo)) + ordered[hi] * (x - lo)

        tail = (1 - confidence) / 2
        return [percentile(tail), percentile(1 - tail)]

    confidence = 1 - .05 / family_size
    return {
        "status": "complete", "accuracy_units": "fraction", "scoring_metric": METRIC,
        "scope": "Post-result diagnostic; not a replacement primary analysis or official benchmark score",
        "cohort": {"questions_or_pairs": len(qids), "scenes": len(scenes),
                   "scene_families": len(families),
                   "question_ids_sha256": hashlib.sha256(
                       json.dumps(qids, separators=(",", ":")).encode()).hexdigest(),
                   "selection": "input identities; independent of outcomes"},
        "bootstrap": {"replicates": replicates, "seed": seed, "cluster": "sceneNNNN family",
                      "family_size": family_size,
                      "correction": "Bonferroni; all listed pairwise and linear contrasts",
                      "intervals_available": len(families) >= 2},
        "conditions": summaries(bank),
        "contrasts": {name: {"weights": contrasts[name], "delta": sum(counts) / len(qids),
                             "ci95": interval(samples[name], .95),
                             "ci_bonferroni": interval(samples[name], confidence),
                             "bonferroni_confidence": confidence,
                             "bootstrap_degenerate": bool(samples[name])
                             and min(samples[name]) == max(samples[name])}
                      for name, counts in prepared.items()}}


def _primary(result, bank, name, weights, family_size, seed, replicates, scope):
    primary = paired_family(bank, {name: weights}, family_size, seed, replicates)
    primary["family_scope"] = scope
    result["cross_configuration_primary_interaction"] = primary
    return result


def instruction_analysis(new, old, seed, replicates=REPLICATES, *, interaction_family_size=None):
    bank = {"new_A": new["A"], "new_T": new["T"], "old_A": old["A"], "old_T": old["T"]}
    result = paired_family(bank, {
        "new_true_minus_absent": {"new_T": 1, "new_A": -1},
        "instruction_at_absent": {"new_A": 1, "old_A": -1},
        "packet_by_instruction": PACKET_INTERACTION}, 3, seed, replicates)
    result["descriptive_new_true_minus_old_true"] = (
        result["conditions"]["new_T"]["correct"]["fraction"]
        - result["conditions"]["old_T"]["correct"]["fraction"])
    result["local_family_role"] = (
        "Three contrasts within this configuration; cross-configuration primary interaction is reported separately")
    if interaction_family_size is not None:
        _require(type(interaction_family_size) is int and interaction_family_size in (4, 8),
                 "Instruction interaction family must be 8 (direct) or 4 (natural QA)")
        _primary(result, bank, "packet_by_instruction", PACKET_INTERACTION,
                 interaction_family_size, seed, replicates,
                 "One primary interaction per configuration/task; separate from the local three-contrast family")
    return result


def instruction_reassigned_analysis(at, continuation, old, seed, replicates=REPLICATES,
                                   *, interaction_family_size=None):
    bank = {"new_T": at["T"], "new_S": continuation["S"], "old_T": old["T"], "old_S": old["S"]}
    result = paired_family(bank, {
        "new_true_minus_reassigned": {"new_T": 1, "new_S": -1},
        "reassignment_by_instruction": ASSIGNMENT_INTERACTION}, 2, seed, replicates)
    if interaction_family_size is not None:
        _require(type(interaction_family_size) is int and interaction_family_size == 8,
                 "Reassignment interaction uses a separate family of 8")
        _primary(result, bank, "reassignment_by_instruction", ASSIGNMENT_INTERACTION,
                 8, seed, replicates, "Separate continuation family across eight direct configuration/task combinations")
    return result


def _pair_index(metadata):
    pairs = metadata.get("pairs", metadata) if isinstance(metadata, Mapping) else metadata
    pairs = list(pairs.values()) if isinstance(pairs, Mapping) else pairs
    _require(isinstance(pairs, list) and pairs, "Nonempty property-pair metadata required")
    result, qids = {}, set()
    for pair in pairs:
        scene = _scene(pair["scene_id"])
        roles = pair["roles"]
        _require(set(roles) == {"A", "B"}, "Pair roles must be A/B")
        sources = [roles[r]["source_id"] for r in ("A", "B")]
        rows = [roles[r]["row_id"] for r in ("A", "B")]
        _require(all(type(v) is int and v >= 0 for v in sources + rows)
                 and len(set(sources)) == len(set(rows)) == 2, "Invalid source-object pair")
        target = pair["target_source_row"]
        _require(type(target) is int and target in rows, "Gold object is outside the pair")
        key = scene, tuple(sorted(sources))
        qid = _name(pair["question_ids"]["direct"])
        _require(key not in result and qid not in qids, "Duplicate source pair or direct question ID")
        result[key] = (qid, sources[rows.index(target)])
        qids.add(qid)
    return result


def align_property_cells(radius, height, radius_height, metadata, *, expected_counts=None):
    """Join by scene and sorted source-object IDs, never by A/B role order.

    Metadata is {radius: pairs, height: pairs}; each value may be a pair list
    or {pairs: {pair_id: pair}}. Each property retains its own gold object.
    Optional expected_counts seals a chosen cohort without hard-coded dataset sizes.
    """
    for bank in (radius, height, radius_height):
        _validate_bank(bank)
    indexes = {target: _pair_index(metadata[target]) for target in ("radius", "height")}
    common = sorted(indexes["radius"].keys() & indexes["height"].keys())
    _require(common, "No shared property pairs")
    cells = {label: {} for label in ("R|R", "R|H", "H|R", "H|H")}
    discordant = set()
    for scene, objects in common:
        rq, rt = indexes["radius"][scene, objects]
        hq, ht = indexes["height"][scene, objects]
        q = scene + "|" + ",".join(map(str, objects))
        for label, source, original_qid in (
            ("R|R", radius["T"], rq), ("R|H", radius_height["T"], rq),
            ("H|R", height["R"], hq), ("H|H", height["T"], hq)):
            _require(original_qid in source and source[original_qid]["scene_id"] == scene,
                     "Property cell pairing incomplete")
            cells[label][q] = source[original_qid]
        if rt != ht:
            discordant.add(q)
    if expected_counts is not None:
        actual = {"radius": len(indexes["radius"]), "height": len(indexes["height"]),
                  "shared": len(common), "scenes": len({key[0] for key in common}),
                  "discordant": len(discordant)}
        _require(isinstance(expected_counts, Mapping) and expected_counts
                 and set(expected_counts) <= actual.keys(), "Unknown expected property count")
        _require(all(type(n) is int and n >= 0 and actual[k] == n
                     for k, n in expected_counts.items()), "Property cohort differs from declared counts")
    return cells, discordant


def property_analysis(cells, discordant, seed, replicates=REPLICATES, *, interaction_family_size=None):
    _validate_bank(cells)
    _require(set(cells) == {"R|R", "R|H", "H|R", "H|H"}, "Four property cells required")
    _require(all(set(rows) == set(next(iter(cells.values()))) for rows in cells.values()),
             "Unpaired property cells")
    _require(not isinstance(discordant, (str, bytes, Mapping)), "Discordant cohort must be an ID sequence")
    discordant = list(discordant)
    _require(len(discordant) == len(set(discordant))
             and set(discordant) <= set(next(iter(cells.values()))), "Invalid discordant cohort")
    effects = {"radius_specificity": {"R|R": 1, "R|H": -1},
               "height_specificity": {"H|H": 1, "H|R": -1},
               "property_by_packet": PROPERTY_INTERACTION}
    output = {}
    for name, chosen in (("all_shared", set(next(iter(cells.values())))),
                         ("discordant_measurement_order", set(discordant))):
        bank = {label: {q: v for q, v in values.items() if q in chosen}
                for label, values in cells.items()}
        result = paired_family(bank, effects, 3, seed, replicates)
        result["target_convention"] = "Each question property retains its own gold object; R|R + H|H - R|H - H|R"
        if interaction_family_size is not None:
            _require(type(interaction_family_size) is int and interaction_family_size == 4,
                     "Property interaction uses a separate family of 4")
            _primary(result, bank, "property_by_packet", PROPERTY_INTERACTION, 4, seed, replicates,
                     "One primary interaction per four configurations; separate per input-defined cohort")
        output[name] = result
    return output


def feature_analysis(new, old, seed, replicates=REPLICATES):
    within = paired_family(new, {"true_minus_absent": {"T": 1, "A": -1},
                                "true_minus_reassigned": {"T": 1, "S": -1}}, 2, seed, replicates)
    bank = {**{"control_" + c: v for c, v in new.items()},
            **{"original_" + c: v for c, v in old.items() if c in ("A", "T", "S")}}
    vs_original = paired_family(bank, {
        "packet_effect_minus_original": {"control_T": 1, "control_A": -1, "original_T": -1, "original_A": 1},
        "binding_effect_minus_original": {"control_T": 1, "control_S": -1, "original_T": -1, "original_S": 1}},
        2, seed, replicates)
    return {"within_feature_control": within, "feature_by_packet_vs_original": vs_original}


def between_feature_controls(zero, permute, seed, replicates=REPLICATES):
    bank = {**{"zero_" + c: v for c, v in zero.items()},
            **{"permute_" + c: v for c, v in permute.items()}}
    return paired_family(bank, {
        "packet_effect_zero_minus_permute": {"zero_T": 1, "zero_A": -1, "permute_T": -1, "permute_A": 1},
        "binding_effect_zero_minus_permute": {"zero_T": 1, "zero_S": -1, "permute_T": -1, "permute_S": 1}},
        2, seed, replicates)


def _keys(value, required, optional=()):
    _require(isinstance(value, Mapping) and set(required) <= value.keys()
             and set(value) <= set(required) | set(optional), "Missing or unknown configuration fields")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def diagnose(config_path, output):
    """Evaluate a config without loading torch, model weights, or raw answers."""
    config_path, output = Path(config_path).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"output must be a new directory: {output}")
    sources = {}

    def read(path):
        path = (Path.cwd() / path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        _require(str(path) not in sources or sources[str(path)] == digest, "Input changed between reads")
        sources[str(path)] = digest
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                          parse_constant=lambda value: _require(False, "Nonfinite JSON number"))

    config = read(config_path)
    _keys(config, {"banks", "analyses", "seed"}, {"replicates"})
    seed, replicates = config["seed"], config.get("replicates", REPLICATES)
    _require(isinstance(config["banks"], Mapping) and config["banks"], "Input banks required")
    banks = {}
    for name, spec in config["banks"].items():
        _name(name)
        _keys(spec, {"path", "conditions"})
        document = read(spec["path"])
        _require(isinstance(document, Mapping)
                 and document.get("schema_version") == "scalestitch_scored_rows/v1",
                 "Unsupported scored-row bank schema")
        available = document.get("conditions")
        _require(isinstance(available, Mapping), "Scored conditions must be an object")
        _require(isinstance(spec["conditions"], Mapping) and spec["conditions"], "Condition aliases required")
        _require(all(c in available for c in spec["conditions"].values()), "Condition alias is absent from bank")
        banks[name] = {alias: available[c] for alias, c in spec["conditions"].items()}
        _validate_bank(banks[name])
    _require(isinstance(config["analyses"], Mapping) and config["analyses"], "Analyses required")
    results = {}
    for name, spec in config["analyses"].items():
        _name(name)
        _require(isinstance(spec, Mapping), "Analysis specification must be an object")
        kind = spec.get("kind")
        common = {"kind"}
        if kind == "instruction":
            _keys(spec, common | {"new", "old", "interaction_family_size"})
            result = instruction_analysis(banks[spec["new"]], banks[spec["old"]], seed, replicates,
                                          interaction_family_size=spec["interaction_family_size"])
        elif kind == "instruction_reassigned":
            _keys(spec, common | {"at", "continuation", "old"})
            result = instruction_reassigned_analysis(banks[spec["at"]], banks[spec["continuation"]],
                                                     banks[spec["old"]], seed, replicates, interaction_family_size=8)
        elif kind in ("feature", "between_features"):
            names = ("new", "old") if kind == "feature" else ("zero", "permute")
            _keys(spec, common | set(names))
            function = feature_analysis if kind == "feature" else between_feature_controls
            result = function(*(banks[spec[key]] for key in names), seed, replicates)
        elif kind == "property":
            _keys(spec, common | {"radius", "height", "radius_height", "metadata"}, {"expected_counts"})
            _keys(spec["metadata"], {"radius", "height"})
            metadata = {target: read(path) for target, path in spec["metadata"].items()}
            cells, discordant = align_property_cells(
                banks[spec["radius"]], banks[spec["height"]], banks[spec["radius_height"]],
                metadata, expected_counts=spec.get("expected_counts"))
            result = property_analysis(cells, discordant, seed, replicates, interaction_family_size=4)
        elif kind == "linear":
            _keys(spec, common | {"bank", "contrasts", "family_size"})
            result = paired_family(banks[spec["bank"]], spec["contrasts"], spec["family_size"], seed, replicates)
        else:
            raise ValueError("Unknown diagnostic kind: " + str(kind))
        results[name] = result
    for path, digest in sources.items():
        _require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, "Input changed during analysis")
    report = {"schema_version": "scalestitch_diagnostic_metrics/v1", "status": "complete",
              "seed": seed, "bootstrap_replicates": replicates, "no_raw_answers_exported": True,
              "analyses": results}
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (("diagnostics.json", report), ("manifest.json", {"sources_sha256": sources,
                            "diagnostic_metrics_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})):
        (output / filename).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    diagnose(args.config, args.output)


if __name__ == "__main__":
    main()
