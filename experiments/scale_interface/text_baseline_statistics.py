"""Fixed paired scene-family contrasts for the ordinary-label baseline."""
from collections import Counter, defaultdict
import hashlib
import math
import random

def _interval(values, confidence):
    ordered = sorted(values)
    def percentile(p):
        index = (len(ordered) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return [percentile((1 - confidence) / 2), percentile(1 - (1 - confidence) / 2)]


def bootstrap_interactions(rows, weights, *, replicates=10000, seed=20260911, family_size=3):
    """Prespecified linear contrasts; resample complete scene families jointly."""
    if type(replicates) is not int or replicates < 1 or type(family_size) is not int or family_size < len(weights):
        raise ValueError("positive bootstrap size and adequate contrast family required")
    names = {condition for coefficients in weights.values() for condition in coefficients}
    for coefficients in weights.values():
        if (len(coefficients) < 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or v == 0 for v in coefficients.values())
                or not math.isclose(sum(coefficients.values()), 0, abs_tol=1e-12)):
            raise ValueError("invalid linear contrast weights")
    families = defaultdict(list)
    seen = set()
    for row in rows:
        qid = str(row["qid"])
        if qid in seen or not names <= row["scores"].keys() or any(
                row["scores"][name] not in (0, 1) for name in names):
            raise ValueError("duplicate question or missing/bad condition scores")
        seen.add(qid)
        families[row["scene_id"].split("_")[0]].append(row)
    if not seen:
        raise ValueError("empty interaction cohort")
    ordered = sorted(families)
    sizes = [len(families[family]) for family in ordered]
    deltas = {name: [sum(sum(coefficient * row["scores"][condition] for condition, coefficient in coefficients.items())
                        for row in families[family]) for family in ordered] for name, coefficients in weights.items()}
    samples = {name: [] for name in weights}
    rng_seed = int.from_bytes(hashlib.blake2b(f"natural-qa-bootstrap\0{seed}".encode(), digest_size=16).digest(), "big")
    rng = random.Random(rng_seed)
    if len(ordered) >= 2:
        for _ in range(replicates):
            draws = Counter(rng.randrange(len(ordered)) for _ in ordered)
            n = sum(weight * sizes[index] for index, weight in draws.items())
            for name, values in deltas.items():
                samples[name].append(sum(weight * values[index] for index, weight in draws.items()) / n)
    confidence = 1 - .05 / family_size
    return dict(questions=len(seen), scene_families=len(ordered), replicates=replicates, seed=seed,
        family_size=family_size, cluster="sceneNNNN family; shared paired resamples",
        contrasts={name: dict(weights=weights[name], delta=sum(values) / len(seen),
            ci95=_interval(samples[name], .95) if samples[name] else None,
            ci_bonferroni=_interval(samples[name], confidence) if samples[name] else None,
            bonferroni_confidence=confidence,
            bootstrap_degenerate=bool(samples[name]) and min(samples[name]) == max(samples[name]))
            for name, values in deltas.items()},
        interpretation="Question-weighted paired effects; no compliant-only exclusions; local family correction.")


FAMILIES = {
    'full': dict(family_size=3, weights={
        'T_minus_A': {'full_true': 1, 'absent': -1},
        'T_minus_C': {'full_true': 1, 'full_constant': -1},
        'T_minus_S': {'full_true': 1, 'full_swapped': -1}}),
    'focused': dict(family_size=3, weights={
        'T_minus_A': {'pair_true': 1, 'absent': -1},
        'T_minus_C': {'pair_true': 1, 'pair_constant': -1},
        'T_minus_S': {'pair_true': 1, 'pair_swapped': -1}}),
    'scope': dict(family_size=2, weights={
        'true_scope_effect': {'pair_true': 1, 'full_true': -1},
        'assignment_interaction': {'pair_true': 1, 'pair_swapped': -1,
                                   'full_true': -1, 'full_swapped': 1}}),
}
