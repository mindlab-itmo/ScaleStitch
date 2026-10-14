"""Synthetic parity fixtures; no model, GPU, collector or private-path dependency.

Golden statistics were generated once from the executed collector's functions
extracted with ast, then compared with this public implementation. Captures from
one scene family and unequal family sizes exercise pairing and weighting.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface import diagnostic_metrics as metrics

GOLDEN_SOURCE_SHA256 = "8e758cf705550d19d7ec22eab8469681cd55ba0ea6909d67ab5d16bf2f4b8386"
GOLDENS = {
    "alignment": "d83c3bb6c291ea8784107eab28ca4c77372810697efca331c824fd5c37a6a41a",
    "assignment_local2": "87da0aafd93715aa31ca8ca428fad7e0e067ec36c5978fdbc04fced10f71e779",
    "assignment_primary8": "9e1f22f9a595f11c5e847eb24154c00d933d2cb3b6f4fa31c197bbdceeeede6a",
    "between_local2": "c4118fcd955e175dbf932b0d1a3a46fa8c0c95a83b66784199597573de00c443",
    "feature_local2": "dcf66b1d0076cc2e8d50391a203d2c5ff704d3f0f668e096df4d2b4d842e6595",
    "instruction_local3": "83e9ec1be49d0fc09688133f718b7c58b6cf775128537c38f9a3310b6f02af14",
    "instruction_primary4": "0f7fb4f3d643d9e0d822503f2dd2f75f683efe387cae0aa19486cdda5d8722fd",
    "instruction_primary8": "23f74d4bd91c260c8e988c0fedd416d38ae6e27c89c0a87cd05cad42d6030a6c",
    "property_local3": "b660a8a70905892b7b63d499e0134611406bcfa2209686bf368b6eae866973d5",
    "property_primary4": "296aadf1a86c2e0f2976d88aeb7a3fc9b9d30a7b11630abdbe6c1132c49e05a1"
}


def vectors(kind):
    scenes = (["scene0001_00"] * 3 + ["scene0001_01"] * 2 + ["scene0002_00"] * 4
              + ["scene0003_00"] + ["scene0004_00"] * 5 + ["scene0005_00"] * 2
              + ["scene0006_00"] * 3)
    rows = {}
    for i, scene in enumerate(scenes):
        hit = int(((i + 1) * (kind + 2) + i // 3) % 11 < 3 + kind % 5)
        rows["q" + str(i)] = {"scene_id": scene, "correct": hit,
                             "candidate_pair_compliant": 1 if hit else int(i % 3 != 0),
                             "native_id_only": 1, "normalized_without_native_guard": hit}
    return rows


def fixture_banks():
    return [{c: vectors(3 * j + i) for i, c in enumerate(("A", "T", "S"))}
            for j in range(4)]


def numeric_report(value):
    if "contrasts" in value:
        return {key: value[key] for key in ("accuracy_units", "scoring_metric", "cohort",
                                           "bootstrap", "conditions", "contrasts")}
    return {key: numeric_report(item) for key, item in value.items() if isinstance(item, dict)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def statistical_cases(module):
    old, new, zero, permute = fixture_banks()
    packet = {"new_A": new["A"], "new_T": new["T"], "old_A": old["A"], "old_T": old["T"]}
    binding = {"new_T": new["T"], "new_S": new["S"], "old_T": old["T"], "old_S": old["S"]}
    cells = {"R|R": new["T"], "R|H": old["T"], "H|R": old["S"], "H|H": new["S"]}
    settings = (73, 257)
    return {
        "instruction_local3": module.instruction_analysis(new, old, *settings),
        "assignment_local2": module.instruction_reassigned_analysis(new, new, old, *settings),
        "property_local3": module.property_analysis(cells, {"q0", "q3", "q6", "q10", "q17"}, *settings),
        "feature_local2": module.feature_analysis(zero, old, *settings),
        "between_local2": module.between_feature_controls(zero, permute, *settings),
        "instruction_primary8": module.paired_family(packet, {"packet_by_instruction":
            {"new_T": 1, "new_A": -1, "old_T": -1, "old_A": 1}}, 8, *settings),
        "instruction_primary4": module.paired_family(packet, {"packet_by_instruction":
            {"new_T": 1, "new_A": -1, "old_T": -1, "old_A": 1}}, 4, *settings),
        "assignment_primary8": module.paired_family(binding, {"reassignment_by_instruction":
            {"new_T": 1, "new_S": -1, "old_T": -1, "old_S": 1}}, 8, *settings),
        "property_primary4": module.paired_family(cells, {"property_by_packet":
            {"R|R": 1, "H|H": 1, "R|H": -1, "H|R": -1}}, 4, *settings),
    }


def alignment_fixture():
    metadata = {target: {"pairs": {}} for target in ("radius", "height")}
    radius, height, crossed = {"T": {}}, {"T": {}, "R": {}}, {"T": {}}
    for target, count in (("radius", 4161), ("height", 4157)):
        for i in range(count):
            identity = i if i < 4131 or target == "radius" else i + 10000
            scene = f"scene{i % 70:04d}_00"
            a, b = 2 * identity, 2 * identity + 1
            roles = ({"A": {"source_id": a, "row_id": 0}, "B": {"source_id": b, "row_id": 1}}
                     if target == "radius" else
                     {"A": {"source_id": b, "row_id": 0}, "B": {"source_id": a, "row_id": 1}})
            qid = target + str(i)
            metadata[target]["pairs"][qid] = {
                "scene_id": scene, "roles": roles,
                "target_source_row": 0 if target == "radius" or i < 1228 else 1,
                "question_ids": {"direct": qid}}
            row = {"scene_id": scene, "correct": 1, "candidate_pair_compliant": 1,
                   "native_id_only": 1, "normalized_without_native_guard": 1}
            wrong = dict(row, correct=0, normalized_without_native_guard=0)
            if target == "radius":
                radius["T"][qid], crossed["T"][qid] = row, wrong
            else:
                height["T"][qid], height["R"][qid] = row, wrong
    return radius, height, crossed, metadata


class DiagnosticMetricsTests(unittest.TestCase):
    def test_exact_executed_kernel_golden_statistics(self):
        actual = statistical_cases(metrics)
        self.assertEqual(set(actual), set(GOLDENS) - {"alignment"})
        for name, result in actual.items():
            with self.subTest(name=name):
                self.assertEqual(fingerprint(numeric_report(result)), GOLDENS[name])
        first = actual["instruction_primary8"]
        self.assertEqual((first["cohort"]["questions_or_pairs"], first["cohort"]["scenes"],
                          first["cohort"]["scene_families"]), (20, 7, 6))
        effect = first["contrasts"]["packet_by_instruction"]
        self.assertEqual(effect["bonferroni_confidence"], .99375)
        self.assertEqual(effect["delta"], .15)
        self.assertEqual(effect["ci95"], [-.1, .43092105263157887])
        self.assertEqual(effect["ci_bonferroni"], [-.17948717948717968, .5529824561403498])
        self.assertNotEqual(effect["ci95"], effect["ci_bonferroni"])
        self.assertLess(effect["ci_bonferroni"][0], effect["ci_bonferroni"][1])

    def test_primary_families_are_separate_from_local_intervals(self):
        old, new, zero, _ = fixture_banks()
        expected = statistical_cases(metrics)
        for family in (4, 8):
            result = metrics.instruction_analysis(new, old, 73, 257, interaction_family_size=family)
            self.assertEqual(result["bootstrap"]["family_size"], 3)
            primary = result["cross_configuration_primary_interaction"]
            self.assertEqual(numeric_report(primary), numeric_report(expected[f"instruction_primary{family}"]))
        result = metrics.instruction_reassigned_analysis(new, new, old, 73, 257, interaction_family_size=8)
        self.assertEqual(result["bootstrap"]["family_size"], 2)
        self.assertEqual(numeric_report(result["cross_configuration_primary_interaction"]),
                         numeric_report(expected["assignment_primary8"]))
        cells = {"R|R": new["T"], "R|H": old["T"], "H|R": old["S"], "H|H": new["S"]}
        result = metrics.property_analysis(cells, {"q0", "q6"}, 73, 257, interaction_family_size=4)
        self.assertEqual(numeric_report(result["all_shared"]["cross_configuration_primary_interaction"]),
                         numeric_report(expected["property_primary4"]))
        self.assertEqual(result["discordant_measurement_order"]["bootstrap"]["family_size"], 3)
        self.assertEqual(result["discordant_measurement_order"]["cross_configuration_primary_interaction"]
                         ["bootstrap"]["family_size"], 4)
        self.assertEqual(metrics.feature_analysis(zero, old, 73, 20)
                         ["feature_by_packet_vs_original"]["bootstrap"]["family_size"], 2)

    def test_alignment_source_identity_and_opposite_gold(self):
        fixture = alignment_fixture()
        cells, discordant = metrics.align_property_cells(*fixture, expected_counts={
            "radius": 4161, "height": 4157, "shared": 4131, "scenes": 70, "discordant": 1228})
        self.assertEqual(fingerprint({"cells": cells, "discordant": sorted(discordant)}), GOLDENS["alignment"])
        self.assertEqual(len(cells["R|R"]), 4131)
        self.assertEqual(len(discordant), 1228)
        self.assertTrue(all(row["correct"] == 1 for row in cells["H|H"].values()))
        with self.assertRaises(ValueError):
            metrics.align_property_cells(*fixture, expected_counts={"shared": 4130})
        fixture[3]["radius"]["pairs"]["duplicate"] = fixture[3]["radius"]["pairs"]["radius0"]
        with self.assertRaises(ValueError):
            metrics.align_property_cells(*fixture)

    def test_single_family_and_natural_candidate_nulls(self):
        old, new, _, _ = fixture_banks()
        for bank in (old, new):
            for rows in bank.values():
                for row in rows.values():
                    row["scene_id"] = "scene0001_00"
                    row["candidate_pair_compliant"] = None
        result = metrics.instruction_analysis(new, old, 73, 257)
        self.assertFalse(result["bootstrap"]["intervals_available"])
        self.assertIsNone(result["contrasts"]["packet_by_instruction"]["ci_bonferroni"])
        self.assertIsNone(result["conditions"]["new_T"]["candidate_pair_compliant"])

    def test_unpaired_invalid_and_mixed_null_inputs_fail(self):
        original = {"T": vectors(1), "A": vectors(0)}
        for fault in ("missing", "scene", "binary", "bool", "null_mix", "family", "weights", "seed"):
            bank, family, weights, seed = deepcopy(original), 2, {"T": 1, "A": -1}, 73
            if fault == "missing": del bank["A"]["q0"]
            elif fault == "scene": bank["A"]["q0"]["scene_id"] = "scene0999_00"
            elif fault == "binary": bank["A"]["q0"]["correct"] = 2
            elif fault == "bool": bank["A"]["q0"]["correct"] = True
            elif fault == "null_mix": bank["A"]["q0"]["candidate_pair_compliant"] = None
            elif fault == "family": family = 0
            elif fault == "weights": weights = {"T": 1, "A": 1}
            else: seed = True
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                metrics.paired_family(bank, {"effect": weights}, family, seed, 30)

    def test_property_analysis_does_not_silently_intersect_unpaired_cells(self):
        cells = {label: vectors(i) for i, label in enumerate(("R|R", "R|H", "H|R", "H|H"))}
        cells["R|H"]["extra"] = dict(cells["R|H"]["q0"])
        with self.assertRaises(ValueError):
            metrics.property_analysis(cells, {"q0"}, 73, 20)

    def test_cli_cwd_paths_compact_output_and_no_torch_import(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "settings").mkdir()
            old, new, _, _ = fixture_banks()
            for name, bank in (("old", old), ("new", new)):
                (directory / (name + ".json")).write_text(json.dumps({
                    "schema_version": "scalestitch_scored_rows/v1", "conditions": bank}), encoding="utf-8")
            config = {"seed": 73, "replicates": 257, "banks": {
                name: {"path": name + ".json", "conditions": {c: c for c in ("A", "T", "S")}}
                for name in ("old", "new")}, "analyses": {
                    "instruction": {"kind": "instruction", "new": "new", "old": "old", "interaction_family_size": 8},
                    "assignment": {"kind": "instruction_reassigned", "at": "new", "continuation": "new", "old": "old"},
                    "feature": {"kind": "feature", "new": "new", "old": "old"},
                    "between": {"kind": "between_features", "zero": "new", "permute": "old"},
                    "linear": {"kind": "linear", "bank": "new", "family_size": 2,
                               "contrasts": {"utility": {"T": 1, "A": -1}}}}}
            path = directory / "settings" / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-B", "-m", "experiments.scale_interface.diagnostic_metrics",
                       "--config", str(path), "--output", "output"]
            result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((directory / "output" / "diagnostics.json").read_text(encoding="utf-8"))
            self.assertTrue(report["no_raw_answers_exported"])
            self.assertEqual(set(report["analyses"]), set(config["analyses"]))
            primary = report["analyses"]["instruction"]["cross_configuration_primary_interaction"]
            self.assertEqual(fingerprint(numeric_report(primary)), GOLDENS["instruction_primary8"])
            manifest = json.loads((directory / "output" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["sources_sha256"]), 3)
            second = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            check = subprocess.run([sys.executable, "-B", "-c",
                "import sys; import experiments.scale_interface.diagnostic_metrics; assert 'torch' not in sys.modules"],
                cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)

    def test_duplicate_json_keys_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"seed":1,"seed":2}', encoding="utf-8")
            with self.assertRaises(ValueError):
                metrics.diagnose(path, Path(tmp) / "output")
            self.assertFalse((Path(tmp) / "output").exists())


if __name__ == "__main__":
    unittest.main()
