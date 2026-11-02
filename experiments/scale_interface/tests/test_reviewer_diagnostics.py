"""Hand-checked reviewer diagnostics, using the frozen raw/mapped prediction schema."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.scale_interface.reviewer_diagnostics import analyze_predictions


MODULE = "experiments.scale_interface.reviewer_diagnostics"


def question(qid="q1"):
    objects = [dict(visible_id=v, source_row=s) for v, s in enumerate([2, 0, 3, 1])]
    # Packet order differs from both visible-ID order and source-row order.
    def packet(values):
        return [dict(visible_id=v, source_row=s, value_cm=values[s])
                for v, s in [(2, 3), (1, 0), (0, 2), (3, 1)]]
    return dict(qid=qid, scene_id="scene0001_00", scene_family="scene0001",
                objects=objects, candidate_source_rows=[0, 1], target_source_row=1,
                direction="max", packets={"A": None,
                    "T": packet({0: "10", 1: "30", 2: "50.0", 3: "50.00"}),
                    "S": packet({0: "40", 1: "20", 2: "60", 3: "20"}),
                    "C": packet({0: "100", 1: "100", 2: "100", 3: "100"})})


def prediction(qid, condition, source=None, *, raw=None, mapped=None, invalid=False):
    if source is not None:
        visible = [2, 0, 3, 1].index(source)
        raw, mapped = f"<OBJ{visible:03d}></s>", f"<OBJ{source:03d}>"
    return dict(qid=qid, scene_id="scene0001_00", condition=condition,
                raw_prediction=raw, prediction=mapped, invalid_object_id=invalid,
                output_token_ids=[], packet_tokens=0, prefix_tokens=20,
                native_tokens=20, numerical_blocks=0, elapsed_seconds=0.1)


def fixture():
    metadata = [question(f"q{i}") for i in range(1, 5)]
    rows = []
    for qid, choices in [("q1", [1, 2, 0, 0]), ("q2", [0, 1, 3, 1]),
                         ("q3", [None, 0, 1, None]), ("q4", [2, None, None, None])]:
        for condition, source in zip(["A", "T", "S", "C"], choices):
            rows.append(prediction(qid, condition, source) if source is not None
                        else prediction(qid, condition, raw="chair", mapped="chair"))
    rows[11] = prediction("q3", "C", raw="<OBJ999></s>", mapped="<OBJ999>")
    rows[13] = prediction("q4", "T", raw="<OBJ003> <OBJ001>", mapped="<OBJ001> <OBJ000>")
    rows[14] = prediction("q4", "S", 1, invalid=True)
    return rows, metadata


class ReviewerDiagnosticsTests(unittest.TestCase):
    def analyze(self, rows=None, metadata=None, **kwargs):
        default_rows, default_metadata = fixture()
        options = dict(configuration="ulip2_full", conditions=["A", "T", "S", "C"],
                       reference_condition="T", end_sym="</s>")
        options.update(kwargs)
        return analyze_predictions(default_rows if rows is None else rows,
                                   default_metadata if metadata is None else metadata, **options)

    def test_full_denominators_include_invalid_and_out_of_pair_answers(self):
        result = self.analyze()
        treatment = result["conditions"]["T"]
        self.assertEqual(treatment["questions"], 4)
        self.assertEqual(treatment["scenes"], 1)
        self.assertEqual(treatment["outcomes"], dict(correct=1, wrong_candidate=1,
                                                    out_of_pair=1, invalid=1))
        self.assertEqual(treatment["accuracy"], dict(count=1, denominator=4, fraction=.25))
        self.assertEqual(treatment["visible_id"], dict(count=3, denominator=4, fraction=.75))
        self.assertEqual(treatment["candidate_compliance"], dict(count=2, denominator=4, fraction=.5))
        self.assertEqual(treatment["in_pair_accuracy_descriptive"]["fraction"], .5)
        self.assertIn("descriptive", result["interpretation"])

    def test_transitions_pair_question_identity_instead_of_row_order(self):
        rows, metadata = fixture()
        result = self.analyze(list(reversed(rows)), metadata)
        transition = result["transitions"]["A->T"]
        self.assertEqual(transition["questions"], 4)
        self.assertEqual(transition["outcome_counts"], {
            "correct->out_of_pair": 1, "wrong_candidate->correct": 1,
            "invalid->wrong_candidate": 1, "out_of_pair->invalid": 1})

    def test_out_of_pair_uses_packet_order_mapping_and_all_tied_extrema(self):
        treatment = self.analyze()["conditions"]["T"]["out_of_pair_diagnostics"]
        self.assertEqual(treatment["eligible_questions"], 4)
        self.assertEqual(treatment["out_of_pair_answers"], 1)
        # The chosen source row 2 is visible ID 0, packet position 2, tied maximum.
        for name in ["maximum", "direction_extreme", "adjacent_to_candidate"]:
            self.assertEqual(treatment[name]["count"], 1)
            self.assertEqual(treatment[name]["fraction_all"], .25)
            self.assertEqual(treatment[name]["fraction_out_of_pair"], 1.)
        self.assertEqual(treatment["maximum"]["random_distractor_rate"], 1.)
        self.assertEqual(treatment["minimum"]["count"], 0)
        self.assertEqual(treatment["first_row"]["count"], 0)
        self.assertEqual(treatment["last_row"]["count"], 0)
        self.assertEqual(treatment["value_spread_cm"], dict(count=4, minimum=40., maximum=40., mean=40.))

    def test_random_distractor_reference_matches_the_selected_out_of_pair_rows(self):
        metadata = [question("q1"), question("q2")]
        metadata[1].update(scene_id="scene0002_00", scene_family="scene0002")
        for meta in metadata:
            meta["packets"] = {condition: meta["packets"][condition] for condition in ["A", "T"]}
        for row in metadata[1]["packets"]["T"]:
            if row["source_row"] == 2:
                row["value_cm"] = "20"  # Only one of two distractors is maximal in q2.
        rows = [prediction(qid, condition, 2 if qid == "q1" else 1)
                for qid in ["q1", "q2"] for condition in ["A", "T"]]
        for row in rows:
            if row["qid"] == "q2":
                row["scene_id"] = "scene0002_00"
        maximum = self.analyze(rows, metadata, conditions=["A", "T"])["conditions"]["T"]["out_of_pair_diagnostics"]["maximum"]
        # Only q1 is out-of-pair. Its two distractors are both maximal: p=1,
        # whereas the unmatched all-question average would incorrectly be .75.
        self.assertEqual(maximum["random_distractor_rate"], 1.)
        self.assertEqual(maximum["random_distractor_questions"], 1)
        self.assertEqual(maximum["expected_random_count_on_outside"], 1.)
        self.assertEqual(maximum["expected_random_fraction_all"], .5)

    def test_by_family_counts_preserve_paired_sufficient_statistics(self):
        rows, metadata = fixture()
        for meta in metadata[2:]:
            meta.update(scene_id="scene0002_00", scene_family="scene0002")
        for row in rows:
            if row["qid"] in ["q3", "q4"]:
                row["scene_id"] = "scene0002_00"
        result = self.analyze(rows, metadata)
        families = result["by_scene_family"]
        self.assertEqual(set(families), {"scene0001", "scene0002"})
        self.assertEqual(families["scene0001"]["conditions"]["T"]["accuracy"],
                         dict(count=1, denominator=2, fraction=.5))
        self.assertEqual(families["scene0002"]["conditions"]["T"]["accuracy"],
                         dict(count=0, denominator=2, fraction=0.))
        self.assertEqual(families["scene0001"]["transitions"]["A->T"]["outcome_counts"],
                         {"correct->out_of_pair": 1, "wrong_candidate->correct": 1})

    def test_packet_agreement_distinguishes_ground_truth_and_constant_ties(self):
        result = self.analyze()
        swapped = result["conditions"]["S"]["packet_following"]
        self.assertEqual(swapped["ranking_counts"], dict(preserved=0, flipped=4, tied=0,
                                                        reference_tied=0, unavailable=0))
        self.assertEqual(swapped["agreement_unique"], dict(count=1, denominator=4, fraction=.25))
        self.assertEqual(swapped["agreement_flipped"], dict(count=1, denominator=4, fraction=.25))
        constant = result["conditions"]["C"]["packet_following"]
        self.assertEqual(constant["ranking_counts"]["tied"], 4)
        self.assertEqual(constant["agreement_unique"], dict(count=0, denominator=0, fraction=None))
        self.assertEqual(result["conditions"]["A"]["packet_following"]["ranking_counts"]["unavailable"], 4)

    def test_nonvisible_but_well_formed_id_is_invalid_not_visible(self):
        constant = self.analyze()["conditions"]["C"]
        self.assertEqual(constant["native_id_syntax"]["count"], 3)
        self.assertEqual(constant["visible_id"]["count"], 2)
        self.assertEqual(constant["outcomes"]["invalid"], 2)

    def test_packet_agreement_exposes_full_cohort_and_unique_winner_denominators(self):
        metadata = [question("q1"), question("q2")]
        for row in metadata[0]["packets"]["S"]:
            if row["source_row"] == 0:
                row["value_cm"] = "20"  # q1 candidates tie; q2 remains flipped.
        rows = [prediction(qid, condition, 0) for qid in ["q1", "q2"]
                for condition in ["A", "T", "S", "C"]]
        result = self.analyze(rows, metadata)
        swapped = result["conditions"]["S"]["packet_following"]
        self.assertEqual(swapped["agreement_all"], dict(count=1, denominator=2, fraction=.5))
        self.assertEqual(swapped["agreement_unique"], dict(count=1, denominator=1, fraction=1.))
        self.assertEqual(swapped["unique_winner_questions"], 1)
        self.assertEqual(result["conditions"]["C"]["packet_following"]["agreement_all"],
                         dict(count=0, denominator=2, fraction=None))

    def test_minimum_direction_and_preserved_or_tied_serialized_rankings(self):
        meta = question()
        meta.update(direction="min", target_source_row=0)
        meta["packets"]["S"] = copy.deepcopy(meta["packets"]["T"])
        rows = [prediction("q1", c, 0) for c in ["A", "T", "S", "C"]]
        result = self.analyze(rows, [meta])
        self.assertEqual(result["conditions"]["S"]["packet_following"]["ranking_counts"]["preserved"], 1)
        self.assertEqual(result["conditions"]["T"]["accuracy"]["count"], 1)
        # Equality is based on serialized decimals, not source float values.
        for row in meta["packets"]["T"]:
            if row["source_row"] == 0:
                row["value_cm"] = "30.000"
        result = self.analyze(rows, [meta])
        self.assertEqual(result["conditions"]["S"]["packet_following"]["ranking_counts"]["reference_tied"], 1)

    def test_rejects_raw_mapped_id_disagreement(self):
        rows, metadata = fixture()
        rows[1]["prediction"] = "<OBJ003>"
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.analyze(rows, metadata)

    def test_rejects_target_disagreement_with_unique_correct_packet_winner(self):
        rows, metadata = fixture()
        metadata[0]["target_source_row"] = 0
        with self.assertRaisesRegex(ValueError, "target"):
            self.analyze(rows, metadata)

    def test_rejects_inconsistent_scene_family_labels(self):
        rows, metadata = fixture()
        metadata[0]["scene_family"] = "different_family"
        with self.assertRaisesRegex(ValueError, "family"):
            self.analyze(rows, metadata)

    def test_rejects_missing_duplicate_and_extra_condition_question_rows(self):
        rows, metadata = fixture()
        for changed in [rows[:-1], rows + [rows[0]], rows + [dict(rows[0], condition="other")],
                        rows + [dict(rows[0], qid="unlisted")]]:
            with self.subTest(rows=len(changed)), self.assertRaises(ValueError):
                self.analyze(changed, metadata)

    def test_requires_explicit_mapping_candidates_packets_and_exact_values(self):
        for field in ["objects", "candidate_source_rows", "target_source_row", "direction", "packets"]:
            rows, metadata = fixture()
            del metadata[0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "metadata"):
                self.analyze(rows, metadata)
        for value in [float("nan"), "NaN", -2, True, "not a value"]:
            rows, metadata = fixture()
            metadata[0]["packets"]["T"][0]["value_cm"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.analyze(rows, metadata)

    def test_rejects_packet_mapping_mismatch_duplicate_rows_and_missing_candidates(self):
        for kind in ["mapping", "duplicate", "candidate", "scene"]:
            rows, metadata = fixture()
            if kind == "mapping":
                metadata[0]["packets"]["T"][0]["source_row"] = 2
            elif kind == "duplicate":
                metadata[0]["packets"]["T"].append(metadata[0]["packets"]["T"][0])
            elif kind == "candidate":
                metadata[0]["packets"]["T"] = metadata[0]["packets"]["T"][:1]
            else:
                rows[0]["scene_id"] = "scene0002_00"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.analyze(rows, metadata)

    def test_absent_packet_and_two_candidate_scope_have_no_invented_distractors(self):
        meta = question()
        for condition in ["T", "S", "C"]:
            meta["packets"][condition] = [row for row in meta["packets"][condition]
                                           if row["source_row"] in [0, 1]]
        rows = [prediction("q1", c, 2) for c in ["A", "T", "S", "C"]]
        result = self.analyze(rows, [meta])
        diagnostic = result["conditions"]["T"]["out_of_pair_diagnostics"]
        self.assertEqual(diagnostic["out_of_pair_answers"], 1)
        self.assertEqual(diagnostic["selected_not_in_packet"], 1)
        self.assertIsNone(diagnostic["maximum"]["random_distractor_rate"])
        self.assertEqual(result["conditions"]["A"]["out_of_pair_diagnostics"]["eligible_questions"], 0)

    def test_input_rows_are_not_modified(self):
        rows, metadata = fixture()
        before = copy.deepcopy((rows, metadata))
        self.analyze(rows, metadata)
        self.assertEqual((rows, metadata), before)

    def test_cli_reads_original_prediction_rows_and_explicit_metadata(self):
        self.analyze()
        rows, metadata = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "predictions.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            completed = subprocess.run([sys.executable, "-B", "-m", MODULE,
                "--predictions", str(path / "predictions.jsonl"),
                "--metadata", str(path / "metadata.json"), "--configuration", "ulip2_full",
                "--conditions", "A", "T", "S", "C", "--reference-condition", "T",
                "--end-sym", "</s>"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["conditions"]["T"]["accuracy"]["denominator"], 4)
        self.assertEqual(set(result["input_sha256"]), {"predictions", "metadata"})


if __name__ == "__main__":
    unittest.main()
