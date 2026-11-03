"""Input-defined populations and all-answer paired interface comparisons."""
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments.scale_interface.packet_scope import focus_scene_packet
from experiments.scale_interface.scene_measurements import build_scene_radius_packet


class InterfaceControlMetricsTests(unittest.TestCase):
    def module(self):
        name = "experiments.scale_interface.interface_control_metrics"
        self.assertIsNotNone(importlib.util.find_spec(name), "public interface analysis is missing")
        from experiments.scale_interface import interface_control_metrics
        return interface_control_metrics

    def metadata(self, index=0):
        core = self.module()
        pair, values, target = [((0, 1), [.1, .2, .3], 1), ((1, 2), [.1, .2, .3], 2),
                                ((0, 2), [.1, .2, .2], 2)][index]
        full, focused = {}, {}
        for mode, symbol in (("true", "T"), ("constant", "C"), ("swapped", "S")):
            packet = build_scene_radius_packet(assigned_rows=[2, 0, 1], scene_mask=[True] * 3,
                radius_m=values, radius_valid=[True] * 3, mode=mode)
            subset = focus_scene_packet(packet, candidate_source_rows=pair)
            normalize = lambda rows: [{**{k: v for k, v in row.items() if k != "radius_cm"},
                "value_cm": row["radius_cm"]} for row in rows]
            full[symbol] = dict(records=normalize(packet.metadata()["records"]),
                text_sha256=hashlib.sha256(packet.text.encode()).hexdigest())
            focused[symbol] = dict(records=normalize(subset.metadata()["records"]),
                full_packet_sha256=subset.metadata()["full_packet_sha256"])
        scene = ("scene0001_00", "scene0001_01", "scene0002_00")[index]
        question = dict(qid=str(index), scene_id=scene, assigned_ids=[2, 0, 1],
            prompt=f"Which <OBJ{pair[0]:03d}> or <OBJ{pair[1]:03d}> is larger?")
        reference = dict(question_id=str(index), scene_id=scene, answers=[f"<OBJ{target:03d}>"])
        return core.prepare_metadata(question, reference, full, focused)

    def bank(self, metadata):
        core = self.module()
        bank = {cell: [] for cell in core.CELLS}
        for cell in core.CELLS:
            restricted = cell.startswith(("ar.", "pr."))
            for item in metadata:
                candidates = item["candidate_source_rows"]
                source = item["target_source_row"]
                if cell == "af.T" and item["qid"] == "0":
                    source = 2
                tie = restricted and cell == "ar.C" and item["qid"] == "0"
                if tie:
                    source = candidates[0]
                inverse = {obj["source_row"]: obj["visible_id"] for obj in item["objects"]}
                row = dict(qid=item["qid"], scene_id=item["scene_id"], condition=cell,
                    raw_prediction=f"<OBJ{inverse[source]:03d}>" + ("" if restricted else "</s>"),
                    prediction=f"<OBJ{source:03d}>", invalid_object_id=False)
                if restricted:
                    logits = [0., 0.] if tie else [float(candidate == source) for candidate in candidates]
                    total = sum(math.exp(value) for value in logits)
                    row["candidate_decision"] = dict(candidate_identifiers=[f"<OBJ{inverse[x]:03d}>" for x in candidates],
                        candidate_token_ids=[128256 + inverse[x] for x in candidates], candidate_logits=logits,
                        candidate_conditional_probabilities=[math.exp(value) / total for value in logits],
                        pair_probability=.01, unrestricted_first_token_id=128256 + inverse[source],
                        tie=tie, score_space="raw model logits before generation processors",
                        decision_rule="maximum first-token logit; exact ties use question order")
                bank[cell].append(row)
        return bank

    def test_full_scene_donors_define_flips_before_answers_and_focus(self):
        core = self.module()
        metadata = [self.metadata(i) for i in range(3)]
        parsed, counts = core.validate_metadata(metadata, expected_count=3,
            expected_rankings=dict(flipped=1, preserved=1, tied=1))
        self.assertEqual(counts, dict(flipped=1, preserved=1, tied=1))
        self.assertEqual(parsed["0"]["s_ranking"], "flipped")
        focused = metadata[0]["focused_packets"]["S"]
        self.assertEqual([row["value_source_row"] for row in focused], [2, 0])
        self.assertEqual([row["value_cm"] for row in focused], [30, 10])
        bad = copy.deepcopy(metadata)
        bad[0]["focused_packets"]["S"][0]["value_source_row"] = 1
        with self.assertRaisesRegex(ValueError, "donor|focus|subset"):
            core.validate_metadata(bad, expected_count=3, expected_rankings=counts)
        with self.assertRaisesRegex(ValueError, "ranking"):
            core.validate_metadata(metadata, expected_count=3, expected_rankings=dict(flipped=0, preserved=2, tied=1))
        bad = copy.deepcopy(metadata)
        bad[0]["packets"]["C"][0].pop("value_source_row")
        with self.assertRaisesRegex(ValueError, "donor"):
            core.validate_metadata(bad, expected_count=3, expected_rankings=counts)

    def test_native_mapping_probabilities_and_question_order_ties_are_checked(self):
        core = self.module()
        metadata = [self.metadata(i) for i in range(3)]
        parsed, _ = core.validate_metadata(metadata, expected_count=3,
            expected_rankings=dict(flipped=1, preserved=1, tied=1))
        bank = self.bank(metadata)
        scores, _ = core.score_bank(bank, parsed, end_sym="</s>", native_token_start=128256, native_token_count=150)
        self.assertEqual(scores["0"]["ar.C"]["outcome"], "wrong_candidate")
        self.assertTrue(scores["0"]["ar.C"]["probability"]["tie"])
        for kind in ("candidate_order", "tokens", "float_tokens", "probabilities", "nonfinite", "tie", "mapped"):
            bad = copy.deepcopy(bank)
            row = bad["ar.T"][0]
            if kind == "candidate_order": row["candidate_decision"]["candidate_identifiers"].reverse()
            if kind == "tokens": row["candidate_decision"]["candidate_token_ids"][0] += 1
            if kind == "float_tokens": row["candidate_decision"]["candidate_token_ids"] = list(map(float, row["candidate_decision"]["candidate_token_ids"]))
            if kind == "probabilities": row["candidate_decision"]["candidate_conditional_probabilities"] = [.5, .6]
            if kind == "nonfinite": row["candidate_decision"]["pair_probability"] = float("nan")
            if kind == "tie": row["candidate_decision"]["tie"] = True
            if kind == "mapped": row["prediction"] = "<OBJ000>"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                core.score_bank(bad, parsed, end_sym="</s>", native_token_start=128256, native_token_count=150)

    def test_missing_duplicate_and_invalid_free_answers_never_shrink_the_cohort(self):
        core = self.module()
        metadata = [self.metadata(i) for i in range(3)]
        parsed, counts = core.validate_metadata(metadata, expected_count=3,
            expected_rankings=dict(flipped=1, preserved=1, tied=1))
        bank = self.bank(metadata)
        bank["af.C"][0].update(raw_prediction="unparseable prose", prediction="unparseable prose")
        scores, _ = core.score_bank(bank, parsed, end_sym="</s>", native_token_start=128256, native_token_count=150)
        self.assertEqual(scores["0"]["af.C"]["outcome"], "invalid")
        self.assertEqual(len(scores), 3)
        for bad in (dict(bank, **{"pf.T": bank["pf.T"][:-1]}),
                    dict(bank, **{"pf.T": bank["pf.T"][:-1] + [bank["pf.T"][0]]})):
            with self.assertRaisesRegex(ValueError, "matrix|cohort|duplicate"):
                core.score_bank(bad, parsed, end_sym="</s>", native_token_start=128256, native_token_count=150)

    def test_cli_runs_global_m8_and_all_secondary_families_on_all_fourteen_cells(self):
        core = self.module()
        metadata = [self.metadata(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, value in (("metadata.json", metadata), ("bank.json", self.bank(metadata))):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            config = dict(metadata={p: "metadata.json" for p in ("radius", "height")},
                banks={p: "bank.json" for p in ("radius", "height")}, expected_counts=dict(radius=3, height=3),
                expected_s_ranking_counts={p: dict(flipped=1, preserved=1, tied=1) for p in ("radius", "height")},
                end_sym="</s>", native_token_start=128256, native_token_count=150)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                core.main(["--config", str(root / "config.json"), "--output", str(root / "result.json")])
            report = json.loads((root / "result.json").read_text())
            self.assertEqual(report["primary_global_family_size"], 8)
            self.assertEqual(report["outputs"], 84)
            for prop in report["properties"].values():
                self.assertEqual(len(prop["conditions"]), 14)
                self.assertTrue(all(c["questions"] == 3 for c in prop["conditions"].values()))
                self.assertEqual(set(prop["conditions"]["ar.T"]["question_position_counts"]), {"first", "second", "outside", "invalid"})
                self.assertEqual(len(prop["primary"]), 4)
                for name, result in prop["primary"].items():
                    self.assertEqual(result["family_size"], 8)
                    self.assertEqual(result["replicates"], 100000)
                    self.assertEqual(result["seed"], 20260911)
                    self.assertEqual(result["bonferroni_confidence"], 1 - .05 / 8)
                p4 = prop["primary"]["reversed_ranking_assignment_pair_restricted"]
                self.assertEqual(p4["questions"], 1)
                self.assertIsNone(p4["ci_bonferroni"])
                self.assertEqual(p4["unavailable_reason"], "fewer_than_two_scene_families")
                self.assertEqual(len(prop["secondary"]["readout"]["contrasts"]), 9)
                self.assertEqual(prop["secondary"]["readout"]["family_size"], 9)
                transitions = prop["transitions"]["all.T"]
                self.assertEqual(sum(sum(row.values()) for row in transitions["counts"].values()), 3)
                self.assertEqual(transitions["counts"]["out_of_pair"]["correct"], 1)
                self.assertEqual(prop["conditions"]["ar.C"]["probability"]["exact_ties"], 1)

    def test_default_is_inert_and_empty_p4_is_unavailable_without_replacement(self):
        core = self.module()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(core.main([]), 0)
        result = core.primary_intervals([dict(qid="q", scene_id="scene0001_00", scores={cell: 0 for cell in core.CELLS})],
            flipped_qids=set())
        p4 = result["reversed_ranking_assignment_pair_restricted"]
        self.assertEqual(p4["questions"], 0)
        self.assertIsNone(p4["delta"])
        self.assertIsNone(p4["ci_bonferroni"])
        self.assertEqual(p4["unavailable_reason"], "empty_input_defined_population")

    def test_cli_validates_both_input_populations_before_opening_answer_banks(self):
        core = self.module()
        metadata = [self.metadata(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            config = dict(metadata={p: "metadata.json" for p in ("radius", "height")},
                banks={p: "not_opened.json" for p in ("radius", "height")},
                expected_counts=dict(radius=3, height=3),
                expected_s_ranking_counts={p: dict(flipped=0, preserved=2, tied=1) for p in ("radius", "height")},
                end_sym="</s>", native_token_start=128256, native_token_count=150)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ranking"):
                core.main(["--config", str(root / "config.json"), "--output", str(root / "result.json")])
            self.assertFalse((root / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
