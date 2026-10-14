"""ScanQA export and lexical strata checks using only the Python standard library."""
from collections.abc import Mapping
import copy
import json
from pathlib import Path
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from scale_interface.scanqa_protocol import (
    build_questions, build_references, lexical_stratum,
    select_lexical_rows, summarize_questions,
)


class GuardedRecord(Mapping):
    """Raise even if a caller tries to inspect target-bearing fields."""
    def __init__(self, question_id=1, scene_id="scene0000_00", question="What color is the chair?"):
        self.fields = dict(question_id=question_id, scene_id=scene_id, question=question)
        self.reads = []

    def __getitem__(self, key):
        if key not in ("question_id", "scene_id", "question"):
            raise AssertionError("target-bearing or unrelated field was read: " + key)
        self.reads.append(key)
        return self.fields[key]

    def __iter__(self):
        raise AssertionError("copying/enumerating the source record is forbidden")

    def __len__(self):
        return 7


class ScanQAProtocolTests(unittest.TestCase):
    def records(self):
        return [
            {"question_id": "val-2", "scene_id": "scene0001_00", "question": "How tall is the table?",
             "answers": ["one meter", "a meter", "one meter"], "object_ids": [94], "object_names": ["table"]},
            {"question_id": 12, "scene_id": "scene0000_00", "question": "  What color is the chair?  ",
             "answers": ["SECRET"], "object_ids": [1000], "caption": "SECRET"},
        ]

    def test_questions_never_read_answers_or_target_objects(self):
        record = GuardedRecord()
        result = build_questions([record], scene_object_counts={"scene0000_00": 3})
        self.assertEqual(result[0]["assigned_ids"], [0, 1, 2])
        self.assertEqual(record.reads, ["question_id", "scene_id", "question"])
        self.assertEqual(set(result[0]), {"question_id", "qid", "scene_id", "question", "prompt", "assigned_ids"})

    def test_original_question_type_order_and_whitespace_preserved(self):
        records = self.records()
        before = copy.deepcopy(records)
        rows = build_questions(records, scene_object_counts={"scene0000_00": 2, "scene0001_00": 3})
        self.assertEqual([row["question_id"] for row in rows], ["val-2", 12])
        self.assertEqual(rows[1]["prompt"], "  What color is the chair?  ")
        self.assertEqual(rows[1]["question"], records[1]["question"])
        self.assertEqual(rows[1]["qid"], 12)
        self.assertNotIn("SECRET", json.dumps(rows))
        self.assertEqual(records, before)

    def test_whitelist_and_counts_are_only_scene_selection_inputs(self):
        rows = build_questions(self.records(), scene_object_counts={"scene0000_00": 2, "scene0001_00": 3},
                               scene_whitelist=["scene0000_00"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["assigned_ids"], [0, 1])
        self.assertEqual(rows[0]["question_id"], 12)

    def test_default_whitelist_is_count_keys_and_empty_selection_is_explicit(self):
        self.assertEqual(len(build_questions(self.records(), scene_object_counts={"scene0000_00": 1})), 1)
        self.assertEqual(build_questions(self.records(), scene_object_counts={}), [])
        self.assertEqual(build_questions(self.records(), scene_object_counts={"scene0000_00": 1},
                                         scene_whitelist=[]), [])

    def test_each_question_gets_an_independent_assignment(self):
        records = [dict(self.records()[1], question_id=i) for i in (1, 2)]
        rows = build_questions(records, scene_object_counts={"scene0000_00": 2})
        rows[0]["assigned_ids"][0] = 1
        self.assertEqual(rows[1]["assigned_ids"], [0, 1])

    def test_duplicate_source_ids_fail_globally_before_filter(self):
        for ids in ((1, 1), (1, "1")):
            records = self.records()
            for record, qid in zip(records, ids):
                record["question_id"] = qid
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "duplicate question_id"):
                build_questions(records, scene_object_counts={"scene0000_00": 1})

    def test_invalid_ids_questions_and_scenes_fail(self):
        for key, value in (("question_id", True), ("question_id", 1.5), ("question_id", " "),
                           ("scene_id", "scene0"), ("question", ""), ("question", 123)):
            records = self.records()
            records[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                build_questions(records, scene_object_counts={"scene0000_00": 1})

    def test_requires_original_scanqa_field_names(self):
        record = {"qid": 1, "scene_id": "scene0000_00", "prompt": "How tall?"}
        with self.assertRaises(KeyError):
            build_questions([record], scene_object_counts={"scene0000_00": 1})

    def test_bad_counts_and_whitelists_fail(self):
        for count in (0, -1, 2.0, True, "2"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                build_questions(self.records(), scene_object_counts={"scene0000_00": count})
        for scenes in (["scene0000_00", "scene0000_00"], ["scene0002_00"], "scene0000_00"):
            with self.subTest(scenes=scenes), self.assertRaises(ValueError):
                build_questions(self.records(), scene_object_counts={"scene0000_00": 2}, scene_whitelist=scenes)

    def test_references_are_separate_preserve_all_answers_and_requested_order(self):
        records = self.records()
        refs = build_references(records, question_ids=[12, "val-2"])
        self.assertEqual([row["question_id"] for row in refs], [12, "val-2"])
        self.assertEqual(refs[1]["answers"], ["one meter", "a meter", "one meter"])
        self.assertEqual(set(refs[1]), {"question_id", "scene_id", "answers"})
        refs[1]["answers"][0] = "changed"
        self.assertEqual(records[0]["answers"][0], "one meter")

    def test_references_do_not_read_unselected_answers(self):
        refs = build_references([GuardedRecord(question_id="skip"), self.records()[0]], question_ids=["val-2"])
        self.assertEqual(len(refs), 1)

    def test_references_reject_missing_repeated_or_malformed_targets(self):
        for ids in (["missing"], [12, 12], [12, "12"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build_references(self.records(), question_ids=ids)
        for answers in ([], "chair", [None], [" "], ["chair", 1]):
            records = self.records()
            records[0]["answers"] = answers
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                build_references(records, question_ids=["val-2"])

    def test_explicit_measurements_keep_distinct_physical_types(self):
        for question, metric in (("What is the radius of the object?", "radius"),
                                 ("What is the diameter of the cylinder?", "diameter"),
                                 ("How tall is the chair?", "height"),
                                 ("How wide is the table?", "width"),
                                 ("How long is the bed?", "length"),
                                 ("How deep is the shelf?", "depth"),
                                 ("What is the volume of the cabinet?", "volume"),
                                 ("How far is the table from the chair?", "distance"),
                                 ("Which object is bigger?", "size_unspecified")):
            with self.subTest(question=question):
                result = lexical_stratum(question)
                self.assertEqual(result["annotation_method"], "lexical_rules")
                self.assertEqual(result["lexical_need"], "required")
                self.assertIn(metric, result["metric_types"])
                if metric != "radius":
                    self.assertNotIn("radius", result["metric_types"])

    def test_comparisons_are_required_wording_not_radius_equivalence(self):
        for question in ("Which chair is tallest?", "Is the table wider than the shelf?",
                         "Which object is closer to the desk?"):
            with self.subTest(question=question):
                result = lexical_stratum(question)
                self.assertEqual(result["lexical_need"], "required")
                self.assertNotIn("radius", result["metric_types"])

    def test_bare_adjectives_polysemy_and_unknown_questions_are_ambiguous(self):
        questions = ("Where is the large chair?", "What is on the long table?",
                     "How high is the lamp?", "Where is the chair?", "What object is this?",
                     "How long has the chair been here?", "What is the volume of the music?",
                     "Is the taller chair red?", "How many chairs fit under the table?",
                     "Which object is next to the larger table?", "Which taller chair is red?")
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(lexical_stratum(question)["lexical_need"], "ambiguous")

    def test_not_required_is_narrow_and_metric_modifiers_override_it(self):
        for question in ("What color is the chair?", "What is the color of the chair?",
                         "What material is the desk made of?", "How many chairs are there?"):
            with self.subTest(question=question):
                self.assertEqual(lexical_stratum(question)["lexical_need"], "not_required")
        for question in ("What color is the taller chair?", "What material is the large table?",
                         "How many chairs are on the left?", "What color is the table nearest the door?"):
            with self.subTest(question=question):
                self.assertEqual(lexical_stratum(question)["lexical_need"], "ambiguous")

    def test_transparent_rules_cues_and_word_boundaries(self):
        result = lexical_stratum("Which object is larger?")
        self.assertEqual(result["rule"], "explicit_metric_comparison")
        self.assertIn({"metric_type": "size_unspecified", "text": "larger"}, result["cues"])
        self.assertEqual(lexical_stratum("What color is the highchair?")["metric_types"], [])

    def test_explicit_filter_and_compact_summary_use_question_only(self):
        records = [GuardedRecord(1, question="How tall is the table?"),
                   GuardedRecord(2), GuardedRecord(3, question="Where is the large chair?")]
        rows = build_questions(records, scene_object_counts={"scene0000_00": 3})
        selected = select_lexical_rows(rows, labels=["required"], metric_types=["height"])
        self.assertEqual([row["question_id"] for row in selected], [1])
        self.assertEqual(select_lexical_rows(rows), rows)
        self.assertEqual(select_lexical_rows(rows, metric_types=["radius"]), [])
        summary = summarize_questions(rows)
        self.assertEqual(summary["questions"], 3)
        self.assertEqual(summary["scenes"], 1)
        self.assertEqual(summary["lexical_need"], {"required": 1, "not_required": 1, "ambiguous": 1})
        self.assertNotIn("How tall", json.dumps(summary))
        self.assertEqual(summary["metric_types"]["height"], 1)
        self.assertEqual(summarize_questions([])["questions"], 0)

    def test_unknown_or_bare_string_filter_categories_fail(self):
        for kwargs in ({"labels": ["semantic"]}, {"labels": "required"},
                       {"metric_types": ["radii"]}, {"metric_types": "radius"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_lexical_rows([], **kwargs)


if __name__ == "__main__":
    unittest.main()
