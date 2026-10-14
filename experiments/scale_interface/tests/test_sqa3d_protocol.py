"""SQA3D input/export checks with guarded target-bearing fields."""
from collections.abc import Mapping
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface.sqa3d_protocol import build_questions, fixed_question_ids, build_references
from scale_interface.scanqa_protocol import lexical_stratum


class GuardedQuestion(Mapping):
    def __init__(self, **values):
        self.fields = dict(question_id=1, scene_id="scene0000_00", situation="  I am sitting.\n",
                           question=" What color is the chair? ")
        self.fields.update(values)
        self.reads = []

    def __getitem__(self, key):
        if key not in ("question_id", "scene_id", "situation", "question"):
            raise AssertionError("non-input field read: " + key)
        self.reads.append(key)
        return self.fields[key]

    def __iter__(self):
        raise AssertionError("source record must not be copied")

    def __len__(self):
        return 12


class SQA3DQuestions(unittest.TestCase):
    def build(self, records):
        return build_questions(records, scene_object_counts={"scene0000_00": 3})

    def test_primary_situation_question_and_whitespace_preserved_without_other_fields(self):
        record = GuardedQuestion()
        row = self.build([record])[0]
        self.assertEqual(record.reads, ["question_id", "scene_id", "situation", "question"])
        self.assertEqual(row["prompt"], "Situation:\n  I am sitting.\n\n\nQuestion:\n What color is the chair? ")
        self.assertEqual(row["assigned_ids"], [0, 1, 2])
        self.assertEqual(row["lexical"], lexical_stratum(row["question"]))
        self.assertEqual(set(row), {"question_id", "qid", "scene_id", "situation", "question", "prompt",
                                   "assigned_ids", "lexical"})

    def test_alternatives_and_ground_truth_never_enter_inputs(self):
        row = dict(GuardedQuestion().fields, alternative_situation=["SECRET alternative"],
                   answers=["SECRET answer"], position="SECRET position", rotation="SECRET rotation")
        before = copy.deepcopy(row)
        result = self.build([row])
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(row, before)

    def test_missing_primary_situation_does_not_fall_back_to_alternatives(self):
        row = dict(GuardedQuestion().fields, alternative_situation=["I stand by a desk."])
        del row["situation"]
        with self.assertRaises(KeyError):
            self.build([row])

    def test_strict_source_fields_and_global_identity_checks(self):
        changes = [("situation", ""), ("situation", ["variant"]), ("question", " "),
                   ("scene_id", "wrong"), ("question_id", True), ("question_id", 1.5)]
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.build([GuardedQuestion(**{key: value})])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.build([GuardedQuestion(), GuardedQuestion(question_id="1", scene_id="scene0001_00")])

    def test_complete_scene_rows_and_source_order_are_kept(self):
        records = [GuardedQuestion(question_id=2), GuardedQuestion(question_id=1),
                   GuardedQuestion(question_id=3, scene_id="scene0001_00")]
        rows = self.build(records)
        self.assertEqual([row["qid"] for row in rows], [2, 1])
        rows[0]["assigned_ids"][0] = 9
        self.assertEqual(rows[1]["assigned_ids"], [0, 1, 2])

    def test_fixed_subset_hash_order_is_independent_of_source_order(self):
        rows = [GuardedQuestion(question_id=i) for i in range(20)]
        selected = fixed_question_ids(rows, per_scene=8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(set(selected), set(fixed_question_ids(list(reversed(rows)), per_scene=8)))
        self.assertEqual(selected, [str(i) for i in range(20) if str(i) in set(selected)])

    def test_all_explicit_metric_questions_included_beyond_scene_limit(self):
        rows = [GuardedQuestion(question_id=i, question="How tall is the chair?") for i in range(10)]
        rows += [GuardedQuestion(question_id=11, question="What color is the chair?")]
        selected = fixed_question_ids(rows, per_scene=1)
        self.assertTrue({str(i) for i in range(10)} <= set(selected))

    def test_situation_metric_words_do_not_change_question_only_sampling(self):
        rows = [GuardedQuestion(question_id=i, situation="The biggest table is nearest me.") for i in range(20)]
        selected = fixed_question_ids(rows, per_scene=3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected, fixed_question_ids([GuardedQuestion(question_id=i) for i in range(20)], per_scene=3))

    def test_bad_subset_limits_and_duplicate_ids_fail(self):
        for value in (0, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fixed_question_ids([GuardedQuestion()], per_scene=value)
        with self.assertRaises(ValueError):
            fixed_question_ids([GuardedQuestion(), GuardedQuestion()])


class GuardedAnnotation(Mapping):
    def __init__(self, question_id=1, scene_id="scene0000_00", answers=None, allow_answers=True):
        self.fields = dict(question_id=question_id, scene_id=scene_id,
                           answers=[{"answer": " one ", "answer_confidence": "unused"}]
                           if answers is None else answers)
        self.allow_answers = allow_answers
        self.reads = []

    def __getitem__(self, key):
        if key not in ("question_id", "scene_id", "answers") or (key == "answers" and not self.allow_answers):
            raise AssertionError("unselected answer or unrelated annotation read: " + key)
        self.reads.append(key)
        return self.fields[key]

    def __iter__(self):
        raise AssertionError("annotation record must not be copied")

    def __len__(self):
        return 7


class SQA3DReferences(unittest.TestCase):
    def test_only_selected_answers_are_read_and_original_strings_are_kept(self):
        selected = GuardedAnnotation(answers=[{"answer": " one "}, {"answer": "1"}, {"answer": "1"}])
        other = GuardedAnnotation(question_id=2, allow_answers=False)
        result = build_references([other, selected], questions=[GuardedQuestion()])
        self.assertEqual(result, [{"question_id": 1, "scene_id": "scene0000_00", "answers": [" one ", "1", "1"]}])
        self.assertEqual(other.reads, ["question_id", "scene_id"])
        self.assertEqual(selected.reads, ["question_id", "scene_id", "answers"])

    def test_question_order_controls_reference_export(self):
        refs = build_references([GuardedAnnotation(2), GuardedAnnotation(1)],
                                questions=[GuardedQuestion(), GuardedQuestion(question_id=2)])
        self.assertEqual([row["question_id"] for row in refs], [1, 2])

    def test_missing_mismatched_or_duplicate_reference_identity_fails(self):
        cases = [[], [GuardedAnnotation(2)], [GuardedAnnotation(scene_id="scene0001_00")],
                 [GuardedAnnotation(), GuardedAnnotation(question_id="1")]]
        for records in cases:
            with self.subTest(records=records), self.assertRaises(ValueError):
                build_references(records, questions=[GuardedQuestion()])

    def test_noncanonical_answer_layout_and_empty_strings_fail(self):
        for answers in ([], ["one"], [{"answer": ""}], [{"answer": 1}], "one"):
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                build_references([GuardedAnnotation(answers=answers)], questions=[GuardedQuestion()])

    def test_answer_mappings_are_not_enumerated_or_augmented_with_pose(self):
        class Answer(Mapping):
            def __getitem__(self, key):
                if key != "answer":
                    raise AssertionError("answer metadata read")
                return "chair"
            def __iter__(self):
                raise AssertionError("answer mapping copied")
            def __len__(self):
                return 3
        result = build_references([GuardedAnnotation(answers=[Answer()])], questions=[GuardedQuestion()])
        self.assertEqual(result[0]["answers"], ["chair"])


if __name__ == "__main__":
    unittest.main()
