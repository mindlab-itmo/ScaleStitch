"""Text-only preparation must preserve values and use ordinary shared labels."""
import copy
import hashlib
import importlib.util
import unittest

from experiments.scale_interface.scene_measurements import build_scene_radius_packet
from experiments.scale_interface.packet_scope import focus_scene_packet


QUESTION = ("Between <OBJ001> and <OBJ002>, which object has the larger measured metric radius? "
            "Answer with exactly one object identifier and no other text.")
OBJECTS = [dict(visible_id=0, source_row=2), dict(visible_id=1, source_row=0), dict(visible_id=2, source_row=1)]
MODULE = "experiments.scale_interface.text_only_baseline"


class PlainTokenizer:
    chat_template = "fixed template"
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "USER: " + messages[0]["content"] + " ASSISTANT: "
    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(c) for c in text]}


class TextOnlyPreparationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec(MODULE), "text-only preparation is not implemented")
        from experiments.scale_interface import text_only_baseline
        return text_only_baseline

    def packet(self):
        return build_scene_radius_packet(assigned_rows=[2, 0, 1], scene_mask=[True] * 3,
            radius_m=[.1, .2, .3], radius_valid=[True] * 3, mode="swapped")

    def test_ordinary_labels_are_bijective_beyond_one_letter(self):
        labels = self.module().ordinary_label
        self.assertEqual([labels(i) for i in [0, 25, 26, 51, 52, 149]], ["A", "Z", "AA", "AZ", "BA", "ET"])
        self.assertEqual(len({labels(i) for i in range(150)}), 150)
        for value in [-1, True, "2"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                labels(value)

    def test_plain_packet_preserves_numeric_bytes_and_full_scene_donors(self):
        module = self.module()
        packet = self.packet()
        prepared = module.prepare_example(qid="q", scene_id="scene1_00", question_text=QUESTION,
            objects=OBJECTS, candidate_source_rows=[0, 1], packet=packet)
        self.assertNotIn("<OBJ", prepared["user_text"])
        self.assertIn("Object B = 30 cm", prepared["user_text"])
        self.assertIn("Object C = 10 cm", prepared["user_text"])
        self.assertIn("Between Object B and Object C", prepared["user_text"])
        self.assertEqual(prepared["candidate_labels"], ["B", "C"])
        self.assertEqual(prepared["label_to_source_row"], {"A": 2, "B": 0, "C": 1})
        self.assertEqual(prepared["source_packet_sha256"], hashlib.sha256(packet.text.encode()).hexdigest())

    def test_absent_example_has_no_invented_scene_description(self):
        prepared = self.module().prepare_example(qid="q", scene_id="scene1_00", question_text=QUESTION,
            objects=OBJECTS, candidate_source_rows=[0, 1], packet=None)
        self.assertNotIn("Measured", prepared["user_text"])
        self.assertNotIn("cm", prepared["user_text"])
        self.assertIsNone(prepared["source_packet_sha256"])

    def test_focused_packet_keeps_original_numeric_unit_bytes_and_external_donor(self):
        module = self.module()
        for unit in ("cm", "mm", "m"):
            with self.subTest(unit=unit):
                full = build_scene_radius_packet(assigned_rows=[2, 0, 1], scene_mask=[True] * 3,
                    radius_m=[.1, .2, .3], radius_valid=[True] * 3, mode="swapped", unit=unit)
                packet = focus_scene_packet(full, candidate_source_rows=[0, 1])
                original = copy.deepcopy(packet.metadata())
                result = module.prepare_example(qid="q", scene_id="s", question_text=QUESTION,
                    objects=OBJECTS, candidate_source_rows=[0, 1], packet=packet)
                expected = packet.text.replace("<OBJ001>", "Object B").replace("<OBJ002>", "Object C")
                self.assertEqual(result["packet_text"].encode(), expected.encode())
                self.assertEqual(result["packet_metadata"], original)
                self.assertIn(2, [r["value_source_row"] for r in result["packet_metadata"]["records"]])
                self.assertEqual(packet.metadata(), original)

    def test_unknown_or_extra_query_ids_and_duplicate_mapping_are_rejected(self):
        module = self.module()
        for text, objects in [(QUESTION.replace("<OBJ002>", "<OBJ999>"), OBJECTS),
                              ("<OBJ000> " + QUESTION, OBJECTS),
                              (QUESTION, OBJECTS + [OBJECTS[0]])]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                module.prepare_example(qid="q", scene_id="s", question_text=text, objects=objects,
                                       candidate_source_rows=[0, 1], packet=self.packet())

    def test_base_and_adapted_receive_identical_rendered_text_and_token_ids(self):
        module = self.module()
        result = module.shared_prefix(PlainTokenizer(), PlainTokenizer(), "Object B = 30 cm")
        self.assertEqual(result["input_ids"], [ord(c) for c in result["rendered_text"]])
        self.assertIn("ASSISTANT", result["rendered_text"])
        self.assertNotIn("<OBJ", result["rendered_text"])

    def test_tokenizer_disagreement_blocks_a_fair_comparison(self):
        class ChangedTokenizer(PlainTokenizer):
            def __call__(self, text, *, add_special_tokens):
                return {"input_ids": [1]}
        with self.assertRaisesRegex(ValueError, "token"):
            self.module().shared_prefix(PlainTokenizer(), ChangedTokenizer(), "Object A")

    def test_chat_rendering_disagreement_blocks_even_with_matching_token_ids(self):
        class ChangedTemplate(PlainTokenizer):
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                return super().apply_chat_template(messages, tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt) + " "
        with self.assertRaisesRegex(ValueError, "text|render|byte"):
            self.module().shared_prefix(PlainTokenizer(), ChangedTemplate(), "Object A")

    def test_packet_mapping_mismatch_is_rejected_without_inventing_a_binding(self):
        objects = [dict(visible_id=0, source_row=0), dict(visible_id=1, source_row=1),
                   dict(visible_id=2, source_row=2)]
        with self.assertRaises(ValueError):
            self.module().prepare_example(qid="q", scene_id="s", question_text=QUESTION,
                objects=objects, candidate_source_rows=[1, 2], packet=self.packet())

    def test_duplicate_source_mapping_and_native_ids_in_shared_text_are_rejected(self):
        module = self.module()
        with self.assertRaises(ValueError):
            module.prepare_example(qid="q", scene_id="s", question_text=QUESTION,
                objects=[dict(visible_id=0, source_row=0), dict(visible_id=1, source_row=0)],
                candidate_source_rows=[0, 1], packet=None)
        with self.assertRaises(ValueError):
            module.shared_prefix(PlainTokenizer(), PlainTokenizer(), "<OBJ001>")

    def test_label_scoring_does_not_extract_letters_from_prose_or_drop_native_ids(self):
        score = self.module().score_output
        options = dict(label_to_source_row={"A": 2, "B": 0, "C": 1}, candidate_labels=["B", "C"], target_source_row=1)
        self.assertTrue(score("C.", **options)["correct"])
        self.assertFalse(score("B", **options)["correct"])
        self.assertFalse(score("A", **options)["candidate_compliant"])
        self.assertTrue(score("A", **options)["visible_label"])
        for text in ["The C", "C because", "C B", "<OBJ002>", "c"]:
            with self.subTest(text=text):
                self.assertFalse(score(text, **options)["format_valid"])
        self.assertFalse(score("ZZZ", **options)["visible_label"])


if __name__ == "__main__":
    unittest.main()
