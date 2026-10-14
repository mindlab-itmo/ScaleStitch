"""CPU checks for the CLI's binding of height declarations to attribute files."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface import __main__ as cli
from scale_interface import natural_qa as qa


class HeightCLI(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.attributes = self.root / "attributes.pt"
        self.attributes.write_bytes(b"fixed attribute bytes")
        self.source_path = self.root / "height_source.json"
        self.source = {
            "measurement": "full_object_axis_aligned_height",
            "input_id": hashlib.sha256(self.attributes.read_bytes()).hexdigest(),
            "source_reference": "geometry-audit.json",
            "verified_scenes": {"scene0001_00": True},
        }
        self.config = {"data": {"attributes": str(self.attributes)}}
        self.options = {"conditions": [{"representation": "absent"}, {"measurement": "height"}],
                        "measurement_source": str(self.source_path)}

    def save_source(self):
        self.source_path.write_text(json.dumps(self.source), encoding="utf-8")

    def test_verified_source_is_forwarded_and_recorded_for_mixed_conditions(self):
        self.save_source()
        prepared = []

        def prepare(core, data, conditions, **options):
            prepared.append(options)
            return SimpleNamespace(
                conditions=conditions, rows=[SimpleNamespace(qid="q1")],
                metadata=lambda: {"measurement_source": options["measurement_source"]})

        def generate(core, data, plan, **selected):
            for name in selected["condition_names"]:
                yield {"qid": "q1", "condition": name, "prediction": "answer"}

        with mock.patch.object(qa, "preflight_natural_qa", side_effect=prepare), \
             mock.patch.object(qa, "iter_natural_qa", side_effect=generate):
            result = cli._generate_scene(object(), object(), self.config, self.root, self.options)
        self.assertEqual(prepared[0]["measurement_source"], self.source)
        self.assertEqual(result["completed_predictions"], 2)
        saved = json.loads((self.root / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["measurement_source"], self.source)
        self.assertEqual(self.options["measurement_source"], str(self.source_path))

    def assert_rejected_before_preflight(self, message):
        with mock.patch.object(qa, "preflight_natural_qa") as prepare, \
             mock.patch.object(qa, "iter_natural_qa") as generate, \
             self.assertRaisesRegex(ValueError, message):
            cli._generate_scene(object(), object(), self.config, self.root, self.options)
        prepare.assert_not_called()
        generate.assert_not_called()
        self.assertFalse((self.root / "predictions.jsonl").exists())

    def test_changed_attributes_refuse_generation(self):
        self.save_source()
        self.attributes.write_bytes(b"different attributes")
        self.assert_rejected_before_preflight("height declaration differs")

    def test_malformed_source_refuses_generation(self):
        self.source["verified_scenes"] = {"scene0001_00": "true"}
        self.save_source()
        self.assert_rejected_before_preflight("actual boolean flags")

    def test_source_option_requires_a_file_path(self):
        for value in (None, {}, ""):
            with self.subTest(value=value):
                self.options["measurement_source"] = value
                self.assert_rejected_before_preflight("JSON file path")


if __name__ == "__main__":
    unittest.main()
