"""End-to-end exports: complete cohorts, answer separation and CPU-only CLI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.scale_interface.controlled_protocol import (
    CANDIDATE_INSTRUCTION, build_native_id_pairs,
)
from experiments.scale_interface.prepare_controlled import prepare


SCENE = "scene0001_00"
REPOSITORY = Path(__file__).resolve().parents[3]


def geometry(count=6):
    return {SCENE: [
        dict(source_id=10 + 2 * i, category="category " + str(i),
             rho=(i + 1) / 10, radius_valid=True,
             height_m=(i + 1) / 5, height_valid=True)
        for i in range(count)
    ]}


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class PrepareControlledTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.objects = self.base / "scene_objects.json"
        write_json(self.objects, geometry())
        self.config = self.base / "config.json"
        self.output = self.base / "prepared"

    def configure(self, **updates):
        config = {"property": "radius", "scene_objects": str(self.objects)}
        config.update(updates)
        write_json(self.config, config)

    def test_complete_export_separates_references_and_verifies_all_hashes(self):
        self.configure()
        manifest = prepare(self.config, self.output)
        self.assertEqual(manifest, read_json(self.output / "manifest.json"))
        self.assertEqual(manifest["pairs"], 15)
        self.assertEqual(manifest["questions"], 30)
        expected = build_native_id_pairs(geometry())
        for task in ("direct", "named"):
            rows = read_json(self.output / (task + ".json"))
            self.assertEqual(rows, expected[task + "_rows"])
            self.assertEqual(read_json(self.output / ("cohort_" + task + ".json")),
                             [row["qid"] for row in rows])
            for row in rows:
                self.assertEqual(set(row), {"qid", "question_id", "scene_id",
                                           "pair_id", "question", "prompt", "assigned_ids"})
        self.assertEqual(read_json(self.output / "scoring/references.json"),
                         expected["references"])
        self.assertEqual(read_json(self.output / "scoring/pair_metadata.json"),
                         expected["pair_metadata"])
        self.assertEqual(read_json(self.output / "scoring/selection_summary.json"),
                         expected["summary"])
        self.assertEqual(read_json(self.output / "scene_measurements.json"),
                         expected["scene_measurements"])
        for artifact in manifest["inputs"].values():
            self.assertEqual(hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest(),
                             artifact["sha256"])
        for relative, digest in manifest["outputs_sha256"].items():
            self.assertEqual(hashlib.sha256((self.output / relative).read_bytes()).hexdigest(),
                             digest)
        self.assertTrue(manifest["code_sha256"]["controlled_protocol.py"])
        with self.assertRaises(FileExistsError):
            prepare(self.config, self.output)

    def test_height_only_geometry_and_required_source_contract(self):
        height_objects = geometry(3)
        for row in height_objects[SCENE]:
            del row["rho"]
            del row["radius_valid"]
        write_json(self.objects, height_objects)
        self.configure(property="height")
        with self.assertRaisesRegex(ValueError, "measurement_source"):
            prepare(self.config, self.output)
        self.assertFalse(self.output.exists())
        source = self.base / "height_source.json"
        declaration = dict(measurement="full_object_axis_aligned_height",
                           input_id="fixture-height", source_reference="synthetic geometry",
                           verified_scenes={SCENE: False})
        write_json(source, declaration)
        self.configure(property="height", measurement_source=str(source))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            prepare(self.config, self.output)
        declaration["verified_scenes"][SCENE] = True
        write_json(source, declaration)
        manifest = prepare(self.config, self.output)
        self.assertEqual(manifest["pairs"], 3)
        self.assertEqual(manifest["measurement_source"], declaration)
        self.assertIn("measurement_source", manifest["inputs"])

    def test_candidate_changes_direct_only_and_never_changes_targets(self):
        self.configure(instruction="candidate")
        manifest = prepare(self.config, self.output)
        expected = build_native_id_pairs(geometry())
        changed = read_json(self.output / "direct.json")
        self.assertEqual(read_json(self.output / "named.json"), expected["named_rows"])
        self.assertEqual(read_json(self.output / "scoring/references.json"), expected["references"])
        self.assertEqual(read_json(self.output / "scoring/pair_metadata.json"), expected["pair_metadata"])
        for original, row in zip(expected["direct_rows"], changed):
            self.assertEqual(row["prompt"], original["prompt"] + " " + CANDIDATE_INSTRUCTION)
            self.assertEqual(row["qid"], original["qid"])
        self.assertEqual(manifest["instruction"], {"direct": "candidate", "named": None})

    def test_zero_eligible_pairs_still_exports_the_exclusion_ledger(self):
        source = geometry(3)
        for row in source[SCENE]:
            row["category"] = "other"
        write_json(self.objects, source)
        self.configure()
        manifest = prepare(self.config, self.output)
        self.assertEqual(manifest["pairs"], 0)
        self.assertEqual(read_json(self.output / "direct.json"), [])
        summary = read_json(self.output / "scoring/selection_summary.json")
        self.assertEqual(summary["per_scene"][SCENE]["excluded_objects_by_first_reason"],
                         {"missing_or_generic_category": 3})

    def test_rejects_unknown_config_and_inapplicable_instruction_or_source(self):
        for updates in ({"pair_cap": 4}, {"instruction": "natural"},
                        {"property": "volume"}, {"measurement_source": str(self.objects)}):
            with self.subTest(updates=updates):
                self.configure(**updates)
                with self.assertRaises(ValueError):
                    prepare(self.config, self.output)
                self.assertFalse(self.output.exists())

    def test_module_cli_works_without_torch_or_site_packages_and_has_no_pair_cap(self):
        write_json(self.objects, geometry(13))
        # A nested config still resolves relative paths against cwd, not its directory.
        self.configure(scene_objects=os.path.relpath(self.objects, REPOSITORY))
        script = """
import importlib.abc
import runpy
import sys
class NoMLImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'numpy'}:
            raise AssertionError('ML dependency imported: ' + fullname)
sys.meta_path.insert(0, NoMLImports())
sys.argv = ['prepare_controlled', '--config', sys.argv[1], '--output', sys.argv[2]]
runpy.run_module('experiments.scale_interface.prepare_controlled', run_name='__main__')
"""
        completed = subprocess.run(
            [sys.executable, "-S", "-c", script, str(self.config), str(self.output)],
            cwd=REPOSITORY, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = read_json(self.output / "manifest.json")
        self.assertEqual(manifest["pairs"], 78)
        self.assertEqual(manifest["questions"], 156)


if __name__ == "__main__":
    unittest.main()
