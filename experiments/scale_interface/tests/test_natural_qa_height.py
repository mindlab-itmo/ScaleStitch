"""Height dispatch preserves old radius names and uses one preflight cohort."""
import unittest

import torch

from test_natural_qa import NaturalHost, dataset, qa
from test_scene_extents import measurement_source


class NaturalQAHeightTests(unittest.TestCase):
    def test_radius_condition_names_are_unchanged_and_height_is_distinct(self):
        self.assertEqual(qa.NaturalQACondition().name, "full_lexical_true_cm_forward")
        self.assertEqual(qa.NaturalQACondition("absent").name, "full_absent_true_cm_forward")
        self.assertEqual(qa.NaturalQACondition(measurement="height").name,
                         "full_lexical_true_cm_forward_height")
        with self.assertRaises(ValueError):
            qa.NaturalQACondition(measurement="volume")
        with self.assertRaises(ValueError):
            qa.NaturalQACondition("absent", measurement="height")

    def test_height_reads_loc_z_extent_not_radius(self):
        data = dataset(rows=1)
        data.attributes["scene0000_00"]["locs"][:, 5] = torch.tensor([1.23, .7, .9])
        _, sid, _, batch = qa._cpu_input(data, 0)
        source = measurement_source(sid)
        height = qa._packet(batch, qa.NaturalQACondition(measurement="height"), 100,
                            measurement_source=source, scene_id=sid)
        radius = qa._packet(batch, qa.NaturalQACondition(), 100)
        self.assertEqual([record.height_cm for record in height.records], [90, 70, 123])
        self.assertEqual([record.radius_cm for record in radius.records], [30, 20, 10])

    def test_all_length_units_and_orders_work_with_height(self):
        core, data = NaturalHost(), dataset(rows=1)
        conditions = [qa.NaturalQACondition(unit=unit, reverse=reverse, measurement="height")
                      for unit in ("cm", "mm", "m") for reverse in (False, True)]
        plan = qa.preflight_natural_qa(core, data, conditions,
                                       measurement_source=measurement_source("scene0000_00"))
        rows = list(qa.iter_natural_qa(core, data, plan))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["condition"].endswith("_height") for row in rows))
        self.assertTrue(all(row["numerical_blocks"] == 3 for row in rows))

    def test_unverified_candidate_scenes_fail_before_any_generation(self):
        core, data = NaturalHost(), dataset()
        for source in (None, measurement_source("scene0000_00")):
            with self.subTest(source=source), self.assertRaises(ValueError):
                qa.preflight_natural_qa(core, data, [qa.NaturalQACondition(measurement="height")],
                                        measurement_source=source)
        self.assertEqual(core.llama_model.calls, [])

    def test_one_bad_height_excludes_same_question_from_radius_and_absent(self):
        core, data = NaturalHost(), dataset()
        data.attributes["scene0001_00"]["locs"][0, 5] = 0
        conditions = [qa.NaturalQACondition("absent"), qa.NaturalQACondition(),
                      qa.NaturalQACondition(measurement="height")]
        source = measurement_source("scene0000_00", "scene0001_00")
        plan = qa.preflight_natural_qa(core, data, conditions, measurement_source=source)
        self.assertEqual([row.qid for row in plan.rows], ["q0"])
        self.assertEqual({exclusion.qid for exclusion in plan.exclusions}, {"q1"})
        rows = list(qa.iter_natural_qa(core, data, plan))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["qid"] for row in rows}, {"q0"})

    def test_one_bad_radius_excludes_same_question_from_height(self):
        core, data = NaturalHost(), dataset()
        data.scene_radii["scene0001_00"][0] = .001
        conditions = [qa.NaturalQACondition(), qa.NaturalQACondition(measurement="height")]
        plan = qa.preflight_natural_qa(core, data, conditions,
                                       measurement_source=measurement_source("scene0000_00", "scene0001_00"))
        self.assertEqual([row.qid for row in plan.rows], ["q0"])

    def test_source_content_is_frozen_and_checked_after_preflight(self):
        core, data = NaturalHost(), dataset(rows=1)
        source = measurement_source("scene0000_00")
        plan = qa.preflight_natural_qa(core, data, [qa.NaturalQACondition(measurement="height")],
                                       measurement_source=source)
        source["verified_scenes"]["scene0000_00"] = False
        self.assertTrue(plan.measurement_source["verified_scenes"]["scene0000_00"])
        plan.metadata()["measurement_source"]["input_id"] = "outside-copy"
        self.assertEqual(plan.measurement_source["input_id"], "fixed-inputs")
        plan.measurement_source["input_id"] = "tampered"
        with self.assertRaisesRegex(ValueError, "source changed"):
            list(qa.iter_natural_qa(core, data, plan))
        self.assertEqual(core.llama_model.calls, [])


if __name__ == "__main__":
    unittest.main()
