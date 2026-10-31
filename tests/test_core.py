import unittest
import numpy as np
from inno_medical_imaging_qc import core


class TestImaging(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)

    def test_clean_image_passes_qc(self):
        passed = sum(core.qc_decision(core.qc_metrics(core.make_image(np.random.default_rng(i))[0]))["pass"] for i in range(20))
        self.assertGreaterEqual(passed, 18)

    def test_each_defect_has_a_reason(self):
        for d in core.DEFECTS:
            hits = sum(not core.qc_decision(core.qc_metrics(core.make_image(np.random.default_rng(i), False, d)[0]))["pass"] for i in range(10))
            self.assertGreaterEqual(hits, 8, d)

    def test_lesion_scores_higher(self):
        a = np.mean([core.suspicion(core.make_image(np.random.default_rng(i), True)[0]) for i in range(15)])
        b = np.mean([core.suspicion(core.make_image(np.random.default_rng(i + 50), False)[0]) for i in range(15)])
        self.assertGreater(a, b)

    def test_worklist_separates_rework_and_orders(self):
        items = [{"id": "a", "qc": {"pass": True}, "score": 3}, {"id": "b", "qc": {"pass": False}, "score": 99}, {"id": "c", "qc": {"pass": True}, "score": 7}]
        w = core.worklist(items)
        self.assertEqual(w["read_order"], ["c", "a"]); self.assertEqual(w["reacquire"], ["b"])

    def test_gauss_preserves_mean(self):
        x = self.rng.random((32, 32))
        self.assertAlmostEqual(core.gauss(x, 2.0).mean(), x.mean(), places=1)
