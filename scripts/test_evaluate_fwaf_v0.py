#!/usr/bin/env python3
"""Numeric checks for the FWAF reference evaluator."""

import unittest

import numpy as np

from evaluate_fwaf_v0 import interpolate, jump_mask, score


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        t = 1000.0 + np.arange(201) * 0.1
        p = np.column_stack((np.sin((t - t[0]) / 5) * 10,
                             (t - t[0]) ** 2 / 25,
                             np.sin((t - t[0]) / 3)))
        self.ref = np.column_stack((t, p))
        self.mask = np.ones(len(t), bool)

    def test_rigid_alignment_and_rpe(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        trace = np.column_stack((self.ref[:, 0], self.ref[:, 1:4] @ rotation + [3, 4, -2]))
        values, _ = score(self.ref, trace[:, 1:4], trace, self.mask,
                          self.mask, 0.5)
        self.assertLess(values["ate_rmse_m"], 1e-9)
        self.assertLess(values["rpe_1s_rmse_m"], 1e-9)
        self.assertLess(values["rpe_5s_rmse_m"], 1e-9)
        self.assertAlmostEqual(values["sim3_scale"], 1.0)

    def test_scale_correction_is_not_metric_ate(self):
        trace = np.column_stack((self.ref[:, 0], self.ref[:, 1:4] * 1.5 + [3, 4, -2]))
        values, _ = score(self.ref, trace[:, 1:4], trace, self.mask,
                          self.mask, 0.5)
        self.assertAlmostEqual(values["sim3_scale"], 2 / 3, places=9)
        self.assertAlmostEqual(values["scale_error_pct"], 50.0, places=8)
        self.assertGreater(values["ate_rmse_m"], 0.1)
        self.assertLess(values["sim3_ate_rmse_m"], 1e-9)

    def test_interpolation_refuses_gaps_and_extrapolation(self):
        trace = np.array([[0., 0., 0., 0.], [1., 1., 0., 0.],
                          [2., 2., 0., 0.], [4., 4., 0., 0.]])
        _, ok = interpolate(trace, np.array([-1., 1., 1.5, 3., 4., 5.]), 1.0)
        self.assertEqual(ok.tolist(), [False, True, True, False, True, False])

    def test_jump_rejection_uses_gt_only(self):
        reference = np.column_stack((np.arange(10) * 0.02, np.zeros((10, 3))))
        reference[5, 1] = 10
        mask, count = jump_mask(reference)
        self.assertEqual(count, 2)
        self.assertFalse(mask[5])


if __name__ == "__main__":
    unittest.main()
