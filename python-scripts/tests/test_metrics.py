"""Run with: python3 -m unittest discover -s python-scripts/tests."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import get_energy_metrics


class EnergyMetricsTests(unittest.TestCase):
    def test_translation_all_four_events_and_input_preserved(self):
        sol = np.array([[0, -.02, 0], [.1, -.01, 1], [.2, 0, 2.]])
        original = sol.copy()
        result = get_energy_metrics(sol, [.1, .2], spring_stiffness=100,
                                    load_mass=.01, closure_index=2)
        for event, pe, ke in [('latch', .02, 0), ('unlatch', .005, .005),
                              ('takeoff', 0, .02), ('close', 0, .02)]:
            self.assertAlmostEqual(result[f'PE_{event}'], pe)
            self.assertAlmostEqual(result[f'KE_{event}'], ke)
        np.testing.assert_array_equal(sol, original)

    def test_rotation_uses_large_angle_geometry(self):
        # theta samples -> stretch y = L(theta) - L0 + y0; yd = omega * dL/dtheta.
        k, I, lam, xp, L0, th0 = 2., 3., .5, 1., 2., .3
        yp = np.sqrt(L0**2 - (xp - lam*np.cos(th0))**2) - lam*np.sin(th0)
        L = lambda t: np.hypot(xp - lam*np.cos(t), yp + lam*np.sin(t))
        dL = lambda t: lam*(yp*np.cos(t) + xp*np.sin(t)) / L(t)
        theta = np.array([th0, .1, -.1])
        omega = np.array([-.1, -.2, -.3])
        sol = np.c_[[0, 1, 2], L(theta) - L(th0), omega*dL(theta)]
        args = dict(spring_stiffness=k, spring_mass=.6, inertia=I, lever_length=lam,
                    pivot_x=xp, spring_length_initial=L0, theta_initial=th0)
        result = get_energy_metrics(sol, [1, 2], theta_closed=.1, **args)
        for event, i in [('latch', 0), ('unlatch', 1), ('takeoff', 2), ('close', 1)]:
            self.assertAlmostEqual(result[f'PE_{event}'], .5*k*sol[i, 1]**2)
            self.assertAlmostEqual(result[f'KE_{event}'],
                                   .5*I*omega[i]**2 + .6*sol[i, 2]**2/6)
        self.assertAlmostEqual(result['t_close'], 1)

    def test_free_rotation_requires_explicit_assumption(self):
        args = dict(spring_stiffness=2, inertia=3, lever_length=.5, pivot_x=1,
                    spring_length_initial=2, theta_initial=.3, theta_closed=-.2)
        yp = np.sqrt(4 - (1 - .5*np.cos(.3))**2) - .5*np.sin(.3)
        L = lambda t: np.hypot(1 - .5*np.cos(t), yp + .5*np.sin(t))
        sol = np.array([[0, 0, -.05], [1, L(.2) - L(.3), -.05]])
        result = get_energy_metrics(sol, [.5, 1], **args)
        self.assertTrue(np.isnan(result['KE_close']))
        result = get_energy_metrics(sol, [.5, 1], **args,
                                    free_rotation_after_takeoff=True)
        self.assertGreater(result['t_close'], result['t_takeoff'])
        self.assertTrue(np.isnan(result['PE_close']))
        result = get_energy_metrics(sol, [.5, 1], **args,
                                    free_rotation_after_takeoff=True,
                                    spring_displacement_at_close=0)
        self.assertEqual(result['PE_close'], 0)

    def test_failed_events_are_nan(self):
        sol = np.array([[0, 0, 0], [1, -.1, -.1]])
        result = get_energy_metrics(sol, [np.inf, np.inf], spring_stiffness=2,
                                    load_mass=3)
        for event in ['unlatch', 'takeoff', 'close']:
            self.assertTrue(np.isnan(result[f'KE_{event}']))
            self.assertTrue(np.isnan(result[f'PE_{event}']))

    def test_duplicate_boundary_uses_last_state(self):
        result = get_energy_metrics([[0, -.2, 0], [1, -.1, 1], [1, -.1, 2]],
                                    [1, 1], spring_stiffness=2, load_mass=1)
        self.assertEqual(result['KE_unlatch'], 2)

    def test_invalid_inputs(self):
        args = dict(spring_stiffness=2, load_mass=1)
        with self.assertRaises(ValueError):
            get_energy_metrics([[1, 0, 0], [0, 0, 0]], [0, 1], **args)
        with self.assertRaises(ValueError):  # closing angle needs a rotating load
            get_energy_metrics([[0, 0, 0]], [0, 0], **args, theta_closed=0)
        with self.assertRaises(ValueError):  # rotation missing geometry
            get_energy_metrics([[0, 0, 0], [1, 2, 1]], [0, 1],
                               spring_stiffness=2, inertia=1, lever_length=1)


if __name__ == '__main__':
    unittest.main()
