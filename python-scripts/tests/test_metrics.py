import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from get_metrics import get_energy_metrics

K, I, LAM, XP, L0, THETA0 = 2.0, 3.0, 0.5, 1.0, 2.0, 0.3
YP = np.sqrt(L0**2 - (XP - LAM * np.cos(THETA0))**2) - LAM * np.sin(THETA0)


def spring_length(theta):
    return np.hypot(XP - LAM * np.cos(theta), YP + LAM * np.sin(theta))


def dL_dtheta(theta):
    return LAM * (YP * np.cos(theta) + XP * np.sin(theta)) / spring_length(theta)


def make_case(theta, omega):
    theta = np.array(theta)
    omega = np.array(omega)

    sol = np.c_[
        np.arange(len(theta)),
        spring_length(theta) - spring_length(THETA0),
        omega * dL_dtheta(theta),
    ]

    args = dict(
        spring_stiffness=K,
        spring_mass=0.6,
        inertia=I,
        lever_length=LAM,
        pivot_x=XP,
        spring_length_initial=L0,
        theta_initial=THETA0,
        theta_closed=-0.2,
    )

    return sol, args


class EnergyMetricsTests(unittest.TestCase):
    def test_energies_at_all_four_events(self):
        omega = [-0.1, -0.2, -0.3, -0.4]
        sol, args = make_case([0.3, 0.1, -0.1, -0.3], omega)

        result = get_energy_metrics(sol, [1, 2], **args)

        for event, i in [("latch", 0), ("unlatch", 1), ("takeoff", 2), ("close", 3)]:
            expected_pe = 0.5 * K * sol[i, 1]**2
            expected_ke = 0.5 * I * omega[i]**2 + 0.6 * sol[i, 2]**2 / 6

            self.assertAlmostEqual(result[f"PE_{event}"], expected_pe)
            self.assertAlmostEqual(result[f"KE_{event}"], expected_ke)

        self.assertEqual(result["t_close"], 3)

    def test_close_is_first_sample_after_takeoff_without_interpolating(self):
        sol, args = make_case([0.3, 0.1, -0.1, -0.3], [-0.1, -0.2, -0.3, -0.4])

        result = get_energy_metrics(sol, [0.5, 1], **{**args, "theta_closed": -0.05})
        self.assertEqual(result["t_close"], 2)

        result = get_energy_metrics(sol, [0.5, 2], **{**args, "theta_closed": 0.2})
        self.assertGreaterEqual(result["t_close"], result["t_takeoff"])

    def test_close_extrapolates_when_angle_is_never_reached(self):
        sol, args = make_case([0.3, 0.2], [-0.1, -0.1])

        result = get_energy_metrics(sol, [0.5, 1], **args)
        self.assertGreater(result["t_close"], 1)
        self.assertGreater(result["KE_close"], 0)
        self.assertTrue(np.isnan(result["PE_close"]))

        result = get_energy_metrics(sol, [0.5, 1], **args, spring_displacement_at_close=0)
        self.assertEqual(result["PE_close"], 0)

    def test_events_that_cannot_be_found_are_nan(self):
        sol, args = make_case([0.3, 0.1, -0.1, -0.3], [-0.1, -0.2, -0.3, -0.4])

        result = get_energy_metrics(sol, [np.inf, np.inf], **args)

        for event in ["unlatch", "takeoff", "close"]:
            self.assertTrue(np.isnan(result[f"PE_{event}"]))
            self.assertTrue(np.isnan(result[f"KE_{event}"]))

    def test_duplicate_time_uses_last_row(self):
        omega = [-0.1, -0.2, -0.3, -0.4]
        sol, args = make_case([0.3, 0.1, -0.1, -0.3], omega)
        args["spring_mass"] = 0

        double_speed_row = [1, sol[1, 1], 2 * sol[1, 2]]
        sol = np.array([sol[0], sol[1], double_speed_row, sol[2], sol[3]])

        result = get_energy_metrics(sol, [1, 2], **args)
        self.assertAlmostEqual(result["KE_unlatch"], 0.5 * I * (2 * omega[1])**2)

    def test_invalid_inputs(self):
        sol, args = make_case([0.3, 0.1, -0.1, -0.3], [-0.1, -0.2, -0.3, -0.4])

        with self.assertRaises(ValueError):
            get_energy_metrics(sol[::-1], [1, 2], **args)

        with self.assertRaises(ValueError):
            get_energy_metrics(sol, [1], **args)

        with self.assertRaises(ValueError):
            get_energy_metrics(sol, [1, 2], **{**args, "inertia": 0})


if __name__ == "__main__":
    unittest.main()
