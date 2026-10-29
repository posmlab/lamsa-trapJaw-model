import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from get_metrics import energy_conservation, get_energy_metrics

#cols: x_p, lam, th0, L0, y0, k, I
PARAMETER_SETS = [
    (0.010, 0.005, 0.3, 0.030, 0.002, 500.0, 1e-8),
    (0.020, 0.008, 0.5, 0.040, 0.003, 300.0, 5e-9),
    (0.010, 0.005, -0.2, 0.030, 0.002, 800.0, 2e-8),
    (0.015, 0.004, 0.6, 0.035, 0.004, 1000.0, 1e-8),
]






# def simulate(xp, lam, th0, L0, y0, k, I):
#     """Integrate Eq. 17 until the spring reaches rest length. Returns sol, theta, theta_dot."""
#     yp = np.sqrt(L0**2 - (xp - lam * np.cos(th0))**2) - lam * np.sin(th0)

#     def rhs(t, state):
#         y, yd, th = state

#         L = np.hypot(xp - lam * np.cos(th), yp + lam * np.sin(th))
#         D = yp * np.cos(th) + xp * np.sin(th)
#         alpha = np.arctan((xp - lam * np.cos(th)) / (yp + lam * np.sin(th))) + np.pi / 2 - th

#         spring_term = -k * y * lam * np.sin(alpha)
#         speed_term = (lam * D**2 + L**2 * (yp * np.sin(th) - xp * np.cos(th))) / (lam**2 * D**3)
#         f_eff = spring_term - speed_term * I * yd**2
#         m_eff = L / (lam * D) * I

#         return [yd, f_eff / m_eff, yd * L / (lam * D)]

#     def spring_relaxed(t, state):
#         return state[0]

#     spring_relaxed.terminal = True
#     spring_relaxed.direction = -1

#     run = solve_ivp(
#         rhs, [0, 1.0], [y0, 0.0, th0],
#         events=spring_relaxed, rtol=1e-11, atol=1e-14, dense_output=True,
#     )

#     if run.t_events[0].size == 0:
#         raise RuntimeError("Spring never reached its rest length")

#     t = np.linspace(0, run.t_events[0][0], 4000)
#     y, yd, th = run.sol(t)

#     L = np.hypot(xp - lam * np.cos(th), yp + lam * np.sin(th))
#     D = yp * np.cos(th) + xp * np.sin(th)

#     return np.c_[t, y, yd], th, yd * L / (lam * D)


# class EnergyConservationTests(unittest.TestCase):
#     def test_total_energy_is_constant_along_trajectory(self):
#         for xp, lam, th0, L0, y0, k, I in PARAMETER_SETS:
#             with self.subTest(xp=xp, lam=lam, th0=th0):
#                 sol, _, theta_dot = simulate(xp, lam, th0, L0, y0, k, I)

#                 energy = 0.5 * I * theta_dot**2 + 0.5 * k * sol[:, 1]**2

#                 self.assertLess(np.ptp(energy) / energy[0], 1e-8)

#     def test_energy_is_conserved_at_the_four_events(self):
#         for xp, lam, th0, L0, y0, k, I in PARAMETER_SETS:
#             with self.subTest(xp=xp, lam=lam, th0=th0):
#                 sol, theta, _ = simulate(xp, lam, th0, L0, y0, k, I)
#                 takeoff = sol[-1, 0]

#                 metrics = get_energy_metrics(
#                     sol, [0.3 * takeoff, takeoff],
#                     spring_stiffness=k,
#                     inertia=I,
#                     lever_length=lam,
#                     pivot_x=xp,
#                     spring_length_initial=L0,
#                     theta_initial=th0,
#                     theta_closed=theta[-1] - 0.2,
#                     spring_displacement_at_close=0,
#                 )

#                 self.assertGreater(metrics["t_close"], metrics["t_takeoff"])

#                 energy = energy_conservation(metrics)

#                 for event in ("unlatch", "takeoff", "close"):
#                     self.assertAlmostEqual(energy[event] / energy["latch"], 1, delta=1e-5)


# if __name__ == "__main__":
#     unittest.main()
