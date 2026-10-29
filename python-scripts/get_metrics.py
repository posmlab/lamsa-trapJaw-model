
import numpy as np


def get_energy_metrics(
    sol,
    transition_times,
    *,
    spring_stiffness,
    inertia,
    lever_length,
    pivot_x,
    spring_length_initial,
    theta_initial,
    theta_closed,
    spring_mass=0.0,
    spring_displacement_at_close=None,
):
    """Calculate PE and KE at latch, unlatch, takeoff, and close."""

    sol = np.asarray(sol, dtype=float)

    if sol.ndim != 2 or len(sol) == 0 or sol.shape[1] < 3:
        raise ValueError("sol must have columns [time, y, ydot]")
    if len(transition_times) != 2:
        raise ValueError("Need unlatch and takeoff times")
    if np.any(np.diff(sol[:, 0]) < 0):
        raise ValueError("Times must be increasing")
    if spring_stiffness < 0 or spring_mass < 0:
        raise ValueError("Spring stiffness and mass must be nonnegative")
    if inertia <= 0 or lever_length <= 0:
        raise ValueError("Inertia and lever length must be positive")

    sol_unique = sol[np.r_[np.diff(sol[:, 0]) != 0, True], :3]
    time, y, v = sol_unique.T
    y0 = sol[0, 1]

    lam = lever_length
    xp = pivot_x
    L0 = spring_length_initial

    yp = np.sqrt(L0**2 - (xp - lam * np.cos(theta_initial))**2) - lam * np.sin(theta_initial)

    def spring_length(theta):
        return np.hypot(xp - lam * np.cos(theta), yp + lam * np.sin(theta))

    def theta_of(y):
        length = y + L0 - y0
        R = np.hypot(xp, yp)
        s = (length**2 - R**2 - lam**2) / (2 * lam * R)

        with np.errstate(invalid="ignore"):
            theta = np.arctan2(xp, yp) + np.arcsin(s)

        return theta

    def omega(y, ydot):
        theta = theta_of(y)
        dL_dtheta = lam * (yp * np.cos(theta) + xp * np.sin(theta)) / spring_length(theta)
        return ydot / dL_dtheta

    def potential_energy(y):
        return 0.5 * spring_stiffness * y**2

    def kinetic_energy(y, ydot):
        ke = 0.5 * inertia * omega(y, ydot)**2
        ke += spring_mass * ydot**2 / 6
        return ke

    def state_at(t):
        if not time[0] <= t <= time[-1]:
            return (np.nan, np.nan, np.nan)

        return (t, np.interp(t, time, y), np.interp(t, time, v))

    unlatch_time, takeoff_time = transition_times

    events = {
        "latch": tuple(sol[0, :3]),
        "unlatch": state_at(unlatch_time),
        "takeoff": state_at(takeoff_time),
        "close": (np.nan, np.nan, np.nan),
    }

    coast = None

    if not np.isnan(events["takeoff"][0]):
        angles = theta_of(y)
        close_indices = np.where((time >= takeoff_time) & (angles <= theta_closed))[0]

        if len(close_indices) > 0:
            i = close_indices[0]
            events["close"] = (time[i], y[i], v[i])

        else:
            w = omega(y[-1], v[-1])

            if w != 0:
                dt = (theta_closed - angles[-1]) / w
            else:
                dt = np.nan

            if dt >= 0:
                pe = np.nan

                if spring_displacement_at_close is not None:
                    pe = potential_energy(spring_displacement_at_close)

                coast = (time[-1] + dt, 0.5 * inertia * w**2, pe)

    metrics = {}

    for name, (t, y_val, v_val) in events.items():
        metrics[f"t_{name}"] = float(t)
        metrics[f"PE_{name}"] = float(potential_energy(y_val))
        metrics[f"KE_{name}"] = float(kinetic_energy(y_val, v_val))

    if coast is not None:
        metrics["t_close"] = float(coast[0])
        metrics["KE_close"] = float(coast[1])
        metrics["PE_close"] = float(coast[2])

    return metrics


def energy_conservation(metrics):
    """Calculate total mechanical energy at each event."""
    energies = {}

    for event in ("latch", "unlatch", "takeoff", "close"):
        energies[event] = metrics[f"PE_{event}"] + metrics[f"KE_{event}"]

    return energies
