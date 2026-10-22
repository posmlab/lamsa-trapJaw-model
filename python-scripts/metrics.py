import numpy as np


def get_energy_metrics(
    sol,
    transition_times,
    *,
    spring_stiffness,
    spring_mass=0.0,
    load_mass=None,
    inertia=None,
    lever_length=None,
    pivot_x=None,
    spring_length_initial=None,
    theta_initial=None,
    theta_closed=None,
    closure_index=None,
    free_rotation_after_takeoff=False,
    spring_displacement_at_close=None,
):
    """Return time [s], PE [J] and KE [J] at latch, unlatch, takeoff and close.

    sol columns: time [s], spring stretch y [m], dy/dt [m/s].
    transition_times: (unlatch, takeoff) times. Angles are in radians.
    Events that can't be found are NaN.

    PE is the spring's ½·k·y². KE is the load's KE plus the spring's, taken as
    ⅙·spring_mass·ẏ² (uniformly stretching spring, one end fixed; assumption).

    Straight-line load: give load_mass.
    Rotating load (large-angle model): give inertia I, lever_length λ, pivot_x
    x_p, spring_length_initial L0 and theta_initial. The spring runs from a
    fixed point to the lever tip, so y = L(θ) − L0 + y0, and KE_load = ½·I·θ̇².

    Close is found from closure_index (a row of sol) or, for rotation, where
    theta reaches theta_closed. If the solution ends first, the free-rotation
    option extrapolates it at constant angular velocity; PE_close then also
    needs spring_displacement_at_close.
    """
    sol = np.asarray(sol, dtype=float)
    rotating = inertia is not None

    if sol.ndim != 2 or len(sol) == 0 or sol.shape[1] < 3:
        raise ValueError("sol must contain time, stretch, and velocity.")
    if len(transition_times) != 2:
        raise ValueError("Supply unlatching and takeoff times.")
    if np.any(np.diff(sol[:, 0]) < 0):
        raise ValueError("Solution times must be nondecreasing.")
    if spring_stiffness < 0 or spring_mass < 0:
        raise ValueError("Spring stiffness and mass must be nonnegative.")
    if closure_index is not None and not 0 <= closure_index < len(sol):
        raise ValueError("closure_index is outside the solution.")
    if rotating:
        needed = (lever_length, pivot_x, spring_length_initial, theta_initial)
        if inertia <= 0 or lever_length is None or lever_length <= 0 or None in needed:
            raise ValueError("Rotation requires inertia, lever_length, pivot_x, "
                             "spring_length_initial and theta_initial.")
    else:
        if load_mass is None or load_mass <= 0:
            raise ValueError("Provide a positive load_mass, or inertia for rotation.")
        if theta_closed is not None or free_rotation_after_takeoff:
            raise ValueError("Closure angles and free rotation require inertia.")

    y0 = sol[0, 1]
    # At a repeated time (phase boundary) keep only the later state.
    time, y, v = sol[np.r_[np.diff(sol[:, 0]) != 0, True], :3].T

    if rotating:
        lam, xp, L0 = lever_length, pivot_x, spring_length_initial
        # Pivot height, from the initial spring length (PDF Eq. 2).
        yp = np.sqrt(L0**2 - (xp - lam * np.cos(theta_initial)) ** 2) - lam * np.sin(theta_initial)

        def spring_length(theta):
            return np.hypot(xp - lam * np.cos(theta), yp + lam * np.sin(theta))

        def stretch(theta):
            return spring_length(theta) - L0 + y0

        def theta_of(y):
            # Invert L(θ)² = xp² + yp² + λ² + 2λ(yp·sinθ − xp·cosθ) for θ.
            length = y + L0 - y0
            r = np.hypot(xp, yp)
            s = (length**2 - xp**2 - yp**2 - lam**2) / (2 * lam * r)
            return np.arctan2(xp, yp) + np.arcsin(s)

        def omega(y, speed):
            # ẏ = (dL/dθ)·θ̇, with dL/dθ = λ(yp·cosθ + xp·sinθ)/L.
            theta = theta_of(y)
            slope = lam * (yp * np.cos(theta) + xp * np.sin(theta)) / spring_length(theta)
            return speed / slope

    def potential_energy(y):
        return 0.5 * spring_stiffness * y**2

    def kinetic_energy(y, speed):
        load = 0.5 * inertia * omega(y, speed) ** 2 if rotating else 0.5 * load_mass * speed**2
        return load + spring_mass * speed**2 / 6

    def state_at(t):
        """(t, y, v) by interpolation, or all NaN if t is outside the solution."""
        if not time[0] <= t <= time[-1]:  # also False for NaN
            return (np.nan,) * 3
        return t, np.interp(t, time, y), np.interp(t, time, v)

    def find_closure():
        """First (t, y, v) where theta reaches theta_closed, else NaN."""
        target = stretch(theta_closed)
        d = y - target
        for i in range(len(time)):
            if d[i] == 0:
                return time[i], y[i], v[i]
            if i + 1 < len(time) and d[i] * d[i + 1] < 0:
                f = -d[i] / (d[i + 1] - d[i])  # fraction of the way to sample i+1
                return (
                    time[i] + f * (time[i + 1] - time[i]),
                    target,
                    v[i] + f * (v[i + 1] - v[i]),
                )
        return (np.nan,) * 3

    unlatch_time, takeoff_time = transition_times
    events = {
        "latch": tuple(sol[0, :3]),
        "unlatch": state_at(unlatch_time),
        "takeoff": state_at(takeoff_time),
        "close": (np.nan,) * 3,
    }
    if closure_index is not None:
        events["close"] = tuple(sol[closure_index, :3])
    elif theta_closed is not None:
        events["close"] = find_closure()

    metrics = {}
    for name, (t, ey, ev) in events.items():
        metrics[f"t_{name}"] = float(t)
        metrics[f"PE_{name}"] = float(potential_energy(ey))
        metrics[f"KE_{name}"] = float(kinetic_energy(ey, ev))

    # The solution ended before closing: coast at constant angular velocity.
    if np.isnan(metrics["t_close"]) and theta_closed is not None and free_rotation_after_takeoff:
        _, takeoff_y, takeoff_v = events["takeoff"]
        w = omega(takeoff_y, takeoff_v)
        if np.isfinite(w) and w != 0:
            duration = (theta_closed - theta_of(takeoff_y)) / w
            if duration >= 0 and takeoff_time + duration > time[-1]:
                metrics["t_close"] = float(takeoff_time + duration)
                metrics["KE_close"] = float(0.5 * inertia * w**2)  # load only; spring is detached
                if spring_displacement_at_close is not None:
                    metrics["PE_close"] = float(potential_energy(spring_displacement_at_close))

    return metrics


def energy_conservation(metrics):
    """Total energy PE + KE [J] at each event, and its fractional change from latch.

    Takes the dict from get_energy_metrics. With only the spring acting, the
    drift should be ~0 at every event; a nonzero drift means energy was lost
    or gained (friction, motor work, solver error, or a model/metrics mismatch).
    Events that weren't found give NaN.
    """
    total = {e: metrics[f"PE_{e}"] + metrics[f"KE_{e}"]
             for e in ("latch", "unlatch", "takeoff", "close")}
    start = total["latch"]
    result = {}
    for event, energy in total.items():
        result[f"E_{event}"] = energy
        result[f"drift_{event}"] = (energy - start) / start if start else np.nan
    return result
