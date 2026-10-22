import numpy as np


def get_energy_metrics(
    sol,
    transition_times,
    *,
    spring_stiffness,
    load_mass,
    lever_length=None,
    theta_initial=None,
    theta_closed=None,
    closure_index=None,
    free_rotation_after_takeoff=False,
    spring_displacement_at_close=None,
):
    """Return PE [J], pre-impact KE [J], and time [s] at four events.

    sol columns: time [s], spring deformation [m], velocity [m/s].
    Angles are in radians. load_mass is the equivalent mass for rotation.
    Unavailable events return NaN. Free rotation assumes zero net torque;
    extrapolated spring PE requires spring_displacement_at_close [m].
    """
    sol = np.asarray(sol, dtype=float)
    rotating = lever_length is not None

    # Input checks
    if sol.ndim != 2 or len(sol) == 0 or sol.shape[1] < 3:
        raise ValueError("sol must contain time, deformation, and velocity.")
    if len(transition_times) != 2:
        raise ValueError("Supply unlatching and takeoff times.")
    if np.any(np.diff(sol[:, 0]) < 0):
        raise ValueError("Solution times must be nondecreasing.")
    if spring_stiffness < 0 or load_mass <= 0:
        raise ValueError("Stiffness must be nonnegative and mass positive.")
    if rotating and (lever_length <= 0 or theta_initial is None):
        raise ValueError("Rotation requires positive lever_length and theta_initial.")
    if not rotating and (theta_closed is not None or free_rotation_after_takeoff):
        raise ValueError("Closure angles and free rotation require lever_length.")
    if closure_index is not None and not 0 <= closure_index < len(sol):
        raise ValueError("closure_index is outside the solution.")
    if rotating:
        if abs(theta_initial) >= np.pi / 2:
            raise ValueError("theta_initial must be inside (-pi/2, pi/2).")
        if theta_closed is not None and abs(theta_closed) >= np.pi / 2:
            raise ValueError("theta_closed must be inside (-pi/2, pi/2).")
        sine = np.sin(theta_initial) + (sol[0, 1] - sol[:, 1]) / lever_length
        if np.any(np.abs(sine) >= 1):
            raise ValueError("Trajectory exceeds the supported angular geometry.")

    # Keep the last state at a repeated phase-boundary time.
    keep = np.r_[np.diff(sol[:, 0]) != 0, True]
    time, displacement, velocity = sol[keep, :3].T
    initial_displacement = sol[0, 1]
    unlatch_time, takeoff_time = transition_times

    def state_at(event_time):
        if not np.isfinite(event_time) or not time[0] <= event_time <= time[-1]:
            return np.nan, np.nan
        return (
            np.interp(event_time, time, displacement),
            np.interp(event_time, time, velocity),
        )

    def angle(y):
        return np.arcsin(
            np.sin(theta_initial) + (initial_displacement - y) / lever_length
        )

    def angular_velocity(y, speed):
        # Differentiate y = y0 + L*(sin(theta0) - sin(theta)).
        return -speed / (lever_length * np.cos(angle(y)))

    def potential_energy(y):
        return 0.5 * spring_stiffness * y**2

    def kinetic_energy(y, speed):
        if rotating:
            inertia = load_mass * lever_length**2
            omega = angular_velocity(y, speed)
            return 0.5 * inertia * omega**2
        return 0.5 * load_mass * speed**2

    latch_y, latch_v = sol[0, 1:3]
    unlatch_y, unlatch_v = state_at(unlatch_time)
    takeoff_y, takeoff_v = state_at(takeoff_time)
    close_time, close_y, close_v = np.nan, np.nan, np.nan

    if closure_index is not None:
        close_time, close_y, close_v = sol[closure_index, :3]
    elif theta_closed is not None:
        # Express the closed angle as a spring deformation to locate its crossing.
        target = initial_displacement + lever_length * (
            np.sin(theta_initial) - np.sin(theta_closed)
        )
        distance = displacement - target
        for i in range(len(time)):
            if distance[i] == 0:
                close_time, close_y, close_v = time[i], displacement[i], velocity[i]
                break
            if i + 1 < len(time) and distance[i] * distance[i + 1] < 0:
                fraction = -distance[i] / (distance[i + 1] - distance[i])
                close_time = time[i] + fraction * (time[i + 1] - time[i])
                close_y = target
                close_v = velocity[i] + fraction * (velocity[i + 1] - velocity[i])
                break

    metrics = {
        "t_latch": float(time[0]),
        "PE_latch": float(potential_energy(latch_y)),
        "KE_latch": float(kinetic_energy(latch_y, latch_v)),

        "t_unlatch": float(unlatch_time) if np.isfinite(unlatch_y) else np.nan,
        "PE_unlatch": float(potential_energy(unlatch_y)),
        "KE_unlatch": float(kinetic_energy(unlatch_y, unlatch_v)),

        "t_takeoff": float(takeoff_time) if np.isfinite(takeoff_y) else np.nan,
        "PE_takeoff": float(potential_energy(takeoff_y)),
        "KE_takeoff": float(kinetic_energy(takeoff_y, takeoff_v)),

        "t_close": float(close_time),
        "PE_close": float(potential_energy(close_y)),
        "KE_close": float(kinetic_energy(close_y, close_v)),
    }

    if np.isnan(close_time) and theta_closed is not None and free_rotation_after_takeoff:
        omega = angular_velocity(takeoff_y, takeoff_v)
        if np.isfinite(omega) and omega != 0:
            duration = (theta_closed - angle(takeoff_y)) / omega
            predicted_time = takeoff_time + duration
            if duration >= 0 and predicted_time > time[-1]:
                metrics["t_close"] = float(predicted_time)
                metrics["KE_close"] = metrics["KE_takeoff"]
                if spring_displacement_at_close is not None:
                    metrics["PE_close"] = float(
                        potential_energy(spring_displacement_at_close)
                    )

    return metrics
