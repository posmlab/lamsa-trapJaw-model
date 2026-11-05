import numpy as np
from scipy.integrate import solve_ivp


def solve_spring_actuation(motor, spring, load, method="Radau"):
    """
    Solve spring actuation with a spring (non-negligible mass) and an ideal latch.
    The motor slowly loads the spring to Fmax = kd, the latch releases instantly at t = 0,
    and the load is driven until the spring reaches its rest length at x = d.

    Returns sol with columns [time, position, velocity, force on load]
    and transition_times = [0, time when the force on the load reaches zero].
    """
    stiffness = motor.max_force / motor.range
    effective_mass = load.mass + spring.mass / 3

    def spring_force(position):
        return stiffness * (motor.range - position)

    def ode(t, state):
        position, velocity = state
        acceleration = spring_force(position) / effective_mass

        return [velocity, acceleration]

    def spring_done(t, state):
        return spring_force(state[0])

    spring_done.terminal = True
    spring_done.direction = -1

    t_end = np.pi * np.sqrt(effective_mass / stiffness)

    for i in range(10):
        run = solve_ivp(
            ode, [0, t_end], [0, 0],
            method=method, events=spring_done, rtol=1e-5, atol=1e-7, dense_output=True,
        )

        spring_released = run.status == 1

        if spring_released:
            break

        t_end *= 10

    else:
        raise RuntimeError("Spring force never reached zero")

    t = np.linspace(0, run.t[-1], 1000)
    position, velocity = run.sol(t)
    force_on_load = load.mass / effective_mass * spring_force(position)

    sol = np.c_[t, position, velocity, force_on_load]
    transition_times = [t[0], t[-1]]

    return sol, transition_times