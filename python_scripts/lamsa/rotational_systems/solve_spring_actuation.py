import numpy as np
from scipy.integrate import solve_ivp


def solve_spring_actuation(spring, load, method="RK45"):
    """
    Solve large-angle spring actuation of a lever with an ideal latch.
    The spring starts stretched, the latch releases instantly at t = 0,
    and the lever is driven until the spring reaches its rest length.

    Returns sol with columns [time, angle, angular velocity].
    """
    lever_length = load.lever_length
    pivot_x = load.pivot_x

    pivot_y = np.sqrt(
        spring.initial_length**2 - (pivot_x - lever_length * np.cos(load.theta_0))**2
    ) - lever_length * np.sin(load.theta_0)

    def spring_length(theta):
        return np.hypot(pivot_x - lever_length * np.cos(theta), pivot_y + lever_length * np.sin(theta))

    def stretch(theta):
        return spring_length(theta) - spring.initial_length + spring.initial_stretch

    def alpha(theta):
        return np.arctan(
            (pivot_x - lever_length * np.cos(theta)) / (pivot_y + lever_length * np.sin(theta))
        ) + np.pi / 2 - theta

    def ode(t, state):
        theta, omega = state

        torque = spring.force(stretch(theta)) * lever_length * np.sin(alpha(theta))

        return [omega, torque / load.inertia]

    def spring_done(t, state):
        return spring.force(stretch(state[0]))

    spring_done.terminal = True
    spring_done.direction = 1

    initial_slope = lever_length * np.sin(alpha(load.theta_0))
    natural_frequency = np.sqrt(spring.stiffness * initial_slope**2 / load.inertia)
    t_end = np.pi / natural_frequency

    for i in range(10):
        run = solve_ivp(
            ode, [0, t_end], [load.theta_0, 0],
            method=method, events=spring_done, rtol=1e-8, atol=1e-10, dense_output=True,
        )

        spring_released = run.status == 1

        if spring_released:
            break

        t_end *= 10

    else:
        raise RuntimeError("Spring force never reached zero")

    t = np.linspace(0, run.t[-1], 1000)
    theta, omega = run.sol(t)

    return np.c_[t, theta, omega]
