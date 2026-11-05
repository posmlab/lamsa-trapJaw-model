import numpy as np
from scipy.integrate import solve_ivp


def solve_direct_actuation(motor, load, method="Radau"):
    """
    Solve direct motor actuation (F = ma) until the load has moved through the motor's range.
    
    Returns sol with columns [time, position, velocity, motor force]
    and transition_times = [0, time when the motor reaches the end of its range].
    """
 
    def ode(t, state):
        velocity = state[1]
        acceleration = motor.force(t, state) / load.mass

        return [velocity, acceleration]

    def motor_done(t, state):
        return motor.range - state[0]

    motor_done.terminal = True
    motor_done.direction = -1

    time_for_speed = motor.velocity * load.mass / motor.max_force
    time_for_range = np.sqrt(2 * motor.range * load.mass / motor.max_force)
    t_end = max(t for t in (time_for_speed, time_for_range) if np.isfinite(t))

    for i in range(10):
        run = solve_ivp(
            ode, [0, t_end], [0, 0],
            method=method, events=motor_done, rtol=1e-5, atol=1e-7, dense_output=True,
        )

        motor_stopped = run.status == 1

        if motor_stopped:
            break

        t_end *= 10

    else:
        raise RuntimeError("Motor never reached the end of its range")

    t = np.linspace(0, run.t[-1], 1000)
    position, velocity = run.sol(t)
    motor_force = []

    for i in range(len(t)):
        state = [position[i], velocity[i]]
        motor_force.append(motor.force(t[i], state))

    sol = np.c_[t, position, velocity, motor_force]
    transition_times = [t[0], t[-1]]

    return sol, transition_times
