import matplotlib.pyplot as plt
import numpy as np

from components.load import Mass
from components.motor import LinearMotor
from components.spring import Spring
from python_scripts.lamsa.linear_systems.solve_direct_actuation import solve_direct_actuation
from python_scripts.lamsa.linear_systems.solve_spring_actuation import solve_spring_actuation


def launch_metrics(sol, mass):
    """Return takeoff velocity, takeoff duration, maximum power, and maximum kinetic energy."""
    t, position, velocity, force = sol.T

    max_power = np.max(force * velocity)
    KE = 0.5 * mass * velocity[-1]**2

    return velocity[-1], t[-1], max_power, KE


if __name__ == "__main__":
    # Parameters 
    max_force = 20            # N
    max_velocity = 5          # m/s
    range_of_motion = 5e-3    # m
    spring_mass = 1e-4        # kg
    min_mass = 1e-4           # kg
    max_mass = 1e-1           # kg
    n_masses = 60

    motor = LinearMotor(F_motor_max=max_force, v_motor_max=max_velocity, range_of_motion=range_of_motion)
    spring = Spring(mass=spring_mass)
    masses = np.logspace(np.log10(min_mass), np.log10(max_mass), n_masses)

    motor_results = []
    spring_results = []

    for mass in masses:
        load = Mass(mass)

        sol, _ = solve_direct_actuation(motor, load)
        motor_results.append(launch_metrics(sol, mass))

        sol, _ = solve_spring_actuation(motor, spring, load)
        spring_results.append(launch_metrics(sol, mass))

    motor_results = np.array(motor_results)
    spring_results = np.array(spring_results)

    labels = ["takeoff velocity [m/s]", "takeoff duration [s]", "max power [W]", "max kinetic energy [J]"]
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5, 9))

    for i, ax in enumerate(axes):
        ax.plot(masses, motor_results[:, i], label="motor-driven")
        ax.plot(masses, spring_results[:, i], "--", label="spring-driven")
        ax.set_xscale("log")
        ax.set_ylabel(labels[i])

    axes[0].legend()
    axes[-1].set_xlabel("projectile mass [kg]")
    fig.tight_layout()
    plt.show()