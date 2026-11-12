import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from python_scripts.lamsa.components.load import RotatingMass
from python_scripts.lamsa.components.spring import LinearSpring
from python_scripts.lamsa.rotational_systems.solve_spring_actuation import solve_spring_actuation


if __name__ == "__main__":
    # Parameters
    mass = 1                       # kg
    lever_length = 1               # m
    pivot_x = 1                    # m
    theta_0 = -0.1                 # rad
    spring_stiffness = 100         # N/m
    spring_rest_length = 0.7       # m
    spring_initial_stretch = 0.2   # m
    spring_mass = 0                # kg

    inertia = mass * lever_length**2
    spring_initial_length = spring_rest_length + spring_initial_stretch

    spring = LinearSpring(spring_stiffness, spring_initial_length, spring_initial_stretch, spring_mass)
    load = RotatingMass(inertia, lever_length, pivot_x, theta_0)

    sol = solve_spring_actuation(spring, load)
    time, theta, omega = sol.T

    fig, ax = plt.subplots()

    ax.plot(time, theta)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lever angle, $\\theta$ [rad]")

    fig.tight_layout()
    plt.show()
