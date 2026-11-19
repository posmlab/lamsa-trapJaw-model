import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from python_scripts.lamsa.components.load import RotatingMass
from python_scripts.lamsa.components.spring import LinearSpring
from python_scripts.lamsa.rotational_systems.solve_spring_actuation import solve_spring_actuation


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[3] / "trapjaw_data.csv"
    with data_path.open(newline="", encoding="utf-8-sig") as data_file:
        rows = list(csv.DictReader(data_file))[1:14]#[1+2:13] 

    density = 1.3                 # kg/m³
    elastic_modulus = 10e9        # Pa
    theta_0 = np.deg2rad(90)
    spring_initial_stretch = 1e-6  # Arbitrary -- should be muscle's optimal length, tendon at rest length

    fig, ax = plt.subplots()

    for row in rows:
        lever_length = float(row["Lin_m"])
        out_lever_length = float(row["Lout_m"])
        mandible_volume = float(row["Vman_m3"])
        apodeme_area = float(row["Aap_m2"])
        apodeme_length = float(row["Lap_m"])

        pivot_x = lever_length  # Assuming this can be approx. by lever_length
        spring_rest_length = apodeme_length  # Assume measured length is relaxed

        mass = density * mandible_volume
        spring_stiffness = elastic_modulus * apodeme_area / apodeme_length
        inertia = mass * out_lever_length**2 / 3  # Uniform rod about one end
        spring_initial_length = spring_rest_length + spring_initial_stretch

        spring = LinearSpring(spring_stiffness, spring_initial_length, spring_initial_stretch)
        load = RotatingMass(inertia, lever_length, pivot_x, theta_0)

        sol = solve_spring_actuation(spring, load)
        time, theta, omega = sol.T

        ax.plot(time, theta, label=row["species"])

    ax.set_xlabel("time [s]")
    ax.set_ylabel("lever angle, $\\theta$ [rad]")
    ax.legend()

    fig.tight_layout()
    plt.show()
