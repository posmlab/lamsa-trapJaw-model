import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from get_metrics import energy_conservation, get_energy_metrics
from test_energy_conservation import PARAMETER_SETS, simulate

SPRING_COLOR = "#2a78d6"
LOAD_COLOR = "#eb6834"
TOTAL_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"

EVENTS = ("latch", "unlatch", "takeoff", "close")
MS = 1e3
MJ = 1e3


def main(index=0):
    """Plot PE and KE over time, and at latch, unlatch, takeoff, and close."""
    xp, lam, th0, L0, y0, k, I = PARAMETER_SETS[index]

    sol, theta, theta_dot = simulate(xp, lam, th0, L0, y0, k, I)
    time = sol[:, 0]
    y = sol[:, 1]
    pe = 0.5 * k * y**2
    ke = 0.5 * I * theta_dot**2

    takeoff = time[-1]

    metrics = get_energy_metrics(
        sol, [0.3 * takeoff, takeoff],
        spring_stiffness=k,
        inertia=I,
        lever_length=lam,
        pivot_x=xp,
        spring_length_initial=L0,
        theta_initial=th0,
        theta_closed=theta[-1] - 0.2,
        spring_displacement_at_close=0,
    )

    total = energy_conservation(metrics)
    events = sorted(EVENTS, key=lambda e: metrics[f"t_{e}"])

    plt.rcParams.update({
        "font.size": 10,
        "axes.edgecolor": MUTED_COLOR,
        "axes.labelcolor": TOTAL_COLOR,
        "xtick.color": MUTED_COLOR,
        "ytick.color": MUTED_COLOR,
        "text.color": TOTAL_COLOR,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 2]})

    ax1.plot(time * MS, pe * MJ, color=SPRING_COLOR, lw=2, label="Spring PE")
    ax1.plot(time * MS, ke * MJ, color=LOAD_COLOR, lw=2, label="Load KE")
    ax1.plot(time * MS, (pe + ke) * MJ, color=TOTAL_COLOR, lw=2, ls="--", label="Total")

    coast_times = [takeoff * MS, metrics["t_close"] * MS]
    ax1.plot(coast_times, [pe[-1] * MJ] * 2, color=SPRING_COLOR, lw=2, ls=":")
    ax1.plot(coast_times, [ke[-1] * MJ] * 2, color=LOAD_COLOR, lw=2, ls=":")
    ax1.plot(coast_times, [(pe[-1] + ke[-1]) * MJ] * 2, color=TOTAL_COLOR, lw=2, ls=":")

    for event in events:
        t = metrics[f"t_{event}"] * MS

        ax1.plot(t, metrics[f"PE_{event}"] * MJ, "o", ms=8, color=SPRING_COLOR,
                 mec="white", mew=2, zorder=3)
        ax1.plot(t, metrics[f"KE_{event}"] * MJ, "o", ms=8, color=LOAD_COLOR,
                 mec="white", mew=2, zorder=3)

        ax1.axvline(t, color=MUTED_COLOR, lw=0.8, ls=":")
        ax1.text(t, 1.02, event, transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", color=MUTED_COLOR)

    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Energy (mJ)")
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, loc="center right")
    ax1.set_title(
        f"Energy at latch, unlatch, takeoff and close "
        f"(x_p={xp * 1e3:g} mm, λ={lam * 1e3:g} mm, θ₀={th0:g} rad)",
        pad=26, loc="left",
    )

    positions = range(len(events))
    pe_bars = [metrics[f"PE_{e}"] * MJ for e in events]
    ke_bars = [metrics[f"KE_{e}"] * MJ for e in events]

    ax2.bar(positions, pe_bars, color=SPRING_COLOR, edgecolor="white", linewidth=2, width=0.6)
    ax2.bar(positions, ke_bars, bottom=pe_bars, color=LOAD_COLOR,
            edgecolor="white", linewidth=2, width=0.6)

    for x, event in zip(positions, events):
        ax2.text(x, total[event] * MJ, f"{total[event] * MJ:.4f} mJ", ha="center", va="bottom")

    ax2.set_xticks(list(positions))
    ax2.set_xticklabels([f"{e}\n{metrics[f't_{e}'] * MS:.3f} ms" for e in events])
    ax2.set_ylabel("Energy (mJ)")
    ax2.set_ylim(0, max(total.values()) * MJ * 1.15)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)

    out = HERE / "energy_conservation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    for event in events:
        print(f"{event:8s} t={metrics[f't_{event}'] * MS:8.4f} ms  "
              f"PE={metrics[f'PE_{event}'] * MJ:.6f} mJ  "
              f"KE={metrics[f'KE_{event}'] * MJ:.6f} mJ  "
              f"total={total[event] * MJ:.6f} mJ")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
