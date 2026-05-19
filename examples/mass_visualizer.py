"""
Interactive mass distribution visualizer.

Shows the rocket as a side view with components plotted as circles sized
by mass. A slider lets you adjust fuel level and see how total mass,
centre of mass, and inertia change in real time.

Usage (from backend directory):

    uv run -m examples.mass_visualizer
    uv run -m examples.mass_visualizer --output mass_dist.png

Close the window to exit.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Slider
import numpy as np

# Add the src directory to the path so we can import simulacra
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
src_dir = os.path.join(backend_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from simulacra.noumena.mass_model import (  # noqa: E402
    FuelComponent,
    MassModel,
    default_sounding_rocket,
)


def create_plot(model: MassModel, length: float = 6.0):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(bottom=0.2, top=0.82)

    # --- HUD text ---
    hud = fig.text(
        0.02, 0.98, "", fontsize=9, family="monospace", va="top",
    )

    # --- Drawing helpers ---
    def draw(fuel_fraction: float):
        ax.clear()

        model.set_fuel_fraction(fuel_fraction)

        total = model.total_mass
        com = model.center_of_mass
        Ix, Iy, Iz = model.inertia_diag
        fuel_mass = model.fuel_mass
        fuel_cap = model.fuel_capacity

        # Update HUD
        hud.set_text(
            f"Total mass: {total:6.1f} kg    "
            f"Fuel: {fuel_fraction * 100:5.1f}% ({fuel_mass:.1f}/{fuel_cap:.1f} kg)\n"
            f"CoM: x={com.x:5.2f} m         "
            f"Inertia: Ix={Ix:.2f}  Iy={Iy:.2f}  Iz={Iz:.2f} kg·m²"
        )

        # Rocket outline
        margin = 0.3
        rocket_rect = FancyBboxPatch(
            (-margin, -0.4), length + 2 * margin, 0.8,
            boxstyle="round,pad=0.1",
            facecolor="#e8e8e8", edgecolor="#888888", linewidth=1.5,
        )
        ax.add_patch(rocket_rect)

        # Plot components
        max_mass = max(c.mass for c in model.components) if model.components else 1.0
        # Use full_mass for fuel components to keep sizing consistent
        max_full = max(
            (c.full_mass if isinstance(c, FuelComponent) else c.mass)
            for c in model.components
        ) if model.components else 1.0
        max_ref = max(max_mass, max_full)

        label_above = True
        for c in model.components:
            x = c.position.x
            is_fuel = isinstance(c, FuelComponent)

            # Circle size proportional to sqrt(mass) for visual weight
            ref_mass = c.full_mass if is_fuel else c.mass
            radius = 0.08 + 0.22 * np.sqrt(ref_mass / max_ref)

            if is_fuel:
                # Show current fill as alpha
                frac = c.mass / c.full_mass if c.full_mass > 0 else 0
                color = plt.cm.YlOrRd(0.3 + 0.7 * frac)  # type: ignore[attr-defined]
                edge = "#cc4400"
            else:
                color = "#6699cc"
                edge = "#335577"

            circle = plt.Circle(
                (x, 0), radius,
                facecolor=color, edgecolor=edge, linewidth=1.5, zorder=3,
            )
            ax.add_patch(circle)

            # Label: alternate above/below
            y_offset = 0.55 if label_above else -0.55
            va = "bottom" if label_above else "top"
            label_text = f"{c.name}\n{c.mass:.1f} kg"
            ax.text(
                x, y_offset, label_text,
                ha="center", va=va, fontsize=7,
                color="#333333",
            )
            label_above = not label_above

        # Centre of mass marker
        ax.plot(
            com.x, 0, marker="D", color="#cc0000", markersize=10,
            zorder=5, markeredgecolor="white", markeredgewidth=1.5,
        )
        ax.annotate(
            f"CoM\n{com.x:.2f} m",
            xy=(com.x, 0), xytext=(com.x, -0.85),
            ha="center", va="top", fontsize=8, fontweight="bold",
            color="#cc0000",
            arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.5),
        )

        # X-axis labels for rocket length
        ax.set_xlim(-0.8, length + 0.8)
        ax.set_ylim(-1.3, 1.1)
        ax.set_xlabel("Body X — distance from base (m)")
        ax.set_aspect("equal")
        ax.set_yticks([])

        # Tick marks at each meter
        ax.set_xticks(range(int(length) + 1))
        ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=1)

        # Legend
        ax.plot([], [], "o", color="#6699cc", label="Structural")
        ax.plot([], [], "o", color="#ee7733", label="Fuel")
        ax.plot([], [], "D", color="#cc0000", label="Centre of mass")
        ax.legend(loc="upper right", fontsize=8)

        fig.canvas.draw_idle()

    # Initial draw
    draw(1.0)

    # --- Slider ---
    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    fuel_slider = Slider(
        slider_ax, "Fuel Level",
        valmin=0.0, valmax=1.0, valinit=1.0,
        valfmt="%4.0f%%",
    )
    # Format the slider value as percentage
    fuel_slider.valtext.set_text("100%")

    def on_change(val: float):
        fuel_slider.valtext.set_text(f"{val * 100:.0f}%")
        draw(val)

    fuel_slider.on_changed(on_change)

    return fig, ax, fuel_slider


def main(output: str | None = None) -> None:
    model = default_sounding_rocket()
    fig, _ax, _slider = create_plot(model)

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved mass distribution visualisation to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive mass distribution visualizer",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Optional output file (PNG, PDF, etc.) for a static snapshot",
    )
    args = parser.parse_args()

    main(args.output)
