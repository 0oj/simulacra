"""
Mid-level example: Custom graphics with setup and frame callbacks.
Shows how to customize graphics elements and add dynamic content.
"""

from ...scenario import Scenario
from ...world import WindyWorld
from ...rocket import Rocket
from ...physics import Vector
from ...graphics import run_scenario_with_graphics


def setup_graphics(renderer, fig, ax):
    """Customize graphics before animation starts."""
    # Add target altitude line
    renderer.add_custom_element(
        "hline", y=50, color="red", linestyle="--", alpha=0.7, label="Target Altitude"
    )

    # Add checkpoint line
    renderer.add_custom_element(
        "vline", x=25, color="green", linestyle=":", alpha=0.7, label="Checkpoint"
    )

    # Add launch zone text
    renderer.add_custom_element(
        "text", x=10, y=60, text="Launch Zone", fontsize=12, color="blue"
    )

    # Customize plot appearance
    renderer.customize(
        title="Rocket Simulation with Custom Graphics",
        xlabel="X Position (m)",
        ylabel="Y Position (m)",
        grid=True,
    )

    # Add legend
    ax.legend()


def frame_update(renderer, scenario):
    """Add dynamic content that updates every frame."""
    # Show current speed
    speed = scenario.rocket.velocity.magnitude
    renderer.ax.text(
        10,
        10,
        f"Speed: {speed:.1f} m/s",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    # Show current altitude
    altitude = scenario.rocket.position.y
    renderer.ax.text(
        10,
        5,
        f"Altitude: {altitude:.1f} m",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )


def main():
    """Mid-level example: Custom graphics."""
    # Create scenario
    scenario = Scenario(
        world=WindyWorld,
        rocket=Rocket(
            mass=1.0,
            thrust=Vector(x=0, y=10),
            position=Vector(x=0, y=0),
            velocity=Vector(x=0, y=0),
            drag_factor=0.1,
        ),
        dt=0.01,
    )

    # Run with custom graphics
    run_scenario_with_graphics(
        scenario,
        max_time=50.0,
        fps=30.0,
        setup_callback=setup_graphics,
        frame_callback=frame_update,
    )


if __name__ == "__main__":
    main()
