"""
Low-level example: Custom renderer using inheritance.
Shows how to override rendering methods for completely custom graphics.
"""

import time
import threading
from ...scenario import Scenario
from ...world import WindyWorld
from ...rocket import Rocket
from ...physics import Vector
from ...graphics import Renderer


class MyCustomRenderer(Renderer):
    """Custom renderer with overridden drawing methods."""

    def _draw_rocket(self, rocket):
        """Draw rocket as a circle with velocity vector."""
        # Draw rocket as a circle
        self.ax.scatter(
            rocket.position.x, rocket.position.y, s=200, c="red", marker="o", zorder=3
        )

        # Draw velocity vector
        scale = 2.0
        self.ax.arrow(
            rocket.position.x,
            rocket.position.y,
            rocket.velocity.x * scale,
            rocket.velocity.y * scale,
            head_width=1,
            head_length=1,
            fc="blue",
            ec="blue",
            zorder=2,
        )

    def _draw_wind(self, world):
        """Draw wind as a single large arrow in the center."""
        if world.wind.x != 0 or world.wind.y != 0:
            # Draw wind as a large arrow in the center
            self.ax.arrow(
                50,
                50,
                world.wind.x * 20,
                world.wind.y * 20,
                head_width=3,
                head_length=3,
                fc="cyan",
                ec="cyan",
                alpha=0.8,
                zorder=1,
                label="Wind",
            )

    def _draw_world_bounds(self, world):
        """Draw world bounds as a thick border."""
        x0, y0 = world.bounds[0].x, world.bounds[0].y
        x1, y1 = world.bounds[1].x, world.bounds[1].y

        # Draw thick border
        self.ax.plot(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", linewidth=3, zorder=0
        )


def main():
    """Low-level example: Custom renderer."""
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

    # Create custom renderer
    renderer = MyCustomRenderer(scenario, fps=30.0, show_plot=True)

    # Customize plot
    renderer.customise(
        title="Custom Renderer Example",
        xlabel="X Position (m)",
        ylabel="Y Position (m)",
        grid=True,
    )

    # Add custom elements
    renderer.add_custom_element(
        "hline", y=50, color="red", linestyle="--", label="Target"
    )
    renderer.add_custom_element(
        "text", x=10, y=90, text="Custom Graphics!", fontsize=16, color="purple"
    )

    print("Starting custom renderer simulation...")
    print("Press Ctrl+C to stop")

    try:
        # Manual simulation loop
        while scenario.world.current_time < 30.0:
            # Step simulation
            scenario.step()

            # Render with custom graphics
            renderer.render_frame()

            # Custom timing
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        print(f"Final altitude: {scenario.rocket.position.y:.2f}m")
        print(f"Final time: {scenario.world.current_time:.2f}s")


if __name__ == "__main__":
    main()
