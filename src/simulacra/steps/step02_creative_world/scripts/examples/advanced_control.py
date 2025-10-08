"""
Low-level example: Manual control over simulation and rendering.
Shows how to have full control over timing, logic, and rendering.
"""

import time
import threading
from ...scenario import Scenario
from ...world import WindyWorld
from ...rocket import Rocket
from ...physics import Vector
from ...graphics import Renderer


def main():
    """Low-level example: Manual control."""
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

    # Create renderer
    renderer = Renderer(scenario, fps=30.0, show_plot=True)

    # Customize renderer
    renderer.customise(
        title="Manual Control Simulation",
        xlabel="X Position (m)",
        ylabel="Y Position (m)",
        grid=True,
    )

    # Add custom elements
    renderer.add_custom_element(
        "hline", y=50, color="red", linestyle="--", label="Target"
    )

    print("Starting manual control simulation...")
    print("Press Ctrl+C to stop")

    try:
        # Manual simulation loop
        while scenario.world.current_time < 50.0:
            # Step simulation
            scenario.step()

            # Custom logic: Check for milestones
            altitude = scenario.rocket.position.y
            if altitude > 25 and not hasattr(scenario, "_milestone_25"):
                print(f"Milestone: Reached 25m at t={scenario.world.current_time:.2f}s")
                scenario._milestone_25 = True

            if altitude > 50 and not hasattr(scenario, "_milestone_50"):
                print(f"Milestone: Reached 50m at t={scenario.world.current_time:.2f}s")
                scenario._milestone_50 = True

            # Custom rendering: Only render every 10 steps for performance
            if int(scenario.world.current_time * 100) % 10 == 0:
                renderer.render_frame()

            # Custom timing: Variable sleep based on altitude
            if altitude > 30:
                time.sleep(0.001)  # Faster when high
            else:
                time.sleep(0.01)  # Slower when low

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        print(f"Final altitude: {scenario.rocket.position.y:.2f}m")
        print(f"Final time: {scenario.world.current_time:.2f}s")


if __name__ == "__main__":
    main()
