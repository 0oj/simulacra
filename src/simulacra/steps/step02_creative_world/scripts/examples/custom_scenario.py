"""
Mid-level example: Scenario callbacks and custom stopping conditions.
Shows how to add custom logic during simulation and use custom stopping conditions.
"""

from ...scenario import Scenario
from ...world import WindyWorld
from ...rocket import Rocket
from ...physics import Vector
from ...graphics import run_scenario_with_graphics


def log_altitude(scenario):
    """Log rocket altitude every step."""
    altitude = scenario.rocket.position.y
    time = scenario.world.current_time
    print(f"t={time:.2f}s: Altitude={altitude:.2f}m")


def log_velocity(scenario):
    """Log rocket velocity every step."""
    velocity = scenario.rocket.velocity.magnitude
    time = scenario.world.current_time
    print(f"t={time:.2f}s: Velocity={velocity:.2f}m/s")


def save_results(scenario):
    """Save results when simulation completes."""
    final_altitude = scenario.rocket.position.y
    final_velocity = scenario.rocket.velocity.magnitude
    final_time = scenario.world.current_time

    print(f"\n=== Simulation Results ===")
    print(f"Final time: {final_time:.2f}s")
    print(f"Final altitude: {final_altitude:.2f}m")
    print(f"Final velocity: {final_velocity:.2f}m/s")


def main():
    """Mid-level example: Scenario callbacks."""
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

    print("=== Running with callbacks ===")
    # Run with callbacks
    scenario.run(max_time=10.0, on_step=log_altitude, on_complete=save_results)

    print("\n=== Running until condition ===")
    # Reset scenario
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

    # Run until rocket reaches altitude 30
    scenario.run_until(lambda s: s.rocket.position.y >= 30)

    print(f"Rocket reached altitude 30 at t={scenario.world.current_time:.2f}s")


if __name__ == "__main__":
    main()
