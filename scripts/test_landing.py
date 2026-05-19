#!/usr/bin/env python3
"""
Test landing scenarios with different controllers.

Generates NDJSON output files for visualization in the frontend.
"""

import math
import os
import sys

from simulacra.landing_env import LandingScenarioGenerator, LandingScenarioConfig
from simulacra.noumena import Environment, Kernel, Scenario
from simulacra.phenomena.actuators import GimbalActuator, ThrottleActuator, Actuators
from simulacra.phenomena.sensors import Sensors
from simulacra.physics import WorldVec3, RocketVec3
from simulacra.onboard.hover_controller import HoverController, HoverControllerConfig
from simulacra.onboard.landing_controller import LandingController, LandingControllerConfig


def run_landing_test(
    controller_type: str = "landing",
    seed: int = 42,
    duration: float = 25.0,
) -> str:
    """
    Run a landing test and write output to file.

    Args:
        controller_type: "landing" or "hover"
        seed: Random seed for scenario generation
        duration: How long to run (seconds)

    Returns:
        Path to output file
    """
    print(f"=" * 60)
    print(f"Landing Test: {controller_type.upper()} controller")
    print(f"=" * 60)

    # Generate scenario
    config = LandingScenarioConfig(
        altitude_range=(100.0, 100.0),
        horizontal_range=(-15.0, 15.0),
        vertical_velocity_range=(-1.0, 1.0),
        horizontal_velocity_range=(-3.0, 3.0),
        tilt_range=(-0.1, 0.1),
        dry_mass=60.0,
        fuel_mass=80.0,  # Plenty of fuel for propulsive landing
        max_thrust=2000.0,
        fuel_burn_rate=4.0,  # kg/s - gives ~20s of full thrust
    )

    generator = LandingScenarioGenerator(config, seed=seed)
    rocket, info = generator.generate()

    print(f"\nInitial conditions:")
    print(f"  Position: ({info['initial_position'][0]:.1f}, "
          f"{info['initial_position'][1]:.1f}, "
          f"{info['initial_position'][2]:.1f}) m")
    print(f"  Velocity: ({info['initial_velocity'][0]:.1f}, "
          f"{info['initial_velocity'][1]:.1f}, "
          f"{info['initial_velocity'][2]:.1f}) m/s")
    print(f"  Distance to pad: {info['distance_to_pad']:.1f} m")

    # Setup simulation
    env = Environment(gravity=WorldVec3(0, -9.81, 0))
    kernel = Kernel(rocket, env)
    sensors = Sensors(kernel)

    gimbal = GimbalActuator(
        rocket=rocket,
        max_outer_angle=math.radians(15),
        max_inner_angle=math.radians(15),
        max_rate=math.radians(60),
        tau=0.05,
    )
    throttle = ThrottleActuator(
        rocket=rocket,
        max_thrust_rate=4000,
    )
    actuators = Actuators(gimbal=gimbal, throttle=throttle)

    # Create controller
    if controller_type == "landing":
        controller = LandingController(
            throttle=throttle,
            gimbal=gimbal,
            config=LandingControllerConfig(
                target_x=0.0,
                target_z=0.0,
                weight=config.mass * 9.81,
                max_thrust=config.max_thrust,
                gravity=9.81,
                thrust_moment_arm=config.length / 2,
                k_descent=3.0,  # Faster descent = more fuel-efficient
                k_horizontal=0.3,
                max_horizontal_velocity=6.0,
                terminal_altitude=10.0,  # Start terminal phase higher
                terminal_descent_rate=2.0,  # Faster terminal descent
                att_Kp=300.0,
                att_Kd=150.0,
            ),
        )
    else:
        controller = HoverController(
            throttle=throttle,
            gimbal=gimbal,
            config=HoverControllerConfig(
                target_alt=10.0,  # Will hover, not land
                weight=config.mass * 9.81,
                max_thrust=config.max_thrust,
                gravity=9.81,
                thrust_moment_arm=config.length / 2,
            ),
        )

    scenario = Scenario(
        kernel=kernel,
        controller=controller,
        sensors=sensors,
        actuators=actuators,
        dt=0.02,
    )

    # Output to file
    os.makedirs("/tmp/simulacra", exist_ok=True)
    output_path = f"/tmp/simulacra/landing_{controller_type}.ndjson"

    print(f"\nRunning for {duration}s...")

    with open(output_path, "w") as f:
        scenario.run(duration=duration, output=f)

    # Report final state
    state = kernel.rocket.state
    ground = kernel.ground_contact

    pos = state.position
    vel = state.velocity
    dist = math.sqrt(pos.x**2 + pos.z**2)

    nose_world = RocketVec3(1, 0, 0).to_world(state.orientation)
    tilt_deg = math.degrees(math.acos(max(-1, min(1, nose_world.y))))

    print(f"\nFinal state:")
    print(f"  Position: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}) m")
    print(f"  Velocity: ({vel.x:.1f}, {vel.y:.1f}, {vel.z:.1f}) m/s")
    print(f"  Speed: {vel.norm:.1f} m/s")
    print(f"  Tilt: {tilt_deg:.1f}°")
    print(f"  Distance to pad: {dist:.1f} m")

    # Landing outcome
    if ground.on_ground:
        if ground.crashed:
            outcome = "CRASHED (impact too hard)"
        elif ground.tipped_over:
            outcome = "TIPPED OVER (too much tilt)"
        else:
            outcome = "SOFT LANDING"
        print(f"\n  Outcome: {outcome}")
        print(f"  Impact speed: {ground.impact_speed:.2f} m/s (max: {ground.MAX_LANDING_SPEED})")
        print(f"  Landing angle: {math.degrees(ground.landing_angle):.1f}° (max: {math.degrees(ground.MAX_LANDING_TILT):.0f}°)")

    print(f"\nOutput: {output_path}")

    return output_path


if __name__ == "__main__":
    # Parse args
    controller = "landing"
    if "--hover" in sys.argv:
        controller = "hover"

    seed = 42
    for arg in sys.argv:
        if arg.startswith("--seed="):
            seed = int(arg.split("=")[1])

    run_landing_test(controller_type=controller, seed=seed)

    print("\n" + "=" * 60)
    print("View in browser:")
    print("  cd frontend && npm run dev")
    print(f"  Open: http://localhost:5173/?file=landing_{controller}.ndjson")
    print("=" * 60)
