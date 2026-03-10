"""
Test the existing LandingController from various altitudes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy.spatial.transform import Rotation as R

from simulacra.noumena import (
    Kernel,
    Rocket,
    RocketState,
    Environment,
    MassModel,
    get_upright_orientation,
)
from simulacra.noumena.mass_model import MassComponent, FuelComponent
from simulacra.physics import WorldVec3, RocketVec3, Orientation
from simulacra.phenomena.sensors import Sensors
from simulacra.phenomena.actuators import Actuators, GimbalActuator, ThrottleActuator
from simulacra.onboard.landing_controller import LandingController, LandingControllerConfig


def create_rocket(altitude: float, vy: float, tilt_deg: float = 0.0):
    """Create a rocket at given altitude with given vertical velocity."""
    # Rocket parameters (matching gym_env)
    dry_mass = 60.0
    fuel_mass = 80.0
    length = 4.0
    max_thrust = 2000.0

    mass_model = MassModel(components=[
        MassComponent("dry", dry_mass, RocketVec3(length / 2, 0, 0)),
        FuelComponent(
            "fuel",
            mass=fuel_mass * 0.8,  # 80% fuel
            position=RocketVec3(length * 0.3, 0, 0),
            full_mass=fuel_mass,
            burn_rate=4.0,
        ),
    ])

    props = mass_model.to_rocket_props(
        length=length,
        thrust_application_offset=RocketVec3(0, 0, 0),
    )

    # Apply tilt
    base_rot = get_upright_orientation()._as_scipy_rotation()
    tilt = R.from_euler("z", np.radians(tilt_deg))
    combined = tilt * base_rot
    x, y, z, w = combined.as_quat()
    orientation = Orientation.from_wxyz(w, x, y, z)

    state = RocketState(
        position=WorldVec3(0.0, altitude, 0.0),
        velocity=WorldVec3(0.0, vy, 0.0),
        orientation=orientation,
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),
    )

    rocket = Rocket(props=props, state=state, mass_model=mass_model)
    return rocket, max_thrust, dry_mass, fuel_mass


def test_landing_controller(
    altitude: float,
    vy: float = 0.0,
    tilt_deg: float = 0.0,
    max_steps: int = 1000,
    dt: float = 0.02,
    debug: bool = False,
) -> dict:
    """Test landing controller from given initial conditions."""
    rocket, max_thrust, dry_mass, fuel_mass = create_rocket(altitude, vy, tilt_deg)
    env = Environment(
        gravity=WorldVec3(0.0, -9.81, 0.0),
        wind=WorldVec3(0.0, 0.0, 0.0),
        air_density=1.225,
    )

    kernel = Kernel(rocket, env)
    sensors = Sensors(kernel)

    # Create actuators
    gimbal = GimbalActuator(
        rocket=rocket,
        max_outer_angle=0.26,
        max_inner_angle=0.26,
        max_rate=1.0,
        tau=0.05,
    )
    throttle = ThrottleActuator(
        rocket=rocket,
        max_thrust_rate=max_thrust * 2,
        min_thrust=0.0,
    )
    actuators = Actuators(gimbal=gimbal, throttle=throttle)

    # Create controller
    config = LandingControllerConfig(
        dry_mass=dry_mass,
        full_fuel_mass=fuel_mass,
        max_thrust=max_thrust,
    )
    controller = LandingController(throttle, gimbal, config).with_sensors(sensors)

    # Run simulation
    time = 0.0
    for step in range(max_steps):
        sensors.set_time(time)
        actuators.set_time(time)

        # Debug output
        if debug and step % 25 == 0:
            state = kernel.rocket.state
            thrust_mag = np.sqrt(state.thrust_body.x**2 + state.thrust_body.y**2 + state.thrust_body.z**2)
            print(f"  t={time:.2f}s  y={state.position.y:.1f}m  vy={state.velocity.y:.1f}m/s  thrust={thrust_mag:.0f}N")

        # Run controller
        controller.run()

        # Step physics
        actuators.step(dt)
        kernel.step(dt)
        sensors.update_attitude_estimator(dt)

        time += dt

        # Check termination
        ground = kernel.ground_contact
        if ground.on_ground:
            state = kernel.rocket.state
            vel_mag = np.sqrt(state.velocity.x**2 + state.velocity.y**2 + state.velocity.z**2)
            if ground.crashed or ground.tipped_over:
                if debug:
                    print(f"  CRASH at t={time:.2f}s  vy={state.velocity.y:.1f}m/s  vel={vel_mag:.1f}m/s")
                return {"outcome": "crash", "time": time, "landing_vel": vel_mag}
            else:
                if debug:
                    print(f"  LANDED at t={time:.2f}s  vel={vel_mag:.1f}m/s")
                return {"outcome": "landed", "time": time, "landing_vel": vel_mag}

        # Check if flew away
        if kernel.rocket.state.position.y > altitude + 50:
            return {"outcome": "flew_away", "time": time}

    return {"outcome": "timeout", "time": time, "final_alt": kernel.rocket.state.position.y}


def main():
    print("Testing LandingController from various altitudes")
    print("=" * 60)

    # Test different altitudes
    altitudes = [20, 40, 60, 80, 100, 150, 200]
    n_trials = 20

    for alt in altitudes:
        successes = 0
        for trial in range(n_trials):
            # Random initial conditions - more challenging
            np.random.seed(trial)
            vy = np.random.uniform(-8.0, 0.0)
            tilt = np.random.uniform(-10.0, 10.0)

            result = test_landing_controller(
                altitude=float(alt),
                vy=vy,
                tilt_deg=tilt,
            )

            if result["outcome"] == "landed":
                successes += 1

        print(f"Altitude {alt:3d}m: {100*successes/n_trials:.0f}% success ({successes}/{n_trials})")

    print("=" * 60)

    # Detailed test at 50m
    print("\nDetailed test at 50m altitude:")
    for trial in range(5):
        np.random.seed(trial + 100)
        vy = np.random.uniform(-5.0, 0.0)
        tilt = np.random.uniform(-5.0, 5.0)

        result = test_landing_controller(
            altitude=50.0,
            vy=vy,
            tilt_deg=tilt,
        )
        print(f"  Trial {trial+1}: {result['outcome']:10s} (vy={vy:.1f}, tilt={tilt:.1f}deg)")

    # Single debug run
    print("\n" + "=" * 60)
    print("DEBUG: Single run from 20m, vy=-2, tilt=0")
    print("=" * 60)
    result = test_landing_controller(
        altitude=20.0,
        vy=-2.0,
        tilt_deg=0.0,
        debug=True,
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
