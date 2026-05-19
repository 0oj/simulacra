"""
PID hover controller test scenario.

Starts rocket at 5m, commands hover at 10m. Tests with clean and noisy sensors.

Usage:
    uv run scenarios/pid_hover_test.py > hover_clean.ndjson
    uv run scenarios/pid_hover_test.py --noisy > hover_noisy.ndjson
"""

import sys
import os

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
src_dir = os.path.join(backend_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from simulacra.noumena import (
    Rocket,
    RocketProps,
    RocketState,
    Environment,
    Kernel,
    Scenario,
    get_upright_orientation,
)
from simulacra.phenomena.actuators import ThrottleActuator, Actuators
from simulacra.phenomena.sensors import Sensors
from simulacra.phenomena.errors import GaussianNoise, Bias, CompositeError
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.pid_hover import PIDHoverController


def make_scenario(noisy: bool = False) -> Scenario:
    mass = 100.0  # kg
    gravity = 9.81  # m/s^2
    weight = mass * gravity
    max_thrust = 2000.0  # N

    rocket = Rocket(
        props=RocketProps(
            length=4.0,
            mass=mass,
            center_of_mass_offset=RocketVec3(2.0, 0.0, 0.0),
            inertia_diag=(50.0, 50.0, 10.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        ),
        state=RocketState(
            position=WorldVec3(0.0, 5.0, 0.0),  # Start at 5m
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=get_upright_orientation(),
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(weight, 0.0, 0.0),  # Start at hover thrust
        ),
    )

    env = Environment(gravity=WorldVec3(0.0, -gravity, 0.0))
    kernel = Kernel(rocket=rocket, environment=env)

    # Sensors: clean or noisy
    if noisy:
        sensors = Sensors(
            kernel,
            gps_errors={"y": GaussianNoise(std=0.5, seed=42)},  # 0.5m altitude noise
            velocity_errors={"y": GaussianNoise(std=0.2, seed=43)},  # 0.2 m/s noise
        )
    else:
        sensors = Sensors(kernel)

    # Actuators with thrust misalignment
    # Real engines have combustion instabilities that cause random thrust direction jitter.
    # Without attitude control (TVC), even this causes gradual attitude drift.
    # Using only noise (no bias) so the rocket doesn't systematically diverge.
    import math
    # Very small noise to show hover works, but drift is visible over time.
    # Real rockets need TVC to counteract even this level of disturbance.
    alignment_errors = {
        "yaw": GaussianNoise(std=math.radians(0.01), seed=44),  # 0.01 deg noise
        "pitch": GaussianNoise(std=math.radians(0.01), seed=45),  # 0.01 deg noise
    }
    throttle = ThrottleActuator(rocket, alignment_errors=alignment_errors)
    actuators = Actuators(throttle=throttle)

    # Controller
    # For critical damping: Kd = 2 * sqrt(Kp * mass)
    # With Kp=100, mass=100: Kd = 2 * sqrt(10000) = 200
    controller = PIDHoverController(
        throttle=throttle,
        target_alt=10.0,  # Hover at 10m
        weight=weight,
        Kp=100.0,
        Kd=200.0,
        max_thrust=max_thrust,
    )

    return Scenario(
        kernel=kernel,
        controller=controller,
        sensors=sensors,
        actuators=actuators,
        dt=0.02,
    )


if __name__ == "__main__":
    import pathlib

    noisy = "--noisy" in sys.argv
    scenario = make_scenario(noisy=noisy)

    # Output to /tmp/simulacra/ for frontend preloading
    output_dir = pathlib.Path("/tmp/simulacra")
    output_dir.mkdir(exist_ok=True)
    suffix = "noisy" if noisy else "clean"
    output_path = output_dir / f"hover_{suffix}.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=10.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
