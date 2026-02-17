"""
Demo: Suicide Burn Landing

Rocket falls from height, then performs a last-second "suicide burn" to land softly.
Demonstrates: Optimal landing trajectory, thrust timing.

Setup:
- Rocket at 100m, falling
- At calculated altitude, full thrust to decelerate
- Goal: touch down at ~0 m/s

Physics:
- Falling from 100m: reaches ~44 m/s at ground if no thrust
- Need to time thrust to stop exactly at ground
- Suicide burn altitude: h = v²/(2*(T/m - g))

Usage:
    uv run scenarios/demos/landing.py
"""

import sys
import os
import math
import pathlib

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(script_dir))
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
from simulacra.phenomena.actuators import ThrottleActuator, Actuators, ThrottleCommand
from simulacra.phenomena.sensors import Sensors
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.controller import Controller


class SuicideBurnController(Controller):
    """
    Simple suicide burn: full thrust when altitude drops below burn altitude.

    Burn altitude calculated so that constant deceleration brings v=0 at h=0.
    h_burn = v² / (2 * a_net) where a_net = (T/m - g)
    """

    def __init__(self, throttle, mass, gravity, max_thrust):
        self.throttle = throttle
        self.mass = mass
        self.gravity = gravity
        self.max_thrust = max_thrust
        self.sensors = None

        # Net deceleration when thrusting
        self.a_net = (max_thrust / mass) - gravity  # m/s²

    def with_sensors(self, sensors):
        self.sensors = sensors
        return self

    def run(self):
        if not self.sensors:
            return

        pos = self.sensors.GPS.read()
        vel = self.sensors.VelocitySensor.read()

        altitude = pos.y
        velocity = vel.y  # Negative when falling

        # Calculate burn altitude for current velocity
        if velocity < 0:  # Falling
            speed = abs(velocity)
            burn_altitude = (speed * speed) / (2 * self.a_net)

            # Add margin for safety
            burn_altitude *= 1.1

            if altitude <= burn_altitude:
                # BURN!
                self.throttle.apply(ThrottleCommand(thrust_N=self.max_thrust))
            else:
                # Coast (free fall)
                self.throttle.apply(ThrottleCommand(thrust_N=0.0))
        else:
            # Rising or stationary - no thrust needed
            # (or we've landed)
            if altitude < 0.5:
                # Near ground and not falling fast - hover
                weight = self.mass * self.gravity
                self.throttle.apply(ThrottleCommand(thrust_N=weight))
            else:
                self.throttle.apply(ThrottleCommand(thrust_N=0.0))


def make_scenario() -> Scenario:
    mass = 100.0
    gravity = 9.81
    max_thrust = 2500.0  # Enough to decelerate (TWR > 1)

    rocket = Rocket(
        props=RocketProps(
            length=4.0,
            mass=mass,
            center_of_mass_offset=RocketVec3(2.0, 0.0, 0.0),
            inertia_diag=(50.0, 50.0, 10.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        ),
        state=RocketState(
            position=WorldVec3(0.0, 100.0, 0.0),  # 100m up
            velocity=WorldVec3(0.0, 0.0, 0.0),     # Start at rest
            orientation=get_upright_orientation(),
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(0.0, 0.0, 0.0),
        ),
    )

    env = Environment(gravity=WorldVec3(0.0, -gravity, 0.0))
    kernel = Kernel(rocket=rocket, environment=env)
    sensors = Sensors(kernel)

    throttle = ThrottleActuator(rocket)
    actuators = Actuators(throttle=throttle)

    controller = SuicideBurnController(
        throttle=throttle,
        mass=mass,
        gravity=gravity,
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
    scenario = make_scenario()

    output_dir = pathlib.Path("/tmp/simulacra")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "demo_landing.ndjson"

    with open(output_path, "w") as f:
        scenario.run(duration=10.0, output=f)

    print(f"Wrote {output_path}", file=sys.stderr)
