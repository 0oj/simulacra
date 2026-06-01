"""
Diagnostic: tipover dynamics after ground impact.

Drops a slightly tilted rocket from 50m with no thrust.
The rocket falls under gravity and hits the ground at ~31 m/s.
We watch how angular velocity evolves after impact.

Question: does the impact impart angular momentum, or does the
rocket tip slowly under gravity alone?

Run:
    cd backend
    uv run ../scenarios/tipover_test.py > ../temp/tipover_test.ndjson
"""

from math import radians

from simulacra.noumena import (
    Environment,
    Kernel,
    Rocket,
    RocketProps,
    RocketState,
    Scenario,
    get_upright_orientation,
)
from simulacra.onboard.controller_legacy import IdentityController
from simulacra.phenomena.sensors import Sensors
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.physics.quat import Orientation
from simulacra.schema import DebugMessage, write_message

from scipy.spatial.transform import Rotation as R


# ── Rocket: 6m, 120kg, tilted 15° from vertical ─────────────────────────────

def make_rocket() -> Rocket:
    # Start upright, then tilt 15° about Z axis
    upright = get_upright_orientation()
    tilt_rot = R.from_euler("z", radians(15))
    combined = tilt_rot * upright._as_scipy_rotation()
    x, y, z, w = combined.as_quat()
    tilt = Orientation.from_wxyz(w, x, y, z)

    props = RocketProps(
        length=6.0,
        mass=120.0,
        center_of_mass_offset=RocketVec3(3.0, 0.0, 0.0),
        inertia_diag=(10.0, 360.0, 360.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )

    state = RocketState(
        position=WorldVec3(0.0, 50.0, 0.0),   # 50m above ground
        velocity=WorldVec3(0.0, 0.0, 0.0),     # released from rest
        orientation=tilt,
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),
    )

    return Rocket(props=props, state=state)


environment = Environment(
    gravity=WorldVec3(0.0, -9.81, 0.0),
    wind=WorldVec3(0.0, 0.0, 0.0),
)

rocket = make_rocket()
kernel = Kernel(rocket=rocket, environment=environment)
sensors = Sensors(kernel)
controller = IdentityController()
scenario = Scenario(kernel=kernel, controller=controller, sensors=sensors)


def announce(text: str):
    def _cb(t, k, c):
        write_message(DebugMessage(timestamp=t, text=text))
    return _cb


scenario.add_event(0.0, announce("Drop — tilted 15deg, 50m"))


if __name__ == "__main__":
    scenario.run(duration=15.0)
