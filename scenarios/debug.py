"""
Canonical 3D rocket flight scenario.

A small sounding rocket launches vertically, performs a pitch-over maneuver,
encounters a crosswind, applies a gimbal correction, then coasts to apogee
and descends back to the ground.

Flight profile:
     0-1s   Pad            Rocket on the ground, no thrust
     1s     Liftoff        Engine ignites at 2 400 N (TWR ≈ 2.04)
     3-5s   Pitch-over     Matched gimbal pair tilts rocket ~12° toward +X
     7s     Crosswind      80 N lateral force in the +Z direction
     8-10s  Correction     Matched gimbal pair steers into the wind
    11s     MECO           Main engine cutoff
    11-38s  Coast          Ballistic arc under gravity + wind → impact

Each gimbal maneuver uses a matched pair: a kick to start rotation, then an
equal counter-kick to arrest it. This leaves the rocket tilted but not
spinning — the same strategy real rockets use, since there is no aerodynamic
damping in this simulation.

Run:
    cd backend
    uv run ../scenarios/debug.py > output.ndjson
    uv run -m examples.plot_wasd_3d output.ndjson
"""

from dataclasses import replace
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
from simulacra.onboard.controller import Controller, IdentityController
from simulacra.phenomena.actuators import (
    GimbalActuator,
    GimbalCommand,
    ThrottleActuator,
    ThrottleCommand,
)
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.schema import DebugMessage, write_message


# ── Rocket definition ─────────────────────────────────────────────────────────

def make_rocket() -> Rocket:
    props = RocketProps(
        length=6.0,                                        # m
        mass=120.0,                                        # kg
        center_of_mass_offset=RocketVec3(3.0, 0.0, 0.0),  # 3 m from base along body +X
        inertia_diag=(10.0, 360.0, 360.0),                 # (Ix, Iy, Iz) kg·m²
        #   Ix=10  : roll (about long axis) — small for a slender cylinder
        #   Iy=360 : pitch — large; matches m·L²/12 for a 6 m, 120 kg uniform rod
        #   Iz=360 : yaw  — same as pitch (axially symmetric)
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),  # nozzle at the base
    )

    state = RocketState(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=get_upright_orientation(),              # body +X → world +Y (nose up)
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),            # no thrust at start
    )

    return Rocket(props=props, state=state)


# ── Environment ───────────────────────────────────────────────────────────────

environment = Environment(
    gravity=WorldVec3(0.0, -9.81, 0.0),  # Earth surface gravity
    wind=WorldVec3(0.0, 0.0, 0.0),       # calm at launch
)


# ── Assemble scenario ─────────────────────────────────────────────────────────

rocket = make_rocket()
kernel = Kernel(rocket=rocket, environment=environment)
controller = IdentityController()
scenario = Scenario(kernel=kernel, controller=controller)

throttle = ThrottleActuator(rocket)
gimbal_actuator = GimbalActuator(
    rocket,
    max_outer_angle=radians(15.0),
    max_inner_angle=radians(15.0),
)


# ── Event helpers ─────────────────────────────────────────────────────────────

def ignite(thrust_N: float):
    """Set thrust magnitude (Newtons) via the throttle actuator."""
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        throttle.apply(ThrottleCommand(thrust_N))
    return _cb


def cutoff():
    """Set thrust to zero via the throttle actuator."""
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        throttle.apply(ThrottleCommand(0.0))
    return _cb


def gimbal(pitch_deg: float = 0.0, yaw_deg: float = 0.0):
    """
    Deflect the nozzle via the gimbal actuator.

    yaw_deg:   outer gimbal angle about body +Z
    pitch_deg: inner gimbal angle about the outer frame's +Y
    """
    cmd = GimbalCommand(
        desired_outer_angle=radians(yaw_deg),
        desired_inner_angle=radians(pitch_deg),
    )
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        gimbal_actuator.apply(cmd)
    return _cb


def set_wind(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    """
    Set a constant wind force (Newtons) in the world frame.

    Note: the current kernel treats wind as a direct additive force, not as
    aerodynamic drag proportional to relative velocity. The values here are
    force magnitudes, not wind speeds.
    """
    wind = WorldVec3(x, y, z)
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        k.environment = replace(k.environment, wind=wind)
    return _cb


def announce(text: str):
    """Emit a DebugMessage at the current simulation time."""
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        write_message(DebugMessage(timestamp=t, text=text))
    return _cb


# ── Flight program ────────────────────────────────────────────────────────────

# Phase 1: Pad → Liftoff
scenario.add_event(0.0, announce("Pad"))
scenario.add_event(1.0, announce("Liftoff — 2400 N"))
scenario.add_event(1.0, ignite(2400.0))

# Phase 2: Pitch-over — matched gimbal pair tilts rocket toward world +X.
#   Kick for 1 s to build angular velocity, then counter-kick for 1 s to
#   arrest rotation. The rocket ends up tilted ~12° from vertical with ω ≈ 0.
scenario.add_event(3.0, announce("Pitch-over"))
scenario.add_event(3.0, gimbal(yaw_deg=0.5))       # start rotating
scenario.add_event(4.0, gimbal(yaw_deg=-0.5))      # arrest rotation
scenario.add_event(5.0, gimbal())                   # nozzle back to center

# Phase 3: Crosswind — 80 N lateral force in world +Z
scenario.add_event(7.0, announce("Crosswind +Z 80 N"))
scenario.add_event(7.0, set_wind(z=80.0))

# Phase 4: Wind correction — matched gimbal pair steers into the wind.
scenario.add_event(8.0, announce("Wind correction"))
scenario.add_event(8.0, gimbal(pitch_deg=0.3))      # start correction
scenario.add_event(9.0, gimbal(pitch_deg=-0.3))     # arrest correction
scenario.add_event(10.0, gimbal())                   # nozzle back to center

# Phase 5: Main engine cutoff → Coast
scenario.add_event(11.0, announce("MECO"))
scenario.add_event(11.0, cutoff())
scenario.add_event(11.01, announce("Coast"))

# Coast, apogee, and descent happen naturally under gravity + wind.

if __name__ == "__main__":
    scenario.run(duration=38.0)
