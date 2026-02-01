"""
Sounding rocket with a compositional mass model.

Uses default_sounding_rocket() so that mass, centre of mass, and inertia
evolve dynamically as fuel burns.  Compare with debug.py, which uses
hand-specified constant RocketProps.

Mass model (120 kg, 6 m):
    50 kg propellant (oxidizer 20 kg @ 4 kg/s, fuel 30 kg @ 6 kg/s)
    70 kg dry mass
    Burn duration: 5 s at full throttle (10 kg/s combined)

Aerodynamics:
    Translational drag  F = -½ ρ Cd A |v| v  (default Cd=0.5, A≈0.03 m²)
    reduces apogee compared to a vacuum trajectory.

Flight profile:
     0-1s    Pad           Sitting on the ground, no thrust
     1s      Liftoff       Engine ignites at 2 400 N  (TWR ≈ 2.04 at launch)
     3-5s    Pitch-over    Matched gimbal pair tilts rocket ~12° toward +X
     6s      MECO          Main engine cutoff — fuel exhausted
     6-30s   Coast         Ballistic arc under gravity → impact & tip-over

As fuel burns the TWR climbs from ~2.0 to ~3.5, the CoM shifts aft (toward
the engine), and the pitch/yaw inertia drops. All of this happens
automatically through the MassModel → Kernel integration.

On landing, the rocket tips over like an inverted pendulum rather than
freezing upright.

Run:
    cd backend
    uv run ../scenarios/sounding_rocket.py > output.ndjson
    uv run -m examples.plot_wasd_3d output.ndjson
"""

from dataclasses import replace
from math import radians

from simulacra.noumena import (
    Environment,
    Kernel,
    Rocket,
    RocketState,
    Scenario,
    default_sounding_rocket,
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


# -- Rocket definition (mass-model driven) ------------------------------------

def make_rocket() -> Rocket:
    model = default_sounding_rocket()

    # Build initial props from the mass distribution.
    # Thrust is applied at the nozzle (base of the rocket, body x=0).
    props = model.to_rocket_props(
        length=6.0,
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )

    state = RocketState(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),
    )

    return Rocket(props=props, state=state, mass_model=model)


# -- Environment ---------------------------------------------------------------

environment = Environment(
    gravity=WorldVec3(0.0, -9.81, 0.0),
    wind=WorldVec3(0.0, 0.0, 0.0),
)


# -- Assemble scenario ---------------------------------------------------------

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


# -- Event helpers -------------------------------------------------------------

def ignite(thrust_N: float):
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        throttle.apply(ThrottleCommand(thrust_N))
    return _cb


def cutoff():
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        throttle.apply(ThrottleCommand(0.0))
    return _cb


def gimbal(pitch_deg: float = 0.0, yaw_deg: float = 0.0):
    cmd = GimbalCommand(
        desired_outer_angle=radians(yaw_deg),
        desired_inner_angle=radians(pitch_deg),
    )
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        gimbal_actuator.apply(cmd)
    return _cb


def announce(text: str):
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        write_message(DebugMessage(timestamp=t, text=text))
    return _cb


# -- Flight program ------------------------------------------------------------

# Phase 1: Pad -> Liftoff
scenario.add_event(0.0, announce("Pad"))
scenario.add_event(1.0, announce("Liftoff — 2400 N"))
scenario.add_event(1.0, ignite(2400.0))

# Phase 2: Pitch-over — matched gimbal pair
scenario.add_event(3.0, announce("Pitch-over"))
scenario.add_event(3.0, gimbal(yaw_deg=0.5))
scenario.add_event(4.0, gimbal(yaw_deg=-0.5))
scenario.add_event(5.0, gimbal())

# Phase 3: MECO — fuel runs out at ~6 s (50 kg / 10 kg/s = 5 s burn)
scenario.add_event(6.0, announce("MECO"))
scenario.add_event(6.0, cutoff())
scenario.add_event(6.01, announce("Coast"))

# Coast to apogee and descent happen under gravity.

if __name__ == "__main__":
    scenario.run(duration=30.0)
