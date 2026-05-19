"""
Engine-heavy rocket: most dry mass at the base.

Demonstrates how mass distribution affects flight dynamics.
Compare with nose_heavy.py — same total mass, same fuel, same flight
program, but the dry mass is concentrated at opposite ends.

Mass model (120 kg, 6 m):
    50 kg propellant (oxidizer 20 kg @ 4 kg/s, fuel 30 kg @ 6 kg/s)
    70 kg dry mass — concentrated at the base (50 kg engine + 10 kg nozzle)
    CoM starts at ~1.5 m, shifts to ~0.9 m as fuel burns

Expected behaviour:
    - Weak gimbal authority (short lever arm: nozzle → CoM ≈ 1.5 m)
    - Sluggish pitch-over from the same gimbal deflection
    - Slow tip-over on landing (heavy base keeps gravity torque small)

Run:
    cd backend
    uv run ../scenarios/engine_heavy.py > engine_heavy.ndjson
    uv run -m examples.plot_wasd_3d engine_heavy.ndjson
"""

from math import radians

from simulacra.noumena import (
    Environment,
    FuelComponent,
    Kernel,
    MassComponent,
    MassModel,
    Rocket,
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


# -- Mass model: engine-heavy --------------------------------------------------

def engine_heavy_model() -> MassModel:
    """
    120 kg rocket, 6 m long.  60 kg of the 70 kg dry mass is at the base.
    CoM ≈ 1.5 m from base at full fuel, ≈ 0.9 m when empty.
    """
    return MassModel(components=[
        MassComponent("Nose cone",        5.0, RocketVec3(5.5, 0.0, 0.0)),
        MassComponent("Avionics",         5.0, RocketVec3(4.5, 0.0, 0.0)),
        FuelComponent("Oxidizer tank",   20.0, RocketVec3(3.0, 0.0, 0.0),
                       full_mass=20.0, burn_rate=4.0),
        FuelComponent("Fuel tank",       30.0, RocketVec3(2.0, 0.0, 0.0),
                       full_mass=30.0, burn_rate=6.0),
        MassComponent("Engine",          50.0, RocketVec3(0.3, 0.0, 0.0)),
        MassComponent("Nozzle",          10.0, RocketVec3(0.0, 0.0, 0.0)),
    ])


# -- Rocket definition ---------------------------------------------------------

def make_rocket() -> Rocket:
    model = engine_heavy_model()
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


# -- Event helpers (identical to nose_heavy.py) --------------------------------

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


# -- Flight program (identical to nose_heavy.py) -------------------------------

scenario.add_event(0.0, announce("Pad — engine-heavy rocket"))
scenario.add_event(1.0, announce("Liftoff — 2400 N"))
scenario.add_event(1.0, ignite(2400.0))

# Pitch-over: same matched gimbal pair in both scenarios
scenario.add_event(3.0, announce("Pitch-over"))
scenario.add_event(3.0, gimbal(yaw_deg=2.0))
scenario.add_event(4.0, gimbal(yaw_deg=-2.0))
scenario.add_event(5.0, gimbal())

scenario.add_event(6.0, announce("MECO"))
scenario.add_event(6.0, cutoff())
scenario.add_event(6.01, announce("Coast"))

if __name__ == "__main__":
    scenario.run(duration=30.0)
