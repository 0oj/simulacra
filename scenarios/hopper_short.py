"""
Hopper scenario: short rocket (3m, 60 kg).

A squat 3m rocket performs the same three-hop profile as hopper_long.py.
Lower inertia means it's more sensitive to perturbations — any asymmetric
thrust or ground-contact torque has a bigger effect.

Perturbations:
    - 2° initial tilt from vertical (more than the long rocket)
    - 5 N crosswind in +Z (orthogonal to the long rocket's wind)

Hop profile (symmetric boost/brake at ±0.5g net):
    Pad  →  10m  →  1m  →  10m  →  1m  →  10m  →  cutoff

Run:
    cd backend
    uv run ../scenarios/hopper_short.py > output.ndjson
"""

import math
from dataclasses import replace
from math import radians

from scipy.spatial.transform import Rotation as R

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
from simulacra.phenomena.actuators import ThrottleActuator, ThrottleCommand, Actuators
from simulacra.phenomena.sensors import Sensors
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.physics.quat import Orientation
from simulacra.schema import DebugMessage, write_message


# ── Physics ───────────────────────────────────────────────────────────────────

G = 9.81


# ── Mass model: 3m rocket, 60 kg ─────────────────────────────────────────────

def make_mass_model() -> MassModel:
    """
    Compact hopper.  60 kg total, 15 kg propellant.
    Burn rate 0.1 kg/s → 150 s of fuel → plenty for 3 hop cycles.
    The short rocket has proportionally more drift than the long one
    because each kilogram of fuel is a bigger fraction of total mass.
    """
    return MassModel(components=[
        MassComponent("Nose cone",   3.0, RocketVec3(2.8, 0.0, 0.0)),
        MassComponent("Avionics",    5.0, RocketVec3(2.2, 0.0, 0.0)),
        MassComponent("Structure",  10.0, RocketVec3(1.5, 0.0, 0.0)),
        FuelComponent("Fuel tank",  15.0, RocketVec3(1.0, 0.0, 0.0),
                      full_mass=15.0, burn_rate=0.1),
        MassComponent("Engine",     20.0, RocketVec3(0.3, 0.0, 0.0)),
        MassComponent("Nozzle",      7.0, RocketVec3(0.0, 0.0, 0.0)),
    ])


# ── Rocket ────────────────────────────────────────────────────────────────────

LENGTH = 3.0

def make_rocket() -> Rocket:
    model = make_mass_model()
    props = model.to_rocket_props(
        length=LENGTH,
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )

    # Start upright, then add a 2° tilt about Y (tips nose toward +Z).
    # Different axis than the long rocket so they diverge in different directions.
    upright = get_upright_orientation()
    tilt = R.from_euler("y", radians(2)) * upright._as_scipy_rotation()
    x, y, z, w = tilt.as_quat()
    orientation = Orientation.from_wxyz(w, x, y, z)

    state = RocketState(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=orientation,
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),
    )
    return Rocket(props=props, state=state, mass_model=model)


# ── Environment ───────────────────────────────────────────────────────────────

environment = Environment(
    gravity=WorldVec3(0.0, -G, 0.0),
    wind=WorldVec3(0.0, 0.0, 0.0),
)


# ── Assemble ──────────────────────────────────────────────────────────────────

rocket = make_rocket()
MASS = rocket.props.mass  # 60 kg at launch

kernel = Kernel(rocket=rocket, environment=environment)
sensors = Sensors(kernel)
throttle = ThrottleActuator(rocket)
actuators = Actuators(throttle=throttle)
controller = IdentityController()
scenario = Scenario(kernel=kernel, controller=controller, sensors=sensors, actuators=actuators)


# ── Event helpers ─────────────────────────────────────────────────────────────

def ignite(thrust_N: float):
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        throttle.apply(ThrottleCommand(thrust_N))
    return _cb

def cutoff():
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        throttle.apply(ThrottleCommand(0.0))
    return _cb

def set_wind(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    wind = WorldVec3(x, y, z)
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        k.environment = replace(k.environment, wind=wind)
    return _cb

def announce(text: str):
    def _cb(t: float, k: Kernel, c: Controller) -> None:
        write_message(DebugMessage(timestamp=t, text=text))
    return _cb


# ── Thrust levels (based on initial mass) ────────────────────────────────────

WEIGHT  = MASS * G             # 588.6 N
T_HOVER = WEIGHT               # TWR 1.0
T_BOOST = WEIGHT * 1.5         # TWR 1.5  (net accel +0.5g)
T_BRAKE = WEIGHT * 0.5         # TWR 0.5  (net accel -0.5g)

A_NET = 0.5 * G                # 4.905 m/s²


# ── Pre-computed timing ──────────────────────────────────────────────────────

def phase_time(height_m: float) -> float:
    return math.sqrt(height_m / A_NET)

T_10 = phase_time(10.0)   # 0→10m: 1.428s per phase
T_9  = phase_time(9.0)    # 1→10m or 10→1m: 1.354s per phase
T_HOLD = 2.0              # hover duration at each waypoint


# ── Flight program ───────────────────────────────────────────────────────────

t = 0.0

scenario.add_event(t, announce("Pad — 3m, 60 kg, 2° tilt"))
scenario.add_event(t, set_wind(z=5.0))  # 5 N crosswind in +Z
t += 1.0

# ─── Hop 1: 0m → 10m ────────────────────────────────────────────────────────

scenario.add_event(t, announce("Hop 1 boost"))
scenario.add_event(t, ignite(T_BOOST))
t += T_10

scenario.add_event(t, announce("Hop 1 brake"))
scenario.add_event(t, ignite(T_BRAKE))
t += T_10

scenario.add_event(t, announce("Hold 10m"))
scenario.add_event(t, ignite(T_HOVER))
t += T_HOLD

# ─── Descend 1: 10m → 1m ────────────────────────────────────────────────────

scenario.add_event(t, announce("Descend 1"))
scenario.add_event(t, ignite(T_BRAKE))
t += T_9

scenario.add_event(t, announce("Descend 1 brake"))
scenario.add_event(t, ignite(T_BOOST))
t += T_9

scenario.add_event(t, announce("Hold 1m"))
scenario.add_event(t, ignite(T_HOVER))
t += T_HOLD

# ─── Hop 2: 1m → 10m ────────────────────────────────────────────────────────

scenario.add_event(t, announce("Hop 2 boost"))
scenario.add_event(t, ignite(T_BOOST))
t += T_9

scenario.add_event(t, announce("Hop 2 brake"))
scenario.add_event(t, ignite(T_BRAKE))
t += T_9

scenario.add_event(t, announce("Hold 10m"))
scenario.add_event(t, ignite(T_HOVER))
t += T_HOLD

# ─── Descend 2: 10m → 1m ────────────────────────────────────────────────────

scenario.add_event(t, announce("Descend 2"))
scenario.add_event(t, ignite(T_BRAKE))
t += T_9

scenario.add_event(t, announce("Descend 2 brake"))
scenario.add_event(t, ignite(T_BOOST))
t += T_9

scenario.add_event(t, announce("Hold 1m"))
scenario.add_event(t, ignite(T_HOVER))
t += T_HOLD

# ─── Hop 3: 1m → 10m ────────────────────────────────────────────────────────

scenario.add_event(t, announce("Hop 3 boost"))
scenario.add_event(t, ignite(T_BOOST))
t += T_9

scenario.add_event(t, announce("Hop 3 brake"))
scenario.add_event(t, ignite(T_BRAKE))
t += T_9

scenario.add_event(t, announce("Hold 10m"))
scenario.add_event(t, ignite(T_HOVER))
t += T_HOLD

# ─── Final cutoff ────────────────────────────────────────────────────────────

scenario.add_event(t, announce("Cutoff — coast to ground"))
scenario.add_event(t, cutoff())

DURATION = t + 10.0


if __name__ == "__main__":
    scenario.run(duration=DURATION)
