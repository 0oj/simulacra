"""
Hopper scenario: tall rocket (12m, 180 kg).

A 12m rocket performs three vertical hops between ground and 10m altitude,
hovering briefly at each waypoint. Thrust is scripted with pre-computed
timing based on initial mass — no controller.

Perturbation: 10 N crosswind in +X. No initial tilt — the tall rocket is
too unstable at ground contact to survive even 0.5° of tilt (high CoM at
~3.7m acts as an inverted pendulum). Wind alone causes growing lateral
drift (0.2m → 13m over 3 hops) without body tilt.

The rocket tips over only on the final uncontrolled descent after cutoff.
Compare with hopper_short.py, where the shorter rocket handles 2° tilt
and wind without tipping.

Hop profile (symmetric boost/brake at ±0.5g net):
    Pad  →  10m  →  1m  →  10m  →  1m  →  10m  →  cutoff

Run:
    cd backend
    uv run ../scenarios/hopper_long.py > output.ndjson
"""

import math
from dataclasses import replace

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
from simulacra.phenomena.actuators import ThrottleActuator, ThrottleCommand
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.schema import DebugMessage, write_message


# ── Physics ───────────────────────────────────────────────────────────────────

G = 9.81


# ── Mass model: 12m rocket, 180 kg ───────────────────────────────────────────

def make_mass_model() -> MassModel:
    """
    Tall sounding-rocket-like hopper.  180 kg total, 30 kg propellant.
    Burn rate 0.2 kg/s → 150 s of fuel → comfortably enough for 3 hops.
    Low burn rate keeps mass loss under ~3% so the open-loop profile
    stays roughly on track, with visible but not catastrophic drift.
    """
    return MassModel(components=[
        MassComponent("Nose cone",   8.0, RocketVec3(11.5, 0.0, 0.0)),
        MassComponent("Avionics",   10.0, RocketVec3(9.0,  0.0, 0.0)),
        MassComponent("Structure",  40.0, RocketVec3(6.0,  0.0, 0.0)),
        FuelComponent("Fuel tank",  30.0, RocketVec3(4.0,  0.0, 0.0),
                      full_mass=30.0, burn_rate=0.2),
        MassComponent("Engine",     70.0, RocketVec3(1.0,  0.0, 0.0)),
        MassComponent("Nozzle",     22.0, RocketVec3(0.0,  0.0, 0.0)),
    ])


# ── Rocket ────────────────────────────────────────────────────────────────────

LENGTH = 12.0

def make_rocket() -> Rocket:
    model = make_mass_model()
    props = model.to_rocket_props(
        length=LENGTH,
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )

    # Start perfectly upright. The wind provides lateral perturbation
    # without tilting the body, so the tall rocket survives ground contact.
    # Even so, each landing imparts angular momentum from lateral drift.
    state = RocketState(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=get_upright_orientation(),
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
MASS = rocket.props.mass  # 180 kg at launch

kernel = Kernel(rocket=rocket, environment=environment)
controller = IdentityController()
scenario = Scenario(kernel=kernel, controller=controller)
throttle = ThrottleActuator(rocket)


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

WEIGHT  = MASS * G            # 1765.8 N
T_HOVER = WEIGHT              # TWR 1.0
T_BOOST = WEIGHT * 1.5        # TWR 1.5  (net accel +0.5g)
T_BRAKE = WEIGHT * 0.5        # TWR 0.5  (net accel -0.5g)

A_NET = 0.5 * G               # 4.905 m/s²


# ── Pre-computed timing ──────────────────────────────────────────────────────
#
# Symmetric boost/brake: each phase covers half the distance.
# For total height H at net acceleration a: t_phase = sqrt(H / a).

def phase_time(height_m: float) -> float:
    return math.sqrt(height_m / A_NET)

T_10 = phase_time(10.0)   # 0→10m: 1.428s per phase
T_9  = phase_time(9.0)    # 1→10m or 10→1m: 1.354s per phase
T_HOLD = 2.0              # hover duration at each waypoint


# ── Flight program ───────────────────────────────────────────────────────────

t = 0.0

scenario.add_event(t, announce("Pad — 12m, 180 kg"))
scenario.add_event(t, set_wind(x=10.0))  # 10 N crosswind in +X
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
