#!/usr/bin/env python3
"""
Diagnose safety envelope signals for PID controller flights.

Runs the PID HopperController through several flight scenarios and measures
the safety envelope signals (actuator saturation, speed ratio, tilt margin,
descent rate) to calibrate r_safety penalty weights.

If PID already triggers heavy penalties, the weights are too aggressive.

Usage:
    cd backend
    uv run python scripts/diagnose_safety_envelope.py
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from simulacra.noumena import (
    Rocket, RocketState, Environment, Kernel, MassModel,
    get_upright_orientation,
)
from simulacra.noumena.mass_model import MassComponent, FuelComponent
from simulacra.phenomena.actuators import Actuators, GimbalActuator, ThrottleActuator
from simulacra.phenomena.sensors import Sensors
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.hopper_controller import HopperController, HopperConfig, Waypoint
from simulacra.onboard.safety_envelope import SafetyEnvelope, default_checks


def make_rocket(pos_y=0.0, vel_y=0.0, vel_x=0.0, fuel_frac=1.0):
    """Create rocket at given state."""
    dry_mass = 60.0
    fuel_mass = 80.0
    length = 4.0
    burn_rate = 0.68

    mass_model = MassModel(components=[
        MassComponent("dry", dry_mass, RocketVec3(length / 2, 0, 0)),
        FuelComponent(
            "fuel",
            mass=fuel_mass * fuel_frac,
            position=RocketVec3(length * 0.3, 0, 0),
            full_mass=fuel_mass,
            burn_rate=burn_rate,
        ),
    ])
    props = mass_model.to_rocket_props(
        length=length,
        thrust_application_offset=RocketVec3(0, 0, 0),
    )
    state = RocketState(
        position=WorldVec3(0.0, pos_y, 0.0),
        velocity=WorldVec3(vel_x, vel_y, 0.0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),
    )
    return Rocket(props=props, state=state, mass_model=mass_model)


def run_pid_flight(name, waypoints, rocket_kwargs=None, duration=60.0):
    """Run a PID flight and collect safety envelope diagnostics."""
    rocket_kwargs = rocket_kwargs or {}
    rocket = make_rocket(**rocket_kwargs)
    environment = Environment(
        gravity=WorldVec3(0.0, -9.81, 0.0),
        wind=WorldVec3(0.0, 0.0, 0.0),
        air_density=1.225,
    )
    kernel = Kernel(rocket, environment)
    actuators = make_actuators(rocket)
    sensors = Sensors(kernel)

    controller = HopperController(
        throttle=actuators._throttle,
        gimbal=actuators._gimbal,
        waypoints=waypoints,
        config=HopperConfig(),
    )
    controller.with_sensors(sensors)

    envelope = SafetyEnvelope(
        throttle=actuators._throttle,
        gimbal=actuators._gimbal,
        checks=default_checks(),
    )

    dt = 0.02
    steps = int(duration / dt)

    # Collectors
    all_signals = []
    all_penalties = []
    thrust_cmds = []
    thrust_actuals = []

    for step in range(steps):
        t = step * dt
        sensors.set_time(t)

        # Run controller
        controller.run()

        # Read actuator state before stepping
        thrust_cmds.append(actuators._throttle._commanded_thrust)

        # Step actuators and physics
        actuators.set_time(t)
        actuators.step(dt)
        kernel.step(dt)

        thrust_actuals.append(actuators._throttle.actual_thrust)

        # Compute safety envelope
        s = kernel.rocket.state
        vel = np.array([s.velocity.x, s.velocity.y, s.velocity.z])
        speed = float(np.linalg.norm(vel))
        alt = s.position.y

        # v_safe from brake margin
        total_mass = kernel.rocket.props.mass
        g = 9.81
        a_max = max(2000.0 / total_mass - g, 1.0)
        dist = float(np.linalg.norm(
            controller.current_target - np.array([s.position.x, s.position.y, s.position.z])
        ))
        v_safe = math.sqrt(2 * a_max * max(0.1, dist))

        # Tilt from quaternion: R[1,0] = 2*(qx*qy + qw*qz)
        q = s.orientation.as_array()  # [w, x, y, z]
        nose_y = 2 * (q[1] * q[2] + q[0] * q[3])
        tilt = math.acos(max(-1.0, min(1.0, nose_y)))

        envelope.update_state(
            speed=speed, v_safe=v_safe, tilt=tilt,
            max_tilt=1.0, vy=s.velocity.y, alt=alt,
        )

        signals = envelope.read()
        penalty = envelope.read_penalty()
        all_signals.append(signals.copy())
        all_penalties.append(penalty)

        if controller.finished:
            break

    signals_arr = np.array(all_signals)  # (T, 6)
    penalties_arr = np.array(all_penalties)  # (T,)
    thrust_cmds = np.array(thrust_cmds)
    thrust_actuals = np.array(thrust_actuals)

    # Thrust reversals
    thrust_diff = np.diff(thrust_actuals)
    reversals = int(np.sum(np.diff(np.sign(thrust_diff)) != 0))

    return {
        "name": name,
        "steps": len(all_signals),
        "signals": signals_arr,
        "penalties": penalties_arr,
        "thrust_cmds": thrust_cmds,
        "thrust_actuals": thrust_actuals,
        "reversals": reversals,
    }


def make_actuators(rocket):
    gimbal = GimbalActuator(
        rocket=rocket,
        max_outer_angle=0.26,
        max_inner_angle=0.26,
        max_rate=1.0,
        tau=0.05,
    )
    throttle = ThrottleActuator(
        rocket=rocket,
        max_thrust_rate=4000.0,
        min_thrust=0.0,
    )
    return Actuators(gimbal=gimbal, throttle=throttle)


SIGNAL_NAMES = [
    "throttle_sat", "gimbal_outer_sat", "gimbal_inner_sat",
    "speed_ratio", "tilt_margin", "descent_ratio",
]

SCENARIOS = [
    ("liftoff_hover", [
        Waypoint(position=np.array([0.0, 20.0, 0.0]), hold_time=5.0),
    ], {"pos_y": 0.5}, 30.0),

    ("vertical_descent", [
        Waypoint(position=np.array([0.0, 1.0, 0.0]), hold_time=0.0),
    ], {"pos_y": 40.0, "vel_y": -3.0}, 30.0),

    ("lateral_move", [
        Waypoint(position=np.array([20.0, 20.0, 0.0]), hold_time=2.0),
        Waypoint(position=np.array([0.0, 20.0, 0.0]), hold_time=2.0),
    ], {"pos_y": 20.0}, 40.0),

    ("aggressive_descent", [
        Waypoint(position=np.array([0.0, 1.0, 0.0]), hold_time=0.0),
    ], {"pos_y": 60.0, "vel_y": -8.0}, 30.0),

    ("landing", [
        Waypoint(position=np.array([0.0, 0.5, 0.0]), hold_time=0.0, acceptance_radius=0.5),
    ], {"pos_y": 15.0, "vel_y": -2.0}, 30.0),
]


def main():
    print("=" * 80)
    print("Safety Envelope Diagnostic — PID Controller Baseline")
    print("=" * 80)
    print()

    for scenario_name, waypoints, rocket_kw, duration in SCENARIOS:
        result = run_pid_flight(scenario_name, waypoints, rocket_kw, duration)
        signals = result["signals"]
        penalties = result["penalties"]

        print(f"--- {scenario_name} ({result['steps']} steps, {result['reversals']} thrust reversals) ---")
        print()

        # Per-signal stats
        print(f"  {'Signal':<20s}  {'Mean':>8s}  {'|Mean|':>8s}  {'Max|x|':>8s}  {'Std':>8s}")
        print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for i, name in enumerate(SIGNAL_NAMES):
            col = signals[:, i]
            print(f"  {name:<20s}  {np.mean(col):8.4f}  {np.mean(np.abs(col)):8.4f}  "
                  f"{np.max(np.abs(col)):8.4f}  {np.std(col):8.4f}")

        # Penalty stats
        mean_pen = np.mean(penalties)
        min_pen = np.min(penalties)
        sum_pen = np.sum(penalties)
        print()
        print(f"  r_safety:  mean={mean_pen:.6f}  min={min_pen:.6f}  "
              f"sum={sum_pen:.2f}  (over {result['steps']} steps)")

        # What the RL agent would see per-episode
        ep_penalty = sum_pen
        print(f"  Estimated episode penalty: {ep_penalty:.1f}")
        print()

    # Summary comparison
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("If mean |r_safety| >> 0.01 per step for PID flights, the penalty")
    print("weights are too aggressive — they'd punish even smooth control.")
    print()
    print("Current r_safety weights:")
    print("  throttle_sat:  -0.02 * sat^2")
    print("  gimbal_sat:    -0.01 * (outer^2 + inner^2)")
    print("  descent_excess: -0.1 * excess^2")
    print()
    print("Compare with typical dense reward magnitude (~0.1-1.0 per step).")


if __name__ == "__main__":
    main()
