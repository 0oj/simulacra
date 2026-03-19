#!/usr/bin/env python3
"""
Evaluate and compare PID vs RL hopper controllers.

Runs both controllers through identical scenarios and produces:
  1. NDJSON flight logs (for 3D viewer replay)
  2. Comparison summary table
  3. Results JSONL

Usage:
    cd backend
    uv run python scripts/eval_hopper.py [--model runs/hopper_rl/final_model]
"""

import sys
import math
import json
import argparse
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from simulacra.noumena import (
    Rocket, RocketState, Environment, Kernel, Scenario,
    MassModel, get_upright_orientation,
)
from simulacra.noumena.mass_model import MassComponent, FuelComponent
from simulacra.phenomena.actuators import Actuators, GimbalActuator, ThrottleActuator
from simulacra.phenomena.sensors import Sensors
from simulacra.physics import RocketVec3, WorldVec3
from simulacra.onboard.hopper_controller import (
    HopperController, HopperConfig, Waypoint, hopper_flight_plan,
)
from simulacra.schema import DebugMessage, write_message


def make_sim(gravity=9.81, mass_scale=1.0, wind_x=0.0, wind_z=0.0):
    """Create simulation components."""
    dry_mass = 60.0 * mass_scale
    fuel_mass = 80.0 * mass_scale
    length = 4.0
    max_thrust = 2000.0

    mass_model = MassModel(components=[
        MassComponent("dry", dry_mass, RocketVec3(length / 2, 0, 0)),
        FuelComponent("fuel", mass=fuel_mass,
                      position=RocketVec3(length * 0.3, 0, 0),
                      full_mass=fuel_mass, burn_rate=0.68),
    ])
    props = mass_model.to_rocket_props(
        length=length,
        thrust_application_offset=RocketVec3(0, 0, 0),
    )
    state = RocketState(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),
    )
    rocket = Rocket(props=props, state=state, mass_model=mass_model)
    environment = Environment(
        gravity=WorldVec3(0.0, -gravity, 0.0),
        wind=WorldVec3(wind_x, 0.0, wind_z),
        air_density=1.225,
    )
    kernel = Kernel(rocket, environment)
    gimbal = GimbalActuator(rocket=rocket, max_outer_angle=0.26,
                            max_inner_angle=0.26, max_rate=1.0, tau=0.05)
    throttle = ThrottleActuator(rocket=rocket, max_thrust_rate=max_thrust * 2,
                                min_thrust=0.0)
    actuators = Actuators(gimbal=gimbal, throttle=throttle)
    sensors = Sensors(kernel)

    config = HopperConfig(
        dry_mass=dry_mass,
        full_fuel_mass=fuel_mass,
        max_thrust=max_thrust,
        gravity=gravity,
    )

    return kernel, actuators, sensors, config


def run_flight(controller_type, waypoints, model_path=None, max_tilt=0.15,
               gravity=9.81, mass_scale=1.0, wind_x=0.0, wind_z=0.0,
               max_duration=200.0, save_ndjson=None):
    """Run a flight with either PID or RL controller and return metrics."""
    kernel, actuators, sensors, config = make_sim(
        gravity=gravity, mass_scale=mass_scale,
        wind_x=wind_x, wind_z=wind_z,
    )

    if controller_type == "pid":
        controller = HopperController(
            throttle=actuators._throttle,
            gimbal=actuators._gimbal,
            waypoints=waypoints,
            config=config,
        )
    elif controller_type == "rl":
        from simulacra.onboard.rl_hopper_controller import RLHopperController
        controller = RLHopperController(
            throttle=actuators._throttle,
            gimbal=actuators._gimbal,
            waypoints=waypoints,
            model_path=model_path,
            config=config,
            max_tilt=max_tilt,
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    dt = 0.02

    # Optional NDJSON recording
    ndjson_file = None
    scenario_for_log = None
    if save_ndjson is not None:
        scenario_for_log = Scenario(
            kernel=kernel, controller=controller, dt=dt,
            sensors=sensors, actuators=actuators,
        )
        ndjson_file = open(save_ndjson, "w")
        scenario_for_log._output = ndjson_file
        scenario_for_log._write_metadata_message(max_duration)
        scenario_for_log._write_constant_message()

    controller = controller.with_sensors(sensors)
    t = 0.0

    max_tilt = 0.0
    max_speed = 0.0
    total_thrust_integral = 0.0  # integral of thrust over time (fuel proxy)
    rms_omega = 0.0
    rms_dthrottle = 0.0
    prev_thrust = 0.0
    step_count = 0
    waypoints_reached = 0
    prev_phase = ""

    while t < max_duration:
        sensors.set_time(t)
        controller.run()
        actuators.set_time(t)
        actuators.step(dt)
        kernel.step(dt)
        t += dt
        step_count += 1

        if scenario_for_log is not None:
            scenario_for_log._time = t
            scenario_for_log._write_constant_if_changed()
            scenario_for_log._write_variable_message()

        s = kernel.rocket.state
        speed = s.velocity.norm
        thrust = s.thrust_body.norm

        # Tilt
        nose_body = RocketVec3(1.0, 0.0, 0.0)
        nose_world = nose_body.to_world(s.orientation)
        cos_tilt = max(-1.0, min(1.0, nose_world.y))
        tilt = math.acos(cos_tilt)
        max_tilt = max(max_tilt, tilt)
        max_speed = max(max_speed, speed)

        # Smoothness metrics
        total_thrust_integral += thrust * dt
        omega_mag = s.angular_velocity_body.norm
        rms_omega += omega_mag ** 2
        dthrottle = abs(thrust - prev_thrust) / dt
        rms_dthrottle += dthrottle ** 2
        prev_thrust = thrust

        # Track waypoint progress + debug annotations
        phase = controller.phase_name
        if "HOLD" in phase and "HOLD" not in prev_phase:
            waypoints_reached += 1
            if ndjson_file is not None:
                wp = waypoints[min(waypoints_reached - 1, len(waypoints) - 1)]
                write_message(DebugMessage(
                    timestamp=t,
                    text=f"WP{waypoints_reached} arrived ({speed:.1f} m/s)"
                ), ndjson_file)
        prev_phase = phase

        if controller.finished and kernel.ground_contact.on_ground:
            if ndjson_file is not None:
                write_message(DebugMessage(
                    timestamp=t,
                    text=f"LANDED ({speed:.2f} m/s)"
                ), ndjson_file)
            break

        if kernel.ground_contact.crashed or kernel.ground_contact.tipped_over:
            if ndjson_file is not None:
                write_message(DebugMessage(
                    timestamp=t,
                    text=f"CRASH ({speed:.1f} m/s, tilt={math.degrees(tilt):.0f}deg)"
                ), ndjson_file)
            break
        if tilt > 1.2:
            if ndjson_file is not None:
                write_message(DebugMessage(
                    timestamp=t,
                    text=f"ABORT tilt={math.degrees(tilt):.0f}deg"
                ), ndjson_file)
            break

    pos = kernel.rocket.state.position
    fuel_remaining = kernel.rocket.mass_model.fuel_fraction
    landed = controller.finished and kernel.ground_contact.on_ground
    crashed = kernel.ground_contact.crashed or kernel.ground_contact.tipped_over
    landing_speed = kernel.ground_contact.impact_speed if landed else None

    rms_omega = math.sqrt(rms_omega / max(1, step_count))
    rms_dthrottle = math.sqrt(rms_dthrottle / max(1, step_count))

    if ndjson_file is not None:
        ndjson_file.close()

    return {
        "completed": landed and not crashed,
        "crashed": crashed,
        "time_s": round(t, 1),
        "fuel_remaining": round(fuel_remaining, 3),
        "fuel_used_pct": round((1.0 - fuel_remaining) * 100, 1),
        "landing_speed_ms": round(landing_speed, 2) if landing_speed else None,
        "max_tilt_deg": round(math.degrees(max_tilt), 1),
        "max_speed_ms": round(max_speed, 2),
        "waypoints_reached": waypoints_reached,
        "waypoints_total": len(waypoints),
        "rms_omega": round(rms_omega, 3),
        "rms_dthrottle": round(rms_dthrottle, 1),
        "thrust_integral": round(total_thrust_integral, 0),
        "final_pos": [round(pos.x, 2), round(pos.y, 2), round(pos.z, 2)],
    }


# ──────────────────────────────────────────────────────────────
# Test scenarios
# ──────────────────────────────────────────────────────────────

def standard_scenario():
    return "standard", hopper_flight_plan(circle_points=12), {}

def simple_vertical_scenario():
    wps = [
        Waypoint(np.array([0., 5., 0.]), hold_time=3.0, max_speed=2.0,
                 acceptance_radius=0.5),
        Waypoint(np.array([0., 15., 0.]), hold_time=3.0, max_speed=3.0,
                 acceptance_radius=0.5),
        Waypoint(np.array([0., 5., 0.]), hold_time=3.0, max_speed=2.0,
                 acceptance_radius=0.5),
        Waypoint(np.array([0., 0.0, 0.]), hold_time=60.0, max_speed=1.0,
                 acceptance_radius=0.5),
    ]
    return "simple_vertical", wps, {}

def tall_vertical_scenario():
    wps = [
        Waypoint(np.array([0., 50., 0.]), hold_time=3.0, max_speed=5.0,
                 acceptance_radius=1.0),
        Waypoint(np.array([0., 0.0, 0.]), hold_time=60.0, max_speed=1.0,
                 acceptance_radius=0.5),
    ]
    return "tall_vertical", wps, {}

def wide_lateral_scenario():
    wps = [
        Waypoint(np.array([0., 20., 0.]), hold_time=2.0, max_speed=4.0,
                 acceptance_radius=1.0),
        Waypoint(np.array([20., 20., 0.]), hold_time=2.0, max_speed=5.0,
                 acceptance_radius=1.5),
        Waypoint(np.array([-20., 20., 0.]), hold_time=2.0, max_speed=5.0,
                 acceptance_radius=1.5),
        Waypoint(np.array([0., 20., 0.]), hold_time=2.0, max_speed=4.0,
                 acceptance_radius=1.0),
        Waypoint(np.array([0., 0.0, 0.]), hold_time=60.0, max_speed=1.0,
                 acceptance_radius=0.5),
    ]
    return "wide_lateral", wps, {}

def precision_land_scenario():
    wps = [
        Waypoint(np.array([0., 30., 0.]), hold_time=2.0, max_speed=4.0,
                 acceptance_radius=1.0),
        Waypoint(np.array([0., 0.0, 0.]), hold_time=60.0, max_speed=0.5,
                 acceptance_radius=0.3),
    ]
    return "precision_land", wps, {}

def fast_sequence_scenario():
    """10 rapid waypoints with short holds."""
    wps = []
    for i in range(5):
        x = 5.0 * (-1) ** i
        y = 10.0 + 3.0 * i
        wps.append(Waypoint(np.array([x, y, 0.]), hold_time=0.5, max_speed=6.0,
                            acceptance_radius=1.5))
    for i in range(4, -1, -1):
        x = 5.0 * (-1) ** (i + 1)
        y = 10.0 + 3.0 * i
        wps.append(Waypoint(np.array([x, y, 0.]), hold_time=0.5, max_speed=6.0,
                            acceptance_radius=1.5))
    wps.append(Waypoint(np.array([0., 0.0, 0.]), hold_time=60.0, max_speed=1.0,
                        acceptance_radius=0.5))
    return "fast_sequence", wps, {}

def wind_scenario():
    wps = [
        Waypoint(np.array([0., 15., 0.]), hold_time=3.0, max_speed=3.0,
                 acceptance_radius=0.5),
        Waypoint(np.array([0., 0.0, 0.]), hold_time=60.0, max_speed=1.0,
                 acceptance_radius=0.5),
    ]
    return "wind_5ms", wps, {"wind_x": 5.0}

ALL_SCENARIOS = [
    standard_scenario,
    simple_vertical_scenario,
    tall_vertical_scenario,
    wide_lateral_scenario,
    precision_land_scenario,
    fast_sequence_scenario,
    wind_scenario,
]


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate PID vs RL hopper controllers")
    parser.add_argument("--model", type=str, default="runs/hopper_rl/final_model",
                        help="Path to trained RL model")
    parser.add_argument("--max-tilt", type=float, default=0.15,
                        help="GuidanceWrapper max_tilt used during training")
    parser.add_argument("--pid-only", action="store_true",
                        help="Only run PID controller")
    parser.add_argument("--save-ndjson", action="store_true", default=True,
                        help="Save NDJSON replay files")
    args = parser.parse_args()

    output_dir = Path("runs/hopper_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    ndjson_dir = output_dir / "ndjson"
    ndjson_dir.mkdir(exist_ok=True)

    controllers = ["pid"]
    if not args.pid_only:
        model_path = Path(args.model)
        if model_path.exists() or (model_path.parent / (model_path.name + ".zip")).exists():
            controllers.append("rl")
        else:
            print(f"RL model not found at {args.model}, running PID only")

    results = []

    # Header
    print(f"\n{'Scenario':<18s} {'Ctrl':<5s} {'OK':>3s} {'Time':>6s} "
          f"{'Fuel%':>6s} {'Vland':>6s} {'WPs':>5s} "
          f"{'Tilt':>6s} {'ω_rms':>6s} {'ΔT_rms':>7s}")
    print("-" * 85)

    for scenario_fn in ALL_SCENARIOS:
        name, waypoints, extra_kwargs = scenario_fn()

        for ctrl_type in controllers:
            ndjson_path = None
            if args.save_ndjson:
                ndjson_path = ndjson_dir / f"{name}_{ctrl_type}.ndjson"

            r = run_flight(
                ctrl_type, waypoints,
                model_path=args.model if ctrl_type == "rl" else None,
                max_tilt=args.max_tilt if ctrl_type == "rl" else 0.15,
                save_ndjson=ndjson_path,
                **extra_kwargs,
            )
            r["scenario"] = name
            r["controller"] = ctrl_type
            results.append(r)

            ok = "YES" if r["completed"] else "NO"
            vl = f"{r['landing_speed_ms']:.2f}" if r["landing_speed_ms"] else "N/A"
            wp_str = f"{r['waypoints_reached']}/{r['waypoints_total']}"
            print(f"{name:<18s} {ctrl_type:<5s} {ok:>3s} {r['time_s']:6.1f} "
                  f"{r['fuel_used_pct']:6.1f} {vl:>6s} {wp_str:>5s} "
                  f"{r['max_tilt_deg']:6.1f} {r['rms_omega']:6.3f} "
                  f"{r['rms_dthrottle']:7.1f}")

        if len(controllers) > 1:
            print()  # blank line between scenarios when comparing

    # Save results
    results_file = output_dir / "eval_results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {results_file}")

    # Summary
    for ctrl_type in controllers:
        ctrl_results = [r for r in results if r["controller"] == ctrl_type]
        completed = sum(1 for r in ctrl_results if r["completed"])
        total = len(ctrl_results)
        avg_fuel = np.mean([r["fuel_used_pct"] for r in ctrl_results])
        avg_omega = np.mean([r["rms_omega"] for r in ctrl_results])
        print(f"\n{ctrl_type.upper()}: {completed}/{total} completed, "
              f"avg fuel={avg_fuel:.1f}%, avg ω_rms={avg_omega:.3f}")

    if args.save_ndjson:
        # Copy NDJSON to frontend
        frontend_dir = Path(__file__).parent.parent.parent / "frontend" / "public" / "episodes"
        if frontend_dir.exists():
            import shutil
            for f in ndjson_dir.glob("*.ndjson"):
                shutil.copy2(f, frontend_dir / f.name)
            print(f"\nNDJSON replays copied to {frontend_dir}")


if __name__ == "__main__":
    main()
