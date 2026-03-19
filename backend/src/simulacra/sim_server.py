"""
Simulation server: run hopper flights on demand via HTTP.

POST /api/simulate — accepts waypoints JSON, runs simulation, returns NDJSON
GET  /api/models   — lists available trained models

Usage:
    cd backend
    uv run python -m simulacra.sim_server [--port 8767]
"""

import io
import json
import math
import sys
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np

from .noumena import (
    Rocket, RocketState, Environment, Kernel, Scenario,
    MassModel, get_upright_orientation,
)
from .noumena.mass_model import MassComponent, FuelComponent
from .phenomena.actuators import Actuators, GimbalActuator, ThrottleActuator
from .phenomena.sensors import Sensors
from .physics import RocketVec3, WorldVec3
from .onboard.hopper_controller import HopperController, HopperConfig, Waypoint
from .schema import DebugMessage, write_message


RUNS_DIR = Path(__file__).parent.parent.parent / "runs"


def find_models() -> list[dict]:
    """Discover available trained models."""
    models = []
    if not RUNS_DIR.exists():
        return models

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        # JAX checkpoint (params.npz in final_model/)
        jax_path = run_dir / "final_model" / "params.npz"
        if jax_path.exists():
            models.append({
                "name": run_dir.name,
                "path": str(run_dir / "final_model"),
                "type": "jax",
            })
            continue
        # SB3 zip
        sb3_path = run_dir / "final_model.zip"
        if sb3_path.exists():
            models.append({
                "name": run_dir.name,
                "path": str(sb3_path),
                "type": "sb3",
            })

    # Sort: hopper models first, then by name
    def sort_key(m):
        name = m["name"]
        if "hopper" in name:
            return (0, name)
        return (1, name)
    models.sort(key=sort_key)
    return models


def make_sim(gravity=9.81, mass_scale=1.0, wind_x=0.0, wind_z=0.0):
    """Create simulation components (same as eval_hopper.py)."""
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
    gimbal = GimbalActuator(
        rocket=rocket, max_outer_angle=0.26,
        max_inner_angle=0.26, max_rate=1.0, tau=0.05,
    )
    throttle = ThrottleActuator(
        rocket=rocket, max_thrust_rate=max_thrust * 2, min_thrust=0.0,
    )
    actuators = Actuators(gimbal=gimbal, throttle=throttle)
    sensors = Sensors(kernel)

    config = HopperConfig(
        dry_mass=dry_mass,
        full_fuel_mass=fuel_mass,
        max_thrust=max_thrust,
        gravity=gravity,
    )

    return kernel, actuators, sensors, config


def run_simulation(waypoints_data: list[dict],
                   controller_type: str = "pid",
                   model_path: str | None = None,
                   max_tilt: float = 0.15,
                   max_duration: float = 120.0) -> str:
    """Run a simulation and return NDJSON text."""
    # Build waypoints
    waypoints = []
    for wp in waypoints_data:
        pos = np.array(wp["pos"], dtype=np.float64)
        waypoints.append(Waypoint(
            position=pos,
            hold_time=wp.get("hold_time", 2.0),
            max_speed=wp.get("max_speed", 3.0),
            acceptance_radius=wp.get("acceptance_radius", 1.0),
        ))

    kernel, actuators, sensors, config = make_sim()

    # Create controller
    if controller_type == "rl":
        from .onboard.rl_hopper_controller import RLHopperController
        if model_path is None:
            # Default to best available model
            default = RUNS_DIR / "v6_hopper" / "final_model"
            if not default.exists():
                default = RUNS_DIR / "hopper_rl" / "final_model.zip"
            model_path = str(default)
        controller = RLHopperController(
            throttle=actuators._throttle,
            gimbal=actuators._gimbal,
            waypoints=waypoints,
            model_path=model_path,
            config=config,
            max_tilt=max_tilt,
        )
    else:
        controller = HopperController(
            throttle=actuators._throttle,
            gimbal=actuators._gimbal,
            waypoints=waypoints,
            config=config,
        )

    dt = 0.02
    output = io.StringIO()

    # Set up scenario for NDJSON recording
    scenario = Scenario(
        kernel=kernel, controller=controller, dt=dt,
        sensors=sensors, actuators=actuators,
    )
    scenario._output = output
    scenario._write_metadata_message(max_duration)
    scenario._write_constant_message()

    controller = controller.with_sensors(sensors)
    t = 0.0
    prev_phase = ""

    while t < max_duration:
        sensors.set_time(t)
        controller.run()
        actuators.set_time(t)
        actuators.step(dt)
        kernel.step(dt)
        t += dt

        scenario._time = t
        scenario._write_constant_if_changed()
        scenario._write_variable_message()

        s = kernel.rocket.state

        # Debug annotations for waypoint progress
        phase = controller.phase_name
        if "HOLD" in phase and "HOLD" not in prev_phase:
            speed = s.velocity.norm
            write_message(DebugMessage(
                timestamp=t, text=f"WP arrived ({speed:.1f} m/s)"
            ), output)
        prev_phase = phase

        # Termination checks
        if controller.finished and kernel.ground_contact.on_ground:
            speed = s.velocity.norm
            write_message(DebugMessage(
                timestamp=t, text=f"LANDED ({speed:.2f} m/s)"
            ), output)
            break

        if kernel.ground_contact.crashed or kernel.ground_contact.tipped_over:
            speed = s.velocity.norm
            nose_body = RocketVec3(1.0, 0.0, 0.0)
            nose_world = nose_body.to_world(s.orientation)
            tilt = math.acos(max(-1.0, min(1.0, nose_world.y)))
            write_message(DebugMessage(
                timestamp=t, text=f"CRASH ({speed:.1f} m/s)"
            ), output)
            break

        nose_body = RocketVec3(1.0, 0.0, 0.0)
        nose_world = nose_body.to_world(s.orientation)
        tilt = math.acos(max(-1.0, min(1.0, nose_world.y)))
        if tilt > 1.2:
            write_message(DebugMessage(
                timestamp=t, text=f"ABORT tilt={math.degrees(tilt):.0f}deg"
            ), output)
            break

    return output.getvalue()


class SimHandler(BaseHTTPRequestHandler):
    """HTTP handler for simulation requests."""

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/models":
            models = find_models()
            body = json.dumps({"models": models}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/simulate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            waypoints = body.get("waypoints", [])
            controller = body.get("controller", "pid")
            model_path = body.get("model_path", None)
            max_tilt = body.get("max_tilt", 0.15)
            max_duration = body.get("max_duration", 120.0)

            try:
                ndjson = run_simulation(
                    waypoints, controller, model_path, max_tilt, max_duration,
                )
                data = ndjson.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self._cors_headers()
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                error = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._cors_headers()
                self.end_headers()
                self.wfile.write(error)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[sim-server] {args[0]}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Simulacra simulation server")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), SimHandler)
    models = find_models()
    hopper_models = [m for m in models if "hopper" in m["name"]]
    print(f"Simulation server on http://localhost:{args.port}")
    print(f"  {len(models)} models found ({len(hopper_models)} hopper)")
    for m in hopper_models:
        print(f"    {m['name']} ({m['type']})")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
