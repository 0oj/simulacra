#!/usr/bin/env python3
"""
Interactive rocket simulation server.

Runs a WebSocket server that:
1. Receives keyboard input from the frontend
2. Runs the physics simulation
3. Streams state back to the frontend in real-time

Usage:
    uv run python scripts/interactive_server.py

Then connect from frontend at ws://localhost:8765
"""

import asyncio
import json
import math
from typing import Set

import websockets
from websockets.server import serve

from simulacra.noumena import Environment, Kernel, Rocket, RocketProps, RocketState, MassModel, get_upright_orientation
from simulacra.noumena.mass_model import MassComponent, FuelComponent
from simulacra.phenomena.actuators import GimbalActuator, ThrottleActuator, Actuators
from simulacra.phenomena.sensors import Sensors
from simulacra.physics import WorldVec3, RocketVec3
from simulacra.onboard.wasd_controller import WASDController, WASDControllerConfig


class InteractiveSimulation:
    """Real-time interactive rocket simulation."""

    def __init__(self):
        self.keys_down: Set[str] = set()
        self.running = True
        self.base_dt = 0.02  # 50 Hz physics
        self.time_scale = 1.0
        self.dt = self.base_dt
        self.reset()

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        # Build rocket
        mass_model = MassModel(components=[
            MassComponent("dry", 60.0, RocketVec3(2.0, 0, 0)),
            FuelComponent("fuel", 80.0, RocketVec3(1.2, 0, 0), full_mass=80.0, burn_rate=4.0),
        ])
        props = mass_model.to_rocket_props(
            length=4.0,
            thrust_application_offset=RocketVec3(0, 0, 0),
        )

        # Start on ground, upright
        state = RocketState(
            position=WorldVec3(0.0, 0.0, 0.0),
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=get_upright_orientation(),
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(0.0, 0.0, 0.0),
        )

        self.rocket = Rocket(props=props, state=state, mass_model=mass_model)
        self.env = Environment(gravity=WorldVec3(0, -9.81, 0))
        self.kernel = Kernel(self.rocket, self.env)
        self.sensors = Sensors(self.kernel)

        self.gimbal = GimbalActuator(
            rocket=self.rocket,
            max_outer_angle=0.26,
            max_inner_angle=0.26,
            max_rate=1.0,
            tau=0.05,
        )
        self.throttle = ThrottleActuator(
            rocket=self.rocket,
            max_thrust_rate=4000,
        )
        self.actuators = Actuators(gimbal=self.gimbal, throttle=self.throttle)

        self.controller = WASDController(
            throttle=self.throttle,
            gimbal=self.gimbal,
            config=WASDControllerConfig(
                max_thrust=2000.0,
                throttle_rate=400.0,  # Smoother throttle
                gimbal_rate=0.3,      # Less sensitive gimbal
            ),
        )
        self.controller.with_sensors(self.sensors)
        self.controller.set_dt(self.dt)

        self.sim_time = 0.0

    def step(self) -> dict:
        """Run one simulation step and return state."""
        # Update controller with current keys and dt
        self.controller.set_keys(self.keys_down)
        self.controller.set_dt(self.dt)
        self.controller.run()

        # Step actuators and physics
        self.actuators.step(self.dt)
        self.kernel.step(self.dt)
        self.sim_time += self.dt

        # Build state message
        s = self.rocket.state
        p = self.rocket.props
        gc = self.kernel.ground_contact

        # Tilt angle
        nose = RocketVec3(1, 0, 0).to_world(s.orientation)
        tilt_deg = math.degrees(math.acos(max(-1, min(1, nose.y))))

        # Gimbal angles (in radians, convert to degrees for display)
        gimbal_outer = self.gimbal.outer_angle
        gimbal_inner = self.gimbal.inner_angle

        return {
            "t": round(self.sim_time, 3),
            "position": {"x": s.position.x, "y": s.position.y, "z": s.position.z},
            "velocity": {"x": s.velocity.x, "y": s.velocity.y, "z": s.velocity.z},
            "orientation": list(s.orientation.as_array()),
            "thrust": s.thrust_body.norm,
            "throttle_percent": self.controller.throttle_percent,
            "fuel_fraction": self.rocket.mass_model.fuel_fraction,
            "tilt_deg": round(tilt_deg, 1),
            "on_ground": gc.on_ground,
            "crashed": gc.crashed,
            "tipped": gc.tipped_over,
            "mass": p.mass,
            "gimbal_outer_deg": round(math.degrees(gimbal_outer), 1),
            "gimbal_inner_deg": round(math.degrees(gimbal_inner), 1),
        }


# Global simulation instance
sim = InteractiveSimulation()


async def handle_client(websocket):
    """Handle a single WebSocket client."""
    print(f"Client connected: {websocket.remote_address}")

    # Send initial state
    await websocket.send(json.dumps({
        "type": "init",
        "state": sim.step(),
    }))

    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "keys":
                # Update key state
                sim.keys_down = set(data.get("keys", []))

            elif msg_type == "timescale":
                # Update time scale (0.1 to 4.0)
                scale = data.get("scale", 1.0)
                sim.time_scale = max(0.1, min(4.0, scale))
                sim.dt = sim.base_dt * sim.time_scale

            elif msg_type == "reset":
                # Reset simulation
                sim.reset()
                await websocket.send(json.dumps({
                    "type": "reset",
                    "state": sim.step(),
                }))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")


async def simulation_loop(websocket):
    """Run simulation and broadcast state."""
    while True:
        if sim.running:
            state = sim.step()
            try:
                await websocket.send(json.dumps({
                    "type": "state",
                    "state": state,
                }))
            except websockets.exceptions.ConnectionClosed:
                break

        # Sleep at base rate (real-time 50 Hz), simulation advances at scaled rate
        await asyncio.sleep(sim.base_dt)


async def handler(websocket):
    """Handle client with both input and output."""
    # Run input handler and simulation loop concurrently
    input_task = asyncio.create_task(handle_client(websocket))
    sim_task = asyncio.create_task(simulation_loop(websocket))

    done, pending = await asyncio.wait(
        [input_task, sim_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()


async def main():
    print("=" * 60)
    print("Interactive Rocket Simulation Server")
    print("=" * 60)
    print()
    print("Controls:")
    print("  W / S     - Throttle up / down")
    print("  A / D     - Gimbal left / right")
    print("  Q / E     - Gimbal forward / back")
    print("  Space     - Cut throttle")
    print("  Shift     - Full throttle")
    print("  R         - Reset simulation")
    print()
    print("Starting server on ws://localhost:8765")
    print("Connect from frontend to start flying!")
    print()

    async with serve(handler, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
