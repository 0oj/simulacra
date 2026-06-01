"""New Scenario runner.

Owns the kernel, bus, guidance, controller, sensors, actuators, and
NDJSON output. Tick loop:

1. Push current sensor readings to the bus.
2. If a controller tick is due (1/rate_hz), snapshot bus,
   guidance.update(frame) -> setpoint, controller.step(frame, setpoint, dt)
   -> command, apply command to actuators.
3. Step kernel by dt.
4. Emit one variable NDJSON message per kernel step.

Deterministic given a seed.
"""
from __future__ import annotations

import random
from typing import Any, TextIO

import numpy as np

from simulacra.noumena import Kernel
from simulacra.onboard.bus import SensorBus
from simulacra.onboard.controller import (
    ActuatorCommand, Controller, VehicleParams,
)
from simulacra.onboard.guidance import Guidance, Mission, WaypointGuidance
from simulacra.physics.vec3 import Vec3, WorldVec3

from .helpers import actuators_for, sensors_for
from .ndjson import NDJSONWriter


class Scenario:
    """Runs a controller against a rocket + environment, writing NDJSON."""

    def __init__(
        self,
        *,
        rocket,
        env,
        mission: Mission,
        controller: Controller,
        guidance: Guidance | None = None,
        sensors: Any = None,
        actuators: Any = None,
        duration: float,
        seed: int = 0,
        dt: float = 0.01,
    ) -> None:
        self.rocket = rocket
        self.env = env
        self.mission = mission
        self.controller = controller
        self.guidance = guidance if guidance is not None else WaypointGuidance()
        self.bus = SensorBus()
        # Kernel must be constructed before sensors (sensors take the kernel).
        self._kernel = Kernel(rocket=rocket, environment=env)
        self.sensors = sensors if sensors is not None else sensors_for(self._kernel)
        self.actuators = actuators if actuators is not None else actuators_for(rocket)
        self.duration = duration
        self.seed = seed
        self.dt = dt

        random.seed(seed)
        np.random.seed(seed)

    def _vehicle_params(self) -> VehicleParams:
        """Derive design-time vehicle params from the rocket.

        v0.1 uses sensible defaults for fields that aren't in
        RocketProps; per-scenario overrides come in v0.2.
        """
        props = self.rocket.props
        return VehicleParams(
            dry_mass=float(props.mass) - 80.0,
            full_fuel_mass=80.0,
            inertia_dry=Vec3(*props.inertia_diag),
            length=float(props.length),
            max_thrust=2000.0,
            max_gimbal_angle=0.349,
            max_gimbal_rate=2.0,
        )

    def _read_sensors(self, t: float) -> None:
        """Read each sensor and push to the bus."""
        # Set the sensor clock so time-dependent errors evaluate correctly.
        self.sensors.set_time(t)

        gps_reading = self.sensors.GPS.read()
        # GPS returns a namedtuple (x, z) or similar; build a Vec3 from baro+gps.
        baro_alt = float(self.sensors.Barometer.read())
        # Build a world-frame position estimate.
        try:
            gps_pos = Vec3(float(gps_reading.x), baro_alt, float(gps_reading.z))
        except AttributeError:
            # Some GPS readings expose .position; fall back.
            arr = np.asarray(gps_reading, dtype=float)
            gps_pos = Vec3(float(arr[0]), baro_alt, float(arr[2] if arr.size > 2 else 0.0))
        self.bus.push_gps(t=t, position=gps_pos)
        self.bus.push_baro(t=t, altitude=baro_alt)

        # IMU: returns (orientation, ang_vel_body) tuple in this codebase.
        imu_raw = self.sensors.IMU.read()
        # We treat the "accel" channel as zero for now (the IMU class here
        # returns orientation+gyro; full accelerometer is via Accelerometer).
        accel_reading = self.sensors.Accelerometer.read()
        accel_vec = Vec3(*[float(x) for x in np.asarray(accel_reading, dtype=float).reshape(3)])
        gyro_reading = self.sensors.GyroSensor.read()
        gyro_vec = Vec3(*[float(x) for x in np.asarray(gyro_reading, dtype=float).reshape(3)])
        self.bus.push_imu(t=t, accel=accel_vec, gyro=gyro_vec)

        vel_reading = self.sensors.VelocitySensor.read()
        vel_arr = np.asarray(vel_reading, dtype=float).reshape(3)
        self.bus.push_velocity(t=t, velocity=Vec3(float(vel_arr[0]), float(vel_arr[1]), float(vel_arr[2])))

        orient = self.sensors.OrientationSensor.read()
        self.bus.push_orientation(t=t, orientation=orient)

    def run(self, out: TextIO) -> None:
        kernel = self._kernel
        writer = NDJSONWriter(out)
        writer.write({
            "message_type": "metadata",
            "schema_version": "0.3.0",
            "dt": self.dt,
            "duration": self.duration,
        })

        vehicle = self._vehicle_params()
        self.guidance.reset(self.mission, vehicle)
        self.controller.reset(self.mission, vehicle)

        from simulacra.phenomena.actuators import GimbalCommand, ThrottleCommand

        controller_period = 1.0 / self.controller.rate_hz
        next_ctrl_t = 0.0
        last_cmd = ActuatorCommand(throttle=0.0, gimbal_pitch=0.0, gimbal_yaw=0.0)

        t = 0.0
        while t < self.duration:
            self._read_sensors(t)
            if t >= next_ctrl_t:
                frame = self.bus.snapshot(t_now=t)
                setpoint = self.guidance.update(frame, t_now=t)
                last_cmd = self.controller.step(frame, setpoint, self.dt)
                next_ctrl_t += controller_period
                # Apply commands.
                self.actuators._throttle.apply(
                    ThrottleCommand(thrust_N=last_cmd.throttle * vehicle.max_thrust),
                )
                self.actuators._gimbal.apply(
                    GimbalCommand(
                        desired_outer_angle=last_cmd.gimbal_pitch,
                        desired_inner_angle=last_cmd.gimbal_yaw,
                    ),
                )
            # Advance actuator dynamics so commands become real thrust.
            self.actuators.set_time(t)
            self.actuators.step(self.dt)
            # Step physics by dt.
            kernel.step(self.dt)
            t += self.dt
            # Per-step variable message.
            writer.write({
                "message_type": "variable",
                "timestamp": t,
                "position": [float(x) for x in self.rocket.state.position],
                "velocity": [float(x) for x in self.rocket.state.velocity],
                "throttle_cmd": last_cmd.throttle,
                "gimbal_cmd": [last_cmd.gimbal_pitch, last_cmd.gimbal_yaw],
            })
            # Early stop on ground contact.
            if float(self.rocket.state.position[1]) <= 0.0:
                break

        writer.close()
