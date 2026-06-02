"""CController — load a C controller library and present it as a Controller.

Engineers write their controller in C against `csdk/include/simulacra.h`,
compile it to a shared library (`.so` / `.dylib` / `.dll`), and load it
into simulacra with:

    from simulacra.onboard.c_controller import CController
    ctrl = CController("path/to/libmy_controller.dylib")
    from simulacra.scenarios import hopper_short
    hopper_short.run(ctrl)

The same `controller_step` they wrote is what runs on the flight
computer later — sim and hardware build the same .c file.

The C structs in this file MUST stay byte-compatible with the structs
declared in `csdk/include/simulacra.h`. If you change one, change both.
"""
from __future__ import annotations

import ctypes
from ctypes import (
    POINTER, Structure, c_float, c_uint8, c_uint32,
)
from typing import Any

import numpy as np

from simulacra.onboard.controller import (
    ActuatorCommand, SensorFrame, VehicleParams,
)
from simulacra.onboard.guidance import Mission, Setpoint, WaypointMission


# ────────────────────── ctypes mirrors of simulacra.h ─────────────

class SimVec3(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float)]


class SimQuat(Structure):
    _fields_ = [("w", c_float), ("x", c_float), ("y", c_float), ("z", c_float)]


class SimStampedVec3(Structure):
    _fields_ = [("value", SimVec3), ("t", c_float), ("seq", c_uint32)]


class SimStampedFloat(Structure):
    _fields_ = [("value", c_float), ("t", c_float), ("seq", c_uint32)]


class SimStampedQuat(Structure):
    _fields_ = [("value", SimQuat), ("t", c_float), ("seq", c_uint32)]


class SimSensorFrame(Structure):
    _fields_ = [
        ("t_now", c_float),
        ("imu_accel", SimStampedVec3),
        ("imu_gyro", SimStampedVec3),
        ("baro", SimStampedFloat),
        ("gps", SimStampedVec3),
        ("gps_valid", c_uint8),
        ("velocity", SimStampedVec3),
        ("orientation", SimStampedQuat),
    ]


class SimActuatorCommand(Structure):
    _fields_ = [
        ("throttle", c_float),
        ("gimbal_pitch", c_float),
        ("gimbal_yaw", c_float),
    ]


class SimVehicleParams(Structure):
    _fields_ = [
        ("dry_mass", c_float),
        ("full_fuel_mass", c_float),
        ("inertia_dry", SimVec3),
        ("length", c_float),
        ("max_thrust", c_float),
        ("max_gimbal_angle", c_float),
        ("max_gimbal_rate", c_float),
    ]


class SimWaypoint(Structure):
    _fields_ = [
        ("position", SimVec3),
        ("hold_time", c_float),
        ("acceptance_radius", c_float),
        ("max_speed", c_float),
    ]


class SimMission(Structure):
    _fields_ = [
        ("waypoints", POINTER(SimWaypoint)),
        ("n_waypoints", c_uint32),
    ]


class SimSetpoint(Structure):
    _fields_ = [
        ("position", SimVec3),
        ("velocity", SimVec3),
        ("accel_ff", SimVec3),
        ("attitude", SimQuat),
    ]


# ─────────────────────── Helpers (Python → C structs) ─────────────

def _vec3_to_c(v) -> SimVec3:
    arr = np.asarray(v, dtype=float).reshape(3)
    return SimVec3(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


def _quat_to_c(orient) -> SimQuat:
    arr = np.asarray(orient.q, dtype=float).reshape(4)
    return SimQuat(w=float(arr[0]), x=float(arr[1]), y=float(arr[2]), z=float(arr[3]))


def _frame_to_c(frame: SensorFrame) -> SimSensorFrame:
    out = SimSensorFrame()
    out.t_now = float(frame.t_now)
    out.imu_accel = SimStampedVec3(
        value=_vec3_to_c(frame.imu_accel.value),
        t=float(frame.imu_accel.t), seq=int(frame.imu_accel.seq),
    )
    out.imu_gyro = SimStampedVec3(
        value=_vec3_to_c(frame.imu_gyro.value),
        t=float(frame.imu_gyro.t), seq=int(frame.imu_gyro.seq),
    )
    out.baro = SimStampedFloat(
        value=float(frame.baro.value),
        t=float(frame.baro.t), seq=int(frame.baro.seq),
    )
    if frame.gps is not None:
        out.gps = SimStampedVec3(
            value=_vec3_to_c(frame.gps.value),
            t=float(frame.gps.t), seq=int(frame.gps.seq),
        )
        out.gps_valid = 1
    else:
        out.gps_valid = 0
    out.velocity = SimStampedVec3(
        value=_vec3_to_c(frame.velocity.value),
        t=float(frame.velocity.t), seq=int(frame.velocity.seq),
    )
    out.orientation = SimStampedQuat(
        value=_quat_to_c(frame.orientation.value),
        t=float(frame.orientation.t), seq=int(frame.orientation.seq),
    )
    return out


def _setpoint_to_c(sp: Setpoint) -> SimSetpoint:
    return SimSetpoint(
        position=_vec3_to_c(sp.position),
        velocity=_vec3_to_c(sp.velocity),
        accel_ff=_vec3_to_c(sp.accel_ff),
        attitude=_quat_to_c(sp.attitude),
    )


def _vehicle_to_c(v: VehicleParams) -> SimVehicleParams:
    return SimVehicleParams(
        dry_mass=float(v.dry_mass),
        full_fuel_mass=float(v.full_fuel_mass),
        inertia_dry=_vec3_to_c(v.inertia_dry),
        length=float(v.length),
        max_thrust=float(v.max_thrust),
        max_gimbal_angle=float(v.max_gimbal_angle),
        max_gimbal_rate=float(v.max_gimbal_rate),
    )


def _mission_to_c(mission: Mission | None) -> tuple[SimMission, Any]:
    """Convert a WaypointMission to C, returning (struct, owned_array_keepalive)."""
    if mission is None or not isinstance(mission, WaypointMission):
        return SimMission(waypoints=None, n_waypoints=0), None
    n = len(mission.waypoints)
    arr = (SimWaypoint * n)()
    for i, wp in enumerate(mission.waypoints):
        arr[i].position = _vec3_to_c(wp.position)
        arr[i].hold_time = float(wp.hold_time)
        arr[i].acceptance_radius = float(wp.acceptance_radius)
        arr[i].max_speed = float(wp.max_speed)
    sm = SimMission(waypoints=arr, n_waypoints=n)
    return sm, arr   # caller must hold onto arr while sm is alive.


# ─────────────────────────── CController ──────────────────────────

class CController:
    """Wrap a C controller shared library as a simulacra Controller.

    The library must export the three functions declared in
    `csdk/include/simulacra.h`:
        float controller_rate_hz(void);
        void  controller_reset(const sim_mission_t*, const sim_vehicle_params_t*);
        void  controller_step (const sim_sensor_frame_t*, const sim_setpoint_t*,
                               float dt, sim_actuator_command_t*);
    """

    def __init__(self, lib_path: str) -> None:
        self._lib = ctypes.CDLL(lib_path)

        self._lib.controller_rate_hz.argtypes = []
        self._lib.controller_rate_hz.restype = c_float

        self._lib.controller_reset.argtypes = [
            POINTER(SimMission), POINTER(SimVehicleParams),
        ]
        self._lib.controller_reset.restype = None

        self._lib.controller_step.argtypes = [
            POINTER(SimSensorFrame), POINTER(SimSetpoint),
            c_float, POINTER(SimActuatorCommand),
        ]
        self._lib.controller_step.restype = None

        self.rate_hz: float = float(self._lib.controller_rate_hz())
        self._mission_keepalive: Any = None
        # Pre-allocated output buffer (no allocations in step()).
        self._out_cmd = SimActuatorCommand()

    def reset(self, mission: Mission, vehicle: VehicleParams) -> None:
        sm, keep = _mission_to_c(mission)
        self._mission_keepalive = keep  # keep the waypoint array alive
        vp = _vehicle_to_c(vehicle)
        self._lib.controller_reset(ctypes.byref(sm), ctypes.byref(vp))

    def step(
        self,
        frame: SensorFrame,
        setpoint: Setpoint,
        dt: float,
    ) -> ActuatorCommand:
        cframe = _frame_to_c(frame)
        csp = _setpoint_to_c(setpoint)
        self._lib.controller_step(
            ctypes.byref(cframe), ctypes.byref(csp),
            c_float(float(dt)), ctypes.byref(self._out_cmd),
        )
        return ActuatorCommand(
            throttle=float(self._out_cmd.throttle),
            gimbal_pitch=float(self._out_cmd.gimbal_pitch),
            gimbal_yaw=float(self._out_cmd.gimbal_yaw),
        )
