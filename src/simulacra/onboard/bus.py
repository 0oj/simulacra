"""SensorBus: the seam between sensor producers and the controller.

Producers (in sim, HIL, or flight) call `push_*` methods at the rate
their sensor produces samples. The controller calls `snapshot(t_now)`
at its own fixed rate and gets a `SensorFrame` of the latest values
with timestamps and sequence counters.

For pure-Python sim this is single-threaded — producer (kernel) and
consumer (controller) interleave on the main loop. In HIL or flight
the producer would run on a separate thread / ISR; the snapshot API
does not change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simulacra.physics.quat import Orientation
from simulacra.physics.vec3 import Vec3

from .controller import SensorFrame, Stamped


@dataclass
class _Slot:
    """Mutable latest-value cell for one topic."""
    value: Any = None
    t: float = -1.0
    seq: int = -1


class SensorBus:
    """Latest-sample registry, one cell per topic."""

    def __init__(self) -> None:
        self._imu_accel = _Slot()
        self._imu_gyro = _Slot()
        self._baro = _Slot()
        self._gps = _Slot()
        self._velocity = _Slot()
        self._orientation = _Slot()

    # --- producers ----------------------------------------------------

    def push_imu(self, t: float, accel: Vec3, gyro: Vec3) -> None:
        self._imu_accel.value = accel
        self._imu_accel.t = t
        self._imu_accel.seq += 1
        self._imu_gyro.value = gyro
        self._imu_gyro.t = t
        self._imu_gyro.seq += 1

    def push_baro(self, t: float, altitude: float) -> None:
        self._baro.value = altitude
        self._baro.t = t
        self._baro.seq += 1

    def push_gps(self, t: float, position: Vec3) -> None:
        self._gps.value = position
        self._gps.t = t
        self._gps.seq += 1

    def push_velocity(self, t: float, velocity: Vec3) -> None:
        self._velocity.value = velocity
        self._velocity.t = t
        self._velocity.seq += 1

    def push_orientation(self, t: float, orientation: Orientation) -> None:
        self._orientation.value = orientation
        self._orientation.t = t
        self._orientation.seq += 1

    # --- consumer -----------------------------------------------------

    def snapshot(self, t_now: float) -> SensorFrame:
        """Snapshot all slots into a SensorFrame.

        Raises RuntimeError if any non-optional topic has never been
        pushed (a wiring bug — the runtime should always push every
        non-GPS topic before snapshot).
        """
        for name in ("_imu_accel", "_imu_gyro", "_baro", "_velocity", "_orientation"):
            slot = getattr(self, name)
            if slot.seq < 0:
                raise RuntimeError(
                    f"SensorBus.snapshot called before any push to '{name[1:]}'"
                )
        gps = (
            Stamped(value=self._gps.value, t=self._gps.t, seq=self._gps.seq)
            if self._gps.seq >= 0 else None
        )
        return SensorFrame(
            t_now=t_now,
            imu_accel=Stamped(
                value=self._imu_accel.value, t=self._imu_accel.t,
                seq=self._imu_accel.seq,
            ),
            imu_gyro=Stamped(
                value=self._imu_gyro.value, t=self._imu_gyro.t,
                seq=self._imu_gyro.seq,
            ),
            baro=Stamped(
                value=self._baro.value, t=self._baro.t, seq=self._baro.seq,
            ),
            gps=gps,
            velocity=Stamped(
                value=self._velocity.value, t=self._velocity.t,
                seq=self._velocity.seq,
            ),
            orientation=Stamped(
                value=self._orientation.value, t=self._orientation.t,
                seq=self._orientation.seq,
            ),
        )
