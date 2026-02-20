"""
The rocket's eyes. Sensors that peek into god's truth and report back,
sometimes accurately, sometimes with error, but always limited.

Each sensor accepts optional per-channel error configurations. Channels are
named consistently: 'x', 'y', 'z' for vector quantities; 'roll', 'pitch', 'yaw'
for Euler perturbations on orientation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Tuple, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..physics import Orientation, RocketVec3, WorldVec3
from .errors import ChannelErrors, IdentityError, SensorError

if TYPE_CHECKING:
    from ..noumena import Kernel


class Sensor(ABC):
    """Base class for all sensors."""

    @abstractmethod
    def read(self) -> Any:
        """Read the sensor value, applying any configured errors."""
        ...


class SensorsInterface(Protocol):
    """
    Minimal interface that sensor bundles satisfy.

    The controller only relies on these attributes existing, not on their
    exact return types.
    """

    GPS: Sensor
    OrientationSensor: Sensor
    GyroSensor: Sensor
    ThrustSensor: Sensor
    VelocitySensor: Sensor

    # Legacy alias for backward compatibility
    IMU: Sensor


def _identity_error() -> SensorError:
    """Factory for default identity error."""
    return IdentityError()


def _default_errors(*channels: str) -> ChannelErrors:
    """Create default error dict with IdentityError for each channel."""
    return {ch: _identity_error() for ch in channels}


class GPS(Sensor):
    """
    Global Positioning System.

    Channels: 'x', 'y', 'z' (world-frame position components).
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("x", "y", "z")
        self._time: float = 0.0

    def read(self) -> WorldVec3:
        """Read position with per-channel errors applied."""
        truth = self._kernel.rocket.state.position
        t = self._time
        return WorldVec3(
            self._errors.get("x", _identity_error()).apply(truth.x, t),
            self._errors.get("y", _identity_error()).apply(truth.y, t),
            self._errors.get("z", _identity_error()).apply(truth.z, t),
        )


class OrientationSensor(Sensor):
    """
    Orientation sensor (attitude reference).

    Channels: 'roll', 'pitch', 'yaw' for Euler perturbations (radians).
    Errors are additive perturbations to the true orientation.
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("roll", "pitch", "yaw")
        self._time: float = 0.0

    def read(self) -> Orientation:
        """Read orientation with Euler perturbations applied."""
        truth = self._kernel.rocket.state.orientation
        t = self._time

        # Get perturbations in roll/pitch/yaw
        droll = self._errors.get("roll", _identity_error()).apply(0.0, t)
        dpitch = self._errors.get("pitch", _identity_error()).apply(0.0, t)
        dyaw = self._errors.get("yaw", _identity_error()).apply(0.0, t)

        if droll == 0.0 and dpitch == 0.0 and dyaw == 0.0:
            return truth

        # Convert true orientation to scipy Rotation
        w, x, y, z = truth.as_array()
        r_true = R.from_quat([x, y, z, w])

        # Create perturbation rotation (ZYX intrinsic = extrinsic XYZ)
        r_perturb = R.from_euler("xyz", [droll, dpitch, dyaw])

        # Compose: perturbed = true * perturb (body-frame perturbation)
        r_result = r_true * r_perturb

        # Convert back to our quaternion format
        x, y, z, w = r_result.as_quat()
        return Orientation.from_wxyz(w, x, y, z)


class GyroSensor(Sensor):
    """
    Gyroscope sensor (angular velocity).

    Channels: 'x', 'y', 'z' (body-frame angular velocity components, rad/s).
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("x", "y", "z")
        self._time: float = 0.0

    def read(self) -> RocketVec3:
        """Read angular velocity with per-channel errors applied."""
        truth = self._kernel.rocket.state.angular_velocity_body
        t = self._time
        return RocketVec3(
            self._errors.get("x", _identity_error()).apply(truth.x, t),
            self._errors.get("y", _identity_error()).apply(truth.y, t),
            self._errors.get("z", _identity_error()).apply(truth.z, t),
        )


class IMU(Sensor):
    """
    Inertial Measurement Unit (legacy combined sensor).

    Wraps OrientationSensor and GyroSensor for backward compatibility.
    """

    def __init__(
        self,
        kernel: Kernel,
        orientation_errors: Optional[ChannelErrors] = None,
        gyro_errors: Optional[ChannelErrors] = None,
    ):
        self._orientation = OrientationSensor(kernel, orientation_errors)
        self._gyro = GyroSensor(kernel, gyro_errors)
        self._time: float = 0.0

    def read(self) -> Tuple[Orientation, RocketVec3]:
        """
        Reads orientation and body-frame angular velocity.

        Returns:
            (Orientation, RocketVec3): Attitude and angular velocity.
        """
        return (self._orientation.read(), self._gyro.read())

    def _set_time(self, t: float) -> None:
        """Update time for child sensors."""
        self._time = t
        self._orientation._time = t
        self._gyro._time = t


class ThrustSensor(Sensor):
    """
    Sensor that reads the thrust magnitude of the rocket.

    Channels: 'magnitude' (scalar thrust in Newtons).
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("magnitude")
        self._time: float = 0.0

    def read(self) -> float:
        """Read thrust magnitude with error applied."""
        truth = self._kernel.rocket.state.thrust_body.norm
        t = self._time
        return self._errors.get("magnitude", _identity_error()).apply(truth, t)


class VelocitySensor(Sensor):
    """
    Sensor that reads the linear velocity of the rocket.

    Channels: 'x', 'y', 'z' (world-frame velocity components, m/s).
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("x", "y", "z")
        self._time: float = 0.0

    def read(self) -> WorldVec3:
        """Read velocity with per-channel errors applied."""
        truth = self._kernel.rocket.state.velocity
        t = self._time
        return WorldVec3(
            self._errors.get("x", _identity_error()).apply(truth.x, t),
            self._errors.get("y", _identity_error()).apply(truth.y, t),
            self._errors.get("z", _identity_error()).apply(truth.z, t),
        )


class Sensors:
    """
    Bundle of sensors attached to a kernel.

    Provides unified time management for all sensors. The Scenario sets
    _time each step, which propagates to all sensors.
    """

    def __init__(
        self,
        kernel: Kernel,
        gps_errors: Optional[ChannelErrors] = None,
        orientation_errors: Optional[ChannelErrors] = None,
        gyro_errors: Optional[ChannelErrors] = None,
        thrust_errors: Optional[ChannelErrors] = None,
        velocity_errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._time: float = 0.0

        # Individual sensors with their error configurations
        self.GPS = GPS(self._kernel, gps_errors)
        self.OrientationSensor = OrientationSensor(self._kernel, orientation_errors)
        self.GyroSensor = GyroSensor(self._kernel, gyro_errors)
        self.ThrustSensor = ThrustSensor(self._kernel, thrust_errors)
        self.VelocitySensor = VelocitySensor(self._kernel, velocity_errors)

        # Legacy combined IMU for backward compatibility
        self.IMU = IMU(self._kernel, orientation_errors, gyro_errors)

    def set_time(self, t: float) -> None:
        """
        Set the current simulation time for all sensors.

        Called by Scenario each step to enable time-dependent errors.
        """
        self._time = t
        self.GPS._time = t
        self.OrientationSensor._time = t
        self.GyroSensor._time = t
        self.ThrustSensor._time = t
        self.VelocitySensor._time = t
        self.IMU._set_time(t)
