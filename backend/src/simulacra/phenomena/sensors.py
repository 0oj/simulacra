"""
The rocket's eyes. Sensors that peek into god's truth and report back,
sometimes accurately, sometimes with error, but always limited.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, Tuple

from ..noumena import Kernel2D, Kernel3D
from ..physics import Orientation, RocketVec3, WorldVec2, WorldVec3


class Sensor(ABC):
    @abstractmethod
    def read(self) -> Any: ...


class SensorsInterface(Protocol):
    """
    Minimal interface that both 2D and 3D sensor bundles satisfy.

    The controller only relies on these attributes existing, not on their
    exact return types.
    """

    GPS: Sensor
    IMU: Sensor
    ThrustSensor: Sensor
    VelocitySensor: Sensor


# ============================================================================
# 2D sensors (legacy)
# ============================================================================


class IMU2D(Sensor):
    """Inertial Measurement Unit (2D)."""

    def __init__(self, kernel: Kernel2D):
        self._kernel = kernel

    def read(self) -> Tuple[float, float]:
        """
        Currently, reads the true orientation and angular velocity of the rocket.

        Returns:
            Tuple[float, float]: The true orientation and angular velocity of the
            rocket in radians.
        """
        return (
            self._kernel.rocket.state.orientation,
            self._kernel.rocket.state.angular_velocity,
        )


class GPS2D(Sensor):
    """Global Positioning System (2D)."""

    def __init__(self, kernel: Kernel2D):
        self._kernel = kernel

    def read(self) -> WorldVec2:
        """
        Currently, reads the true position of the rocket.

        Returns:
            WorldVec2: The true position of the rocket in the world frame.
        """
        return self._kernel.rocket.state.position


class ThrustSensor2D(Sensor):
    """Sensor that reads the thrust magnitude of the rocket (2D)."""

    def __init__(self, kernel: Kernel2D):
        self._kernel = kernel

    def read(self) -> float:
        """
        Reads the thrust of the rocket.

        Returns:
            float: The thrust of the rocket in Newtons.
        """
        return self._kernel.rocket.state.thrust_magnitude


class VelocitySensor2D(Sensor):
    """Sensor that reads the linear velocity of the rocket (2D)."""

    def __init__(self, kernel: Kernel2D):
        self._kernel = kernel

    def read(self) -> WorldVec2:
        """
        Currently, reads the true linear velocity of the rocket.

        Returns:
            WorldVec2: The true linear velocity of the rocket in the world frame (m/s).
        """
        return self._kernel.rocket.state.velocity


class Sensors2D:
    """Bundle of 2D sensors."""

    def __init__(self, kernel: Kernel2D):
        self._kernel = kernel
        self.GPS = GPS2D(self._kernel)
        self.IMU = IMU2D(self._kernel)
        self.ThrustSensor = ThrustSensor2D(self._kernel)
        self.VelocitySensor = VelocitySensor2D(self._kernel)


# Backwards-compatible alias for existing code that expects `Sensors`.
Sensors = Sensors2D


# ============================================================================
# 3D sensors
# ============================================================================


class IMU3D(Sensor):
    """Inertial Measurement Unit (3D)."""

    def __init__(self, kernel: Kernel3D):
        self._kernel = kernel

    def read(self) -> Tuple[Orientation, RocketVec3]:
        """
        Reads the true orientation and body-frame angular velocity of the rocket.

        Returns:
            (Orientation, RocketVec3): True attitude and angular velocity.
        """
        return (
            self._kernel.rocket.state.orientation,
            self._kernel.rocket.state.angular_velocity_body,
        )


class GPS3D(Sensor):
    """Global Positioning System (3D)."""

    def __init__(self, kernel: Kernel3D):
        self._kernel = kernel

    def read(self) -> WorldVec3:
        """
        Reads the true position of the rocket in the world frame.
        """
        return self._kernel.rocket.state.position


class ThrustSensor3D(Sensor):
    """Sensor that reads the thrust magnitude of the rocket (3D)."""

    def __init__(self, kernel: Kernel3D):
        self._kernel = kernel

    def read(self) -> float:
        """
        Reads the thrust magnitude of the rocket.

        Returns:
            float: The thrust of the rocket in Newtons.
        """
        return self._kernel.rocket.state.thrust_magnitude


class VelocitySensor3D(Sensor):
    """Sensor that reads the linear velocity of the rocket (3D)."""

    def __init__(self, kernel: Kernel3D):
        self._kernel = kernel

    def read(self) -> WorldVec3:
        """
        Reads the true linear velocity of the rocket.

        Returns:
            WorldVec3: True linear velocity (m/s) in the world frame.
        """
        return self._kernel.rocket.state.velocity


class Sensors3D:
    """Bundle of 3D sensors."""

    def __init__(self, kernel: Kernel3D):
        self._kernel = kernel
        self.GPS = GPS3D(self._kernel)
        self.IMU = IMU3D(self._kernel)
        self.ThrustSensor = ThrustSensor3D(self._kernel)
        self.VelocitySensor = VelocitySensor3D(self._kernel)
