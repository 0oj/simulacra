"""
The rocket's eyes. Sensors that peek into god's truth and report back,
sometimes accurately, sometimes with error, but always limited.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, Tuple

from ..noumena import Kernel
from ..physics import Orientation, RocketVec3, WorldVec3


class Sensor(ABC):
    @abstractmethod
    def read(self) -> Any: ...


class SensorsInterface(Protocol):
    """
    Minimal interface that sensor bundles satisfy.

    The controller only relies on these attributes existing, not on their
    exact return types.
    """

    GPS: Sensor
    IMU: Sensor
    ThrustSensor: Sensor
    VelocitySensor: Sensor


class IMU(Sensor):
    """Inertial Measurement Unit."""

    def __init__(self, kernel: Kernel):
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


class GPS(Sensor):
    """Global Positioning System."""

    def __init__(self, kernel: Kernel):
        self._kernel = kernel

    def read(self) -> WorldVec3:
        """
        Reads the true position of the rocket in the world frame.
        """
        return self._kernel.rocket.state.position


class ThrustSensor(Sensor):
    """Sensor that reads the thrust magnitude of the rocket."""

    def __init__(self, kernel: Kernel):
        self._kernel = kernel

    def read(self) -> float:
        """
        Reads the thrust magnitude of the rocket.

        Returns:
            float: The thrust of the rocket in Newtons.
        """
        return self._kernel.rocket.state.thrust_body.norm


class VelocitySensor(Sensor):
    """Sensor that reads the linear velocity of the rocket."""

    def __init__(self, kernel: Kernel):
        self._kernel = kernel

    def read(self) -> WorldVec3:
        """
        Reads the true linear velocity of the rocket.

        Returns:
            WorldVec3: True linear velocity (m/s) in the world frame.
        """
        return self._kernel.rocket.state.velocity


class Sensors:
    """Bundle of sensors."""

    def __init__(self, kernel: Kernel):
        self._kernel = kernel
        self.GPS = GPS(self._kernel)
        self.IMU = IMU(self._kernel)
        self.ThrustSensor = ThrustSensor(self._kernel)
        self.VelocitySensor = VelocitySensor(self._kernel)
