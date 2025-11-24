"""
The rocket's eyes. Sensors that peek into god's truth and report back,
sometimes accurately, sometimes with error, but always limited.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from ..noumena import Kernel
from ..physics import WorldVec2


class Sensor(ABC):
    @abstractmethod
    def read(self) -> Any: ...


class IMU(Sensor):
    """Inertial Measurement Unit"""

    def __init__(self, kernel: Kernel):
        self._kernel = kernel

    def read(self) -> Tuple[float, float]:
        """
        Currently, reads the true orientation and angular velocity of the rocket.

        Returns:
            Tuple[float, float]: The true orientation and angular velocity of the rocket in radians.
        """
        return (
            self._kernel.rocket.state.orientation,
            self._kernel.rocket.state.angular_velocity,
        )


class GPS(Sensor):
    """Global Positioning System"""

    def __init__(self, kernel: Kernel):
        self._kernel = kernel

    def read(self) -> WorldVec2:
        """
        Currently, reads the true position of the rocket.
        Returns:
            WorldVec2: The true position of the rocket in the world frame.
        """

        return self._kernel.rocket.state.position


class ThrustSensor(Sensor):
    """Sensor that reads the thrust of the rocket."""

    def __init__(self, kernel: Kernel):
        self._kernel = kernel

    def read(self) -> float:
        """
        Reads the thrust of the rocket.
        Returns:
            float: The thrust of the rocket in Newtons.
        """
        return self._kernel.rocket.state.thrust_magnitude


class VelocitySensor(Sensor):
    """Sensor that reads the linear velocity of the rocket."""

    def __init__(self, kernel: Kernel):
        self._kernel = kernel

    def read(self) -> WorldVec2:
        """
        Currently, reads the true linear velocity of the rocket.
        Returns:
            WorldVec2: The true linear velocity of the rocket in the world frame (m/s).
        """
        return self._kernel.rocket.state.velocity


class Sensors:
    def __init__(self, kernel: Kernel):
        self._kernel = kernel
        self.GPS = GPS(self._kernel)
        self.IMU = IMU(self._kernel)
        self.ThrustSensor = ThrustSensor(self._kernel)
        self.VelocitySensor = VelocitySensor(self._kernel)
