from typing import Tuple

from noumena import Kernel
from physics import WorldVec2


class IMU:
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

class GPS:
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

class ThrustSensor:
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

class VelocitySensor:
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
        self.GPS = GPS(kernel)
        self.IMU = IMU(kernel)
        self.ThrustSensor = ThrustSensor(kernel)
        self.VelocitySensor = VelocitySensor(kernel)
