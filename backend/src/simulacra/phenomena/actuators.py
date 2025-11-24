"""
The rocket's effectors. Commands from the controller become physical changes,
but only if god allows it (saturation, rate limits, failures).
"""

from dataclasses import dataclass

import numpy as np

from ..noumena import Kernel


@dataclass
class GimbalCommand:
    desired_gimbal_angle: float  # radians, desired gimbal angle


class GimbalActuator:
    max_gimbal_angle: float  # radians, maximum gimbal angle

    def __init__(self, kernel: Kernel, max_gimbal_angle: float):
        self.max_gimbal_angle = max_gimbal_angle
        self._kernel = kernel

    def apply(self, command: GimbalCommand) -> None:
        ...  # TODO: implement rate limiting, saturation, etc.
        self._kernel.rocket.state.gimbal_angle = np.clip(
            command.desired_gimbal_angle,
            -self.max_gimbal_angle,
            self.max_gimbal_angle,
        )

@dataclass
class ThrustCommand:
    desired_thrust_magnitude: float  # Newtons, desired thrust magnitude


class ThrustActuator:
    max_thrust_magnitude: float  # Newtons, maximum thrust magnitude

    def __init__(self, kernel: Kernel, max_thrust_magnitude: float):
        self.max_thrust_magnitude = max_thrust_magnitude
        self._kernel = kernel

    def apply(self, command: ThrustCommand) -> None:
        self._kernel.rocket.state.thrust_magnitude = np.clip(
            command.desired_thrust_magnitude,
            0.0,
            self.max_thrust_magnitude,
        )