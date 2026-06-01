"""Default sensor and actuator constructors for scenarios."""
from __future__ import annotations

from simulacra.noumena import Kernel
from simulacra.phenomena.sensors import Sensors
from simulacra.phenomena.actuators import (
    Actuators, GimbalActuator, ThrottleActuator,
)


def sensors_for(kernel: Kernel) -> Sensors:
    """Default sensor suite wired to `kernel`.

    Sensors take the kernel (not the rocket) per the phenomena API.
    The Scenario tick loop reads each sensor and pushes into the bus.
    """
    return Sensors(kernel)


def actuators_for(rocket) -> Actuators:
    """Default actuator suite (throttle + 20° gimbal) for the given rocket."""
    return Actuators(
        throttle=ThrottleActuator(rocket),
        gimbal=GimbalActuator(rocket, max_outer_angle=0.349, max_inner_angle=0.349),
    )
