"""
PID hover controller. Holds altitude using sensor feedback.
"""

from ..phenomena.actuators import ThrottleActuator, ThrottleCommand
from ..phenomena.sensors import Sensors
from .controller import Controller


class PIDHoverController(Controller):
    """
    PD altitude controller using sensor readings.

    Controls throttle to maintain target altitude. No gimbal control
    (assumes vertical flight only).
    """

    def __init__(
        self,
        throttle: ThrottleActuator,
        target_alt: float,
        weight: float,
        Kp: float = 10.0,
        Kd: float = 5.0,
        max_thrust: float = float("inf"),
    ):
        """
        Args:
            throttle: Throttle actuator to command.
            target_alt: Target altitude (meters).
            weight: Rocket weight (mass * gravity, in Newtons).
            Kp: Proportional gain.
            Kd: Derivative gain.
            max_thrust: Maximum thrust limit (Newtons).
        """
        self.throttle = throttle
        self.target_alt = target_alt
        self.weight = weight
        self.Kp = Kp
        self.Kd = Kd
        self.max_thrust = max_thrust
        self.sensors: Sensors | None = None

    def with_sensors(self, sensors: Sensors) -> "PIDHoverController":
        self.sensors = sensors
        return self

    def run(self) -> None:
        if self.sensors is None:
            raise ValueError("Sensors not attached to controller")

        # Read from sensors (not truth)
        pos = self.sensors.GPS.read()
        vel = self.sensors.VelocitySensor.read()

        alt = pos.y
        vy = vel.y

        # PD control: thrust = weight + Kp*(target - alt) - Kd*vy
        error = self.target_alt - alt
        thrust = self.weight + self.Kp * error - self.Kd * vy

        # Clamp to valid range
        thrust = max(0.0, min(thrust, self.max_thrust))

        self.throttle.apply(ThrottleCommand(thrust_N=thrust))
