"""
The test scenario. God orchestrates the simulation, running the controller,
advancing physics, and recording the truth for later inspection.
"""
from typing import Callable, Optional

from ..noumena import Kernel
from ..onboard.controller import Controller
from ..phenomena.sensors import Sensors
from ..schema import SimState, write_sim_state

Callback = Callable[[float, Kernel, Controller], None]


def default_callback(t: float, kernel: Kernel, controller: Controller): ...


class Scenario:
    _time: float = 0.0

    def __init__(
        self,
        kernel: Kernel,
        controller: Controller,
        callback: Optional[Callback] = default_callback,
        dt: float = 0.01,
    ):
        """
        Initialize scenario.

        Args:
            kernel: Kernel instance (physics engine)
            sensors: Sensors object providing access to GPS, IMU, etc.
            controller: Controller instance (engineer's code)
            callback: Optional callback function called each step
            dt: Simulation timestep in seconds
        """
        self.kernel = kernel
        self.sensors = Sensors(self.kernel)
        self.controller = controller
        self.callback = callback
        self.dt = dt

    def step(self) -> None:
        self.controller.run()
        self.kernel.step(self.dt)
        self.callback(self._time, self.kernel, self.controller)
        self._time += self.dt

    def run(self, duration: float):
        while self._time < duration:
            self.step()

            write_sim_state(
                SimState(
                    t=self._time,
                    pos_x=self.kernel.rocket.state.position.x,
                    pos_y=self.kernel.rocket.state.position.y,
                    gimbal_angle=self.kernel.rocket.state.gimbal_angle,
                    orientation=self.kernel.rocket.state.orientation,
                    angular_velocity=self.kernel.rocket.state.angular_velocity,
                    thrust_magnitude=self.kernel.rocket.state.thrust_magnitude,
                    gps_x=self.sensors.GPS.read().x,
                    gps_y=self.sensors.GPS.read().y,
                    imu_orientation=self.sensors.IMU.read()[0],
                    imu_angular_velocity=self.sensors.IMU.read()[1],
                    thrust_sensor=self.sensors.ThrustSensor.read(),
                    rocket_length=self.kernel.rocket.properties.length,
                )
            )
