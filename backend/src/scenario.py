from noumena import Kernel
from onboard.controller import Controller
from typing import Callable, Optional
from plot.schema import SimState, write_sim_state
from phenomena.sensors import Sensors

Callback = Callable[[float, Kernel, Controller], None]


class Scenario:
    def __init__(
        self,
        kernel: Kernel,
        sensors: Sensors,
        controller: Controller,
        callback: Optional[Callback] = None,
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
        self.sensors = sensors
        self.controller = controller
        self.callback = callback or (lambda t, kernel, controller: None)
        self.dt = dt

    def run(self, duration: float):
        t = 0.0

        while t < duration:
            self.controller.run()
            self.kernel.step(self.dt)
            self.callback(t, self.kernel, self.controller)
            write_sim_state(
                SimState(
                    t=t,
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
            t += self.dt
