"""
The test scenario. God orchestrates the simulation, running the controller,
advancing physics, and recording the truth for later inspection.
"""
import heapq
from itertools import count
from typing import Callable, Optional

from ..noumena import Kernel
from ..onboard.controller import Controller
from ..phenomena.sensors import Sensors
from ..schema import SimState, write_sim_state

Callback = Callable[[float, Kernel, Controller], None]


def default_callback(t: float, kernel: Kernel, controller: Controller): ...


class Event:
    """An event scheduled to execute at a specific timestamp."""

    _seq_incr = count(-1)

    def __init__(self, at_time: float, callback: Callback):
        """
        Initialize event.

        Args:
            at_time: Time at which the event should execute
            counter: Event counter for tie-breaking (maintains insertion order)
            callback: Callback function with signature (t, kernel, controller)
        """
        self.at_time = at_time
        self.seq_no = next(self._seq_incr)
        self.callback = callback

    def __lt__(self, other: "Event") -> bool:
        """Comparison for priority queue ordering."""
        if self.at_time != other.at_time:
            return self.at_time < other.at_time
        return self.seq_no < other.seq_no


class Scenario:
    _time: float = 0.0
    _events: list[Event] = []
    
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

    def add_event(self, at_time: float, callback: Callback) -> None:
        """
        Schedule an event to execute at a specific timestamp.

        Args:
            at_time: Time at which the event should execute
            callback: Callback function with signature (t, kernel, controller)
        """
        event = Event(at_time, callback)
        heapq.heappush(self._events, event)

    def _exec_due_events(self) -> None:
        """Process all events that should execute at the current time."""
        while self._events and self._events[0].at_time <= self._time:
            event: Event = heapq.heappop(self._events)
            event.callback(self._time, self.kernel, self.controller)

    def step(self) -> None:
        self.controller.run()
        self.kernel.step(self.dt)
        self._exec_due_events()
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
