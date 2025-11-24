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
from ..schema import ConstantMessage, MetadataMessage, VariableMessage, write_message

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
            controller: Controller instance (engineer's code)
            callback: Optional callback function called each step
            dt: Simulation timestep in seconds
        """
        self.kernel = kernel
        self.sensors = Sensors(self.kernel)
        self.controller = controller.with_sensors(self.sensors)
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
        self._write_variable_message()
        self._time += self.dt

    def _write_metadata_message(self, duration: float) -> None:
        message = MetadataMessage(
            dt=self.dt,
            duration=duration,
        )
        write_message(message)

    def _write_constant_message(self) -> None:
        message = ConstantMessage(
            timestamp=self._time,
            world_gravity=self.kernel.environment.gravity,
            world_wind=self.kernel.environment.wind,
            world_air_density=self.kernel.environment.air_density,
            rocket_length=self.kernel.rocket.properties.length,
            rocket_mass=self.kernel.rocket.properties.mass,
            rocket_center_of_mass_offset=self.kernel.rocket.properties.center_of_mass_offset,
            rocket_moment_of_inertia=self.kernel.rocket.properties.moment_of_inertia,
            rocket_thrust_application_offset=self.kernel.rocket.properties.thrust_application_offset,
        )
        write_message(message)

    def _write_variable_message(self) -> None:
        message = VariableMessage(
            timestamp=self._time,
            position=self.kernel.rocket.state.position,
            velocity=self.kernel.rocket.state.velocity,
            orientation=self.kernel.rocket.state.orientation,
            angular_velocity=self.kernel.rocket.state.angular_velocity,
            thrust_vector=self.kernel.rocket.state.thrust_vector,
            gimbal_angle=self.kernel.rocket.state.gimbal_angle,
            gps=self.sensors.GPS.read(),
            thrust_sensor=self.sensors.ThrustSensor.read(),
            velocity_sensor=self.sensors.VelocitySensor.read(),
        )
        write_message(message)

    def run(self, duration: float):
        self._write_metadata_message(duration)
        self._write_constant_message()
        while self._time < duration:
            self.step()
