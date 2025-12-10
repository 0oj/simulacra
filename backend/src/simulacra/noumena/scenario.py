"""
The test scenario. God orchestrates the simulation, running the controller,
advancing physics, and recording the truth for later inspection.
"""

import heapq
from itertools import count
from typing import Callable, Optional

from ..noumena import Kernel2D, Kernel3D
from ..onboard.controller import Controller
from ..phenomena.sensors import Sensors2D, Sensors3D
from ..schema import ConstantMessage, MetadataMessage, VariableMessage, write_message


Callback2D = Callable[[float, Kernel2D, Controller], None]
Callback3D = Callable[[float, Kernel3D, Controller], None]


def default_callback_2d(t: float, kernel: Kernel2D, controller: Controller) -> None: ...


def default_callback_3d(t: float, kernel: Kernel3D, controller: Controller) -> None: ...


class Event:
    """An event scheduled to execute at a specific timestamp."""

    _seq_incr = count(-1)

    def __init__(
        self,
        at_time: float,
        callback: Callback2D | Callback3D,
    ):
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


class Scenario2D: 
    """
    2D scenario (legacy).

    It runs the controller, advances the physics, and records the truth for later inspection.
    """

    _time: float = 0.0
    _events: list[Event] = []

    def __init__(
        self,
        kernel: Kernel2D,
        controller: Controller,
        callback: Optional[Callback2D] = default_callback_2d,
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
        self.sensors = Sensors2D(self.kernel)
        self.controller = controller.with_sensors(self.sensors)
        self.callback = callback
        self.dt = dt

    def add_event(self, at_time: float, callback: Callback2D) -> None:
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
            rocket_length=self.kernel.rocket.props.length,
            rocket_mass=self.kernel.rocket.props.mass,
            rocket_center_of_mass_offset=self.kernel.rocket.props.center_of_mass_offset,
            rocket_moment_of_inertia=self.kernel.rocket.props.moment_of_inertia,
            rocket_thrust_application_offset=self.kernel.rocket.props.thrust_application_offset,
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


class Scenario3D:
    """
    3D scenario.

    It runs the controller, advances the physics, and (eventually) records the
    truth for later inspection. For now, logging is minimal and focused on
    orchestration; the 3D schema will evolve separately.
    """

    _time: float = 0.0
    _events: list[Event] = []

    def __init__(
        self,
        kernel: Kernel3D,
        controller: Controller,
        callback: Optional[Callback3D] = default_callback_3d,
        dt: float = 0.01,
    ):
        self.kernel = kernel
        self.sensors = Sensors3D(self.kernel)
        self.controller = controller.with_sensors(self.sensors)
        self.callback = callback
        self.dt = dt

    def add_event(self, at_time: float, callback: Callback3D) -> None:
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
            world_gravity_3d=self.kernel.environment.gravity,
            world_wind_3d=self.kernel.environment.wind,
            world_air_density=self.kernel.environment.air_density,
            rocket_length=self.kernel.rocket.props.length,
            rocket_mass=self.kernel.rocket.props.mass,
            rocket_center_of_mass_offset_3d=self.kernel.rocket.props.center_of_mass_offset,
            rocket_inertia_diag_3d=self.kernel.rocket.props.inertia_diag,
            rocket_thrust_application_offset_3d=self.kernel.rocket.props.thrust_application_offset,
        )
        write_message(message)

    def _write_variable_message(self) -> None:
        # True 3D state (noumena)
        state = self.kernel.rocket.state
        gps_reading = self.sensors.GPS.read()
        thrust_reading = self.sensors.ThrustSensor.read()
        velocity_reading = self.sensors.VelocitySensor.read()

        message = VariableMessage(
            timestamp=self._time,
            position_3d=state.position,
            velocity_3d=state.velocity,
            orientation_quat=tuple(state.orientation.as_array()),
            angular_velocity_body_3d=state.angular_velocity_body,
            thrust_vector_body_3d=state.thrust_direction,
            gps_3d=gps_reading,
            thrust_sensor=thrust_reading,
            velocity_sensor_3d=velocity_reading,
        )
        write_message(message)

    def run(self, duration: float) -> None:
        """
        Run the 3D scenario for the given duration.
        """
        self._write_metadata_message(duration)
        self._write_constant_message()
        while self._time < duration:
            self.step()
