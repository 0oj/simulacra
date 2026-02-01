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
from ..schema import ConstantMessage, MetadataMessage, VariableMessage, collect, write_message


Callback = Callable[[float, Kernel, Controller], None]


def default_callback(t: float, kernel: Kernel, controller: Controller) -> None: ...


class Event:
    """An event scheduled to execute at a specific timestamp."""

    _seq_incr = count(-1)

    def __init__(
        self,
        at_time: float,
        callback: Callback,
    ):
        """
        Initialize event.

        Args:
            at_time: Time at which the event should execute
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
    """
    Scenario orchestrator.

    It runs the controller, advances the physics, and records the
    truth for later inspection.
    """

    _time: float = 0.0
    _events: list[Event] = []

    def __init__(
        self,
        kernel: Kernel,
        controller: Controller,
        callback: Optional[Callback] = default_callback,
        dt: float = 0.01,
    ):
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
        message = collect(
            ConstantMessage, self._time,
            props=self.kernel.rocket.props,
            env=self.kernel.environment,
        )
        write_message(message)

    def _write_variable_message(self) -> None:
        message = collect(
            VariableMessage, self._time,
            state=self.kernel.rocket.state,
            sensors=self.sensors,
            props=self.kernel.rocket.props,
            model=self.kernel.rocket.mass_model,
            ground=self.kernel.ground_contact,
        )
        write_message(message)

    def run(self, duration: float) -> None:
        """
        Run the scenario for the given duration.
        """
        self._write_metadata_message(duration)
        self._write_constant_message()
        while self._time < duration:
            self.step()
