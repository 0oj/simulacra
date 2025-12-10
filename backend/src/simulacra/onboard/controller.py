"""
The rocket's control logic. The prisoner's mind that reads sensors
and commands actuators, never knowing the true state of things.
"""

from abc import ABC, abstractmethod

from ..phenomena.sensors import Sensors


class Controller(ABC):
    @abstractmethod
    def run(self) -> None: ...

    @abstractmethod
    def with_sensors(self, sensors: Sensors) -> "Controller": ...


class IdentityController(Controller):
    def __init__(self):
        self.sensors = None

    def with_sensors(self, sensors: Sensors) -> "Controller":
        self.sensors = sensors
        return self

    def run(self) -> None:
        if self.sensors is None:
            raise ValueError("Sensors not attached to controller")
        else:
            ...
