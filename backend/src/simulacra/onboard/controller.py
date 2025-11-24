"""
The rocket's control logic. The prisoner's mind that reads sensors
and commands actuators, never knowing the true state of things.
"""
from abc import ABC, abstractmethod
from ..phenomena.sensors import Sensors

class Controller(ABC):
    @abstractmethod
    def run(self) -> None: ...

class IdentityController(Controller):
    def __init__(self, sensors: Sensors = None):
        self.sensors = sensors

    def run(self) -> None: ...
