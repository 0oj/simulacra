"""
The corruption of truth. Sensor errors that veil reality from the rocket,
introducing noise, bias, and other imperfections into the prisoner's perception.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

class SensorError(ABC):
    @abstractmethod
    def apply(self, truth: float, t: float) -> float: ...


@dataclass
class GaussianNoise(SensorError):
    mean: float = 0.0
    std: float = 0.0

    def apply(self, truth: float, t: float) -> float:
        falsity = truth + np.random.normal(self.mean, self.std)
        return falsity

@dataclass
class Bias(SensorError):
    offset: float = 0.0

    def apply(self, truth: float, t: float) -> float:
        falsity = truth + self.offset
        return falsity

@dataclass
class IdentityError(SensorError):
    def apply(self, truth: float, t: float) -> float:
        return truth
