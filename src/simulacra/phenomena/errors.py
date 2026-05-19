"""
The corruption of truth. Sensor errors that veil reality from the rocket,
introducing noise, bias, and other imperfections into the prisoner's perception.

All errors operate on scalar values. Multi-dimensional sensors (GPS, IMU)
decompose into named channels, each with its own error chain.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


class SensorError(ABC):
    """Base class for sensor error models."""

    @abstractmethod
    def apply(self, truth: float, t: float) -> float:
        """
        Apply error to a true scalar value.

        Args:
            truth: The true value from the noumenal world.
            t: Current simulation time (seconds).

        Returns:
            The corrupted value as perceived by the sensor.
        """
        ...


@dataclass
class IdentityError(SensorError):
    """No error: returns truth unchanged. Default for backward compatibility."""

    def apply(self, truth: float, t: float) -> float:
        return truth


@dataclass
class GaussianNoise(SensorError):
    """
    Additive Gaussian noise with configurable mean and standard deviation.

    Uses a seeded random number generator for reproducibility.
    """

    mean: float = 0.0
    std: float = 0.0
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def apply(self, truth: float, t: float) -> float:
        noise = self._rng.normal(self.mean, self.std)
        return truth + noise


@dataclass
class Bias(SensorError):
    """Constant additive offset."""

    offset: float = 0.0

    def apply(self, truth: float, t: float) -> float:
        return truth + self.offset


@dataclass
class ScaleBias(SensorError):
    """Multiplicative scale error (e.g., sensor reads 1.02x true value)."""

    scale: float = 1.0

    def apply(self, truth: float, t: float) -> float:
        return truth * self.scale


@dataclass
class RandomWalk(SensorError):
    """
    Random walk (integrated white noise) for drift modeling.

    The drift accumulates over time: drift += sigma * sqrt(dt) * N(0,1) each step.
    """

    sigma: float = 0.0  # drift rate (units/sqrt(s))
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _drift: float = field(init=False, default=0.0, repr=False)
    _last_t: float = field(init=False, default=-1.0, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._drift = 0.0
        self._last_t = -1.0

    def apply(self, truth: float, t: float) -> float:
        if self._last_t >= 0:
            dt = t - self._last_t
            if dt > 0:
                self._drift += self.sigma * np.sqrt(dt) * self._rng.standard_normal()
        self._last_t = t
        return truth + self._drift

    def reset(self) -> None:
        """Reset drift state (for scenario reset)."""
        self._drift = 0.0
        self._last_t = -1.0


@dataclass
class CompositeError(SensorError):
    """
    Chain multiple errors in sequence.

    Errors are applied left-to-right: errors[0].apply() first, then errors[1], etc.
    """

    errors: Sequence[SensorError] = field(default_factory=list)

    def apply(self, truth: float, t: float) -> float:
        value = truth
        for error in self.errors:
            value = error.apply(value, t)
        return value


@dataclass
class Quantization(SensorError):
    """
    Quantization error (ADC resolution).

    Rounds to nearest multiple of step_size.
    """

    step_size: float = 1.0

    def apply(self, truth: float, t: float) -> float:
        if self.step_size <= 0:
            return truth
        return round(truth / self.step_size) * self.step_size


@dataclass
class Saturation(SensorError):
    """Clamp output to [min_val, max_val] range."""

    min_val: float = float("-inf")
    max_val: float = float("inf")

    def apply(self, truth: float, t: float) -> float:
        return max(self.min_val, min(self.max_val, truth))


# Type alias for error configuration dictionaries
ChannelErrors = dict[str, SensorError]
