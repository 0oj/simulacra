"""
Onboard safety envelope — flight computer logic for actuator feedback and state awareness.

Reads actuator saturation state (exposed by Phenomena) and sensor-derived values
(computed by the controller) to produce observation signals and reward penalties.
This is an onboard computation, not a sensor simulation — the flight computer
processing actuator + sensor data into derived safety margins.

The check definitions (SafetyCheck ABC, concrete checks) live here because they
define onboard flight envelope protection logic. The actuator saturation properties
they reference are exposed by the Phenomena layer (actuators know when they're
saturated), but the decision to aggregate and act on them is an onboard concern.

Usage:
    from simulacra.onboard.safety_envelope import SafetyEnvelope, default_checks
    envelope = SafetyEnvelope(throttle, gimbal, default_checks())
    envelope.update_state(speed=..., v_safe=..., ...)
    signals = envelope.read()       # shape: (6,)
    penalty = envelope.read_penalty()  # scalar ≤ 0
"""

from abc import ABC, abstractmethod

import numpy as np

from ..phenomena.actuators import ThrottleActuator, GimbalActuator


# ──────────────────────────────────────────────────────────────
# Check ABC and config
# ──────────────────────────────────────────────────────────────

class SafetyCheck(ABC):
    """One safety envelope check. Produces obs signals and reward penalties."""

    name: str
    signal_dim: int

    @abstractmethod
    def compute_signal(self, ctx: dict) -> np.ndarray:
        """Return signal for observation. Shape: (signal_dim,)."""

    @abstractmethod
    def compute_penalty(self, ctx: dict) -> float:
        """Return reward penalty (≤ 0)."""

    @abstractmethod
    def obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (low, high) for this check's obs dimensions."""


class SafetyEnvelopeConfig:
    """Composable registry of safety checks."""

    def __init__(self, checks: list[SafetyCheck]):
        self.checks = checks
        self.total_signal_dim = sum(c.signal_dim for c in checks)

    def compute_all_signals(self, ctx: dict) -> np.ndarray:
        return np.concatenate([c.compute_signal(ctx) for c in self.checks])

    def compute_total_penalty(self, ctx: dict) -> float:
        return sum(c.compute_penalty(ctx) for c in self.checks)

    def obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lows = np.concatenate([c.obs_bounds()[0] for c in self.checks])
        highs = np.concatenate([c.obs_bounds()[1] for c in self.checks])
        return lows, highs


# ──────────────────────────────────────────────────────────────
# Concrete checks
# ──────────────────────────────────────────────────────────────

class ThrottleSaturationCheck(SafetyCheck):
    """Throttle actuator saturation feedback (signal_dim=1).

    Signal: clipped throttle saturation ratio from actuator.
    Penalty: quadratic in saturation — discourages commands the actuator can't follow.
    """

    name = "throttle_sat"
    signal_dim = 1

    def __init__(self, penalty_weight: float = 0.02):
        self.weight = penalty_weight

    def compute_signal(self, ctx: dict) -> np.ndarray:
        sat = np.clip(ctx["throttle_saturation"], -3.0, 3.0)
        return np.array([sat], dtype=np.float32)

    def compute_penalty(self, ctx: dict) -> float:
        sat = min(3.0, max(-3.0, ctx["throttle_saturation"]))
        return -self.weight * sat ** 2

    def obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-3.0], dtype=np.float32), np.array([3.0], dtype=np.float32)


class GimbalSaturationCheck(SafetyCheck):
    """Gimbal actuator saturation feedback (signal_dim=2).

    Signal: clipped outer/inner gimbal saturation ratios.
    Penalty: quadratic sum of both axes.
    """

    name = "gimbal_sat"
    signal_dim = 2

    def __init__(self, penalty_weight: float = 0.01):
        self.weight = penalty_weight

    def compute_signal(self, ctx: dict) -> np.ndarray:
        outer = np.clip(ctx["gimbal_outer_saturation"], -3.0, 3.0)
        inner = np.clip(ctx["gimbal_inner_saturation"], -3.0, 3.0)
        return np.array([outer, inner], dtype=np.float32)

    def compute_penalty(self, ctx: dict) -> float:
        outer = max(-3.0, min(3.0, ctx["gimbal_outer_saturation"]))
        inner = max(-3.0, min(3.0, ctx["gimbal_inner_saturation"]))
        return -self.weight * (outer ** 2 + inner ** 2)

    def obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([-3.0, -3.0], dtype=np.float32),
            np.array([3.0, 3.0], dtype=np.float32),
        )


class SpeedEnvelopeCheck(SafetyCheck):
    """Speed relative to safe envelope (signal_dim=1).

    Signal: speed / v_safe, clipped to [0, 3]. >1 means exceeding safe speed.
    Penalty: 0 (avoid double-counting with r_brake in the reward).
    """

    name = "speed_envelope"
    signal_dim = 1

    def compute_signal(self, ctx: dict) -> np.ndarray:
        v_safe = max(ctx["v_safe"], 0.1)
        ratio = np.clip(ctx["speed"] / v_safe, 0.0, 3.0)
        return np.array([ratio], dtype=np.float32)

    def compute_penalty(self, ctx: dict) -> float:
        return 0.0

    def obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0.0], dtype=np.float32), np.array([3.0], dtype=np.float32)


class TiltMarginCheck(SafetyCheck):
    """Proximity to tilt limit (signal_dim=1).

    Signal: 1 - tilt/max_tilt. 1 = perfectly upright, 0 = at limit.
    Penalty: 0 (avoid double-counting with r_tilt).
    """

    name = "tilt_margin"
    signal_dim = 1

    def compute_signal(self, ctx: dict) -> np.ndarray:
        max_tilt = max(ctx["max_tilt"], 1e-6)
        margin = np.clip(1.0 - ctx["tilt"] / max_tilt, 0.0, 1.0)
        return np.array([margin], dtype=np.float32)

    def compute_penalty(self, ctx: dict) -> float:
        return 0.0

    def obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)


class DescentRateCheck(SafetyCheck):
    """Vertical descent speed vs altitude budget (signal_dim=1).

    Signal: descent_speed / v_safe_vertical, clipped to [0, 3].
    Penalty: quadratic when exceeding safe descent rate.
    """

    name = "descent_rate"
    signal_dim = 1

    def __init__(self, penalty_weight: float = 0.1, a_max: float = 4.0):
        self.weight = penalty_weight
        self.a_max = a_max

    def compute_signal(self, ctx: dict) -> np.ndarray:
        descent_speed = max(0.0, -ctx["vy"])
        v_safe_vert = max(0.1, (2.0 * self.a_max * max(0.1, ctx["alt"])) ** 0.5)
        ratio = np.clip(descent_speed / v_safe_vert, 0.0, 3.0)
        return np.array([ratio], dtype=np.float32)

    def compute_penalty(self, ctx: dict) -> float:
        descent_speed = max(0.0, -ctx["vy"])
        v_safe_vert = max(0.1, (2.0 * self.a_max * max(0.1, ctx["alt"])) ** 0.5)
        excess = max(0.0, descent_speed - v_safe_vert)
        return -self.weight * excess ** 2

    def obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0.0], dtype=np.float32), np.array([3.0], dtype=np.float32)


def default_checks() -> list[SafetyCheck]:
    """Standard 6-dim safety envelope (throttle + gimbal + speed + tilt + descent)."""
    return [
        ThrottleSaturationCheck(),
        GimbalSaturationCheck(),
        SpeedEnvelopeCheck(),
        TiltMarginCheck(),
        DescentRateCheck(),
    ]


# ──────────────────────────────────────────────────────────────
# Safety envelope aggregator
# ──────────────────────────────────────────────────────────────

class SafetyEnvelope:
    """Onboard flight envelope protection — aggregates actuator feedback and state margins.

    Reads actuator saturation properties (Phenomena-layer, the actuator knows
    when it's clipping) and sensor-derived values (computed by the controller)
    to produce a fixed-size signal vector for the observation space.

    Not a sensor. This is flight computer logic.
    """

    def __init__(self, throttle: ThrottleActuator, gimbal: GimbalActuator,
                 checks: list[SafetyCheck] | None = None):
        self._throttle = throttle
        self._gimbal = gimbal
        self._config = SafetyEnvelopeConfig(checks or default_checks())
        # State-derived values (set by controller before read())
        self._speed: float = 0.0
        self._v_safe: float = 0.0
        self._tilt: float = 0.0
        self._max_tilt: float = 1.0
        self._vy: float = 0.0
        self._alt: float = 0.0

    @property
    def signal_dim(self) -> int:
        return self._config.total_signal_dim

    def obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._config.obs_bounds()

    def update_state(self, speed: float, v_safe: float, tilt: float,
                     max_tilt: float, vy: float, alt: float) -> None:
        """Set sensor-derived values before calling read(). Called by controller."""
        self._speed = speed
        self._v_safe = v_safe
        self._tilt = tilt
        self._max_tilt = max_tilt
        self._vy = vy
        self._alt = alt

    def _build_ctx(self) -> dict:
        return {
            "throttle_saturation": self._throttle.saturation_ratio,
            "gimbal_outer_saturation": self._gimbal.outer_saturation,
            "gimbal_inner_saturation": self._gimbal.inner_saturation,
            "speed": self._speed,
            "v_safe": self._v_safe,
            "tilt": self._tilt,
            "max_tilt": self._max_tilt,
            "vy": self._vy,
            "alt": self._alt,
        }

    def read(self) -> np.ndarray:
        """Return safety signal vector. Shape: (signal_dim,)."""
        return self._config.compute_all_signals(self._build_ctx())

    def read_penalty(self) -> float:
        """Return total reward penalty for current state."""
        return self._config.compute_total_penalty(self._build_ctx())
