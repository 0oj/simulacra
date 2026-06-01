"""
WASD Controller for interactive rocket control.

Translates keyboard input into thrust and gimbal commands.
Designed to receive real-time input from a frontend via WebSocket.

Controls:
    W / S     - Increase / decrease throttle
    A / D     - Gimbal left / right (yaw)
    Q / E     - Gimbal up / down (pitch)
    Space     - Kill throttle (emergency stop)
    Shift     - Boost throttle to max
"""

from dataclasses import dataclass
from typing import Set

from ..phenomena.actuators import (
    ThrottleActuator,
    ThrottleCommand,
    GimbalActuator,
    GimbalCommand,
)
from ..phenomena.sensors import Sensors
from .controller_legacy import Controller


@dataclass
class WASDControllerConfig:
    """Configuration for WASD controller."""

    max_thrust: float = 2000.0  # N
    throttle_rate: float = 500.0  # N/s - how fast throttle changes
    gimbal_rate: float = 0.5  # rad/s - how fast gimbal moves
    max_gimbal_angle: float = 0.26  # rad (~15 degrees)

    # Initial throttle (fraction of max)
    initial_throttle: float = 0.0


class WASDController(Controller):
    """
    Interactive controller that responds to keyboard input.

    Call `key_down(key)` and `key_up(key)` to update the key state,
    then `run()` each simulation step to apply commands.
    """

    def __init__(
        self,
        throttle: ThrottleActuator,
        gimbal: GimbalActuator,
        config: WASDControllerConfig | None = None,
    ):
        self.throttle_actuator = throttle
        self.gimbal_actuator = gimbal
        self.config = config or WASDControllerConfig()
        self.sensors: Sensors | None = None

        # Current state
        self._throttle = self.config.initial_throttle * self.config.max_thrust
        self._gimbal_outer = 0.0  # yaw (A/D)
        self._gimbal_inner = 0.0  # pitch (Q/E)

        # Keys currently held down
        self._keys_down: Set[str] = set()

        # Time tracking for rate-based control
        self._dt = 0.02  # Will be updated by run()

    def with_sensors(self, sensors: Sensors) -> "WASDController":
        self.sensors = sensors
        return self

    def set_dt(self, dt: float) -> None:
        """Set the timestep for rate calculations."""
        self._dt = dt

    def key_down(self, key: str) -> None:
        """Register a key press."""
        self._keys_down.add(key.lower())

    def key_up(self, key: str) -> None:
        """Register a key release."""
        self._keys_down.discard(key.lower())

    def set_keys(self, keys: Set[str]) -> None:
        """Set the complete key state (for WebSocket updates)."""
        self._keys_down = {k.lower() for k in keys}

    def run(self) -> None:
        """Update commands based on current key state."""
        cfg = self.config
        dt = self._dt

        # --- Throttle control ---

        # Space = kill throttle
        if "space" in self._keys_down or " " in self._keys_down:
            self._throttle = 0.0
        # Shift = max throttle
        elif "shift" in self._keys_down:
            self._throttle = cfg.max_thrust
        else:
            # W/S = increase/decrease
            if "w" in self._keys_down:
                self._throttle += cfg.throttle_rate * dt
            if "s" in self._keys_down:
                self._throttle -= cfg.throttle_rate * dt

        # Clamp throttle
        self._throttle = max(0.0, min(cfg.max_thrust, self._throttle))

        # --- Gimbal control ---

        # A/D = yaw (outer gimbal)
        if "a" in self._keys_down:
            self._gimbal_outer += cfg.gimbal_rate * dt
        if "d" in self._keys_down:
            self._gimbal_outer -= cfg.gimbal_rate * dt

        # Q/E = pitch (inner gimbal)
        if "q" in self._keys_down:
            self._gimbal_inner += cfg.gimbal_rate * dt
        if "e" in self._keys_down:
            self._gimbal_inner -= cfg.gimbal_rate * dt

        # Return gimbal to center when keys released
        if "a" not in self._keys_down and "d" not in self._keys_down:
            # Decay toward zero
            self._gimbal_outer *= 0.9
        if "q" not in self._keys_down and "e" not in self._keys_down:
            self._gimbal_inner *= 0.9

        # Clamp gimbal
        self._gimbal_outer = max(-cfg.max_gimbal_angle, min(cfg.max_gimbal_angle, self._gimbal_outer))
        self._gimbal_inner = max(-cfg.max_gimbal_angle, min(cfg.max_gimbal_angle, self._gimbal_inner))

        # --- Apply commands ---

        self.throttle_actuator.apply(ThrottleCommand(thrust_N=self._throttle))
        self.gimbal_actuator.apply(GimbalCommand(
            desired_outer_angle=self._gimbal_outer,
            desired_inner_angle=self._gimbal_inner,
        ))

    @property
    def throttle_percent(self) -> float:
        """Current throttle as percentage."""
        return 100.0 * self._throttle / self.config.max_thrust

    @property
    def state_summary(self) -> dict:
        """Current controller state for debugging/display."""
        return {
            "throttle_percent": self.throttle_percent,
            "gimbal_outer_deg": self._gimbal_outer * 57.3,
            "gimbal_inner_deg": self._gimbal_inner * 57.3,
            "keys": list(self._keys_down),
        }
