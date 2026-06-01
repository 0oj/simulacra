"""
Noumena-layer events: things that happen to the rocket from god's perspective.

Events are scheduled on a Scenario and fire at specified times. Each receives
(t, kernel, controller) and may modify the true rocket state directly.

Usage:
    from simulacra.noumena.events import bird_strike, random_perturbation

    scenario = Scenario(kernel, controller, ...)
    scenario.add_event(5.0, bird_strike(impulse_mps=3.0, direction=(0.5, 0.2, -0.3)))
    scenario.add_event(8.0, random_perturbation(vel_std=2.0, omega_std=0.5, seed=42))
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

from .scenario import Callback

if TYPE_CHECKING:
    from .kernel import Kernel
    from ..onboard.controller_legacy import Controller


def bird_strike(
    impulse_mps: float = 3.0,
    direction: Optional[Tuple[float, float, float]] = None,
    angular_impulse_rps: float = 0.5,
    seed: Optional[int] = None,
) -> Callback:
    """Create a bird strike event that applies a velocity + angular velocity kick.

    Simulates a significant external perturbation — a bird collision, debris impact,
    or being "chucked by hand". The rocket receives an instantaneous velocity change
    and angular velocity change in the world/body frames respectively.

    Args:
        impulse_mps: Magnitude of the velocity kick (m/s). A 3 kg bird at 15 m/s
            hitting a 140 kg rocket imparts ~0.3 m/s. We default to 3.0 m/s for
            a dramatic, recovery-testable event.
        direction: Unit direction of velocity kick in world frame (x, y, z).
            If None, a random direction is sampled.
        angular_impulse_rps: Magnitude of the angular velocity kick (rad/s).
            Applied in a random body-frame direction. Default 0.5 rad/s (~29 deg/s).
        seed: Random seed for direction sampling. If None, uses system entropy.

    Returns:
        Callback suitable for Scenario.add_event().
    """
    rng = np.random.default_rng(seed)

    # Pre-compute direction if random
    if direction is not None:
        d = np.array(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        vel_kick = d * impulse_mps
    else:
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d)
        vel_kick = d * impulse_mps

    # Random body-frame angular kick direction
    omega_dir = rng.normal(size=3)
    omega_dir = omega_dir / np.linalg.norm(omega_dir)
    omega_kick = omega_dir * angular_impulse_rps

    def _callback(t: float, kernel: Kernel, controller: Controller) -> None:
        from ..physics import WorldVec3, RocketVec3

        state = kernel.rocket.state
        # Apply velocity impulse (world frame)
        state.velocity = WorldVec3(
            state.velocity.x + vel_kick[0],
            state.velocity.y + vel_kick[1],
            state.velocity.z + vel_kick[2],
        )
        # Apply angular velocity impulse (body frame)
        state.angular_velocity_body = RocketVec3(
            state.angular_velocity_body.x + omega_kick[0],
            state.angular_velocity_body.y + omega_kick[1],
            state.angular_velocity_body.z + omega_kick[2],
        )

    return _callback


def random_perturbation(
    vel_std: float = 2.0,
    omega_std: float = 0.3,
    seed: Optional[int] = None,
) -> Callback:
    """Create a random perturbation event with Gaussian-distributed kicks.

    Lighter-weight than bird_strike — useful for scheduling multiple small
    disturbances throughout a flight to test continuous recovery.

    Args:
        vel_std: Standard deviation of velocity kick per axis (m/s).
        omega_std: Standard deviation of angular velocity kick per axis (rad/s).
        seed: Random seed.

    Returns:
        Callback suitable for Scenario.add_event().
    """
    rng = np.random.default_rng(seed)
    vel_kick = rng.normal(scale=vel_std, size=3)
    omega_kick = rng.normal(scale=omega_std, size=3)

    def _callback(t: float, kernel: Kernel, controller: Controller) -> None:
        from ..physics import WorldVec3, RocketVec3

        state = kernel.rocket.state
        state.velocity = WorldVec3(
            state.velocity.x + vel_kick[0],
            state.velocity.y + vel_kick[1],
            state.velocity.z + vel_kick[2],
        )
        state.angular_velocity_body = RocketVec3(
            state.angular_velocity_body.x + omega_kick[0],
            state.angular_velocity_body.y + omega_kick[1],
            state.angular_velocity_body.z + omega_kick[2],
        )

    return _callback


def schedule_random_bird_strikes(
    scenario,
    t_start: float = 2.0,
    t_end: float = 25.0,
    count: int = 3,
    impulse_range: Tuple[float, float] = (1.0, 5.0),
    angular_range: Tuple[float, float] = (0.2, 0.8),
    seed: Optional[int] = None,
) -> list[float]:
    """Schedule multiple random bird strikes throughout a flight.

    Convenience function that adds `count` bird strike events at random times
    between t_start and t_end with random intensities.

    Args:
        scenario: Scenario instance to add events to.
        t_start: Earliest possible strike time (s).
        t_end: Latest possible strike time (s).
        count: Number of strikes to schedule.
        impulse_range: (min, max) velocity impulse magnitude (m/s).
        angular_range: (min, max) angular impulse magnitude (rad/s).
        seed: Random seed for reproducibility.

    Returns:
        List of scheduled event times.
    """
    rng = np.random.default_rng(seed)
    times = sorted(rng.uniform(t_start, t_end, size=count))

    for t in times:
        impulse = rng.uniform(*impulse_range)
        angular = rng.uniform(*angular_range)
        event = bird_strike(
            impulse_mps=impulse,
            angular_impulse_rps=angular,
            seed=rng.integers(0, 2**31),
        )
        scenario.add_event(float(t), event)

    return [float(t) for t in times]
