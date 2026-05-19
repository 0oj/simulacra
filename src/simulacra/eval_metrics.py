"""Smoothness and control quality metrics for eval episodes."""

import numpy as np


def compute_smoothness_metrics(actions: np.ndarray) -> dict:
    """Compute smoothness metrics from an episode's action sequence.

    Args:
        actions: shape (T, 3) — [throttle, tilt_x, tilt_z] per step.
                 T must be >= 3 for reversal counting (need diff of diff).

    Returns:
        dict with keys:
            thrust_reversals: int — number of sign changes in throttle delta
            rms_throttle_rate: float — sqrt(mean(diff(throttle)^2))
            mean_gimbal_rate: float — mean(|diff(tilt_x)| + |diff(tilt_z)|)
    """
    T = actions.shape[0]

    if T < 2:
        return {
            "thrust_reversals": 0,
            "rms_throttle_rate": 0.0,
            "mean_gimbal_rate": 0.0,
        }

    throttle = actions[:, 0]
    tilt_x = actions[:, 1]
    tilt_z = actions[:, 2]

    # ── Thrust reversals ──
    # A reversal is a sign change in the throttle *delta* (not throttle itself).
    # This catches oscillatory commanding: accel → decel → accel.
    dthrottle = np.diff(throttle)
    if len(dthrottle) >= 2:
        sign_changes = np.diff(np.sign(dthrottle))
        reversals = int(np.sum(sign_changes != 0))
    else:
        reversals = 0

    # ── RMS throttle rate ──
    # Measures overall command smoothness — lower is smoother.
    rms_throttle_rate = float(np.sqrt(np.mean(dthrottle ** 2)))

    # ── Mean absolute gimbal rate ──
    # Combined attitude command rate across both gimbal axes.
    dtilt_x = np.diff(tilt_x)
    dtilt_z = np.diff(tilt_z)
    mean_gimbal_rate = float(np.mean(np.abs(dtilt_x) + np.abs(dtilt_z)))

    return {
        "thrust_reversals": reversals,
        "rms_throttle_rate": rms_throttle_rate,
        "mean_gimbal_rate": mean_gimbal_rate,
    }


def aggregate_smoothness_metrics(episode_metrics: list[dict]) -> dict:
    """Average smoothness metrics across multiple episodes.

    Args:
        episode_metrics: list of dicts from compute_smoothness_metrics()

    Returns:
        dict with keys: mean_reversals, mean_rms_throttle_rate, mean_gimbal_rate
    """
    if not episode_metrics:
        return {
            "mean_reversals": 0.0,
            "mean_rms_throttle_rate": 0.0,
            "mean_gimbal_rate": 0.0,
        }

    reversals = [m["thrust_reversals"] for m in episode_metrics]
    rms_rates = [m["rms_throttle_rate"] for m in episode_metrics]
    gimbal_rates = [m["mean_gimbal_rate"] for m in episode_metrics]

    return {
        "mean_reversals": float(np.mean(reversals)),
        "mean_rms_throttle_rate": float(np.mean(rms_rates)),
        "mean_gimbal_rate": float(np.mean(gimbal_rates)),
    }
