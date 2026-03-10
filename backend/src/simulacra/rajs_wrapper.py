"""
Random Annealing Jump Start (RAJS) wrapper for Gymnasium environments.

Based on: "Rocket Landing Control with Random Annealing Jump Start
Reinforcement Learning" (LandSpace, 2024)

The wrapper uses a guide policy (e.g., LandingController) to bootstrap
exploration, gradually annealing its influence to zero during training.
"""

from typing import Any, Callable, Optional, Tuple
import gymnasium as gym
import numpy as np


class RAJSWrapper(gym.Wrapper):
    """
    RAJS wrapper that uses a guide policy to bootstrap RL training.

    At each episode:
    1. Sample guide horizon H ~ U(0, H_max * beta)
    2. For first H steps, use guide policy
    3. After H steps, use RL policy (action passed to step())

    Beta anneals from 1 to 0 during training based on success rate.

    Args:
        env: The base Gymnasium environment
        guide_policy: Callable that takes observation and returns action
        max_guide_horizon: Maximum guide steps at start of training
        anneal_threshold: Success rate threshold to trigger annealing
        anneal_step: How much to reduce beta per trigger (default 1/1500)
    """

    def __init__(
        self,
        env: gym.Env,
        guide_policy: Callable[[np.ndarray], np.ndarray],
        max_guide_horizon: int = 200,  # ~4 seconds at 50Hz
        anneal_threshold: float = 0.3,
        anneal_step: float = 1 / 1500,
    ):
        super().__init__(env)

        self.guide_policy = guide_policy
        self.max_guide_horizon = max_guide_horizon
        self.anneal_threshold = anneal_threshold
        self.anneal_step = anneal_step

        # Annealing state
        self.beta = 1.0  # Starts at 1, anneals to 0

        # Episode tracking
        self._step_count = 0
        self._guide_horizon = 0
        self._using_guide = False

        # Success tracking for annealing
        self._episode_outcomes: list[bool] = []
        self._window_size = 100  # Rolling window for success rate

    @property
    def success_rate(self) -> float:
        """Rolling success rate over recent episodes."""
        if not self._episode_outcomes:
            return 0.0
        return sum(self._episode_outcomes[-self._window_size:]) / min(
            len(self._episode_outcomes), self._window_size
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset and sample new guide horizon."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Sample guide horizon: H ~ U(0, H_max * beta)
        effective_max = int(self.max_guide_horizon * self.beta)
        if effective_max > 0:
            self._guide_horizon = self.np_random.integers(0, effective_max + 1)
        else:
            self._guide_horizon = 0

        self._step_count = 0
        self._using_guide = self._guide_horizon > 0

        # Add info about RAJS state
        info["rajs_guide_horizon"] = self._guide_horizon
        info["rajs_beta"] = self.beta
        info["rajs_success_rate"] = self.success_rate

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment.

        If within guide horizon, use guide policy instead of provided action.
        """
        self._step_count += 1

        # Decide which action to use
        if self._step_count <= self._guide_horizon:
            # Use guide policy
            obs = self._get_current_obs()
            actual_action = self.guide_policy(obs)
            self._using_guide = True
        else:
            # Use RL action
            actual_action = action
            self._using_guide = False

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(actual_action)

        # Add RAJS info
        info["rajs_using_guide"] = self._using_guide
        info["rajs_step"] = self._step_count

        # Track episode outcome for annealing
        if terminated or truncated:
            # Determine if this was a success
            # You may need to customize this based on your env's info dict
            success = info.get("is_success", False) or reward > 50  # Adjust threshold
            self._episode_outcomes.append(success)

            # Check if we should anneal
            self._maybe_anneal()

        return obs, reward, terminated, truncated, info

    def _get_current_obs(self) -> np.ndarray:
        """Get current observation from the unwrapped env."""
        # This accesses the internal state - may need adjustment
        # depending on your env implementation
        return self.env.unwrapped._get_obs()

    def _maybe_anneal(self) -> None:
        """Reduce beta if success rate exceeds threshold."""
        if self.beta > 0 and self.success_rate >= self.anneal_threshold:
            self.beta = max(0.0, self.beta - self.anneal_step)

    def set_beta(self, beta: float) -> None:
        """Manually set beta (useful for curriculum or manual scheduling)."""
        self.beta = max(0.0, min(1.0, beta))


class RAJSCallbackMixin:
    """
    Mixin for SB3 callbacks to log RAJS metrics.

    Usage:
        class MyCallback(RAJSCallbackMixin, BaseCallback):
            def _on_step(self):
                self.log_rajs_metrics()
                return True
    """

    def log_rajs_metrics(self) -> None:
        """Log RAJS metrics to tensorboard."""
        # Get the wrapper from the env
        env = self.training_env.envs[0]
        while hasattr(env, 'env'):
            if isinstance(env, RAJSWrapper):
                self.logger.record("rajs/beta", env.beta)
                self.logger.record("rajs/success_rate", env.success_rate)
                break
            env = env.env
