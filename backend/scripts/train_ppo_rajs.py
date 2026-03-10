"""
PPO + RAJS training on RocketLandingEnvV2.

Uses LandingController as guide policy to bootstrap exploration.

Usage:
    python scripts/train_ppo_rajs.py

Logs to tensorboard:
    tensorboard --logdir=runs/

Episodes saved to /tmp/simulacra for frontend visualization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_ppo import EpisodeRecorderCallback, OUTPUT_DIR

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from simulacra.gym_env import RocketLandingEnvV2
from simulacra.rajs_wrapper import RAJSWrapper
from simulacra.onboard.landing_controller import LandingController, LandingControllerConfig
from simulacra.phenomena.actuators import ThrottleActuator, GimbalActuator


class GuidePolicy:
    """
    Wraps LandingController to work as a guide policy for RAJS.

    The guide policy needs access to the environment's internal state
    to run the controller. This is a bit hacky but necessary.
    """

    def __init__(self, env: RocketLandingEnvV2):
        self.env = env

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Run LandingController and return action in env's action space.

        Args:
            obs: Current observation (not directly used - we read from sensors)

        Returns:
            action: [throttle, gimbal_outer, gimbal_inner] normalized
        """
        # Get the unwrapped env to access internals
        unwrapped = self.env.unwrapped

        # Create controller with current actuators
        # Use dry_mass and full_fuel_mass for dynamic mass estimation
        config = LandingControllerConfig(
            dry_mass=unwrapped.dry_mass,
            full_fuel_mass=unwrapped.fuel_mass,
            max_thrust=unwrapped.max_thrust,
        )
        controller = LandingController(
            throttle=unwrapped._actuators._throttle,
            gimbal=unwrapped._actuators._gimbal,
            config=config,
        )
        controller = controller.with_sensors(unwrapped._sensors)

        # Run controller (this applies commands to actuators internally)
        controller.run()

        # Extract what the controller commanded
        # We need to convert back to normalized action space
        throttle_cmd = unwrapped._actuators._throttle._commanded_thrust
        gimbal_outer_cmd = unwrapped._actuators._gimbal._commanded_outer
        gimbal_inner_cmd = unwrapped._actuators._gimbal._commanded_inner

        # Normalize to [-1, 1] or [0, 1]
        throttle_norm = throttle_cmd / unwrapped.max_thrust
        gimbal_outer_norm = gimbal_outer_cmd / unwrapped.max_gimbal_angle
        gimbal_inner_norm = gimbal_inner_cmd / unwrapped.max_gimbal_angle

        return np.array([
            np.clip(throttle_norm, 0, 1),
            np.clip(gimbal_outer_norm, -1, 1),
            np.clip(gimbal_inner_norm, -1, 1),
        ], dtype=np.float32)


class RAJSLoggingCallback(BaseCallback):
    """Callback to log RAJS metrics during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._last_beta = 1.0

    def _on_step(self) -> bool:
        # Try to get RAJS metrics from info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "rajs_beta" in info:
                beta = info["rajs_beta"]
                if beta != self._last_beta:
                    print(f"RAJS beta: {beta:.4f}")
                    self._last_beta = beta
                self.logger.record("rajs/beta", beta)
            if "rajs_success_rate" in info:
                self.logger.record("rajs/success_rate", info["rajs_success_rate"])
        return True


def make_env_with_rajs(seed: int = 0, max_guide_horizon: int = 200):
    """Factory function for creating RAJS-wrapped environments."""

    def _init():
        # Create base env
        env = RocketLandingEnvV2()

        # Create guide policy
        guide = GuidePolicy(env)

        # Wrap with RAJS
        env = RAJSWrapper(
            env,
            guide_policy=guide,
            max_guide_horizon=max_guide_horizon,
            anneal_threshold=0.3,  # Start annealing at 30% success
            anneal_step=1 / 1500,  # Gradual annealing
        )

        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def make_eval_env(seed: int = 100):
    """Create eval env WITHOUT RAJS (pure RL evaluation)."""
    def _init():
        env = RocketLandingEnvV2()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    # === Configuration ===
    n_envs = 8
    total_timesteps = 100_000_000  # 100M timesteps
    max_guide_horizon = 200  # ~4 seconds at 50Hz
    save_dir = Path("runs/ppo_rajs_100m")
    save_dir.mkdir(parents=True, exist_ok=True)

    # === Create environments ===
    print(f"Creating {n_envs} RAJS-wrapped environments...")
    print(f"Max guide horizon: {max_guide_horizon} steps")

    # Training envs with RAJS
    train_envs = SubprocVecEnv([
        make_env_with_rajs(seed=i, max_guide_horizon=max_guide_horizon)
        for i in range(n_envs)
    ])

    # Eval env WITHOUT RAJS (test pure RL performance)
    eval_env = DummyVecEnv([make_eval_env(seed=100)])

    # === Create model ===
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_envs,
        # PPO hyperparameters (from RAJS paper Table II)
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=0.007,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # Network
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # RAJS used (256, 256)
        ),
        verbose=1,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # === Callbacks ===
    callbacks = [
        EpisodeRecorderCallback(save_freq=100, prefix="rajs", verbose=1),
        RAJSLoggingCallback(),
        EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best_model"),
            log_path=str(save_dir / "eval_logs"),
            eval_freq=50_000 // n_envs,
            n_eval_episodes=50,  # More episodes for reliable success rate
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=200_000 // n_envs,
            save_path=str(save_dir / "checkpoints"),
            name_prefix="ppo_rajs",
        ),
    ]

    # === Train ===
    print(f"Training for {total_timesteps:,} timesteps...")
    print(f"Logs: tensorboard --logdir={save_dir / 'tensorboard'}")
    print(f"Episodes saved to: {OUTPUT_DIR} (prefix: rajs_)")
    print("Watch for beta annealing as success rate improves!")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # === Save final model ===
    model.save(save_dir / "final_model")
    print(f"Saved final model to {save_dir / 'final_model'}")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
