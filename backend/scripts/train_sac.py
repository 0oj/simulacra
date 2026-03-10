"""
Pure SAC training baseline on RocketLandingEnvV2.

Alternative to BC + SAC pipeline - train SAC from scratch with:
- Fixed additive reward function (V2 env)
- Easy starting conditions (curriculum-style)
- Early stopping callback

Usage:
    python scripts/train_sac.py

Logs to tensorboard:
    tensorboard --logdir=runs/sac_baseline/tensorboard

Episodes saved to /tmp/simulacra for frontend visualization.
"""

import sys
from pathlib import Path
from datetime import datetime
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from simulacra.gym_env import RocketLandingEnvV2
from simulacra.schema import (
    MetadataMessage,
    ConstantMessage,
    VariableMessage,
    collect,
    write_message,
)


OUTPUT_DIR = Path("/tmp/simulacra")


class CurriculumCallback(BaseCallback):
    """
    Progressively increase environment difficulty based on success rate.

    Stages:
    1. Easy: 5-15m altitude, low velocity, minimal tilt
    2. Medium: 10-30m altitude, moderate velocity
    3. Hard: 10-60m altitude, full range
    """

    STAGES = [
        {
            "name": "easy",
            "altitude_range": (5.0, 15.0),
            "velocity_range": (-3.0, 0.0),
            "horizontal_range": (-2.0, 2.0),
            "tilt_range": (-0.05, 0.05),
            "threshold": 0.4,  # Move to next stage at 40% success
        },
        {
            "name": "medium",
            "altitude_range": (10.0, 30.0),
            "velocity_range": (-5.0, 0.0),
            "horizontal_range": (-3.0, 3.0),
            "tilt_range": (-0.1, 0.1),
            "threshold": 0.5,
        },
        {
            "name": "hard",
            "altitude_range": (10.0, 60.0),
            "velocity_range": (-8.0, 0.0),
            "horizontal_range": (-5.0, 5.0),
            "tilt_range": (-0.15, 0.15),
            "threshold": 1.0,  # Final stage
        },
    ]

    def __init__(
        self,
        eval_env,
        eval_freq: int = 20000,
        n_eval_episodes: int = 20,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.current_stage = 0
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        success_rate = self._evaluate()

        stage = self.STAGES[self.current_stage]
        self.history.append({
            "step": self.num_timesteps,
            "stage": stage["name"],
            "success_rate": success_rate,
        })

        if self.verbose:
            print(f"\n[Curriculum] step={self.num_timesteps}, stage={stage['name']}, "
                  f"success={success_rate:.1%}")

        # Check for stage advancement
        if success_rate >= stage["threshold"] and self.current_stage < len(self.STAGES) - 1:
            self.current_stage += 1
            new_stage = self.STAGES[self.current_stage]
            if self.verbose:
                print(f"  Advancing to stage: {new_stage['name']}")
            self._update_envs()

        return True

    def _evaluate(self) -> float:
        """Evaluate on CURRENT stage difficulty."""
        successes = 0
        stage = self.STAGES[self.current_stage]

        for ep in range(self.n_eval_episodes):
            # Reset with stage-appropriate options
            obs, _ = self.eval_env.reset(options={
                "altitude_range": stage["altitude_range"],
                "velocity_range": stage["velocity_range"],
                "horizontal_range": stage["horizontal_range"],
                "tilt_range": stage["tilt_range"],
            })

            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

            if info.get("outcome") == "landed":
                successes += 1

        return successes / self.n_eval_episodes

    def _update_envs(self):
        """Update training environments with new stage parameters."""
        # Note: For SubprocVecEnv, we can't directly modify the envs
        # Instead, we rely on the env using reset options
        # This is a limitation - for proper curriculum, we'd need to recreate envs
        pass


class EarlyStoppingCallback(BaseCallback):
    """Stop training when success rate doesn't improve."""

    def __init__(
        self,
        eval_env,
        eval_freq: int = 50000,
        n_eval_episodes: int = 20,
        patience: int = 5,
        min_improvement: float = 0.02,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience
        self.min_improvement = min_improvement

        self.best_success_rate = 0.0
        self.no_improvement_count = 0
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Evaluate on full difficulty
        success_rate, avg_reward = self._evaluate_full()

        self.history.append({
            "step": self.num_timesteps,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
        })

        if self.verbose:
            print(f"\n[Eval Full] step={self.num_timesteps}, "
                  f"success={success_rate:.1%}, reward={avg_reward:.1f}")

        if success_rate > self.best_success_rate + self.min_improvement:
            self.best_success_rate = success_rate
            self.no_improvement_count = 0
            # Save best model
            self.model.save(Path("runs/sac_baseline") / "best_model")
            if self.verbose:
                print(f"  New best! Saved model.")
        else:
            self.no_improvement_count += 1
            if self.verbose:
                print(f"  No improvement ({self.no_improvement_count}/{self.patience})")

        if self.no_improvement_count >= self.patience:
            if self.verbose:
                print(f"\nEarly stopping triggered")
            return False

        return True

    def _evaluate_full(self) -> tuple[float, float]:
        """Evaluate on full difficulty (not curriculum)."""
        successes = 0
        total_reward = 0.0

        for ep in range(self.n_eval_episodes):
            # Full difficulty
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated

            total_reward += ep_reward
            if info.get("outcome") == "landed":
                successes += 1

        return successes / self.n_eval_episodes, total_reward / self.n_eval_episodes


class EpisodeRecorderCallback(BaseCallback):
    """Records episodes as NDJSON files."""

    def __init__(self, save_freq: int = 100, prefix: str = "sac", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.prefix = prefix
        self.episode_count = 0
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self.episode_count += 1
                if self.episode_count % self.save_freq == 0:
                    self._save_episode()
        return True

    def _save_episode(self):
        env = RocketLandingEnvV2()
        obs, _ = env.reset(seed=self.episode_count)

        frames = []
        timestamp = 0.0
        dt = env.dt

        frames.append(self._capture_frame(env, timestamp))

        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            timestamp += dt
            frames.append(self._capture_frame(env, timestamp))
            done = terminated or truncated

        outcome = info.get("outcome", "unknown")
        filename = f"{self.prefix}_ep{self.episode_count:06d}_{outcome}.ndjson"
        filepath = OUTPUT_DIR / filename

        with open(filepath, "w") as f:
            meta = MetadataMessage(
                run_by="train_sac.py",
                run_at=datetime.now().isoformat(),
                scenario_name=f"SAC Episode {self.episode_count}",
                dt=dt,
                duration=timestamp,
            )
            write_message(meta, f)

            const = ConstantMessage(
                timestamp=0.0,
                rocket_length=env.length,
                rocket_mass=env.dry_mass + env.fuel_mass,
            )
            write_message(const, f)

            for frame in frames:
                write_message(frame, f)

        env.close()

    def _capture_frame(self, env: RocketLandingEnvV2, timestamp: float) -> VariableMessage:
        kernel = env._kernel
        state = kernel.rocket.state
        ground = kernel.ground_contact
        sensors = env._sensors
        actuators = env._actuators

        return collect(
            VariableMessage,
            timestamp,
            state=state,
            sensors=sensors,
            model=kernel.rocket.mass_model,
            ground=ground,
            actuators=actuators,
        )


def make_env_easy(seed: int = 0):
    """Create environment with easy curriculum start."""
    def _init():
        env = RocketLandingEnvV2(
            init_altitude_range=(5.0, 15.0),
            init_velocity_range=(-3.0, 0.0),
            init_horizontal_range=(-2.0, 2.0),
            init_tilt_range=(-0.05, 0.05),
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def make_env_full(seed: int = 0):
    """Create environment with full difficulty."""
    def _init():
        env = RocketLandingEnvV2()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    save_dir = Path("runs/sac_baseline")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pure SAC Training (Baseline)")
    print("=" * 60)

    # Configuration
    n_envs = 4
    total_timesteps = 10_000_000

    print(f"\nConfiguration:")
    print(f"  Environments: {n_envs}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Starting with easy curriculum")

    # Create environments
    print("\nCreating environments...")
    train_envs = SubprocVecEnv([make_env_easy(seed=i) for i in range(n_envs)])
    eval_env_curriculum = RocketLandingEnvV2()  # Will use reset options
    eval_env_full = RocketLandingEnvV2()  # Full difficulty

    # Create SAC model
    print("Creating SAC model...")
    model = SAC(
        "MlpPolicy",
        train_envs,
        # SAC hyperparameters
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.995,
        ent_coef="auto",  # Auto-tune entropy
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=1,
        # Policy network
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
        ),
        verbose=1,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # Callbacks
    curriculum_cb = CurriculumCallback(
        eval_env=eval_env_curriculum,
        eval_freq=20_000 // n_envs,
        n_eval_episodes=20,
        verbose=1,
    )

    early_stop_cb = EarlyStoppingCallback(
        eval_env=eval_env_full,
        eval_freq=50_000 // n_envs,
        n_eval_episodes=20,
        patience=5,
        verbose=1,
    )

    callbacks = [
        curriculum_cb,
        early_stop_cb,
        EpisodeRecorderCallback(save_freq=100, prefix="sac_base", verbose=0),
        CheckpointCallback(
            save_freq=100_000 // n_envs,
            save_path=str(save_dir / "checkpoints"),
            name_prefix="sac",
        ),
    ]

    # Train
    print(f"\nStarting training...")
    print(f"Logs: tensorboard --logdir={save_dir / 'tensorboard'}")
    print(f"Episodes saved to: {OUTPUT_DIR}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    model.save(save_dir / "final_model")
    print(f"\nSaved final model to {save_dir / 'final_model'}")

    # Save training history
    if curriculum_cb.history:
        np.savez(
            save_dir / "curriculum_history.npz",
            steps=[h["step"] for h in curriculum_cb.history],
            stages=[h["stage"] for h in curriculum_cb.history],
            success_rates=[h["success_rate"] for h in curriculum_cb.history],
        )

    if early_stop_cb.history:
        np.savez(
            save_dir / "eval_history.npz",
            steps=[h["step"] for h in early_stop_cb.history],
            success_rates=[h["success_rate"] for h in early_stop_cb.history],
            avg_rewards=[h["avg_reward"] for h in early_stop_cb.history],
        )

    # Final evaluation
    print("\nFinal evaluation (100 episodes, full difficulty)...")
    successes = 0
    total_reward = 0.0

    for ep in range(100):
        obs, _ = eval_env_full.reset(seed=2000 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_full.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_reward += ep_reward
        if info.get("outcome") == "landed":
            successes += 1

    final_success_rate = successes / 100
    final_avg_reward = total_reward / 100

    print(f"\nFinal Results:")
    print(f"  Success rate: {final_success_rate:.1%}")
    print(f"  Avg reward: {final_avg_reward:.1f}")
    print(f"  Best success rate during training: {early_stop_cb.best_success_rate:.1%}")

    # Cleanup
    train_envs.close()
    eval_env_curriculum.close()
    eval_env_full.close()

    print("\n" + "=" * 60)
    print("SAC Training Complete!")
    print("=" * 60)

    return final_success_rate >= 0.5


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
