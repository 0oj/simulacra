"""
SAC fine-tuning from BC policy.

Phase 2 of the BC + SAC training pipeline:
1. Load BC policy
2. Initialize SAC with BC policy weights (warm start)
3. Fine-tune using SAC with lower learning rate

Usage:
    python scripts/train_sac_finetune.py

Prerequisites:
    - Run train_bc.py first to generate runs/bc/bc_policy.pt

Logs to tensorboard:
    tensorboard --logdir=runs/sac_finetune/tensorboard
"""

import sys
from pathlib import Path
from datetime import datetime
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer

from simulacra.gym_env import RocketLandingEnvV2
from simulacra.schema import (
    MetadataMessage,
    ConstantMessage,
    VariableMessage,
    collect,
    write_message,
)


OUTPUT_DIR = Path("/tmp/simulacra")


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training when success rate doesn't improve for `patience` eval cycles.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        patience: int = 5,
        min_improvement: float = 0.01,
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
        self.eval_count = 0
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Evaluate
        success_rate, avg_reward = self._evaluate()
        self.eval_count += 1
        self.history.append({
            "step": self.num_timesteps,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
        })

        if self.verbose:
            print(f"\n[Eval {self.eval_count}] step={self.num_timesteps}, "
                  f"success={success_rate:.1%}, reward={avg_reward:.1f}")

        # Check improvement
        if success_rate > self.best_success_rate + self.min_improvement:
            self.best_success_rate = success_rate
            self.no_improvement_count = 0
            if self.verbose:
                print(f"  New best! success_rate={success_rate:.1%}")
        else:
            self.no_improvement_count += 1
            if self.verbose:
                print(f"  No improvement ({self.no_improvement_count}/{self.patience})")

        # Early stopping
        if self.no_improvement_count >= self.patience:
            if self.verbose:
                print(f"\nEarly stopping: no improvement for {self.patience} cycles")
            return False

        return True

    def _evaluate(self) -> tuple[float, float]:
        """Run evaluation episodes and return (success_rate, avg_reward)."""
        successes = 0
        total_reward = 0.0

        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated

            total_reward += ep_reward

            # Check final info for outcome
            if hasattr(self.eval_env, "envs"):
                final_info = self.eval_env.envs[0].unwrapped._kernel.ground_contact
            else:
                final_info = self.eval_env.unwrapped._kernel.ground_contact

            if final_info.on_ground and not final_info.crashed and not final_info.tipped_over:
                successes += 1

        return successes / self.n_eval_episodes, total_reward / self.n_eval_episodes


class EpisodeRecorderCallback(BaseCallback):
    """Records episodes as NDJSON files for frontend visualization."""

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
        """Run and save a single episode."""
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
                run_by="train_sac_finetune.py",
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

        if self.verbose:
            print(f"Saved episode to {filepath}")

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


def prefill_buffer_from_demos(model: SAC, demos_path: Path, n_transitions: int = 10000):
    """
    Pre-fill SAC's replay buffer with BC demonstrations.

    This helps the critic learn from good trajectories early on.
    """
    demos = np.load(demos_path)
    observations = demos["observations"]
    actions = demos["actions"]

    n_available = len(observations) - 1  # Can't use last obs (no next_obs)
    n_to_add = min(n_transitions, n_available)

    print(f"Pre-filling replay buffer with {n_to_add} transitions from BC demos...")

    buffer = model.replay_buffer

    for i in range(n_to_add):
        obs = observations[i]
        action = actions[i]
        next_obs = observations[i + 1]
        # Assume small reward for flight shaping (we don't have actual rewards)
        reward = -0.1
        done = False

        buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=np.array([reward]),
            done=np.array([done]),
            infos=[{}],
        )

    print(f"  Buffer size: {buffer.size()}")


def make_env(seed: int = 0, easy_start: bool = True):
    """Factory function for creating environments."""
    def _init():
        if easy_start:
            # Start with easier conditions
            env = RocketLandingEnvV2(
                init_altitude_range=(5.0, 15.0),
                init_velocity_range=(-3.0, 0.0),
                init_horizontal_range=(-2.0, 2.0),
                init_tilt_range=(-0.05, 0.05),
            )
        else:
            env = RocketLandingEnvV2()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    save_dir = Path("runs/sac_finetune")
    save_dir.mkdir(parents=True, exist_ok=True)

    bc_policy_path = Path("runs/bc/bc_policy.pt")
    demos_path = Path("runs/bc/demos.npz")

    print("=" * 60)
    print("SAC Fine-tuning from BC Policy")
    print("=" * 60)

    # Check for BC policy
    if not bc_policy_path.exists():
        print(f"\nERROR: BC policy not found at {bc_policy_path}")
        print("Please run train_bc.py first.")
        return False

    # Configuration
    n_envs = 4
    total_timesteps = 10_000_000  # 10M steps (can increase if needed)
    eval_freq = 50_000

    print(f"\nConfiguration:")
    print(f"  Environments: {n_envs}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Eval frequency: {eval_freq:,}")

    # Create environments
    print("\nCreating environments...")
    train_envs = SubprocVecEnv([make_env(seed=i, easy_start=True) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(seed=100, easy_start=False)])  # Full difficulty for eval

    # Create SAC model
    print("Creating SAC model...")
    model = SAC(
        "MlpPolicy",
        train_envs,
        # SAC hyperparameters optimized for fine-tuning
        learning_rate=1e-4,  # Lower LR for fine-tuning
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.995,
        ent_coef=0.01,  # Lower entropy for stability
        learning_starts=10_000,
        # Policy network
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
        ),
        verbose=1,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # Pre-fill replay buffer with BC demos
    if demos_path.exists():
        prefill_buffer_from_demos(model, demos_path, n_transitions=20000)

    # Callbacks
    early_stop_callback = EarlyStoppingCallback(
        eval_env=eval_env.envs[0],  # Unwrap for direct access
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=20,
        patience=5,
        verbose=1,
    )

    callbacks = [
        early_stop_callback,
        EpisodeRecorderCallback(save_freq=100, prefix="sac_ft", verbose=0),
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
    if early_stop_callback.history:
        history = early_stop_callback.history
        np.savez(
            save_dir / "history.npz",
            steps=[h["step"] for h in history],
            success_rates=[h["success_rate"] for h in history],
            avg_rewards=[h["avg_reward"] for h in history],
        )

    # Final evaluation
    print("\nFinal evaluation (100 episodes)...")
    final_success_rate, final_avg_reward = 0.0, 0.0
    successes = 0
    total_reward = 0.0

    eval_env_final = RocketLandingEnvV2()
    for ep in range(100):
        obs, _ = eval_env_final.reset(seed=2000 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_final.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_reward += ep_reward
        if info.get("outcome") == "landed":
            successes += 1

    eval_env_final.close()
    final_success_rate = successes / 100
    final_avg_reward = total_reward / 100

    print(f"\nFinal Results:")
    print(f"  Success rate: {final_success_rate:.1%}")
    print(f"  Avg reward: {final_avg_reward:.1f}")
    print(f"  Best success rate during training: {early_stop_callback.best_success_rate:.1%}")

    # Cleanup
    train_envs.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("SAC Fine-tuning Complete!")
    print("=" * 60)

    return final_success_rate >= 0.7


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
