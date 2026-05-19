"""
Baseline PPO training on RocketLandingEnv.

Usage:
    python scripts/train_ppo.py

Logs to tensorboard:
    tensorboard --logdir=runs/

Episodes saved to /tmp/simulacra for frontend visualization.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from simulacra.gym_env import RocketLandingEnvV2 as RocketLandingEnv
from simulacra.schema import (
    MetadataMessage,
    ConstantMessage,
    VariableMessage,
    collect,
    write_message,
)


# Output directory for frontend visualization
OUTPUT_DIR = Path("/tmp/simulacra")


class EpisodeRecorderCallback(BaseCallback):
    """
    Records episodes as NDJSON files for frontend visualization.

    Saves one episode every `save_freq` episodes to OUTPUT_DIR.
    """

    def __init__(self, save_freq: int = 100, prefix: str = "ppo", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.prefix = prefix
        self.episode_count = 0
        self.current_episode_frames = []
        self.recording = False

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if any environment finished an episode
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                self.episode_count += 1

                # Save every save_freq episodes
                if self.episode_count % self.save_freq == 0:
                    self._save_episode_from_eval()

        return True

    def _save_episode_from_eval(self):
        """Run a single episode and save it as NDJSON."""
        # Create a fresh env for recording
        env = RocketLandingEnv()
        obs, info = env.reset(seed=self.episode_count)

        # Get the model for inference
        model = self.model

        frames = []
        timestamp = 0.0
        dt = env.dt

        # Record initial state
        frames.append(self._capture_frame(env, timestamp))

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            timestamp += dt
            frames.append(self._capture_frame(env, timestamp))
            done = terminated or truncated

        # Save to file
        outcome = info.get("outcome", "unknown")
        filename = f"{self.prefix}_ep{self.episode_count:06d}_{outcome}.ndjson"
        filepath = OUTPUT_DIR / filename

        with open(filepath, "w") as f:
            # Metadata
            meta = MetadataMessage(
                run_by="train_ppo.py",
                run_at=datetime.now().isoformat(),
                scenario_name=f"PPO Episode {self.episode_count}",
                dt=dt,
                duration=timestamp,
            )
            write_message(meta, f)

            # Constants
            const = ConstantMessage(
                timestamp=0.0,
                rocket_length=env.length,
                rocket_mass=env.dry_mass + env.fuel_mass,
            )
            write_message(const, f)

            # Variable frames
            for frame in frames:
                write_message(frame, f)

        if self.verbose:
            print(f"Saved episode to {filepath}")

        env.close()

    def _capture_frame(self, env: RocketLandingEnv, timestamp: float) -> VariableMessage:
        """Capture current state as a VariableMessage."""
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


def make_env(seed: int = 0):
    """Factory function for creating environments."""
    def _init():
        env = RocketLandingEnv()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    # === Configuration ===
    n_envs = 8  # Parallel environments
    total_timesteps = 100_000_000  # 100M timesteps
    save_dir = Path("runs/ppo_baseline_100m")
    save_dir.mkdir(parents=True, exist_ok=True)

    # === Create environments ===
    print(f"Creating {n_envs} parallel environments...")

    # Training envs
    train_envs = SubprocVecEnv([make_env(seed=i) for i in range(n_envs)])

    # Eval env (single, for evaluation callbacks)
    eval_env = DummyVecEnv([make_env(seed=100)])

    # === Create model ===
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_envs,
        # PPO hyperparameters (similar to RAJS paper)
        learning_rate=3e-4,
        n_steps=2048,  # Steps per env before update
        batch_size=64,
        n_epochs=10,
        gamma=0.995,  # Discount factor (RAJS used 0.995)
        gae_lambda=0.97,  # GAE lambda (RAJS used 0.97)
        clip_range=0.2,
        ent_coef=0.007,  # Entropy coefficient (RAJS used 0.007)
        vf_coef=0.5,
        max_grad_norm=0.5,
        # Logging
        verbose=1,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # === Callbacks ===
    callbacks = [
        # Record episodes for frontend visualization
        EpisodeRecorderCallback(save_freq=100, verbose=1),
        # Evaluate every 50k steps
        EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best_model"),
            log_path=str(save_dir / "eval_logs"),
            eval_freq=50_000 // n_envs,
            n_eval_episodes=20,
            deterministic=True,
        ),
        # Save checkpoint every 100k steps
        CheckpointCallback(
            save_freq=100_000 // n_envs,
            save_path=str(save_dir / "checkpoints"),
            name_prefix="ppo",
        ),
    ]

    # === Train ===
    print(f"Training for {total_timesteps:,} timesteps...")
    print(f"Logs: tensorboard --logdir={save_dir / 'tensorboard'}")
    print(f"Episodes saved to: {OUTPUT_DIR} (view in frontend BROWSE menu)")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # === Save final model ===
    model.save(save_dir / "final_model")
    print(f"Saved final model to {save_dir / 'final_model'}")

    # Cleanup
    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
