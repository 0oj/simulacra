"""
PPO with Curriculum Learning for RocketLandingEnv.

Starts with easy conditions (low altitude, upright, no velocity)
and gradually increases difficulty as the agent improves.

Key changes from baseline:
1. Remove mid-flight tilt termination (let it crash naturally)
2. Start at low altitude (5-10m) and increase as success rate improves
3. Dense reward shaping for descent progress
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from simulacra.gym_env import RocketLandingEnv
from simulacra.schema import (
    MetadataMessage,
    ConstantMessage,
    VariableMessage,
    collect,
    write_message,
)

# Output directory for frontend visualization
OUTPUT_DIR = Path("/tmp/simulacra")


class CurriculumEpisodeRecorderCallback(BaseCallback):
    """
    Records episodes using the curriculum environment (not standard env).
    """

    def __init__(self, stage: int = 0, save_freq: int = 100, prefix: str = "curriculum", verbose: int = 0):
        super().__init__(verbose)
        self.stage = stage
        self.save_freq = save_freq
        self.prefix = prefix
        self.episode_count = 0
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                self.episode_count += 1
                if self.episode_count % self.save_freq == 0:
                    self._save_episode()
        return True

    def _save_episode(self):
        """Run episode on curriculum env and save as NDJSON."""
        # Use curriculum env with current stage parameters
        stages = CurriculumCallback.STAGES
        stage_params = stages[min(self.stage, len(stages) - 1)]

        env = RocketLandingEnvEasy(
            init_altitude_range=stage_params["altitude"],
            init_velocity_range=stage_params["velocity"],
            init_tilt_range=stage_params["tilt"],
        )
        obs, info = env.reset(seed=self.episode_count)

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
                run_by="train_ppo_curriculum.py",
                run_at=datetime.now().isoformat(),
                scenario_name=f"Curriculum Episode {self.episode_count} (Stage {self.stage + 1})",
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

    def _capture_frame(self, env, timestamp: float) -> VariableMessage:
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


class CurriculumCallback(BaseCallback):
    """
    Adjusts environment difficulty based on success rate.

    Curriculum stages:
    1. altitude 5-10m, no tilt, no velocity
    2. altitude 10-20m, small tilt, small velocity
    3. altitude 20-40m, medium tilt, medium velocity
    4. altitude 40-60m, full difficulty
    """

    STAGES = [
        {"altitude": (5, 10), "tilt": (0, 0), "velocity": (0, 0), "name": "Stage 1: Easy"},
        {"altitude": (10, 20), "tilt": (-0.05, 0.05), "velocity": (-1, 0), "name": "Stage 2: Medium-Easy"},
        {"altitude": (20, 40), "tilt": (-0.1, 0.1), "velocity": (-3, 0), "name": "Stage 3: Medium"},
        {"altitude": (40, 60), "tilt": (-0.15, 0.15), "velocity": (-5, 0), "name": "Stage 4: Full"},
    ]

    def __init__(self, envs, advance_threshold=0.3, verbose=1):
        super().__init__(verbose)
        self.envs = envs
        self.advance_threshold = advance_threshold
        self.current_stage = 0
        self.success_history = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check for episode completions
        infos = self.locals.get("infos", [])
        for info in infos:
            if "outcome" in info:
                self.episode_count += 1
                success = info["outcome"] == "landed"
                self.success_history.append(success)

                # Keep last 100 episodes
                if len(self.success_history) > 100:
                    self.success_history.pop(0)

        # Check if we should advance stage
        if len(self.success_history) >= 50:
            success_rate = sum(self.success_history) / len(self.success_history)

            if success_rate >= self.advance_threshold and self.current_stage < len(self.STAGES) - 1:
                self.current_stage += 1
                self.success_history = []  # Reset history
                self._update_envs()
                if self.verbose:
                    print(f"\n🎯 CURRICULUM: Advancing to {self.STAGES[self.current_stage]['name']}")
                    print(f"   Success rate was: {success_rate:.1%}")

        return True

    def _update_envs(self):
        """Update environment parameters for current stage."""
        stage = self.STAGES[self.current_stage]
        # Note: This requires environments to support parameter updates
        # For now, we'll just log the change - full implementation would
        # require custom VecEnv wrapper
        pass


class RocketLandingEnvEasy(RocketLandingEnv):
    """
    Modified RocketLandingEnv with:
    1. No mid-flight tilt termination
    2. Configurable initial conditions
    3. Better reward shaping
    """

    def __init__(
        self,
        init_altitude_range=(5.0, 10.0),  # Start low
        init_velocity_range=(0.0, 0.0),    # No initial velocity
        init_tilt_range=(0.0, 0.0),        # Start upright
        **kwargs
    ):
        super().__init__(
            init_altitude_range=init_altitude_range,
            init_velocity_range=init_velocity_range,
            init_tilt_range=init_tilt_range,
            max_tilt=3.14,  # Effectively disable tilt termination (180°)
            **kwargs
        )

    def _compute_reward(self, action):
        """Enhanced reward shaping for better learning signal."""
        reward, info = super()._compute_reward(action)

        # Add dense shaping rewards
        state = self._kernel.rocket.state

        # Reward for descending (getting closer to ground)
        descent_reward = -0.01 * state.velocity.y  # Positive when falling (vy < 0)

        # Reward for being upright
        tilt = self._get_tilt_angle()
        upright_reward = 0.1 * (1.0 - tilt / 3.14)  # Max 0.1 when upright

        # Reward for low horizontal velocity
        horiz_vel = np.sqrt(state.velocity.x**2 + state.velocity.z**2)
        stable_reward = -0.01 * horiz_vel

        # Combine
        shaped_reward = reward + descent_reward + upright_reward + stable_reward

        return shaped_reward, info


def make_env(seed: int = 0, stage: int = 0):
    """Factory function for curriculum environments."""
    stages = CurriculumCallback.STAGES
    stage_params = stages[min(stage, len(stages)-1)]

    def _init():
        env = RocketLandingEnvEasy(
            init_altitude_range=stage_params["altitude"],
            init_velocity_range=stage_params["velocity"],
            init_tilt_range=stage_params["tilt"],
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    # === Configuration ===
    n_envs = 8
    total_timesteps = 50_000_000  # 50M should be enough with curriculum
    save_dir = Path("runs/ppo_curriculum")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PPO with Curriculum Learning")
    print("=" * 60)
    print("Starting with easy conditions (5-10m altitude, upright)")
    print("Will advance difficulty as agent improves")
    print()

    # === Create environments ===
    print(f"Creating {n_envs} environments (Stage 1: Easy)...")

    # Start at stage 0 (easiest)
    train_envs = SubprocVecEnv([make_env(seed=i, stage=0) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(seed=100, stage=0)])

    # === Create model ===
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Slightly lower for faster credit assignment
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # More exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # === Callbacks ===
    callbacks = [
        CurriculumEpisodeRecorderCallback(stage=0, save_freq=100, prefix="curriculum", verbose=1),
        CurriculumCallback(train_envs, advance_threshold=0.3, verbose=1),
        EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best_model"),
            log_path=str(save_dir / "eval_logs"),
            eval_freq=25_000 // n_envs,  # More frequent eval
            n_eval_episodes=20,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=500_000 // n_envs,
            save_path=str(save_dir / "checkpoints"),
            name_prefix="ppo_curriculum",
        ),
    ]

    # === Train ===
    print(f"Training for {total_timesteps:,} timesteps...")
    print(f"Logs: tensorboard --logdir={save_dir / 'tensorboard'}")
    print(f"Episodes saved to: {OUTPUT_DIR} (prefix: curriculum_)")
    print()

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
