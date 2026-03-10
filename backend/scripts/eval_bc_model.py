"""
Evaluate the trained BC model for rocket landing.

Usage:
    python scripts/eval_bc_model.py [--record] [--n-episodes N]

The model achieves ~96% success on easy conditions (3-8m altitude).
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from simulacra.gym_env import RocketLandingEnvV2
from simulacra.schema import (
    MetadataMessage,
    ConstantMessage,
    VariableMessage,
    collect,
    write_message,
)


class BCNet(nn.Module):
    """Behavioral cloning network for rocket landing."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(model_path: Path = None) -> BCNet:
    """Load the trained BC model."""
    if model_path is None:
        model_path = Path(__file__).parent.parent / "runs" / "bc_working" / "bc_model.pt"

    model = BCNet()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def evaluate(
    model: BCNet,
    n_episodes: int = 50,
    difficulty: str = "easy",
    record: bool = False,
    output_dir: Path = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate the model.

    Args:
        model: The trained BC model
        n_episodes: Number of episodes to evaluate
        difficulty: "easy", "medium", or "hard"
        record: Whether to record episodes as NDJSON
        output_dir: Where to save recordings
        verbose: Print per-episode results

    Returns:
        Dictionary with success_rate, avg_reward, etc.
    """
    # Difficulty settings
    if difficulty == "easy":
        env = RocketLandingEnvV2(
            init_altitude_range=(3.0, 8.0),
            init_velocity_range=(-2.0, 0.0),
            init_horizontal_range=(-1.0, 1.0),
            init_tilt_range=(-0.02, 0.02),
        )
    elif difficulty == "medium":
        env = RocketLandingEnvV2(
            init_altitude_range=(5.0, 15.0),
            init_velocity_range=(-4.0, 0.0),
            init_horizontal_range=(-2.0, 2.0),
            init_tilt_range=(-0.05, 0.05),
        )
    else:  # hard
        env = RocketLandingEnvV2(
            init_altitude_range=(10.0, 30.0),
            init_velocity_range=(-6.0, 0.0),
            init_horizontal_range=(-4.0, 4.0),
            init_tilt_range=(-0.1, 0.1),
        )

    if output_dir is None:
        output_dir = Path("/tmp/simulacra")
    output_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    total_reward = 0
    landing_vels = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        frames = []
        time = 0.0
        ep_reward = 0.0
        done = False

        if record:
            frames.append(_capture_frame(env, time))

        while not done:
            with torch.no_grad():
                action = model(torch.tensor(obs).unsqueeze(0)).numpy()[0]
            action = np.clip(action, [0, -1, -1], [1, 1, 1])

            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            time += env.dt
            done = term or trunc

            if record:
                frames.append(_capture_frame(env, time))

        outcome = info.get("outcome", "timeout")
        total_reward += ep_reward

        if outcome == "landed":
            successes += 1
            landing_vels.append(info.get("landing_vel", 0))

        if verbose:
            print(f"Ep {ep+1:3d}: {outcome:8s} reward={ep_reward:7.1f}")

        if record:
            _save_recording(env, frames, time, ep, outcome, output_dir)

    env.close()

    return {
        "success_rate": successes / n_episodes,
        "n_successes": successes,
        "n_episodes": n_episodes,
        "avg_reward": total_reward / n_episodes,
        "avg_landing_vel": np.mean(landing_vels) if landing_vels else 0,
        "difficulty": difficulty,
    }


def _capture_frame(env, t):
    """Capture a frame, copying mutable state to avoid aliasing."""
    kernel = env._kernel
    state = kernel.rocket.state

    # Create a snapshot of the state by copying mutable fields
    from simulacra.noumena import RocketState
    from simulacra.physics import WorldVec3, RocketVec3, Orientation

    state_copy = RocketState(
        position=WorldVec3(*state.position),
        velocity=WorldVec3(*state.velocity),
        orientation=Orientation.from_wxyz(*state.orientation.as_array()),
        angular_velocity_body=RocketVec3(*state.angular_velocity_body),
        thrust_body=RocketVec3(*state.thrust_body),
    )

    return collect(
        VariableMessage,
        t,
        state=state_copy,
        sensors=env._sensors,
        model=kernel.rocket.mass_model,
        ground=kernel.ground_contact,
        actuators=env._actuators,
    )


def _save_recording(env, frames, duration, ep, outcome, output_dir):
    filepath = output_dir / f"bc_eval_ep{ep}_{outcome}.ndjson"
    with open(filepath, "w") as f:
        meta = MetadataMessage(
            run_by="eval_bc_model.py",
            run_at=datetime.now().isoformat(),
            scenario_name=f"BC Eval Episode {ep}",
            dt=env.dt,
            duration=duration,
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC landing model")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None
    model = load_model(model_path)

    print(f"Evaluating BC model on {args.difficulty} conditions")
    print(f"Episodes: {args.n_episodes}")
    print("=" * 50)

    results = evaluate(
        model,
        n_episodes=args.n_episodes,
        difficulty=args.difficulty,
        record=args.record,
        verbose=True,
    )

    print("=" * 50)
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Avg reward: {results['avg_reward']:.1f}")
    if results["avg_landing_vel"] > 0:
        print(f"Avg landing velocity: {results['avg_landing_vel']:.2f} m/s")


if __name__ == "__main__":
    main()
