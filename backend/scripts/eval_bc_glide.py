"""
Evaluate and record the BC glide path model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
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

    def __init__(self, obs_dim: int = 13, act_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(model_path: Path = None) -> BCNet:
    if model_path is None:
        model_path = Path(__file__).parent.parent / "runs" / "bc_glide" / "bc_model.pt"

    model = BCNet()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def evaluate(
    model: BCNet,
    n_episodes: int = 50,
    altitude_range: tuple = (10.0, 30.0),
    record: bool = False,
    output_dir: Path = None,
    verbose: bool = True,
) -> dict:
    """Evaluate the BC model."""
    env = RocketLandingEnvV2(
        init_altitude_range=altitude_range,
        init_velocity_range=(-5.0, 0.0),
        init_horizontal_range=(-2.0, 2.0),
        init_tilt_range=(-0.05, 0.05),
    )

    if output_dir is None:
        output_dir = Path("/tmp/simulacra")
    output_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    crashes = 0
    timeouts = 0
    landing_vels = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        init_alt = info["init_altitude"]
        frames = []
        time = 0.0
        done = False

        if record:
            frames.append(_capture_frame(env, time))

        while not done:
            with torch.no_grad():
                action = model(torch.tensor(obs).unsqueeze(0)).numpy()[0]
            action = np.clip(action, [0, -1, -1], [1, 1, 1])

            obs, reward, term, trunc, info = env.step(action)
            time += env.dt
            done = term or trunc

            if record:
                frames.append(_capture_frame(env, time))

        outcome = info.get("outcome", "timeout")

        if outcome == "landed":
            successes += 1
            landing_vels.append(info.get("landing_vel", 0))
        elif outcome == "crash":
            crashes += 1
        else:
            timeouts += 1

        if verbose:
            print(f"Ep {ep + 1:3d}: {outcome:8s} from {init_alt:5.1f}m")

        if record:
            _save_recording(env, frames, time, ep, outcome, init_alt, output_dir)

    env.close()

    return {
        "success_rate": successes / n_episodes,
        "n_successes": successes,
        "n_crashes": crashes,
        "n_timeouts": timeouts,
        "n_episodes": n_episodes,
        "avg_landing_vel": np.mean(landing_vels) if landing_vels else 0,
        "altitude_range": altitude_range,
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


def _save_recording(env, frames, duration, ep, outcome, init_alt, output_dir):
    filepath = output_dir / f"bc_glide_ep{ep}_{outcome}_from{int(init_alt)}m.ndjson"
    with open(filepath, "w") as f:
        meta = MetadataMessage(
            run_by="eval_bc_glide.py",
            run_at=datetime.now().isoformat(),
            scenario_name=f"BC Glide Landing from {init_alt:.1f}m",
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
    print(f"  -> Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC glide landing model")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--altitude", type=str, default="10-30",
                        help="Altitude range, e.g., '10-30' or '30-50'")
    parser.add_argument("--record", action="store_true", help="Record episodes")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    # Parse altitude range
    alt_parts = args.altitude.split("-")
    altitude_range = (float(alt_parts[0]), float(alt_parts[1]))

    model_path = Path(args.model) if args.model else None
    model = load_model(model_path)

    print(f"Evaluating BC Glide model: {altitude_range[0]}-{altitude_range[1]}m")
    print(f"Episodes: {args.n_episodes}")
    print("=" * 60)

    results = evaluate(
        model,
        n_episodes=args.n_episodes,
        altitude_range=altitude_range,
        record=args.record,
        verbose=True,
    )

    print("=" * 60)
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Crashes: {results['n_crashes']}, Timeouts: {results['n_timeouts']}")
    if results["avg_landing_vel"] > 0:
        print(f"Avg landing velocity: {results['avg_landing_vel']:.2f} m/s")


if __name__ == "__main__":
    main()
