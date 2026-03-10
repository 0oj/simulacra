"""
Train a behavioral cloning model from the glide path landing controller.

The glide path controller achieves ~85% success from 10-20m altitude.
This script collects demonstrations and trains a neural network to imitate it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from simulacra.gym_env import RocketLandingEnvV2


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


def glide_path_expert(obs: np.ndarray, env: RocketLandingEnvV2, mass: float) -> np.ndarray:
    """
    Glide path controller - the expert policy for BC.

    Same implementation as in test_suicide_burn.py.
    """
    alt = obs[1]
    vy = obs[4]
    wx, wy, wz = obs[10], obs[11], obs[12]

    g = 9.81
    max_thrust = env.max_thrust
    hover_throttle = (mass * g) / max_thrust

    k = 1.5
    min_descent_speed = -0.8
    max_descent_speed = -15.0

    if alt > 0.5:
        target_vy = max(max_descent_speed, min(-k * np.sqrt(alt), min_descent_speed))
    else:
        target_vy = -1.0

    if alt < 0.4 and abs(vy) < 2.5:
        throttle = 0.0
    else:
        vy_error = target_vy - vy

        if alt < 2.0:
            Kp = 0.25
        elif alt < 10.0:
            Kp = 0.30
        else:
            Kp = 0.35

        throttle = hover_throttle + Kp * vy_error
        throttle = np.clip(throttle, 0.0, 1.0)

    gimbal_outer = np.clip(-0.5 * wz, -0.3, 0.3)
    gimbal_inner = np.clip(-0.5 * wy, -0.3, 0.3)

    return np.array([throttle, gimbal_outer, gimbal_inner], dtype=np.float32)


def collect_demonstrations(
    n_episodes: int = 500,
    altitude_range: tuple = (5.0, 30.0),
    velocity_range: tuple = (-5.0, 0.0),
    verbose: bool = True,
) -> tuple:
    """
    Collect (obs, action) pairs from the expert policy.

    Returns:
        observations: np.ndarray of shape (N, 13)
        actions: np.ndarray of shape (N, 3)
        success_rate: float
    """
    env = RocketLandingEnvV2(
        init_altitude_range=altitude_range,
        init_velocity_range=velocity_range,
        init_horizontal_range=(-2.0, 2.0),
        init_tilt_range=(-0.05, 0.05),
    )

    all_obs = []
    all_actions = []
    successes = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        mass = env._kernel.rocket.mass_model.total_mass
        done = False
        ep_obs = []
        ep_actions = []

        while not done:
            action = glide_path_expert(obs, env, mass)
            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            mass = env._kernel.rocket.mass_model.total_mass

        outcome = info.get("outcome", "timeout")
        if outcome == "landed":
            successes += 1
            # Only include successful episodes in training data
            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)

        if verbose and (ep + 1) % 50 == 0:
            print(f"Collected {ep + 1}/{n_episodes} episodes, "
                  f"{len(all_obs)} transitions, "
                  f"{100*successes/(ep+1):.0f}% success rate")

    env.close()

    return (
        np.array(all_obs, dtype=np.float32),
        np.array(all_actions, dtype=np.float32),
        successes / n_episodes,
    )


def train_bc(
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    validation_split: float = 0.1,
    verbose: bool = True,
) -> BCNet:
    """
    Train a BC model on the collected demonstrations.
    """
    # Split into train/val
    n = len(observations)
    n_val = int(n * validation_split)
    indices = np.random.permutation(n)

    train_obs = observations[indices[n_val:]]
    train_act = actions[indices[n_val:]]
    val_obs = observations[indices[:n_val]]
    val_act = actions[indices[:n_val]]

    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(train_obs),
        torch.tensor(train_act),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_obs),
        torch.tensor(val_act),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model and optimizer
    model = BCNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for obs_batch, act_batch in train_loader:
            pred = model(obs_batch)
            loss = loss_fn(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(obs_batch)

        train_loss /= len(train_obs)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                pred = model(obs_batch)
                loss = loss_fn(pred, act_batch)
                val_loss += loss.item() * len(obs_batch)

        val_loss /= len(val_obs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # Load best model
    model.load_state_dict(best_state)
    return model


def evaluate_model(
    model: BCNet,
    n_episodes: int = 50,
    altitude_range: tuple = (10.0, 30.0),
    verbose: bool = True,
) -> dict:
    """Evaluate the trained BC model."""
    env = RocketLandingEnvV2(
        init_altitude_range=altitude_range,
        init_velocity_range=(-5.0, 0.0),
        init_horizontal_range=(-2.0, 2.0),
        init_tilt_range=(-0.05, 0.05),
    )

    model.eval()
    successes = 0
    crashes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 1000)  # Different seeds than training
        done = False

        while not done:
            with torch.no_grad():
                action = model(torch.tensor(obs).unsqueeze(0)).numpy()[0]
            action = np.clip(action, [0, -1, -1], [1, 1, 1])

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

        outcome = info.get("outcome", "timeout")
        if outcome == "landed":
            successes += 1
        elif outcome == "crash":
            crashes += 1

        if verbose:
            print(f"Ep {ep + 1:3d}: {outcome}")

    env.close()

    return {
        "success_rate": successes / n_episodes,
        "crashes": crashes,
        "n_episodes": n_episodes,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train BC model for rocket landing")
    parser.add_argument("--n-demos", type=int, default=500, help="Number of demo episodes")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--output", type=str, default="runs/bc_glide/bc_model.pt")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Behavioral Cloning for Rocket Landing (Glide Path Expert)")
    print("=" * 60)

    # Collect demonstrations
    print("\nPhase 1: Collecting demonstrations...")
    obs, actions, expert_success = collect_demonstrations(
        n_episodes=args.n_demos,
        altitude_range=(5.0, 30.0),
        velocity_range=(-5.0, 0.0),
    )
    print(f"\nCollected {len(obs)} transitions from {args.n_demos} episodes")
    print(f"Expert success rate: {100*expert_success:.1f}%")

    # Train BC model
    print("\nPhase 2: Training BC model...")
    model = train_bc(obs, actions, epochs=args.epochs)

    # Save model
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved model to {output_path}")

    # Evaluate on training conditions
    print("\nPhase 3: Evaluating on training conditions (10-30m)...")
    results = evaluate_model(model, n_episodes=50, altitude_range=(10.0, 30.0))
    print(f"\nBC Model Success Rate: {100*results['success_rate']:.1f}%")

    # Evaluate on harder conditions
    print("\nPhase 4: Evaluating on harder conditions (30-50m)...")
    results_hard = evaluate_model(model, n_episodes=50, altitude_range=(30.0, 50.0))
    print(f"\nBC Model Success Rate (30-50m): {100*results_hard['success_rate']:.1f}%")


if __name__ == "__main__":
    main()
