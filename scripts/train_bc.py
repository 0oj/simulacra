"""
Behavioral Cloning (BC) training using expert demonstrations from LandingController.

Phase 1 of the BC + SAC training pipeline:
1. Collect expert demonstrations from LandingController
2. Train an MLP policy to imitate the expert
3. Save the policy for SAC fine-tuning

KNOWN LIMITATIONS:
    - The LandingController has issues completing landings from higher altitudes
    - It tends to hover at ~1m instead of touching down
    - Success rate is ~20% with narrow initial conditions
    - Consider using pure SAC (train_sac.py) instead, which doesn't require expert demos

Usage:
    python scripts/train_bc.py

Output:
    runs/bc/bc_policy.pt - Trained BC policy
    runs/bc/demos.npz - Collected demonstrations (for debugging)
"""

import sys
from pathlib import Path
from datetime import datetime
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from simulacra.gym_env import RocketLandingEnvV2
from simulacra.noumena import Kernel, Rocket, RocketState, Environment, MassModel
from simulacra.noumena.mass_model import MassComponent, FuelComponent
from simulacra.noumena import get_upright_orientation, RocketProps
from simulacra.phenomena.sensors import Sensors
from simulacra.phenomena.actuators import (
    Actuators,
    GimbalActuator,
    ThrottleActuator,
    GimbalCommand,
    ThrottleCommand,
)
from simulacra.physics import RocketVec3, WorldVec3, Orientation
from simulacra.onboard.landing_controller import LandingController, LandingControllerConfig

from scipy.spatial.transform import Rotation as R


class BCPolicy(nn.Module):
    """
    MLP policy for behavioral cloning.

    Input: 13-dim observation (same as RocketLandingEnvV2)
    Output: 3-dim action (throttle, gimbal_outer, gimbal_inner)
    """

    def __init__(self, obs_dim: int = 13, act_dim: int = 3, hidden_dims: tuple = (256, 256)):
        super().__init__()

        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, act_dim))
        # Output activation: sigmoid for throttle [0,1], tanh for gimbal [-1,1]
        # We'll handle this in forward()

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.net(obs)
        # Split: throttle uses sigmoid, gimbals use tanh
        throttle = torch.sigmoid(raw[:, 0:1])
        gimbal = torch.tanh(raw[:, 1:3])
        return torch.cat([throttle, gimbal], dim=1)


def collect_expert_demos(
    n_episodes: int = 500,
    seed_start: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Collect expert demonstrations from LandingController.

    Returns:
        observations: (N, 13) array of observations
        actions: (N, 3) array of actions (throttle [0,1], gimbal [-1,1])
        episode_stats: list of per-episode statistics
    """
    # Use the same parameters as RocketLandingEnvV2
    dt = 0.02
    dry_mass = 60.0
    fuel_mass = 80.0
    length = 4.0
    max_thrust = 2000.0
    burn_rate = 4.0
    max_gimbal_angle = 0.26

    # Initial conditions tuned for LandingController success
    # The controller works best at low altitude with moderate downward velocity
    # (higher velocity triggers earlier braking)
    altitude_range = (5.0, 10.0)  # Low altitude where controller can brake
    velocity_range = (-5.0, -2.0)  # Moderate downward - triggers braking
    horizontal_range = (-2.0, 2.0)  # Small horizontal offset
    tilt_range = (-0.05, 0.05)  # Minimal initial tilt
    fuel_range = (0.8, 1.0)

    all_obs = []
    all_actions = []
    episode_stats = []

    rng = np.random.default_rng(seed_start)

    iterator = range(n_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Collecting demos")

    for ep in iterator:
        # Sample initial conditions
        init_alt = rng.uniform(*altitude_range)
        init_vy = rng.uniform(*velocity_range)
        init_vx = rng.uniform(*horizontal_range) * 0.1
        init_vz = rng.uniform(*horizontal_range) * 0.1
        init_x = rng.uniform(*horizontal_range)
        init_z = rng.uniform(*horizontal_range)
        init_tilt_y = rng.uniform(*tilt_range)
        init_tilt_z = rng.uniform(*tilt_range)
        init_fuel = rng.uniform(*fuel_range)

        # Build perturbed orientation
        base_rot = get_upright_orientation()._as_scipy_rotation()
        perturb_y = R.from_euler("y", init_tilt_y)
        perturb_z = R.from_euler("z", init_tilt_z)
        combined = perturb_z * perturb_y * base_rot
        x, y, z, w = combined.as_quat()
        init_orient = Orientation.from_wxyz(w, x, y, z)

        # Create rocket
        mass_model = MassModel(components=[
            MassComponent("dry", dry_mass, RocketVec3(length / 2, 0, 0)),
            FuelComponent(
                "fuel",
                mass=fuel_mass * init_fuel,
                position=RocketVec3(length * 0.3, 0, 0),
                full_mass=fuel_mass,
                burn_rate=burn_rate,
            ),
        ])
        props = mass_model.to_rocket_props(
            length=length,
            thrust_application_offset=RocketVec3(0, 0, 0),
        )

        init_state = RocketState(
            position=WorldVec3(init_x, init_alt, init_z),
            velocity=WorldVec3(init_vx, init_vy, init_vz),
            orientation=init_orient,
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(0.0, 0.0, 0.0),
        )

        rocket = Rocket(props=props, state=init_state, mass_model=mass_model)
        environment = Environment(
            gravity=WorldVec3(0.0, -9.81, 0.0),
            wind=WorldVec3(0.0, 0.0, 0.0),
            air_density=1.225,
        )

        kernel = Kernel(rocket, environment)
        sensors = Sensors(kernel)

        gimbal = GimbalActuator(
            rocket=rocket,
            max_outer_angle=max_gimbal_angle,
            max_inner_angle=max_gimbal_angle,
            max_rate=1.0,
            tau=0.05,
        )
        throttle_actuator = ThrottleActuator(
            rocket=rocket,
            max_thrust_rate=max_thrust * 2,
            min_thrust=0.0,
        )
        actuators = Actuators(gimbal=gimbal, throttle=throttle_actuator)

        # Setup controller
        controller_config = LandingControllerConfig(
            weight=(dry_mass + fuel_mass * init_fuel) * 9.81,
            max_thrust=max_thrust,
            max_gimbal_angle=max_gimbal_angle,
        )
        controller = LandingController(
            throttle=throttle_actuator,
            gimbal=gimbal,
            config=controller_config,
        ).with_sensors(sensors)

        # Run episode
        ep_obs = []
        ep_actions = []
        time = 0.0
        max_steps = 1000

        for step in range(max_steps):
            # Get observation (same as RocketLandingEnvV2._get_obs)
            gps = sensors.GPS.read()
            alt = sensors.Barometer.read()
            vel = sensors.VelocitySensor.read()
            orient = sensors.AttitudeEstimator.read()
            omega = sensors.GyroSensor.read()
            qw, qx, qy, qz = orient.as_array()

            obs = np.array([
                gps.x, alt, gps.z,
                vel.x, vel.y, vel.z,
                qw, qx, qy, qz,
                omega.x, omega.y, omega.z,
            ], dtype=np.float32)

            # Run controller (this modifies actuator commands internally)
            controller.run()

            # Extract action from actuator commands
            # Get the commanded thrust and angles
            commanded_thrust = throttle_actuator._commanded_thrust
            commanded_outer = gimbal.outer_angle  # actual angle after rate limiting
            commanded_inner = gimbal.inner_angle

            # Normalize to action space
            action = np.array([
                commanded_thrust / max_thrust,  # [0, 1]
                commanded_outer / max_gimbal_angle,  # [-1, 1]
                commanded_inner / max_gimbal_angle,  # [-1, 1]
            ], dtype=np.float32)

            # Clip actions to valid range
            action = np.clip(action, [-0.0, -1.0, -1.0], [1.0, 1.0, 1.0])

            ep_obs.append(obs)
            ep_actions.append(action)

            # Step simulation
            sensors.set_time(time)
            actuators.set_time(time)
            actuators.step(dt)
            kernel.step(dt)
            sensors.update_attitude_estimator(dt)

            time += dt

            # Check termination
            ground = kernel.ground_contact
            if ground.on_ground:
                break

            # Check failure conditions
            tilt = _get_tilt_angle(kernel)
            if tilt > 1.0 or kernel.rocket.state.position.y > 100.0:
                break
            if kernel.rocket.mass_model.fuel_fraction <= 0:
                break

        # Record episode stats
        ground = kernel.ground_contact
        outcome = "landed" if ground.on_ground and not ground.crashed else "crash"
        stats = {
            "episode": ep,
            "outcome": outcome,
            "steps": len(ep_obs),
            "init_alt": init_alt,
            "final_vel": kernel.rocket.state.velocity.norm,
        }
        episode_stats.append(stats)

        # Only add successful episodes to training data
        if outcome == "landed":
            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)

    observations = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)

    return observations, actions, episode_stats


def _get_tilt_angle(kernel: Kernel) -> float:
    """Compute tilt angle from vertical (radians)."""
    orient = kernel.rocket.state.orientation
    nose_body = RocketVec3(1.0, 0.0, 0.0)
    nose_world = nose_body.to_world(orient)
    cos_angle = max(-1.0, min(1.0, nose_world.y))
    return math.acos(cos_angle)


def train_bc_policy(
    observations: np.ndarray,
    actions: np.ndarray,
    hidden_dims: tuple = (256, 256),
    batch_size: int = 256,
    n_epochs: int = 100,
    lr: float = 1e-3,
    val_split: float = 0.1,
    verbose: bool = True,
) -> tuple[BCPolicy, dict]:
    """
    Train BC policy using MSE loss.

    Returns:
        policy: Trained BCPolicy
        history: Training history (losses)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Split into train/val
    n_samples = len(observations)
    n_val = int(n_samples * val_split)
    indices = np.random.permutation(n_samples)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_obs = torch.tensor(observations[train_idx], dtype=torch.float32, device=device)
    train_act = torch.tensor(actions[train_idx], dtype=torch.float32, device=device)
    val_obs = torch.tensor(observations[val_idx], dtype=torch.float32, device=device)
    val_act = torch.tensor(actions[val_idx], dtype=torch.float32, device=device)

    train_dataset = TensorDataset(train_obs, train_act)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model
    policy = BCPolicy(
        obs_dim=observations.shape[1],
        act_dim=actions.shape[1],
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    iterator = range(n_epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training BC")

    for epoch in iterator:
        # Training
        policy.train()
        epoch_loss = 0.0
        n_batches = 0

        for obs_batch, act_batch in train_loader:
            optimizer.zero_grad()
            pred_act = policy(obs_batch)
            loss = nn.functional.mse_loss(pred_act, act_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches

        # Validation
        policy.eval()
        with torch.no_grad():
            pred_val = policy(val_obs)
            val_loss = nn.functional.mse_loss(pred_val, val_act).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    return policy, history


def evaluate_bc_policy(
    policy: BCPolicy,
    n_episodes: int = 100,
    seed: int = 9999,
    verbose: bool = True,
) -> dict:
    """
    Evaluate BC policy on RocketLandingEnvV2.

    Returns:
        stats: Dictionary with success rate, avg reward, etc.
    """
    device = next(policy.parameters()).device
    policy.eval()

    env = RocketLandingEnvV2()

    successes = 0
    total_rewards = []
    landing_vels = []
    landing_tilts = []

    iterator = range(n_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating BC")

    for ep in iterator:
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_tensor).cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_rewards.append(ep_reward)

        if info.get("outcome") == "landed":
            successes += 1
            landing_vels.append(info.get("landing_vel", 0))
            landing_tilts.append(info.get("landing_tilt", 0))

    env.close()

    success_rate = successes / n_episodes

    return {
        "success_rate": success_rate,
        "n_successes": successes,
        "n_episodes": n_episodes,
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_landing_vel": np.mean(landing_vels) if landing_vels else 0,
        "avg_landing_tilt": np.mean(landing_tilts) if landing_tilts else 0,
    }


def main():
    save_dir = Path("runs/bc")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Behavioral Cloning Training")
    print("=" * 60)

    # Phase 1: Collect demonstrations
    print("\n[Phase 1] Collecting expert demonstrations...")
    observations, actions, episode_stats = collect_expert_demos(
        n_episodes=500,
        seed_start=0,
        verbose=True,
    )

    # Compute stats
    successes = sum(1 for s in episode_stats if s["outcome"] == "landed")
    print(f"\nDemo collection complete:")
    print(f"  Episodes: {len(episode_stats)}")
    print(f"  Successes: {successes} ({100*successes/len(episode_stats):.1f}%)")
    print(f"  Transitions: {len(observations)}")

    # Save demonstrations (for debugging)
    np.savez(
        save_dir / "demos.npz",
        observations=observations,
        actions=actions,
    )
    print(f"  Saved demos to {save_dir / 'demos.npz'}")

    # Phase 2: Train BC policy
    print("\n[Phase 2] Training BC policy...")
    policy, history = train_bc_policy(
        observations,
        actions,
        hidden_dims=(256, 256),
        batch_size=256,
        n_epochs=100,
        lr=1e-3,
        verbose=True,
    )

    # Save policy
    policy_path = save_dir / "bc_policy.pt"
    torch.save({
        "state_dict": policy.state_dict(),
        "hidden_dims": (256, 256),
        "obs_dim": 13,
        "act_dim": 3,
    }, policy_path)
    print(f"  Saved policy to {policy_path}")

    # Save training history
    np.savez(
        save_dir / "history.npz",
        train_loss=np.array(history["train_loss"]),
        val_loss=np.array(history["val_loss"]),
    )

    # Phase 3: Evaluate
    print("\n[Phase 3] Evaluating BC policy...")
    eval_stats = evaluate_bc_policy(policy, n_episodes=100, verbose=True)

    print(f"\nEvaluation results:")
    print(f"  Success rate: {eval_stats['success_rate']*100:.1f}%")
    print(f"  Avg reward: {eval_stats['avg_reward']:.2f} +/- {eval_stats['std_reward']:.2f}")
    print(f"  Avg landing velocity: {eval_stats['avg_landing_vel']:.2f} m/s")
    print(f"  Avg landing tilt: {eval_stats['avg_landing_tilt']:.1f} deg")

    # Save eval stats
    with open(save_dir / "eval_stats.txt", "w") as f:
        for k, v in eval_stats.items():
            f.write(f"{k}: {v}\n")

    print("\n" + "=" * 60)
    print("BC Training Complete!")
    print(f"Policy saved to: {policy_path}")
    print("=" * 60)

    return eval_stats["success_rate"] >= 0.5


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
