# Training Approaches for Rocket Landing

This guide documents different approaches for training a rocket landing policy, their tradeoffs, and implementation details.

---

## Executive Summary (Handoff Notes)

**Current State (March 2026):**
- `LandingController` achieves **100% success** from 20-400m altitude with wind
- All full 3D RL approaches have failed due to gimbal exploration destroying stability
- **Currently running:** Throttle-only PPO with curriculum (approach 8)
- **Roadmap:** Throttle-only → Residual gimbal learning → Full control

**What's Been Tried:**

| Approach | Steps | Best Reward | Landings | Outcome |
|----------|-------|-------------|----------|---------|
| PPO Baseline | 100M | -11 | 0 | Plateaued |
| PPO + RAJS | 21M+ | -18 | 0 | Plateaued at 3M |
| Curriculum PPO | 32M | -8 | 40% at 5M, then 0 | Catastrophic forgetting |
| BC (glide path) | N/A | N/A | 0% deploy | Distribution shift |
| DAgger + PPO | 3 rounds | N/A | 34% → 0% | DAgger degraded BC |
| **Throttle-only PPO** | **running** | **TBD** | **TBD** | **Current approach** |

**Root Causes Identified:**
1. **Gimbal exploration is catastrophic**: PPO's Gaussian noise (std ~0.5-1.0) on gimbal actions creates random torques → rocket tumbles in ~40 steps. This is the #1 failure mode across all RL approaches.
2. **BC distribution shift**: BC model's throttle output is biased ~0.05-0.1 higher than expert → rocket hovers instead of descending → all timeouts at deployment.
3. **DAgger makes it worse**: Adding expert-labeled data from the learner's failure states (mostly tumbling/crashing) overwhelms the dataset with non-representative transitions.
4. **Reward function issues**: Multiplicative terminal rewards, sparse signal, imbalanced penalties (all fixed in V2 env).
5. **Curriculum bug (old)**: Difficulty never actually increased (env update was no-op — fixed).

**Key diagnostic finding:** Constant hover thrust (~0.6) with zero gimbal lands **70% of the time**. The rocket is naturally stable — the problem is that RL exploration actively destabilizes it.

**Current Strategy — Throttle-only PPO:**
- Fix gimbals to proportional damping (`gimbal = -0.5 * omega`), only learn throttle (1D action)
- Eliminates the gimbal exploration problem entirely
- 4-level curriculum: L0 (3-8m) → L3 (10-60m)
- Dense glide-path-following reward

---

## Problem Statement

**Goal:** Land a rocket from 10-200m altitude with varying initial conditions:
- Vertical velocity: 0 to -25 m/s
- Horizontal offset: up to 20m from target
- Tilt: up to ±15°
- Wind: up to 10 m/s crosswind

**Observation space (13-dim):**
```
[x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
```

**Action space (3-dim continuous):**
```
[throttle (0-1), gimbal_outer (-1,1), gimbal_inner (-1,1)]
```

**Success criteria:**
- Landing velocity < 3 m/s
- Tilt < 15°
- Position near target

---

## Approach 1: Classical Control (Baseline)

**Status:** ✅ Working (100% success)

The `LandingController` uses proportional guidance with dynamic mass estimation:

```python
# Vertical guidance: glide path velocity profile
target_vy = -k_descent * sqrt(altitude)  # k=2.0

# Velocity tracking
vy_error = target_vy - vy
thrust = hover_thrust + Kp * vy_error

# Attitude control: point thrust opposite to desired acceleration
# Gimbal damping
gimbal = -Kd * angular_velocity
```

**Pros:**
- 100% success rate
- Interpretable, verifiable
- No training required
- Fast inference

**Cons:**
- Hand-tuned gains
- May not be globally optimal
- Doesn't adapt to model uncertainty

**Use this when:** You need a working controller now, or as expert policy for imitation learning.

---

## Approach 2: Behavioral Cloning (BC)

**Status:** ✅ Working (84% success at 10-30m)

Train a neural network to imitate the classical controller.

```python
# 1. Collect demonstrations
expert = LandingController(...)
dataset = []
for episode in range(300):
    obs = env.reset()
    while not done:
        action = expert(obs)
        dataset.append((obs, action))
        obs, _, done, _ = env.step(action)

# 2. Train network
model = MLP([128, 128])  # Small is fine
loss = MSE(model(obs), expert_action)
```

**Implementation:** `scripts/train_bc_glide.py`

**Pros:**
- Simple, fast to train
- Inherits expert's structure
- Works with small networks

**Cons:**
- Bounded by expert performance
- Distribution shift (compounding errors)
- Can't exceed expert

**Use this when:** Expert policy exists and you want a neural network approximation for SINDy distillation.

---

## Approach 3: BC + SAC Fine-tuning ⭐ RECOMMENDED

**Status:** 🔄 Not yet implemented - THIS IS THE RECOMMENDED NEXT STEP

Initialize policy from BC, then fine-tune with SAC to exceed expert.

### Why This Approach

| Factor | BC + SAC | Pure RL (PPO/RAJS) |
|--------|----------|-------------------|
| Starting point | Working 100% policy | Random policy |
| Exploration needed | Minimal | Massive |
| Time to first success | Minutes (BC phase) | Never achieved |
| Risk | Forgetting | Never learning |

### Phase 1: Behavioral Cloning

**Create file:** `scripts/train_bc.py`

```python
import torch
import torch.nn as nn
import numpy as np
from simulacra.gym_env import RocketLandingEnv
from simulacra.onboard.landing_controller import LandingController, LandingControllerConfig

class BCPolicy(nn.Module):
    def __init__(self, obs_dim=13, act_dim=3, hidden=[256, 256]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        layers.append(nn.Tanh())  # Actions in [-1, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def collect_demonstrations(n_episodes=500):
    """Collect expert demonstrations from LandingController."""
    env = RocketLandingEnv(
        altitude_range=(10, 100),  # Varied conditions
        velocity_range=(-15, 0),
        tilt_range=(-0.15, 0.15),
    )

    config = LandingControllerConfig(
        dry_mass=env.unwrapped.dry_mass,
        full_fuel_mass=env.unwrapped.fuel_mass,
        max_thrust=env.unwrapped.max_thrust,
    )
    expert = LandingController(env.unwrapped.sensors, config)

    observations, actions = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = expert.update(0.05)  # dt=0.05
            action_normalized = np.array([
                action[0] * 2 - 1,  # throttle 0-1 -> -1 to 1
                action[1],          # gimbal already -1 to 1
                action[2],
            ])
            observations.append(obs)
            actions.append(action_normalized)
            obs, _, terminated, truncated, _ = env.step(action_normalized)
            done = terminated or truncated

    return np.array(observations), np.array(actions)

def train_bc(observations, actions, epochs=100, batch_size=256, lr=1e-3):
    """Train BC policy on demonstrations."""
    policy = BCPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(observations),
        torch.FloatTensor(actions)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for obs_batch, act_batch in loader:
            pred = policy(obs_batch)
            loss = nn.MSELoss()(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

    return policy

if __name__ == "__main__":
    print("Collecting demonstrations...")
    obs, acts = collect_demonstrations(500)
    print(f"Collected {len(obs)} transitions")

    print("Training BC policy...")
    policy = train_bc(obs, acts)

    torch.save(policy.state_dict(), "runs/bc_policy.pt")
    print("Saved to runs/bc_policy.pt")
```

**Expected outcome:** >70% landing rate in ~5 minutes of training

### Phase 2: SAC Fine-tuning

**Create file:** `scripts/train_sac_finetune.py`

```python
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import torch

# Load BC policy weights into SAC
def load_bc_into_sac(sac_model, bc_weights_path):
    """Initialize SAC actor from BC policy."""
    bc_state = torch.load(bc_weights_path)
    # Map BC weights to SAC actor network
    # (implementation depends on exact architecture match)
    pass

# Create env with FIXED reward function
class FixedRewardWrapper(gym.Wrapper):
    """Replace broken multiplicative rewards with additive."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Override reward
        if info.get("landed"):
            vel = info.get("landing_velocity", 3.0)
            tilt = info.get("landing_tilt", 0.26)
            reward = 50.0  # Base landing reward
            reward += 30.0 * max(0, 1 - vel/3.0)      # Velocity bonus
            reward += 15.0 * max(0, 1 - tilt/0.26)    # Tilt bonus
        elif info.get("crashed"):
            reward = -100.0
        else:
            # Dense shaping during flight
            alt = obs[1]
            vy = obs[4]
            target_vy = -2.0 * np.sqrt(max(0.5, alt))
            vy_error = abs(vy - target_vy)

            reward = -0.1 * vy_error  # Glide path following
            reward -= 0.05 * compute_tilt(obs[6:10])  # Stay upright

        return obs, reward, terminated, truncated, info

# Training
env = FixedRewardWrapper(RocketLandingEnv(...))

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=1e-4,        # Lower for fine-tuning
    buffer_size=1_000_000,
    batch_size=256,
    ent_coef=0.01,             # Lower entropy (don't destroy BC init)
    learning_starts=10_000,
    verbose=1,
)

# Load BC weights
load_bc_into_sac(model, "runs/bc_policy.pt")

# Early stopping
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="runs/sac_best",
    eval_freq=10_000,
    callback_after_eval=StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        verbose=1
    ),
)

model.learn(total_timesteps=5_000_000, callback=eval_callback)
```

**Key hyperparameters:**
- `learning_rate=1e-4`: Lower than default to preserve BC initialization
- `ent_coef=0.01`: Low entropy to not randomize away from working policy
- `learning_starts=10_000`: Let replay buffer fill before training

### Success Criteria

| Phase | Target |
|-------|--------|
| BC | >50% landing rate |
| SAC Fine-tune | >80% landing rate |
| SAC + curriculum | >90% on full altitude range |

---

## Approach 4: PPO + RAJS (Current)

**Status:** ❌ Failed - Plateaued at -20 reward after 21M+ steps, 0 landings

Random Annealing Jump Start uses guide policy to bootstrap exploration.

### Experimental Results (March 2026)

**Training run:** `scripts/train_ppo_rajs.py` with `LandingController` as guide

| Metric | Value |
|--------|-------|
| Total steps | 21M+ (still running) |
| Rollout reward | -20 avg (range -29 to -13) |
| Eval reward | -5 (stable entire run) |
| Successful landings | 0 |
| Episode length | ~110 steps |
| FPS | 2581 |

**Key observations:**
- Reward improved from -60 to -20 in first 3M steps
- **Completely flat since 3M steps** - no further improvement
- All episodes labeled "unknown" (timeout, no land/crash)
- RAJS beta stays at 1.0 (guide always active, never hands off)
- Eval reward (-5) much better than rollout (-20) because eval uses deterministic actions

```python
# During rollout
if step < guide_horizon:  # H ~ U(0, H_max * beta)
    action = guide_policy(obs)
else:
    action = rl_policy(obs)

# Beta anneals 1.0 → 0.0 as success rate increases
```

**Implementation:** `scripts/train_ppo_rajs.py`

**Current issues:**
1. Beta not annealing (success rate threshold not met)
2. Agent learns to rely on guide
3. Sparse terminal rewards provide weak gradient
4. Rollout rewards plateau while eval stays flat

**Recommended fixes:**

### Fix 4.1: Force beta annealing on schedule
```python
# Instead of success-based annealing:
beta = max(0, 1.0 - (timestep / anneal_steps))  # Linear decay over 50M steps
```

### Fix 4.2: Add dense reward shaping
```python
def shaped_reward(obs, action, next_obs):
    alt = next_obs[1]
    vy = next_obs[4]

    # Reward following glide path
    target_vy = -2.0 * np.sqrt(max(0.5, alt))
    vy_error = abs(vy - target_vy)
    glide_reward = -0.1 * vy_error  # Dense reward every step

    # Reward being upright
    tilt = compute_tilt(next_obs[6:10])
    tilt_reward = -0.05 * tilt

    return glide_reward + tilt_reward  # Add to environment reward
```

### Fix 4.3: Physics-informed observations
```python
def augment_observation(obs):
    alt, vy = obs[1], obs[4]

    # Add computed features
    time_to_impact = alt / max(0.1, -vy) if vy < 0 else 999
    target_vy = -2.0 * np.sqrt(max(0.5, alt))
    vy_error = vy - target_vy

    # Stopping distance at max deceleration
    max_decel = (max_thrust / mass) - g
    stopping_dist = vy**2 / (2 * max_decel) if vy < 0 else 0

    return np.concatenate([obs, [time_to_impact, vy_error, stopping_dist]])
```

---

## Approach 4b: Curriculum Learning

**Status:** ❌ Failed - Catastrophic forgetting after initial success

Start with easy conditions, gradually increase difficulty.

### Experimental Results (March 2026)

**Training run:** `scripts/train_ppo_curriculum.py`

| Phase | Steps | Altitude | Success Rate |
|-------|-------|----------|--------------|
| Initial | 0-5M | 5-10m | **40%** ✅ |
| Mid | 5-15M | 5-10m (stuck) | 20% |
| Late | 15-32M | 5-10m (stuck) | **0%** ❌ |

**What went wrong:**

1. **Curriculum never progressed**: The env difficulty update was a no-op
   ```python
   # BUG: This didn't actually change the env
   env.set_difficulty(new_level)  # Method existed but did nothing
   ```

2. **No early stopping**: Training continued as performance degraded

3. **Catastrophic forgetting**: Agent unlearned successful policy
   - At 5M: 40% landing rate
   - At 32M: 0% landing rate (worse than random!)

**Lessons learned:**
- Curriculum CAN work (proved by 40% success at 5M)
- Need actual difficulty progression
- Need early stopping when performance drops
- Need checkpointing to recover from forgetting

**If retrying curriculum:**
```python
# Fix 1: Actually update env difficulty
env = make_env(altitude_range=(min_alt, max_alt))  # Recreate env

# Fix 2: Early stopping
if success_rate < best_success_rate - 0.1:
    load_checkpoint(best_checkpoint)
    break

# Fix 3: Progressive thresholds
difficulties = [
    {"alt": (5, 10), "threshold": 0.7},   # Must hit 70% before advancing
    {"alt": (10, 30), "threshold": 0.6},
    {"alt": (30, 60), "threshold": 0.5},
]
```

---

## Approach 5: Model-Based RL

**Status:** 📋 Not implemented

Learn dynamics model, then plan through it.

```python
# 1. Learn dynamics model
dynamics = MLP([256, 256])
dynamics.fit(obs, action, next_obs)  # Supervised learning

# 2. Plan using model
def plan(obs, horizon=20):
    best_actions = None
    best_reward = -inf
    for _ in range(1000):  # Random shooting
        actions = sample_actions(horizon)
        reward = simulate(dynamics, obs, actions)
        if reward > best_reward:
            best_actions = actions
    return best_actions[0]
```

**Better variant: MPC with learned model**
```python
# Cross-entropy method or CMA-ES for action optimization
optimizer = CEM(action_dim=3, horizon=20)
actions = optimizer.optimize(dynamics, obs, reward_fn)
```

**Pros:**
- Sample efficient
- Can plan ahead
- Interpretable dynamics

**Cons:**
- Model errors compound
- Planning is slow
- Harder to implement

---

## Approach 6: Hierarchical RL

**Status:** 📋 Not implemented

Decompose into high-level decisions and low-level control.

```python
# High-level policy: when to brake, target attitude
class HighLevelPolicy:
    def __call__(self, obs):
        # Outputs: target_altitude_rate, target_attitude
        return [target_vy, target_roll, target_pitch]

# Low-level policy: track targets (can be classical PD)
class LowLevelPolicy:
    def __call__(self, obs, targets):
        target_vy, target_roll, target_pitch = targets
        throttle = pd_control(obs.vy, target_vy)
        gimbal = pd_control(obs.attitude, [target_roll, target_pitch])
        return [throttle, gimbal_outer, gimbal_inner]

# Train high-level with RL, use classical low-level
```

**Pros:**
- Simpler credit assignment
- Can reuse classical low-level control
- More interpretable

**Cons:**
- Needs careful interface design
- May limit expressiveness

---

## Approach 7: Direct SINDy on Controller

**Status:** 📋 Not implemented (but promising)

Skip RL entirely. Apply SINDy to extract symbolic policy from LandingController.

```python
# 1. Collect (state, action) pairs from LandingController
states = []
actions = []
for episode in range(1000):
    obs = env.reset()
    while not done:
        action = controller(obs)
        states.append(obs)
        actions.append(action)
        obs, _, done, _ = env.step(action)

# 2. Apply SINDy to find symbolic mapping
from pysindy import SINDy

# For throttle
model_throttle = SINDy(feature_library=PolynomialLibrary(degree=2))
model_throttle.fit(states, actions[:, 0])
print(model_throttle.equations())
# Output: throttle = 0.5 + 0.3*vy - 0.1*sqrt(y) + ...

# 3. Verify symbolic policy with Lyapunov analysis
```

**Pros:**
- No RL training needed
- Directly interpretable
- Easier to verify

**Cons:**
- Bounded by controller performance
- SINDy may not find good fit
- Need diverse demonstration data

---

## Approach 8: Throttle-Only PPO with Curriculum ⭐ CURRENT

**Status:** 🔄 Running — `scripts/train_landing_rl.py`

Fixes the core problem: gimbal exploration kills the rocket. Instead of learning all 3 actions, only learn throttle (1D). Gimbals are fixed to proportional damping.

### Architecture

```
ThrottleOnlyWrapper:
  action_space = Box([0], [1])  # Just throttle
  gimbal_outer = clip(-0.5 * omega_z, -0.3, 0.3)  # Proportional damping
  gimbal_inner = clip(-0.5 * omega_y, -0.3, 0.3)

GlideRewardWrapper:
  reward = 0.05 (alive)
         - 0.10 * |vy - target_vy|  (glide path error)
         - 0.15 * tilt²              (stay upright)
         - 0.05 * omega              (angular damping)
         - 0.01 * v_horiz            (drift)
         - 0.002 * alt               (progress)
  terminal: +100 (land) / -50 (crash) / -30 (tilt/fuel) / -5-alt*0.1 (timeout)

CurriculumWrapper:
  L0: alt 3-8m,   vel -2 to 0,  tilt ±0.02  (promote at 60%)
  L1: alt 5-15m,  vel -4 to 0,  tilt ±0.05  (promote at 55%)
  L2: alt 8-30m,  vel -6 to 0,  tilt ±0.10  (promote at 50%)
  L3: alt 10-60m, vel -8 to 0,  tilt ±0.15  (final)
```

### PPO Config
- `net_arch = [64, 64]` (small — 1D action is simple)
- `ent_coef = 0.01` (moderate exploration for 1D)
- `gamma = 0.99`, `gae_lambda = 0.95`
- 4 parallel envs, 10M total steps

### Why This Should Work
- Hover thrust alone lands 70% — the agent just needs to learn the descent profile
- 1D action space: PPO exploration is safe (only varies throttle)
- Proportional gimbal damping is the expert's strategy and keeps rocket stable
- Dense glide-path reward gives gradient every step

---

## Approach 8b: BC (Glide Path Expert)

**Status:** ❌ Failed — 77% expert success, 0% at deployment

**Implementation:** `scripts/train_bc_glide.py`

Trained BC on a simpler obs-based glide path expert (not the full LandingController):
```python
target_vy = -k * sqrt(alt)  # k=1.5
throttle = hover_throttle + Kp * (target_vy - vy)
gimbal = -0.5 * omega  # proportional damping
```

**Results:** Expert achieves 77% success rate (not 84% as previously claimed — retested). BC model gets 0% at deployment due to distribution shift: the learned throttle is systematically ~0.05-0.1 higher than expert, causing the rocket to hover indefinitely instead of descending. All episodes end as timeouts.

**Lesson:** BC distribution shift on throttle is fatal for descent. Even tiny bias compounds across hundreds of steps.

---

## Approach 8c: DAgger + PPO

**Status:** ❌ Failed — DAgger degraded performance from 34% to 0%

**Implementation:** `scripts/train_landing_rl.py` (old version, replaced by throttle-only)

Attempted to fix BC distribution shift with DAgger (Dataset Aggregation):
1. Train initial BC on 300 expert episodes → 34% success
2. Roll out BC policy, label every visited state with expert action
3. Retrain on aggregated dataset
4. Repeat

**Results:** Round 0 BC: 34%. After 3 DAgger rounds: 0%. Adding expert-labeled data from the BC's failure states (mostly tumbling/crashing) overwhelmed the dataset and made the model worse, not better.

**Lesson:** DAgger assumes the expert's actions are meaningful on the learner's states. But when the learner visits catastrophic states (tumbling at 45°), the expert's "ideal" action from that state is irrelevant — the episode is already lost.

---

## Roadmap: Throttle-Only → Full Control

### Phase 1: Throttle-only (current)
Learn descent profile with 1D action, gimbals on autopilot.

### Phase 2: Residual gimbal learning
```python
gimbal = proportional_damping(omega) + policy(obs) * bound
```
Start with `bound = 0.1`, increase gradually as agent improves.
The agent learns small corrections that improve on the proportional baseline.

### Phase 3: Full control (optional)
Remove proportional damping baseline. Agent controls throttle + gimbals directly.
By this point it has a good initialization from phase 2.

### Phase 4: Fuel-optimal trajectories
Add fuel efficiency term to reward. With full gimbal authority,
RL may rediscover gravity-turn-like maneuvers (tilting thrust into velocity vector).

---

## Recommended Path Forward

### For research (RL → SINDy → Verification):

1. **Throttle-only PPO** (current — establish basic RL landing)
2. **Residual gimbal learning** (extend to 3D control)
3. **Add fuel penalty** (discover efficient trajectories)
4. **Apply SINDy** to extract symbolic policy
5. **Verify** with Lyapunov/CBF analysis

### For deployment:

1. **Use LandingController directly** (100% success)
2. **Apply SINDy** to get symbolic form for verification
3. **Deploy** with runtime monitor and fallback

---

## Hyperparameter Reference

### PPO (from RAJS paper)
```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.995
gae_lambda = 0.97
clip_range = 0.2
ent_coef = 0.007  # Low entropy for fine-tuning
vf_coef = 0.5
max_grad_norm = 0.5
net_arch = [256, 256]
```

### SAC
```python
learning_rate = 1e-4  # Lower for fine-tuning
buffer_size = 1_000_000
batch_size = 256
tau = 0.005
gamma = 0.99
ent_coef = "auto"  # Or 0.01 for fine-tuning
```

### BC
```python
epochs = 100
batch_size = 256
learning_rate = 1e-3
hidden_dims = [128, 128]  # Small network
```

---

## Critical Fixes Made to LandingController

The `LandingController` was broken (0% success) until these fixes were applied:

### Fix 1: Dynamic Mass Estimation

**Problem:** Fixed weight didn't account for fuel burn, causing thrust overshoot.

```python
# BEFORE (broken):
mass = cfg.weight / cfg.gravity  # Fixed mass

# AFTER (working):
fuel_fraction = self.sensors.FuelSensor.read()
current_fuel_mass = fuel_fraction * cfg.full_fuel_mass
mass = cfg.dry_mass + current_fuel_mass
```

**Location:** `src/simulacra/onboard/landing_controller.py:45-50`

### Fix 2: Smooth Terminal Transition

**Problem:** Discontinuous jump from sqrt profile to terminal descent rate caused oscillations.

```python
# BEFORE (broken):
if altitude > cfg.terminal_altitude:
    vy_desired = -cfg.k_descent * math.sqrt(altitude)
else:
    vy_desired = -cfg.terminal_descent_rate  # Jump!

# AFTER (working):
sqrt_velocity = -cfg.k_descent * math.sqrt(altitude)
if altitude > cfg.terminal_altitude:
    vy_desired = sqrt_velocity
else:
    # Smooth blend over terminal_altitude range
    blend = 1.0 - (altitude / cfg.terminal_altitude)
    blend = max(0.0, min(1.0, blend))
    vy_desired = (1.0 - blend) * sqrt_velocity + blend * (-cfg.terminal_descent_rate)
```

**Location:** `src/simulacra/onboard/landing_controller.py:60-70`

### Fix 3: LandingControllerConfig

Added required config parameters:

```python
@dataclass
class LandingControllerConfig:
    dry_mass: float           # Rocket mass without fuel (kg)
    full_fuel_mass: float     # Full fuel mass (kg)
    max_thrust: float         # Maximum thrust (N)
    gravity: float = 9.81
    k_descent: float = 2.0    # Glide path gain
    terminal_altitude: float = 3.0
    terminal_descent_rate: float = 1.0
```

---

## Files Reference

| File | Description | Status |
|------|-------------|--------|
| `scripts/train_landing_rl.py` | **Throttle-only PPO + curriculum** | 🔄 Running |
| `scripts/train_ppo_rajs.py` | PPO + RAJS training | ❌ Plateaued |
| `scripts/train_ppo.py` | Baseline PPO | ❌ Plateaued |
| `scripts/train_ppo_curriculum.py` | Curriculum PPO | ❌ Catastrophic forgetting |
| `scripts/train_bc.py` | Behavioral cloning (LandingController) | ⚠️ Config mismatch |
| `scripts/train_bc_glide.py` | BC (glide path expert) | ❌ 77% expert, 0% deploy |
| `scripts/diagnose_env.py` | Environment diagnostics | ✅ Useful |
| `scripts/test_landing_controller.py` | Test classical controller | ✅ Working |
| `src/simulacra/rajs_wrapper.py` | RAJS environment wrapper | ❌ Beta never anneals |
| `src/simulacra/gym_env.py` | Gymnasium environments (V2 fixed) | ✅ Working |
| `src/simulacra/onboard/landing_controller.py` | Classical controller | ✅ 100% success |

---

## Quick Start

1. **Verify LandingController works:**
   ```bash
   cd backend && uv run python scripts/test_landing_controller.py
   ```

2. **Run throttle-only training (current approach):**
   ```bash
   cd backend && uv run python scripts/train_landing_rl.py --steps 10000000
   ```

3. **Resume interrupted training:**
   ```bash
   cd backend && uv run python scripts/train_landing_rl.py --resume
   ```

4. **Evaluate trained models:**
   ```bash
   cd backend && uv run python scripts/train_landing_rl.py --eval-only
   ```

5. **Monitor with tensorboard:**
   ```bash
   cd backend && tensorboard --logdir=runs/throttle_ppo/tensorboard
   ```
