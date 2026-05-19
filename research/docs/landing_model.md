# Rocket Landing Model

This document describes the trained behavioral cloning (BC) models for autonomous rocket landing.

## Current Best: Glide Path Model

The **glide path model** achieves **84% success rate** on realistic landing conditions (10-30m altitude).

### Key Insight

Instead of a suicide burn (free fall then max braking), the glide path approach uses a velocity profile that decreases smoothly with altitude:

```
target_velocity = -k * sqrt(altitude)
```

This creates a stable descent path where the rocket naturally slows as it approaches the ground.

### Performance

| Altitude Range | Success Rate |
|----------------|--------------|
| 10-30m         | **84%**      |
| 30-50m         | 48%          |
| 50-80m         | ~25%         |

### Files

| Path | Description |
|------|-------------|
| `runs/bc_glide/bc_model.pt` | Trained glide path model |
| `scripts/train_bc_glide.py` | Training script |
| `scripts/eval_bc_glide.py` | Evaluation and recording |
| `scripts/test_suicide_burn.py` | Expert policy testing |

### Usage

```bash
cd backend

# Evaluate the model
uv run python scripts/eval_bc_glide.py --n-episodes 50 --altitude 10-30

# Record episodes for visualization
uv run python scripts/eval_bc_glide.py --n-episodes 5 --altitude 10-30 --record

# View in frontend
# 1. Start frontend: cd frontend && npm run dev
# 2. Open http://localhost:5173/
# 3. Browse to /tmp/simulacra/ and select a .ndjson file
```

---

## Model Architecture

Both models use the same architecture:

```
Input:  13-dim observation [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
Hidden: 128 -> ReLU -> 128 -> ReLU
Output: 3-dim action [throttle, gimbal_outer, gimbal_inner]
```

~18k parameters, saved as a PyTorch state dict.

---

## Expert Policies

### Glide Path Controller (Current Best)

```python
def glide_path_expert(obs, mass, max_thrust):
    alt = obs[1]  # altitude
    vy = obs[4]   # vertical velocity
    wx, wy, wz = obs[10], obs[11], obs[12]

    g = 9.81
    hover_throttle = (mass * g) / max_thrust

    # Glide slope: target velocity decreases with altitude
    k = 1.5
    if alt > 0.5:
        target_vy = max(-15.0, min(-k * sqrt(alt), -0.8))
    else:
        target_vy = -1.0

    # Terminal landing: cut engine close to ground
    if alt < 0.4 and abs(vy) < 2.5:
        throttle = 0.0
    else:
        # P-controller on velocity error
        vy_error = target_vy - vy
        Kp = 0.25 if alt < 2.0 else (0.30 if alt < 10.0 else 0.35)
        throttle = clip(hover_throttle + Kp * vy_error, 0.0, 1.0)

    # Angular rate damping
    gimbal_outer = clip(-0.5 * wz, -0.3, 0.3)
    gimbal_inner = clip(-0.5 * wy, -0.3, 0.3)

    return [throttle, gimbal_outer, gimbal_inner]
```

### Simple Hover Controller (Legacy)

The original expert only works at very low altitudes (3-8m):

```python
def simple_expert(obs):
    alt, vy = obs[1], obs[4]
    target_vy = -0.8 - 0.3 * sqrt(max(0.1, alt))
    if alt < 0.5:
        target_vy = -0.3
    throttle = clip(0.5 + 0.5 * (target_vy - vy), 0.0, 1.0)
    # ... gimbal damping
```

---

## Training History

### Why Behavioral Cloning?

Pure RL approaches failed:
- **PPO** (100M steps): Plateaued at -11 reward, 0 landings
- **SAC** (100k steps): 0% success rate
- **Curriculum RL**: Catastrophic forgetting

The issue was sparse rewards and exploration difficulty.

### Glide Path BC Training

- **Demo Episodes**: 300 (77% expert success rate)
- **Transitions**: ~53,000
- **Conditions**: 5-30m altitude, -5 to 0 m/s velocity
- **Epochs**: 100
- **Final Loss**: 0.000167

---

## Environment Details

Uses `RocketLandingEnvV2` with:
- **Observation**: 13-dim (position, velocity, quaternion, angular velocity)
- **Action**: 3-dim (throttle [0,1], gimbal_outer [-1,1], gimbal_inner [-1,1])
- **Reward**: Additive terminal (+50 to +100 for landing, -100 for crash)
- **Episode length**: Up to 1000 steps (20 seconds)
- **Landing thresholds**: velocity < 3 m/s, tilt < 15 degrees

---

## Future Work

To improve beyond 84% at 10-30m and extend to higher altitudes:

1. **Better attitude control**: The current gimbal controller only damps angular rates. Adding position feedback could reduce drift.
2. **RL fine-tuning**: Use the BC model to initialize SAC/PPO for further optimization.
3. **Domain randomization**: Add wind, sensor noise, mass uncertainty during training.
4. **Curriculum on altitude**: Progressively train on harder conditions.
