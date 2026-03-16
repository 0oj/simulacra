# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Simulacra is a rocket flight simulator with a three-layer architecture inspired by Kantian epistemology:

- **Noumena** (`backend/src/simulacra/noumena/`) — God's view: true world state, physics integration, no noise or bias. Contains `Environment`, `Rocket` (`RocketProps` + `RocketState`), `Kernel` (physics engine), and `Scenario` (orchestration with event scheduling via heapq).
- **Phenomena** (`backend/src/simulacra/phenomena/`) — Interfaces: sensors (`GPS`, `IMU`, `ThrustSensor`, `VelocitySensor`) that add noise/bias to noumenal truth, and actuators (`GimbalActuator`) that translate imperfect commands into forces/torques.
- **Onboard** (`backend/src/simulacra/onboard/`) — Flight computer: sees only sensor outputs, never reads noumena directly. Issues commands through actuators.

The onboard controller must never access noumenal state. Phenomena is the only bridge between truth and the controller.

## Commands

### Backend (Python, managed with `uv`)

All commands run from `backend/`:

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_quat.py       # run a single test file
uv run pytest tests/test_quat.py -k "test_name"  # run a single test
uv run -m examples.rocket_wasd_3d      # run WASD example (outputs NDJSON to stdout)
uv run -m examples.plot_wasd_3d output.ndjson  # plot from NDJSON log
```

### Frontend (TypeScript/Three.js, managed with npm)

From `frontend/`:

```bash
npm run dev      # start Vite dev server
npm run build    # typecheck + build
```

## Key Conventions

### Coordinate System (universal across all code)

- **Right-handed**: X (red), Y (green), Z (blue), where x × y = z
- **World frame**: fixed reference; gravity along -Y
- **Body frame**: attached to rocket; thrust/torques expressed here
- **Quaternions**: stored as (w, x, y, z) = w·1 + x·i + y·j + z·k; maps body → world
- **Angular velocity**: body-frame vector [ωx, ωy, ωz] in rad/s

### Type-Safe Frame Separation

Vec3 has frame-tagged subclasses (`RocketVec3`/`WorldVec3`) that prevent accidental mixing of reference frames at the type level. Decorators `@check_frame_compatibility` and `@preserve_type` enforce this.

### Data Flow

Controller reads sensors → issues actuator commands → actuators modify rocket (with imperfection) → kernel steps physics → new true state feeds sensors → repeat. Both noumenal truth and sensor readings are recorded as NDJSON messages (`schema.py`: MetadataMessage, ConstantMessage, VariableMessage).

## Development Philosophy

- **Learning over speed**: prefer explicit, step-by-step implementations over hiding ideas behind heavy libraries.
- **Visual-first**: back ideas with plots and visualizations (trajectories, attitude, thrust over time).
- **Small, safe steps**: introduce one concept at a time; avoid giant refactors.
- **Be explicit about approximations**: document toy models vs. realistic ones.
- **Concept → code → test → visualization**: write the math down, implement minimally, test it, plot it.

## UI Design Principles

Design interfaces as precise, didactic technical diagrams rather than decorative UI.

### Core axioms

- **Purpose**: help a human understand a system. Every component justifies itself in clarity gained per pixel spent.
- **Cognitive stance**: precision instrument with a teacher's restraint. No hype, no dramatization. Expose ambiguity rather than hiding it.
- **Audience**: curious, technically literate reader — a strong undergrad or an engineer outside their domain. Labels are explicit but sparse.

### Geometry and layout

- All elements align to explicit or implicit grids (Cartesian by default). "Almost aligned" is a bug.
- Use precise vector primitives: lines, circles, rectangles, arcs, splines.
- Favor exploded views, layers, construction lines, anchors, and ticks.
- Nothing is arbitrarily placed.

### Color

- Limited semantic palette (2-4 colors total). Color encodes role or state, never decoration.
- **One saturated color per screen by default.** Violating this should feel wrong.
- Dark mode is canonical for this project. No glow, no neon. Contrast comes from value, not saturation.

### Typography

- Text is **subordinate to geometry**. Labels attach to objects; explanations live at margins.
- Neutral, technical, and unobtrusive. Mathematical typesetting / engineering documentation feel.
- Numbers are exact when known, approximate when estimated, symbolic when explanatory. Units are mandatory when ambiguity exists.

### Motion and interaction

- Motion exists only to explain change or causality. No decorative animation.
- Fast, consistent, and restrained. Same action = same duration.
- Three modes: View (observe), Inspect (probe/hover/reveal), Manipulate (change parameters). Modes must be obvious.

### Avoid

- Cyberpunk or neon aesthetics, glassmorphism, skeuomorphism
- Decorative gradients, textures, or expressive fonts
- Trend-driven UI patterns
- Over-smoothing or hiding uncertainty

### Target aesthetic

Timeless, slightly archival. Should feel like a future technical manual or a very good one from 1975.

## RL Training Pipeline

### Current Architecture: GuidanceWrapper

The RL agent outputs 3-dim actions `[throttle, tilt_x, tilt_z]`. An inner PD controller (`LandingController.attitude_law`) converts tilt commands into gimbal angles. This decouples "what direction to point" from "how to achieve that attitude."

### Observation Space

Base: 13-dim `[x, baro_y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]`
Physics features are appended via `FeatureWrapper` or `PhysicsFeatureWrapper`.

### Feature Catalogue

| ID | Name | Formula | When Useful |
|----|------|---------|-------------|
| F1 | `sqrt_y` | √y | Descent velocity scale |
| F2 | `decel_required` | vy²/(2y) | Near ground only |
| F3 | `horiz_dist` | √(x²+z²) | All altitudes |
| F4 | `glide_slope` | horiz_dist/y | Mid-altitude |
| F5 | `brake_margin` | y - vy²/(2·a_max) | **All altitudes** — key for brake timing |
| F6 | `vel_error` | vy + k·√y | **All altitudes** — control error signal |
| F7 | `nose_y` | 2(qw·qy + qx·qz) | Uprightness |
| F8 | `horiz_speed` | √(vx²+vz²) | Redundant |
| F9 | `specific_energy` | vy²/2 + g·y | Energy budget |

Constants: `a_max=4.0`, `k_descent=2.0`, `g=9.81`

### Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_guidance_v3_features.py` | Original 4-feature training (baseline) |
| `scripts/train_guidance_v3_bignet.py` | No-features, 256×256 net (capacity baseline) |
| `scripts/train_feature_ablation.py {A\|B\|C\|D}` | Feature ablation study |
| `scripts/train_dr.py` | DR training: features + bigger net + physics randomization |

### Curriculum (3 stages, identical across experiments)

| Stage | Alt | Vel | Horiz | Tilt | Steps |
|-------|-----|-----|-------|------|-------|
| 1 | 5-80m | -9 to 0 | ±7m | ±0.17 | 2M |
| 2 | 5-130m | -12 to 0 | ±12m | ±0.25 | 2.5M |
| 3 | 5-200m | -15 to 0 | ±20m | ±0.30 | 3.5M |

### Experiment Results

### Domain Randomization

`DomainRandomization` dataclass in `gym_env.py` — pass to `RocketLandingEnvV2(domain_randomization=...)`.
Each episode samples new physics, actuator, and sensor parameters:

- **Physics**: mass, thrust, burn_rate, length (multiplicative scale)
- **Actuators**: gimbal rate limit, gimbal lag tau (multiplicative scale)
- **Environment**: wind (direction + magnitude), gravity, air density
- **Sensors**: GPS noise, baro noise, velocity noise, orientation noise, gyro noise + bias

DR params are logged in `info["dr_params"]` from `reset()` and in `ConstantMessage` for NDJSON replay.

### Experiment Results

Results saved to `backend/runs/`. Each run contains:
- `config.json` — hyperparameters and feature list
- `results.jsonl` — structured eval results per stage
- `training_log_stage{N}.jsonl` — per-minute training dynamics
- `tensorboard/` — training curves
- `checkpoints/` — periodic model snapshots

Comprehensive study design: `backend/docs/feature_ablation_study.md`

## Dependencies

- **Backend**: Python >=3.13, numpy, scipy, matplotlib, pydantic, stable-baselines3, gymnasium
- **Frontend**: Node.js 22.19.0, Three.js, TypeScript, Vite
