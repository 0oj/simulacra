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

## Dependencies

- **Backend**: Python >=3.13, numpy, scipy, matplotlib, pydantic
- **Frontend**: Node.js 22.19.0, Three.js, TypeScript, Vite
