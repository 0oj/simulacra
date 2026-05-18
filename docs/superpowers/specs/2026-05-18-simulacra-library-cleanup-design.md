# Simulacra library cleanup & controller SDK — design

**Status:** Draft for review
**Author:** Ooj
**Date:** 2026-05-18
**Scope:** v0.1 of `simulacra` as a shareable library that lets a small team write, run, compare, and eventually port flight-control algorithms.

---

## 1. Motivation

Simulacra today is a research monorepo: simulator, RL training infrastructure, FYP report, NEAT experiments, and a 3D viewer entangled in one tree. We want to share it with two engineers and onboard them onto the hopper control campaign. To do that without forcing them to read the entire research history, the codebase needs three things at once:

1. A **clean, minimal `Controller` interface** they can implement on day one.
2. A **runtime architecture** that survives the migration to hardware-in-the-loop and to a C `.h` controller without rewrites.
3. A **library-shaped repository** in which the engineering-relevant code is obvious and the FYP-era research code is preserved but out of the way.

## 2. Goals & non-goals

### Goals
- A `Controller` protocol with explicit data in / data out, suitable for both Python and C.
- A sensor bus that decouples sensor production rate from controller consumption rate, so the same controller code runs in pure sim, HIL, and (eventually) flight.
- A `Guidance` layer that converts a mission into setpoints, so different controllers can be compared fairly.
- A repo layout in which engineers writing controllers see a small, well-bounded API surface; research code is archived in-tree under `research/`.
- The existing reference controllers (`pid_hover`, `hopper_controller`, `rl_hopper_controller`, `neat_hopper_controller`) adapted to the new interface as importable reference implementations.
- The existing NDJSON contract and 3D viewer preserved unchanged.

### Non-goals (v0.1)
- The C `.h` interface itself. The Python design preserves portability; the header generation/audit is a v0.2 task.
- HIL transport (UDP/serial packet format).
- A full four-layer GNC stack (Mission Management / Guidance / Navigation / Control). v0.1 collapses to two layers; the bus pattern keeps the door open.
- Per-sensor callback subscribers (push-driven control loops). The bus is designed to accommodate them later.
- Publishing to a private package index. v0.1 stays as a monorepo with library-shaped layout.
- License selection. Default for now: internal use, source-only.

## 3. Core architecture

The flight-software stack collapses to two layers (Guidance, Control) connected by a sensor bus. Producers push sensor samples into the bus at their own rates; the controller pulls a snapshot of the bus at a fixed rate.

```
sensor producers ─push─►  ┌─────────────────┐
(sim kernel, HIL          │  sensor bus     │
 socket, or hw ISRs)      │  (latest sample │ ◄── controller ticks at rate_hz,
                          │   per topic +   │     snapshots the bus, computes
                          │   timestamps)   │     command, writes actuator.
                          └─────────────────┘

┌─────────────┐  Mission   ┌──────────────┐  Setpoint  ┌──────────────┐
│  Scenario   │ ─────────► │   Guidance   │ ─────────► │  Controller  │
│ (waypoints) │            │ (waypoint    │            │  (engineer's │
└─────────────┘            │  sequencer)  │            │   algorithm) │
                           └──────────────┘            └──────────────┘
                                  ▲                            │
                                  │     SensorFrame snapshot   │
                                  └────── (from bus) ──────────┘
                                                                │
                                                                ▼
                                                       ActuatorCommand
```

Why this shape:
- **Sensor bus** is the only design that survives all three deployment modes (sim, HIL, flight) without code changes on the controller side. In each mode, only the *producer* differs.
- **Pull-at-fixed-rate** is forced on us by the actuator side: gimbal servos and throttle valves want deterministic command cadence, not bursts.
- **Guidance** separation lets engineers writing modern-control-theory tracking algorithms (LQR, MPC, geometric control, sliding mode) be compared *apples-to-apples* — same mission, same setpoints, only the tracking algorithm differs.
- **Closed-loop guidance** (guidance sees the bus too) lets it detect arrival, stall, GPS-stale, and aborts — the kind of supervisor logic that lives one layer above a pure controller in real flight code.

## 4. Public API (in `simulacra.onboard`)

### 4.1 `controller.py`

Existing types reused: `Vec3` and `Quat` from `simulacra.physics`. New types in `controller.py`:

```python
@dataclass(frozen=True, slots=True)
class Stamped[T]:
    value: T
    t: float      # seconds since boot
    seq: int      # incrementing counter — controller can detect "did this update?"

@dataclass(frozen=True, slots=True)
class SensorFrame:
    t_now: float                          # snapshot time
    imu_accel:    Stamped[Vec3]           # body-frame, ~200 Hz
    imu_gyro:     Stamped[Vec3]           # body-frame, ~200 Hz
    baro:         Stamped[float]          # altitude, ~50 Hz
    gps:          Stamped[Vec3] | None    # may be None when no GPS lock
    velocity:     Stamped[Vec3]           # fused/sensed
    orientation:  Stamped[Quat]           # fused/sensed

@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    throttle:      float    # 0..1
    gimbal_pitch:  float    # rad
    gimbal_yaw:    float    # rad

class Controller(Protocol):
    rate_hz: float
    def reset(self, mission: Mission, vehicle: VehicleParams) -> None: ...
    def step(self, frame: SensorFrame, setpoint: Setpoint, dt: float) -> ActuatorCommand: ...

@dataclass(frozen=True, slots=True)
class VehicleParams:
    """Vehicle properties the controller is allowed to know at design time.
    Distinct from the runtime mass (which changes as fuel burns) — those values
    must be sensed or estimated, not read from here."""
    dry_mass:    float          # kg, structure + payload, no fuel
    full_fuel_mass: float       # kg, fuel at t=0
    inertia_dry: Vec3           # kg·m², principal axes, dry
    length:      float          # m, nose-to-base
    max_thrust:  float          # N, at full throttle, sea level
    max_gimbal_angle: float     # rad
    max_gimbal_rate:  float     # rad/s
```

Rationale:
- `Stamped[T]` exposes timestamps so controllers can detect stale data and degrade gracefully (a real flight requirement; not optional).
- `frozen=True, slots=True` dataclasses are stack-friendly in spirit and translate directly to C structs.
- `reset` takes both `mission` and `vehicle` so controllers that linearise around hover (LQR) can compute their gains using actual vehicle mass / inertia, not hard-coded constants.
- `rate_hz` is the controller's declared tick rate — the runtime obeys this regardless of physics timestep.

### 4.2 `bus.py`

```python
class SensorBus:
    def push_imu(self, t: float, accel: Vec3, gyro: Vec3) -> None: ...
    def push_baro(self, t: float, altitude: float) -> None: ...
    def push_gps(self, t: float, position: Vec3) -> None: ...
    def push_velocity(self, t: float, velocity: Vec3) -> None: ...
    def push_orientation(self, t: float, quat: Quat) -> None: ...

    def snapshot(self, t_now: float) -> SensorFrame: ...
```

- Single writer (the producer side) and single reader (the controller) per topic in v0.1; threading not yet relevant in pure-Python sim.
- Each topic stores the latest sample plus its timestamp and seq counter.
- `snapshot()` is a cheap value copy; no locks needed because producers and consumers in sim are sequential on one thread.
- The C analogue: `void sensor_bus_push_imu(float t, vec3_t accel, vec3_t gyro)` writes to a volatile struct; `sensor_bus_snapshot(sensor_frame_t* out)` copies it. Identical shape, identical mental model.

### 4.3 `guidance.py`

```python
@dataclass(frozen=True, slots=True)
class Setpoint:
    position: Vec3      # target position (world frame)
    velocity: Vec3      # target velocity (world frame)
    accel_ff: Vec3      # feed-forward acceleration the controller can apply
    attitude: Quat      # nominal nose direction (controllers may ignore)

@dataclass
class Waypoint:
    position: Vec3
    hold_time: float
    acceptance_radius: float = 1.0
    max_speed: float = 5.0

class Mission(Protocol):
    """A scenario-level objective. Polymorphic for future mission types."""

class WaypointMission(Mission):
    waypoints: list[Waypoint]

class Guidance(Protocol):
    def reset(self, mission: Mission, vehicle: VehicleParams) -> None: ...
    def update(self, frame: SensorFrame, t_now: float) -> Setpoint: ...

class WaypointGuidance(Guidance):
    """Sequences a WaypointMission into setpoints, closed-loop on sensed state."""
    def reset(self, mission: WaypointMission, vehicle: VehicleParams) -> None: ...
    def update(self, frame: SensorFrame, t_now: float) -> Setpoint: ...
```

- `WaypointGuidance` does double duty as Mission Management *and* Guidance in v0.1. This is documented technical debt; see §8.
- Guidance is closed-loop on `SensorFrame`, not open-loop on time. This is what allows abort logic, arrival detection from sensed position, and stall handling.

### 4.4 `runtime/`

```python
class Scenario:
    def __init__(self, *, rocket, env, mission, controller,
                 guidance=None,           # defaults to WaypointGuidance
                 sensors=None,            # defaults to sensors_for(rocket)
                 actuators=None,          # defaults to actuators_for(rocket)
                 duration: float, seed: int = 0):
        ...
    def run(self, out) -> None:           # writes NDJSON to out
        ...

def sensors_for(rocket) -> SensorSuite: ...     # default GPS+IMU+baro+vel+orientation
def actuators_for(rocket) -> ActuatorSuite: ... # default throttle + gimbal
```

`sensors_for` / `actuators_for` are convenience constructors over the existing `phenomena.sensors` / `phenomena.actuators` modules, wired to the `SensorBus`. They produce the same noisy sensors and rate-limited actuators used today; scenarios that need non-default sensor noise pass their own.

The runner ticks the kernel at the physics dt, ticks sensors at their configured rates (pushing to the bus), ticks guidance + controller at `controller.rate_hz`, and writes NDJSON messages matching the existing `schema.py` contract.

There is **no CLI**. Scenarios are Python scripts (see §6).

## 5. Repository layout (target)

```
simulacra/                                project root
├── pyproject.toml                        name="simulacra", v0.1.0-dev
├── README.md                             library-facing onboarding
├── src/simulacra/
│   ├── __init__.py
│   ├── noumena/                          God's view (unchanged module contents)
│   ├── phenomena/                        the interfaces (unchanged module contents)
│   ├── onboard/
│   │   ├── __init__.py
│   │   ├── controller.py                 Controller, SensorFrame, ActuatorCommand,
│   │   │                                 Stamped[T], VehicleParams
│   │   ├── bus.py                        SensorBus
│   │   ├── guidance.py                   Guidance, Setpoint, Mission,
│   │   │                                 WaypointGuidance, WaypointMission, Waypoint
│   │   └── reference/                    full reference controllers, importable
│   │       ├── pid_hopper.py             cascaded PID (adapted from hopper_controller)
│   │       ├── rl_hopper.py              SB3 / JAX policy wrapper
│   │       └── neat_hopper.py            NEAT genome wrapper
│   ├── scenarios/
│   │   ├── hopper_short.py
│   │   ├── hopper_long.py
│   │   └── ...
│   └── runtime/
│       ├── scenario.py                   Scenario class
│       └── ndjson.py                     log writer (preserves schema.py messages)
├── examples/                             minimal skeletons engineers copy
│   ├── lqr_skeleton.py
│   ├── mpc_skeleton.py
│   └── run_lqr_on_hopper_short.py        sample runner script
├── research/                             FYP-era code, preserved not maintained
│   ├── landing/                          landing_controller, GFOLD, landing scenarios
│   ├── sounding_rocket/                  sounding_rocket scenario, pitch_over, etc.
│   ├── training/                         train_*.py, NEAT infra, eval scripts
│   ├── jax/                              jax_kernel, jax_env, jax_ppo, jax_hopper_env
│   └── docs/                             FYP report, figures, session reports
├── frontend/                             3D viewer (unchanged, NDJSON contract preserved)
├── scripts/                              dev utilities, not part of the library
│   ├── compare_runs.py                   prints metrics table across NDJSON files
│   └── plot_*.py                         (existing plot scripts)
└── tests/                                library tests
```

Key consequences:
- The top-level `backend/` directory is removed. Its Python package moves to top-level `src/simulacra/`. This is the conventional library layout and what `uv build` expects without configuration.
- Top-level `scenarios/` moves *into* the package as `simulacra.scenarios`, becoming importable.
- The frontend stays exactly as it is. The NDJSON schema is unchanged.
- `research/` is one-way: the library does not import from it, and it does not get packaged when (later) we publish.

## 6. Scenarios as Python scripts

Scenarios are self-contained Python scripts that:
1. Construct the rocket, environment, mission.
2. Pick a controller.
3. Run via `Scenario(...).run(stdout)`.
4. Are invokable directly via `uv run -m simulacra.scenarios.<name>`.

Each scenario module exposes a parametrised `run()` function so engineers can use the same scenario with their own controller without copying the file:

```python
# src/simulacra/scenarios/hopper_short.py
"""Short hopper: three vertical hops to 10 m with crosswind."""
import sys
from simulacra.noumena import Rocket, Environment, ...
from simulacra.onboard.controller import Controller
from simulacra.onboard.guidance import WaypointGuidance, WaypointMission, Waypoint
from simulacra.runtime import Scenario


def build_mission() -> WaypointMission: ...
def build_world() -> tuple[Rocket, Environment]: ...

def run(controller: Controller, out=sys.stdout, seed: int = 42) -> None:
    rocket, env = build_world()
    Scenario(
        rocket=rocket, env=env,
        mission=build_mission(), controller=controller,
        seed=seed, duration=60.0,
    ).run(out)


if __name__ == "__main__":
    from simulacra.onboard.reference.pid_hopper import PIDHopperController
    run(PIDHopperController())
```

To use a different controller, engineers write a 5-line runner:

```python
# any_path/run_lqr_hopper_short.py
from simulacra.scenarios import hopper_short
from my_lqr import LQRHopperController
hopper_short.run(LQRHopperController(Q_diag=..., R_diag=...))
```

```bash
uv run any_path/run_lqr_hopper_short.py > runs/lqr.ndjson
```

Rationale:
- Matches the existing project pattern (`scenarios/hopper_short.py` already runs this way today).
- Self-documenting: anyone reading a runner script sees the full experiment (rocket, mission, controller, seed) in one file.
- No CLI machinery, no `--controller foo:Bar` string parsing, no controller registry.

## 7. Day-one engineer journey

### Journey 1 — run existing controllers, compare in frontend

```bash
git clone <repo> simulacra && cd simulacra
uv sync

# Each scenario's default controller (PID), via its __main__ block:
uv run -m simulacra.scenarios.hopper_short > runs/pid.ndjson

# Same scenario with a different reference controller:
cat > runs/run_rl.py <<'EOF'
from simulacra.scenarios import hopper_short
from simulacra.onboard.reference.rl_hopper import RLHopperController
hopper_short.run(RLHopperController.from_checkpoint("models/v10c_step1210M.zip"))
EOF
uv run runs/run_rl.py > runs/rl.ndjson

# View one
cd frontend && npm install && npm run dev
# browser: localhost:5173/?file=../runs/pid.ndjson

# Compare
# browser: localhost:5173/?file=../runs/pid.ndjson&file=../runs/rl.ndjson

# Quantitative comparison
uv run python scripts/compare_runs.py runs/pid.ndjson runs/rl.ndjson
```

### Journey 2 — engineer writes a new controller (LQR example)

The engineer creates **one file** and imports **two modules**:

```python
# my_lqr.py
import numpy as np
from scipy.linalg import solve_continuous_are
from simulacra.onboard.controller import Controller, SensorFrame, ActuatorCommand
from simulacra.onboard.guidance import Setpoint, Mission, VehicleParams


class LQRHopperController(Controller):
    rate_hz: float = 100.0

    def __init__(self, Q_diag: np.ndarray, R_diag: np.ndarray):
        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        self.K: np.ndarray | None = None

    def reset(self, mission: Mission, vehicle: VehicleParams) -> None:
        A, B = self._hover_linearisation(vehicle)
        P = solve_continuous_are(A, B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ B.T @ P

    def step(self, frame: SensorFrame, setpoint: Setpoint, dt: float) -> ActuatorCommand:
        x_err = self._state_error(frame, setpoint)
        if frame.t_now - frame.gps.t > 0.5:
            x_err[0:3] = 0.0    # GPS stale — drop horizontal position errors
        u = -self.K @ x_err
        return ActuatorCommand(
            throttle     = float(np.clip(u[0], 0.0, 1.0)),
            gimbal_pitch = float(np.clip(u[1], -0.3, 0.3)),
            gimbal_yaw   = float(np.clip(u[2], -0.3, 0.3)),
        )
    # ...internal helpers
```

```bash
uv run runs/run_lqr.py > runs/lqr.ndjson
```

To run their controller across a suite:

```python
# runs/lqr_suite.py
from simulacra.scenarios import hopper_short, hopper_long, hopper_orbit
from my_lqr import LQRHopperController

ctrl = LQRHopperController(...)
for scenario, name in [(hopper_short,"short"),(hopper_long,"long"),(hopper_orbit,"orbit")]:
    with open(f"runs/lqr_{name}.ndjson","w") as f:
        scenario.run(ctrl, out=f)
```

## 8. Known evolution (deliberate technical debt)

The following are *not* problems to be fixed in v0.1, but expected directions in which the design will evolve. They are listed here so engineers reading the code recognise them as known rather than confusing.

1. **`WaypointGuidance` does double duty as Mission Management + Guidance.** When a second mission type (sounding-rocket profile, controlled descent, abort) is added, a `MissionManager` layer will emerge above `Guidance`. Both will fit the existing protocols; the controller interface does not change.
2. **Sensor bus is single-threaded.** Pure sim is sequential, so locks aren't needed. When HIL arrives, the producer thread will need atomic publishes; the consumer API does not change.
3. **No explicit Navigation layer.** Controllers may currently do their own filtering inside `step()`. When a shared EKF is needed, it becomes a bus producer that publishes `EstimatedState`; controllers consume that instead of raw sensor topics.
4. **No push-driven subscribers on the bus.** Controllers that want to process every IMU sample (a high-rate EKF inside the controller, for example) currently can't. Subscriber callbacks can be added to the bus without changing the producer or controller-step sides.
5. **The C `.h` is not generated.** v0.2 task. The Python ABI is preserved as a parallel shape: `Stamped`, `SensorFrame`, `ActuatorCommand`, `Mission`, `Setpoint`, `VehicleParams` all map to C structs; `reset` and `step` map to C functions.

## 9. Migration of existing code

Each existing controller is adapted to the new interface and moved to `simulacra.onboard.reference`. The adaptation is the proof that the design can accommodate a real cascaded PID, an RL policy, and a NEAT genome.

| Existing file | New location | Notes |
|---|---|---|
| `onboard/pid_hover.py` | `onboard/reference/pid_hover.py` (kept) | Trivial adaptation; simplest reference. |
| `onboard/hopper_controller.py` | `onboard/reference/pid_hopper.py` | **Canary**: cascaded PID, the canonical "modern PID hopper". Inner loops kept; outer waypoint sequencing moves to `WaypointGuidance`. |
| `onboard/rl_hopper_controller.py` | `onboard/reference/rl_hopper.py` | Wraps SB3/JAX policy; reads `SensorFrame` to build observation. |
| `onboard/neat_hopper_controller.py` | `onboard/reference/neat_hopper.py` | Wraps NEAT genome. |
| `onboard/safety_envelope.py` | `onboard/reference/safety_envelope.py` | Stays as reference; future home: dedicated supervisor layer. |
| `onboard/landing_controller.py`, `gfold_controller.py`, `hover_controller.py`, `wasd_controller.py` | `research/landing/` | Not adapted; preserved as research artefacts. |
| `onboard/controller.py` (current Controller ABC + IdentityController) | Replaced. Old IdentityController preserved temporarily to ease the migration. | |

Scenarios:
- `scenarios/hopper_short.py`, `hopper_long.py` → `src/simulacra/scenarios/hopper_short.py`, `hopper_long.py`, adapted to use new `Controller` / `Mission` interface.
- `scenarios/sounding_rocket.py`, `pitch_over.py`, `nose_heavy.py`, `engine_heavy.py`, `tipover_test.py` → `research/sounding_rocket/`.

Training & research scripts (`backend/scripts/train_*.py`, `backend/scripts/analyze_*.py`, `backend/scripts/launch_*.sh`, NEAT infra, GFOLD code, FYP figures) → `research/training/` and `research/docs/`. Preserved unchanged.

## 10. Out of scope

- **C `.h` generation.** v0.2.
- **HIL transport.** v0.2 or later, depends on hardware availability.
- **Publishing to a private package index.** Layout is library-shaped now; flipping to `uv add simulacra` for engineers is a small task when desired.
- **License selection.** Internal-only until a sharing decision is made.
- **Mission Management as a distinct layer.** Folded into `WaypointGuidance` for v0.1.
- **Explicit Navigation layer.** Controllers self-filter for v0.1.
- **Per-sensor subscriber callbacks on the bus.** Snapshot-only for v0.1.
- **Domain randomisation / training scripts.** Preserved under `research/training/`, not part of the library API.
- **A docs site.** A clear `README.md` is enough for two engineers; mkdocs/sphinx is for when the audience grows.

## 11. Acceptance criteria for v0.1

A v0.1 release is ready when all of these are true:

1. A new engineer can clone the repo, run `uv sync`, and execute `uv run -m simulacra.scenarios.hopper_short > runs/pid.ndjson` successfully on a clean machine.
2. The NDJSON output of `hopper_short` opens in the frontend and renders the same trajectory the current code produces (within numerical tolerance).
3. All four reference controllers (`pid_hover`, `pid_hopper`, `rl_hopper`, `neat_hopper`) are adapted to the new interface and runnable via `simulacra.scenarios.hopper_short.run(<controller>)`.
4. The `examples/lqr_skeleton.py` controller can be made to run on `hopper_short` by filling in two stub methods (`_hover_linearisation`, `_state_error`).
5. Two NDJSON runs can be compared in the frontend via `?file=A&file=B` (existing capability, must not regress).
6. `scripts/compare_runs.py` prints a metrics table across N NDJSON inputs.
7. The repository is reorganised under the layout in §5; `research/` contains the FYP-era code and is not imported by the library.
8. `README.md` walks a new engineer through Journey 1 and Journey 2 in §7.
9. The library's own tests (`tests/`) cover: bus push+snapshot+staleness, WaypointGuidance arrival detection, a smoke test that each reference controller produces well-formed NDJSON on hopper_short, and the controller protocol contract (a no-op `IdentityController` runs cleanly).
