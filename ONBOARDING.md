# Onboarding to simulacra

Welcome. This document is how you get from "I cloned the repo" to "I'm writing a controller and watching it fly" in under an hour. It also tells you what's stable, what's about to change, and how to write your code so the upcoming changes don't bite you.

If you only read one other file, read the design spec for the v0.1 controller SDK:
`docs/superpowers/specs/2026-05-18-simulacra-library-cleanup-design.md` — that's where this repo is heading over the next week or two.

---

## 1. What simulacra is

A rocket flight simulator with a clean three-layer architecture, modelled after Kantian epistemology:

- **Noumena** (`backend/src/simulacra/noumena/`) — God's view: true world state, physics integration, no noise or bias.
- **Phenomena** (`backend/src/simulacra/phenomena/`) — the interface: sensors that add noise/bias to the truth, actuators that translate imperfect commands into forces and torques.
- **Onboard** (`backend/src/simulacra/onboard/`) — the flight computer: sees only sensor outputs, issues actuator commands. **This is where your controller lives.**

The onboard layer is the only part you need to write. The simulator's job is to give your controller a realistic sandbox: noisy GPS, rate-limited gimbal, finite thrust, asymmetric mass as fuel burns, wind disturbances. If your controller passes simulation, it has a real chance of passing hardware.

---

## 2. Fifteen-minute setup

You need Python ≥ 3.13 (managed by `uv`) and Node 22.x.

```bash
git clone git@github.com:Rocketry-IN/simulacra.git
cd simulacra
git checkout onboarding              # this branch

# Backend
cd backend
uv sync                              # installs Python deps in a local venv
uv run pytest -q                     # sanity check: should pass

# Run the canonical hopper scenario
uv run ../scenarios/hopper_short.py > /tmp/hopper_pid.ndjson

# Frontend (new shell)
cd ../frontend
npm install
npm run dev                          # serves the 3D viewer on localhost:5173
```

Open `http://localhost:5173/?file=/tmp/hopper_pid.ndjson` in a browser. You should see a 3 m rocket performing three vertical hops to 10 m with a 5 N crosswind. Drag to orbit, scroll to zoom, scrub the timeline at the bottom.

If anything in that sequence fails, tell us before you write any controller code — onboarding bugs are bugs.

---

## 3. Where to read first

Read these files in this order. They're each small. Together they're the entire mental model of how the simulator works.

| File | What you learn |
|------|----------------|
| `backend/src/simulacra/noumena/rocket.py` | How a rocket is described: mass components, inertia, fuel, geometry. The "physical truth". |
| `backend/src/simulacra/noumena/kernel.py` | The physics integrator. RK4 over rigid-body 6-DOF dynamics. |
| `backend/src/simulacra/phenomena/sensors.py` | How noisy sensors are modelled. GPS, barometer, IMU, velocity, orientation. **Your controller only ever sees this side.** |
| `backend/src/simulacra/phenomena/actuators.py` | Gimbal + throttle. Rate limits and actuator lag live here. |
| `backend/src/simulacra/onboard/controller.py` | The current `Controller` abstract base class. (Note: this is changing — see §5.) |
| `backend/src/simulacra/onboard/hopper_controller.py` | A working cascaded PID hopper. Three nested loops (position → velocity → attitude → gimbal). Read this top to bottom. |
| `scenarios/hopper_short.py` | A full scenario: rocket built, environment configured, controller plugged in, NDJSON written to stdout. |

If you want a second worked example with an entirely different algorithm:
- `backend/src/simulacra/onboard/rl_hopper_controller.py` — a stable-baselines / JAX policy wrapped as a `Controller`.
- `backend/src/simulacra/onboard/neat_hopper_controller.py` — a neuroevolution genome wrapped as a `Controller`.

The point of reading three different controllers (PID, RL, NEAT) is to see how differently the *same interface* can be filled in. Your own modern-control-theory implementation (LQR, MPC, geometric, sliding mode — whichever you pick) becomes a fourth point in that space.

---

## 4. How to write a controller right now

There is an interface change coming (see §5). To save yourself a painful rewrite, structure your code so that **the algorithm is separated from the wrapper that adapts it to the simulator's controller interface.** The algorithm is interface-independent — it's just math. The wrapper is what changes.

### Pattern: algorithm class + thin controller wrapper

```python
# my_lqr/algorithm.py — pure math, no simulacra imports
import numpy as np
from scipy.linalg import solve_continuous_are

class LQRAlgorithm:
    """LQR for the hopper, linearised about hover. Interface-free."""

    def __init__(self, Q_diag, R_diag, vehicle_mass, vehicle_inertia, max_thrust):
        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        self.A, self.B = self._linearise(vehicle_mass, vehicle_inertia, max_thrust)
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ P

    def control(self, state_error: np.ndarray) -> np.ndarray:
        """state_error: 13-vector (pos, vel, quat_err, ang_vel) → 3-vector (throttle, gimbal_y, gimbal_z)."""
        return -self.K @ state_error

    def _linearise(self, m, I, T_max):
        # ... your linearisation here
        ...
```

```python
# my_lqr/controller.py — the wrapper. This file will change when v0.1 lands.
import numpy as np
from simulacra.onboard.controller import Controller
from simulacra.phenomena.sensors import Sensors
from simulacra.phenomena.actuators import GimbalActuator, ThrottleActuator, ...
from my_lqr.algorithm import LQRAlgorithm

class LQRHopperController(Controller):
    def __init__(self, throttle, gimbal, target, vehicle, Q_diag, R_diag):
        self.throttle = throttle
        self.gimbal   = gimbal
        self.target   = target
        self.algo     = LQRAlgorithm(Q_diag, R_diag, **vehicle)
        self.sensors  = None

    def with_sensors(self, sensors: Sensors) -> "LQRHopperController":
        self.sensors = sensors
        return self

    def run(self) -> None:
        x_err = self._state_error_from_sensors()
        u     = self.algo.control(x_err)
        self.throttle.apply(...)
        self.gimbal.apply(...)
```

**When v0.1 lands**, only `controller.py` changes — it becomes a `step(frame, setpoint, dt)` implementation. `algorithm.py` is untouched. Your LQR design work is preserved.

### Run your controller

Copy `scenarios/hopper_short.py` to `scenarios/hopper_short_lqr.py`, swap the controller construction:

```python
from my_lqr.controller import LQRHopperController
controller = LQRHopperController(
    throttle=throttle_actuator, gimbal=gimbal_actuator,
    target=Vec3(0, 10, 0), vehicle=...,
    Q_diag=[10,10,10, 1,1,1, 5,5,5, 1,1,1, 1],
    R_diag=[1, 1, 1],
)
```

Then:
```bash
uv run scenarios/hopper_short_lqr.py > /tmp/hopper_lqr.ndjson
```

### Compare against PID

```
localhost:5173/?file=/tmp/hopper_pid.ndjson&file=/tmp/hopper_lqr.ndjson
```

The viewer overlays both runs. You can scrub time, see thrust profiles, attitude excursions, and trajectory in 3D side-by-side.

---

## 5. What's coming in v0.1 (and what it means for your code)

The current `Controller` interface (`with_sensors(s)` + `run()`) is being replaced with a cleaner one designed to (a) port to a C `.h` for the flight computer, (b) support hardware-in-the-loop, and (c) make controller comparison fair. The new shape:

```python
class Controller(Protocol):
    rate_hz: float
    def reset(self, mission: Mission, vehicle: VehicleParams) -> None: ...
    def step(self, frame: SensorFrame, setpoint: Setpoint, dt: float) -> ActuatorCommand: ...
```

- `SensorFrame` is a snapshot of all sensor topics with per-field timestamps (you can detect stale GPS, missing baro, etc.).
- `Setpoint` is what `WaypointGuidance` wants you to track — position, velocity, feedforward acceleration, attitude.
- `ActuatorCommand` is plain data — throttle in [0,1], gimbal pitch/yaw in radians.

The architecture diagram, the full API, the migration plan, and the deliberate technical debt are all in `docs/superpowers/specs/2026-05-18-simulacra-library-cleanup-design.md`. Read it when you have 20 minutes — it'll save you guessing.

**If you've followed the algorithm/wrapper pattern in §4**, porting your controller to v0.1 is a single new file (`controller_v01.py`) that calls into your unchanged `algorithm.py`. The math survives.

**Rough timeline**: v0.1 lands on a `feature/v0.1-sdk` branch over the next week or two and is merged into `main` after we've adapted the four reference controllers and a smoke test passes on `hopper_short`. We'll tag a `v0.1.0` and notify you to rebase.

---

## 6. Working agreement

- **Branch**: work on `onboarding` or a personal branch from it. Don't push to `main`.
- **Your code lives** under `controllers/<your_name>/` at the repo root. Each engineer gets their own folder. When v0.1 lands, this convention may shift to per-engineer external repos, but for now in-tree is simpler.
- **Sharing a run**: commit only the runner script (not the NDJSON — those are large). Mention the commit SHA in chat when asking for feedback. We'll all be looking at the same scenario, same seed, same controller version.
- **Questions in code**: if you read a file and something doesn't make sense, leave a comment with `# QUESTION(<your_name>):` — we'll grep for these and discuss in the weekly.
- **The frontend** is read-only; you don't need to touch TypeScript to do your work. If you find a viewer limitation that's blocking you, raise it — don't work around it.

---

## 7. What success looks like in week 1

- You've cloned the repo and run `hopper_short` with the default PID controller.
- You've opened the result in the viewer and understood the three hops.
- You've read the four files in §3 (the kernel, sensors, actuators, hopper PID).
- You've sketched the algorithm class for whatever modern control approach you're choosing (LQR / MPC / sliding mode / geometric / your-favourite), and committed it under `controllers/<your_name>/algorithm.py`.

Don't worry about getting the controller working in the simulator in week 1 — that's week 2. Week 1 is for understanding the simulator's contract with you (what's noisy, what's rate-limited, what's truth) and writing the math that's interface-independent.

---

Questions, surprises, "this doesn't make sense" — say so early. The cost of asking is low; the cost of pretending it's clear is real.
