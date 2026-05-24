# Simulacra v0.1 — Plan 2: API Primitives + Runtime

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the new `Controller` SDK primitives and runtime — `Stamped[T]`, `SensorFrame`, `ActuatorCommand`, `VehicleParams`, `Controller` Protocol, `SensorBus`, `Setpoint`, `Mission`/`Waypoint`/`WaypointMission`, `Guidance` Protocol, `WaypointGuidance`, NDJSON writer, `Scenario` runner — and prove they compose end-to-end with an `IdentityController` smoke test. Old controllers and scenarios are **untouched** by this plan; they continue to work in parallel. Reference-controller migration is Plan 3.

**Architecture:** Each new module lives at the location specified by the design spec (`src/simulacra/onboard/{controller,bus,guidance}.py`, `src/simulacra/runtime/{scenario,ndjson}.py`). All new types are immutable `@dataclass(frozen=True, slots=True)` so they map cleanly to C structs. The SDK reuses existing `Vec3` and `Orientation` from `simulacra.physics` — no parallel type system. The new `Scenario` wraps `noumena.Kernel` but owns the controller-tick loop and bus-push cadence.

**Tech Stack:** Python 3.13, `dataclasses`, `typing.Protocol`, `pytest`, existing `numpy`/`scipy` stack. No new dependencies.

---

## Verified API conventions (use these constructors)

Discovered by reading `src/simulacra/noumena/`, `src/simulacra/phenomena/`, and existing `scenarios/hopper_short.py`:

```python
# Rocket has three pieces — NOT a single max_thrust/length kwarg.
from simulacra.noumena import (
    Rocket, RocketProps, RocketState, get_upright_orientation,
    MassModel, MassComponent, FuelComponent, Environment, Kernel,
)
from simulacra.physics.vec3 import RocketVec3, WorldVec3

props = RocketProps(
    length=4.0, mass=140.0,                       # mass = dry + fuel at t=0
    center_of_mass_offset=RocketVec3(2.0, 0, 0),
    inertia_diag=(10.0, 360.0, 360.0),
    thrust_application_offset=RocketVec3(0, 0, 0),
)
state = RocketState(
    position=WorldVec3(0, 10, 0), velocity=WorldVec3(0, 0, 0),
    orientation=get_upright_orientation(),
    angular_velocity=RocketVec3(0, 0, 0), time=0.0,
    thrust_body=RocketVec3(0, 0, 0),
)
mass_model = MassModel(
    components=[MassComponent(name="dry", mass=60.0, position=RocketVec3(2, 0, 0))],
    fuel=FuelComponent(name="fuel", mass=80.0, position=RocketVec3(1.5, 0, 0), burn_rate=1.0),
)
rocket = Rocket(props=props, state=state, mass_model=mass_model)

# Environment uses WorldVec3 and the kwarg is 'wind' (not 'wind_vec')
env = Environment(gravity=WorldVec3(0, -9.81, 0), wind=WorldVec3(0, 0, 0))

# Kernel takes the rocket AND `environment` (not `env`)
kernel = Kernel(rocket=rocket, environment=env)

# Sensors take the KERNEL, not the rocket
from simulacra.phenomena.sensors import Sensors
sensors = Sensors(kernel)
# sensors.GPS, sensors.Barometer, sensors.IMU, sensors.VelocitySensor,
# sensors.OrientationSensor, sensors.Accelerometer, sensors.GyroSensor exist.

# Actuators are individually constructed and bundled
from simulacra.phenomena.actuators import (
    Actuators, ThrottleActuator, GimbalActuator,
    ThrottleCommand, GimbalCommand,
)
throttle = ThrottleActuator(rocket)
gimbal = GimbalActuator(rocket, max_outer_angle=0.349, max_inner_angle=0.349)
actuators = Actuators(throttle=throttle, gimbal=gimbal)
```

**Where this affects tasks:** Task 8 (`sensors_for` / `actuators_for`), Task 9 (`Scenario.__init__`), and test fixtures in Tasks 8/9/10 must use these signatures. If any later step's code block contradicts the above, prefer the verified signatures here.

Vehicle params for the SDK Protocol have no perfect mapping into `RocketProps` — `max_thrust` isn't a Rocket attribute (it's enforced by the controller or scenario, not by physics). For v0.1, `Scenario._vehicle_params()` derives what it can from `rocket.props` and uses sensible defaults for the rest (`max_thrust = 2000.0`, `max_gimbal_angle = 0.349`, `max_gimbal_rate = 2.0`). Per-scenario overrides become possible in v0.2.

## Pre-flight

This plan runs on `feature/v0.1-sdk` after Plan 1's `v0.1-restructure` tag. The branch contains the new layout (`src/simulacra/`) but the old `Controller(ABC)` and old reference controllers still exist — they're not touched by this plan. The new SDK is added **alongside** the old one.

Verify starting state:
```bash
git status --short | grep -E '^[MARD]' | head     # only user FYP edits expected
git log --oneline | head -3                        # tip is at or after v0.1-restructure
uv run pytest -q --tb=no 2>&1 | tail -1            # baseline: 297 passed, 1 warning, 1 error
```

---

## Task 1: New types in `controller.py` (Stamped, SensorFrame, ActuatorCommand, VehicleParams)

**Files:**
- Create: `src/simulacra/onboard/controller.py` (new file; the existing `controller.py` will be moved to `controller_legacy.py` in Step 1.1 to avoid losing the old `Controller(ABC)` + `IdentityController` until Plan 3 migrates consumers).
- Create: `tests/onboard/__init__.py` (empty)
- Create: `tests/onboard/test_controller_types.py`

- [ ] **Step 1.1: Move the old `controller.py` to `controller_legacy.py`**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
git mv src/simulacra/onboard/controller.py src/simulacra/onboard/controller_legacy.py
```

Now update every importer to reference `controller_legacy` so nothing breaks. The current importers (all reference the old `Controller`/`IdentityController`):

```bash
grep -rn "from .controller import\|from simulacra.onboard.controller import\|from ..onboard.controller import" src/ scenarios/ tests/ 2>/dev/null
```

Replace each match: change `from ...controller import Controller, IdentityController` (or just `Controller`) to `from ...controller_legacy import Controller, IdentityController`. Use the Edit tool per file.

Verify nothing imports `simulacra.onboard.controller` directly any more (the new file we're about to create will have a different shape):

```bash
grep -rn "from .controller import\|from simulacra.onboard.controller import\|from ..onboard.controller import" src/ scenarios/ tests/ 2>/dev/null
# expected: no output
```

Run tests to confirm the rename didn't break anything:

```bash
uv run pytest -q --tb=no 2>&1 | tail -3
# expected: 297 passed (same as baseline)
```

- [ ] **Step 1.2: Write the failing test**

Create `tests/onboard/__init__.py` (empty) and `tests/onboard/test_controller_types.py`:

```python
"""Unit tests for the SDK's plain-data types in onboard/controller.py."""
from __future__ import annotations

import pytest

from simulacra.physics.vec3 import Vec3
from simulacra.physics.quat import Orientation

from simulacra.onboard.controller import (
    Stamped,
    SensorFrame,
    ActuatorCommand,
    VehicleParams,
)


def test_stamped_carries_value_and_timestamp():
    s = Stamped(value=1.5, t=0.42, seq=7)
    assert s.value == 1.5
    assert s.t == 0.42
    assert s.seq == 7


def test_stamped_is_frozen():
    s = Stamped(value=1.0, t=0.0, seq=0)
    with pytest.raises((AttributeError, Exception)):
        s.value = 2.0  # frozen dataclass disallows assignment


def test_sensor_frame_holds_per_topic_stamps():
    accel = Vec3(0.0, -9.81, 0.0)
    gyro = Vec3(0.0, 0.0, 0.0)
    pos = Vec3(0.0, 100.0, 0.0)
    vel = Vec3(0.0, -5.0, 0.0)
    ori = Orientation.from_quaternion(1.0, 0.0, 0.0, 0.0)
    frame = SensorFrame(
        t_now=1.234,
        imu_accel=Stamped(value=accel, t=1.230, seq=10),
        imu_gyro=Stamped(value=gyro, t=1.230, seq=10),
        baro=Stamped(value=100.0, t=1.220, seq=5),
        gps=Stamped(value=pos, t=1.000, seq=2),
        velocity=Stamped(value=vel, t=1.220, seq=5),
        orientation=Stamped(value=ori, t=1.230, seq=10),
    )
    assert frame.t_now == 1.234
    assert frame.gps.t == 1.000
    # GPS is "stale" by 234 ms
    assert frame.t_now - frame.gps.t == pytest.approx(0.234)


def test_sensor_frame_gps_may_be_none():
    frame = SensorFrame(
        t_now=0.0,
        imu_accel=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        imu_gyro=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        baro=Stamped(value=0.0, t=0.0, seq=0),
        gps=None,  # GPS lock lost
        velocity=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        orientation=Stamped(value=Orientation.from_quaternion(1, 0, 0, 0), t=0.0, seq=0),
    )
    assert frame.gps is None


def test_actuator_command_holds_throttle_and_gimbal():
    cmd = ActuatorCommand(throttle=0.7, gimbal_pitch=0.05, gimbal_yaw=-0.03)
    assert cmd.throttle == 0.7
    assert cmd.gimbal_pitch == 0.05
    assert cmd.gimbal_yaw == -0.03


def test_actuator_command_is_frozen():
    cmd = ActuatorCommand(throttle=0.0, gimbal_pitch=0.0, gimbal_yaw=0.0)
    with pytest.raises((AttributeError, Exception)):
        cmd.throttle = 1.0


def test_vehicle_params_holds_design_time_constants():
    inertia = Vec3(10.0, 360.0, 360.0)
    vp = VehicleParams(
        dry_mass=60.0,
        full_fuel_mass=80.0,
        inertia_dry=inertia,
        length=4.0,
        max_thrust=2000.0,
        max_gimbal_angle=0.349,   # ~20 degrees
        max_gimbal_rate=2.0,
    )
    assert vp.dry_mass == 60.0
    assert vp.full_fuel_mass == 80.0
    assert vp.max_thrust == 2000.0
```

- [ ] **Step 1.3: Run test to verify it fails**

```bash
uv run pytest tests/onboard/test_controller_types.py -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'simulacra.onboard.controller'` (the old file was renamed in Step 1.1; the new one doesn't exist yet).

- [ ] **Step 1.4: Write minimal implementation**

Create `src/simulacra/onboard/controller.py`:

```python
"""SDK data types and Controller Protocol.

These types are the public contract that engineers writing controllers
program against. They are deliberately plain — frozen dataclasses with
no methods — so they map directly to C structs when the controller is
later ported to a flight computer.

The Controller Protocol itself is defined in this file but lives a
separate file from the implementations: `reference/` for the shipped
reference controllers, and engineer-written modules elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from simulacra.physics.quat import Orientation
from simulacra.physics.vec3 import Vec3


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Stamped(Generic[T]):
    """A sensor sample with its production timestamp.

    `t` is seconds since simulator/flight boot. `seq` is a monotonically
    increasing counter the controller can use to detect "did this update
    since I last looked?" without comparing timestamps directly.
    """
    value: T
    t: float
    seq: int


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """Snapshot of all sensor topics as the controller sees them at one tick.

    Each field carries its own `Stamped` so the controller knows how
    fresh each reading is. `gps` may be None when GPS lock is lost.
    """
    t_now: float
    imu_accel: Stamped[Vec3]
    imu_gyro: Stamped[Vec3]
    baro: Stamped[float]
    gps: Stamped[Vec3] | None
    velocity: Stamped[Vec3]
    orientation: Stamped[Orientation]


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    """Single command output from the controller for one tick."""
    throttle: float          # 0..1
    gimbal_pitch: float      # rad
    gimbal_yaw: float        # rad


@dataclass(frozen=True, slots=True)
class VehicleParams:
    """Vehicle properties the controller may know at design time.

    Runtime mass changes as fuel burns and must be sensed or estimated;
    those values are not here.
    """
    dry_mass: float          # kg, structure + payload, no fuel
    full_fuel_mass: float    # kg, fuel at t=0
    inertia_dry: Vec3        # kg·m², principal-axis diagonal, dry
    length: float            # m, nose-to-base
    max_thrust: float        # N, at full throttle
    max_gimbal_angle: float  # rad
    max_gimbal_rate: float   # rad/s


# Controller Protocol added in Task 2.
```

- [ ] **Step 1.5: Run test to verify it passes**

```bash
uv run pytest tests/onboard/test_controller_types.py -v 2>&1 | tail -10
```

Expected: all 7 tests pass.

- [ ] **Step 1.6: Commit**

```bash
git add src/simulacra/onboard/controller.py src/simulacra/onboard/controller_legacy.py \
        tests/onboard/__init__.py tests/onboard/test_controller_types.py
git status --short | head -20    # verify expected staged files + legacy importer edits
git add -u src/ scenarios/ tests/   # pick up the controller_legacy import rewrites from Step 1.1
git commit -m "$(cat <<'EOF'
sdk: add plain-data types in onboard/controller.py (Stamped, SensorFrame, ActuatorCommand, VehicleParams)

The old Controller ABC + IdentityController moved to controller_legacy.py
so existing scenarios and reference controllers keep working until
Plan 3 migrates them. New types are frozen+slots dataclasses that map
cleanly to C structs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `Controller` Protocol + `IdentityController` reference impl

**Files:**
- Modify: `src/simulacra/onboard/controller.py` (append the Protocol and the IdentityController)
- Modify: `tests/onboard/test_controller_types.py` (add Protocol/Identity tests)

- [ ] **Step 2.1: Write failing tests**

Append to `tests/onboard/test_controller_types.py`:

```python
from simulacra.onboard.controller import Controller, IdentityController
from simulacra.onboard.guidance import Setpoint   # used at runtime; deferred import is fine
# NOTE: Setpoint import in tests will only resolve after Task 5/6 defines it.
# If Setpoint doesn't exist yet, this test gets imported but skipped via
# pytest.importorskip below.

import pytest


def test_identity_controller_returns_zero_command():
    pytest.importorskip("simulacra.onboard.guidance", reason="guidance not yet implemented")
    ctrl = IdentityController()
    assert ctrl.rate_hz == 100.0
    # reset is a no-op
    vp = VehicleParams(
        dry_mass=60.0, full_fuel_mass=80.0, inertia_dry=Vec3(10.0, 360.0, 360.0),
        length=4.0, max_thrust=2000.0, max_gimbal_angle=0.349, max_gimbal_rate=2.0,
    )
    ctrl.reset(mission=None, vehicle=vp)  # IdentityController ignores mission
    frame = SensorFrame(
        t_now=0.0,
        imu_accel=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        imu_gyro=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        baro=Stamped(value=0.0, t=0.0, seq=0),
        gps=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        velocity=Stamped(value=Vec3(0.0, 0.0, 0.0), t=0.0, seq=0),
        orientation=Stamped(
            value=Orientation.from_quaternion(1.0, 0.0, 0.0, 0.0), t=0.0, seq=0,
        ),
    )
    # The "setpoint" passed to IdentityController is ignored, so any
    # placeholder works; use None for now since Setpoint may not exist
    # at test-collection time when running this file in isolation.
    cmd = ctrl.step(frame, setpoint=None, dt=0.01)
    assert cmd.throttle == 0.0
    assert cmd.gimbal_pitch == 0.0
    assert cmd.gimbal_yaw == 0.0
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
uv run pytest tests/onboard/test_controller_types.py::test_identity_controller_returns_zero_command -v 2>&1 | tail -3
```

Expected: `ImportError: cannot import name 'Controller'` (or `IdentityController`) from `simulacra.onboard.controller`.

- [ ] **Step 2.3: Write minimal implementation**

Append to `src/simulacra/onboard/controller.py`:

```python
class Controller(Protocol):
    """Protocol for any controller.

    Implementations must:
    - Declare a fixed tick rate via the class-level `rate_hz` attribute.
    - Implement `reset(mission, vehicle)`, called once before the scenario starts.
    - Implement `step(frame, setpoint, dt)`, called every 1/rate_hz seconds.

    `step` must be pure-ish: same inputs should give the same outputs
    given the same internal state. No I/O, no time-of-day reads, no
    allocations in the hot loop (in C, this becomes a timer ISR).
    """
    rate_hz: float

    def reset(self, mission, vehicle: VehicleParams) -> None: ...
    def step(self, frame: SensorFrame, setpoint, dt: float) -> ActuatorCommand: ...


class IdentityController:
    """A controller that outputs zero throttle and zero gimbal every tick.

    Useful for plumbing tests: it lets a scenario run end-to-end without
    a real control algorithm and proves the runtime calls the controller
    correctly. The rocket will of course fall down.
    """
    rate_hz: float = 100.0

    def reset(self, mission, vehicle: VehicleParams) -> None:
        return None

    def step(self, frame: SensorFrame, setpoint, dt: float) -> ActuatorCommand:
        return ActuatorCommand(throttle=0.0, gimbal_pitch=0.0, gimbal_yaw=0.0)
```

- [ ] **Step 2.4: Run test to verify it passes**

```bash
uv run pytest tests/onboard/test_controller_types.py -v 2>&1 | tail -10
```

Expected: 8 tests pass (7 from Task 1 + 1 from Task 2). The `setpoint=None` placeholder works because the type annotation on `Controller.step` uses no concrete type for it (`setpoint` is unannotated in the Protocol).

Wait — the Protocol *does* have type annotations. Let me leave `setpoint` annotated as the forward-declared `Setpoint` and add it in Task 5; for Task 2, leave it unannotated to keep types clean:

Re-examine the file: `def step(self, frame: SensorFrame, setpoint, dt: float) -> ActuatorCommand: ...` — `setpoint` has no annotation. Good. Move on.

- [ ] **Step 2.5: Commit**

```bash
git add src/simulacra/onboard/controller.py tests/onboard/test_controller_types.py
git commit -m "$(cat <<'EOF'
sdk: add Controller Protocol and IdentityController reference impl

IdentityController returns zero throttle and zero gimbal each tick;
exists for end-to-end runtime smoke tests where the actual control
logic isn't the thing under test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `SensorBus` — push/snapshot, basic shape

**Files:**
- Create: `src/simulacra/onboard/bus.py`
- Create: `tests/onboard/test_bus.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/onboard/test_bus.py`:

```python
"""Unit tests for SensorBus push/snapshot semantics."""
from __future__ import annotations

import pytest

from simulacra.physics.vec3 import Vec3
from simulacra.physics.quat import Orientation

from simulacra.onboard.bus import SensorBus
from simulacra.onboard.controller import SensorFrame, Stamped


def test_snapshot_before_any_push_raises_or_returns_none():
    bus = SensorBus()
    with pytest.raises(RuntimeError):
        bus.snapshot(t_now=0.0)


def test_single_push_then_snapshot_yields_value():
    bus = SensorBus()
    bus.push_imu(t=0.01, accel=Vec3(0.0, -9.81, 0.0), gyro=Vec3(0.0, 0.0, 0.0))
    bus.push_baro(t=0.02, altitude=100.0)
    bus.push_gps(t=0.05, position=Vec3(0.0, 100.0, 0.0))
    bus.push_velocity(t=0.02, velocity=Vec3(0.0, -5.0, 0.0))
    bus.push_orientation(
        t=0.02, orientation=Orientation.from_quaternion(1.0, 0.0, 0.0, 0.0),
    )
    frame = bus.snapshot(t_now=0.10)
    assert isinstance(frame, SensorFrame)
    assert frame.t_now == 0.10
    assert frame.imu_accel.value[1] == pytest.approx(-9.81)
    assert frame.baro.value == 100.0
    assert frame.gps is not None
    assert frame.gps.value[1] == pytest.approx(100.0)


def test_repeated_push_increments_seq():
    bus = SensorBus()
    bus.push_baro(t=0.01, altitude=100.0)
    bus.push_baro(t=0.02, altitude=101.0)
    bus.push_baro(t=0.03, altitude=102.0)
    # Other topics need at least one push for snapshot to work
    bus.push_imu(t=0.01, accel=Vec3(0, 0, 0), gyro=Vec3(0, 0, 0))
    bus.push_velocity(t=0.01, velocity=Vec3(0, 0, 0))
    bus.push_orientation(t=0.01, orientation=Orientation.from_quaternion(1, 0, 0, 0))
    bus.push_gps(t=0.01, position=Vec3(0, 0, 0))

    frame = bus.snapshot(t_now=0.04)
    assert frame.baro.value == 102.0
    assert frame.baro.seq == 2   # 0-indexed: pushes 0, 1, 2
    assert frame.baro.t == 0.03


def test_gps_can_be_marked_unavailable():
    bus = SensorBus()
    bus.push_imu(t=0.01, accel=Vec3(0, 0, 0), gyro=Vec3(0, 0, 0))
    bus.push_baro(t=0.01, altitude=0.0)
    bus.push_velocity(t=0.01, velocity=Vec3(0, 0, 0))
    bus.push_orientation(t=0.01, orientation=Orientation.from_quaternion(1, 0, 0, 0))
    # No GPS push.
    frame = bus.snapshot(t_now=0.02)
    assert frame.gps is None
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
uv run pytest tests/onboard/test_bus.py -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'simulacra.onboard.bus'`.

- [ ] **Step 3.3: Write minimal implementation**

Create `src/simulacra/onboard/bus.py`:

```python
"""SensorBus: the seam between sensor producers and the controller.

Producers (in sim, HIL, or flight) call `push_*` methods at the rate
their sensor produces samples. The controller calls `snapshot(t_now)`
at its own fixed rate and gets a `SensorFrame` of the latest values
with timestamps and sequence counters.

For pure-Python sim this is single-threaded — producer (kernel) and
consumer (controller) interleave on the main loop. In HIL or flight
the producer would run on a separate thread / ISR; the snapshot API
does not change.
"""
from __future__ import annotations

from dataclasses import dataclass

from simulacra.physics.quat import Orientation
from simulacra.physics.vec3 import Vec3

from .controller import SensorFrame, Stamped


@dataclass
class _Slot:
    """Mutable latest-value cell for one topic."""
    value: object | None = None
    t: float = -1.0
    seq: int = -1


class SensorBus:
    """Latest-sample registry, one cell per topic."""

    def __init__(self) -> None:
        self._imu_accel = _Slot()
        self._imu_gyro = _Slot()
        self._baro = _Slot()
        self._gps = _Slot()
        self._velocity = _Slot()
        self._orientation = _Slot()

    # --- producers ----------------------------------------------------

    def push_imu(self, t: float, accel: Vec3, gyro: Vec3) -> None:
        self._imu_accel.value = accel
        self._imu_accel.t = t
        self._imu_accel.seq += 1
        self._imu_gyro.value = gyro
        self._imu_gyro.t = t
        self._imu_gyro.seq += 1

    def push_baro(self, t: float, altitude: float) -> None:
        self._baro.value = altitude
        self._baro.t = t
        self._baro.seq += 1

    def push_gps(self, t: float, position: Vec3) -> None:
        self._gps.value = position
        self._gps.t = t
        self._gps.seq += 1

    def push_velocity(self, t: float, velocity: Vec3) -> None:
        self._velocity.value = velocity
        self._velocity.t = t
        self._velocity.seq += 1

    def push_orientation(self, t: float, orientation: Orientation) -> None:
        self._orientation.value = orientation
        self._orientation.t = t
        self._orientation.seq += 1

    # --- consumer -----------------------------------------------------

    def snapshot(self, t_now: float) -> SensorFrame:
        """Snapshot all slots into a SensorFrame.

        Raises RuntimeError if any non-optional topic has never been
        pushed (would indicate a wiring bug in the runtime).
        """
        for name in ("_imu_accel", "_imu_gyro", "_baro", "_velocity", "_orientation"):
            slot = getattr(self, name)
            if slot.seq < 0:
                raise RuntimeError(
                    f"SensorBus.snapshot called before any push to '{name[1:]}'"
                )
        gps = (
            Stamped(value=self._gps.value, t=self._gps.t, seq=self._gps.seq)
            if self._gps.seq >= 0 else None
        )
        return SensorFrame(
            t_now=t_now,
            imu_accel=Stamped(
                value=self._imu_accel.value, t=self._imu_accel.t,
                seq=self._imu_accel.seq,
            ),
            imu_gyro=Stamped(
                value=self._imu_gyro.value, t=self._imu_gyro.t,
                seq=self._imu_gyro.seq,
            ),
            baro=Stamped(
                value=self._baro.value, t=self._baro.t, seq=self._baro.seq,
            ),
            gps=gps,
            velocity=Stamped(
                value=self._velocity.value, t=self._velocity.t,
                seq=self._velocity.seq,
            ),
            orientation=Stamped(
                value=self._orientation.value, t=self._orientation.t,
                seq=self._orientation.seq,
            ),
        )
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
uv run pytest tests/onboard/test_bus.py -v 2>&1 | tail -10
```

Expected: all 4 tests pass.

- [ ] **Step 3.5: Commit**

```bash
git add src/simulacra/onboard/bus.py tests/onboard/test_bus.py
git commit -m "$(cat <<'EOF'
sdk: add SensorBus with per-topic push and snapshot

Producers call push_imu / push_baro / push_gps / push_velocity /
push_orientation at sensor rate; consumer (controller) calls snapshot
at its own fixed rate and gets a SensorFrame with stamped values.

GPS may be unavailable; snapshot returns frame.gps=None in that case.
A snapshot before any push to a non-optional topic raises RuntimeError
as a wiring-bug check.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `Waypoint`, `Mission`, `WaypointMission` in `guidance.py`

**Files:**
- Create: `src/simulacra/onboard/guidance.py`
- Create: `tests/onboard/test_guidance_mission.py`

- [ ] **Step 4.1: Write failing tests**

Create `tests/onboard/test_guidance_mission.py`:

```python
"""Tests for mission data types."""
from __future__ import annotations

import pytest

from simulacra.physics.vec3 import Vec3
from simulacra.onboard.guidance import Waypoint, WaypointMission


def test_waypoint_holds_position_and_hold_time():
    wp = Waypoint(position=Vec3(0.0, 10.0, 0.0), hold_time=2.0)
    assert wp.position[1] == 10.0
    assert wp.hold_time == 2.0
    assert wp.acceptance_radius == 1.0   # default
    assert wp.max_speed == 5.0           # default


def test_waypoint_mission_holds_list():
    wps = [
        Waypoint(position=Vec3(0.0,  5.0, 0.0), hold_time=1.0),
        Waypoint(position=Vec3(0.0, 10.0, 0.0), hold_time=2.0),
        Waypoint(position=Vec3(0.0,  1.0, 0.0), hold_time=0.5),
    ]
    mission = WaypointMission(waypoints=wps)
    assert len(mission.waypoints) == 3
    assert mission.waypoints[1].position[1] == 10.0


def test_waypoint_mission_rejects_empty():
    with pytest.raises(ValueError):
        WaypointMission(waypoints=[])
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
uv run pytest tests/onboard/test_guidance_mission.py -v 2>&1 | tail -3
```

Expected: `ModuleNotFoundError: No module named 'simulacra.onboard.guidance'`.

- [ ] **Step 4.3: Write minimal implementation**

Create `src/simulacra/onboard/guidance.py`:

```python
"""Guidance: turn a Mission into a stream of Setpoints for the Controller.

A Mission is a scenario-level objective (e.g. a list of waypoints to
visit). A Guidance implementation owns the logic for converting that
objective plus the live sensor state into a Setpoint at each controller
tick.

For v0.1 only WaypointMission + WaypointGuidance ship. The Mission
Protocol is polymorphic so future mission types (sounding-rocket
profile, controlled descent, abort) can be added without changing
the Controller interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from simulacra.physics.quat import Orientation
from simulacra.physics.vec3 import Vec3

from .controller import SensorFrame, VehicleParams


@dataclass(frozen=True)
class Waypoint:
    """A target the rocket should reach and optionally hold."""
    position: Vec3
    hold_time: float = 0.0
    acceptance_radius: float = 1.0   # m
    max_speed: float = 5.0           # m/s


class Mission(Protocol):
    """Scenario-level objective consumed by a Guidance implementation."""


@dataclass(frozen=True)
class WaypointMission:
    waypoints: list[Waypoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.waypoints) == 0:
            raise ValueError("WaypointMission requires at least one waypoint")
```

- [ ] **Step 4.4: Run tests to verify they pass**

```bash
uv run pytest tests/onboard/test_guidance_mission.py -v 2>&1 | tail -5
```

Expected: 3 tests pass.

- [ ] **Step 4.5: Commit**

```bash
git add src/simulacra/onboard/guidance.py tests/onboard/test_guidance_mission.py
git commit -m "$(cat <<'EOF'
sdk: add Waypoint, Mission Protocol, WaypointMission

Mission is polymorphic to accommodate future mission types
(sounding-rocket profile, controlled descent) without changing
the Controller interface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `Setpoint` and `Guidance` Protocol

**Files:**
- Modify: `src/simulacra/onboard/guidance.py` (append Setpoint + Guidance Protocol)
- Create: `tests/onboard/test_guidance_setpoint.py`

- [ ] **Step 5.1: Write failing tests**

Create `tests/onboard/test_guidance_setpoint.py`:

```python
"""Tests for Setpoint and the Guidance Protocol shape."""
from __future__ import annotations

from simulacra.physics.vec3 import Vec3
from simulacra.physics.quat import Orientation

from simulacra.onboard.guidance import Setpoint


def test_setpoint_holds_pos_vel_accel_attitude():
    sp = Setpoint(
        position=Vec3(0.0, 10.0, 0.0),
        velocity=Vec3(0.0, 0.0, 0.0),
        accel_ff=Vec3(0.0, 0.0, 0.0),
        attitude=Orientation.from_quaternion(1.0, 0.0, 0.0, 0.0),
    )
    assert sp.position[1] == 10.0
    assert sp.velocity[0] == 0.0
    assert sp.accel_ff[0] == 0.0
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
uv run pytest tests/onboard/test_guidance_setpoint.py -v 2>&1 | tail -3
```

Expected: `ImportError: cannot import name 'Setpoint'`.

- [ ] **Step 5.3: Write minimal implementation**

Append to `src/simulacra/onboard/guidance.py`:

```python
@dataclass(frozen=True, slots=True)
class Setpoint:
    """What Guidance wants the rocket to be doing right now.

    Controllers may ignore individual fields. A pure-position tracker
    can ignore `velocity` and `accel_ff`; an attitude-only controller
    can ignore everything except `attitude`.
    """
    position: Vec3
    velocity: Vec3
    accel_ff: Vec3
    attitude: Orientation


class Guidance(Protocol):
    """A guidance implementation.

    `reset` is called once before the scenario starts. `update` is
    called every controller tick (so guidance runs at controller rate).
    Guidance sees the same SensorFrame the controller does, which lets
    it react to sensed state (e.g. detect arrival from sensed position,
    abort on stale GPS).
    """
    def reset(self, mission: Mission, vehicle: VehicleParams) -> None: ...
    def update(self, frame: SensorFrame, t_now: float) -> Setpoint: ...
```

- [ ] **Step 5.4: Run test to verify it passes**

```bash
uv run pytest tests/onboard/test_guidance_setpoint.py -v 2>&1 | tail -3
```

Expected: 1 test passes.

- [ ] **Step 5.5: Commit**

```bash
git add src/simulacra/onboard/guidance.py tests/onboard/test_guidance_setpoint.py
git commit -m "$(cat <<'EOF'
sdk: add Setpoint dataclass and Guidance Protocol

Setpoint carries position+velocity+accel_ff+attitude so trackers can
use as much or as little as they want. Guidance.update is called at
controller rate with the same SensorFrame the controller sees.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `WaypointGuidance` — arrival detection + setpoint generation

**Files:**
- Modify: `src/simulacra/onboard/guidance.py` (append WaypointGuidance)
- Create: `tests/onboard/test_waypoint_guidance.py`

- [ ] **Step 6.1: Write failing tests (arrival + sequencing)**

Create `tests/onboard/test_waypoint_guidance.py`:

```python
"""Tests for WaypointGuidance: arrival detection, sequencing, setpoint output."""
from __future__ import annotations

import math

import pytest

from simulacra.physics.vec3 import Vec3
from simulacra.physics.quat import Orientation

from simulacra.onboard.controller import (
    SensorFrame, Stamped, VehicleParams,
)
from simulacra.onboard.guidance import (
    Waypoint, WaypointMission, WaypointGuidance, Setpoint,
)


def make_frame(t: float, pos: Vec3, vel: Vec3 = Vec3(0, 0, 0)) -> SensorFrame:
    return SensorFrame(
        t_now=t,
        imu_accel=Stamped(value=Vec3(0, 0, 0), t=t, seq=0),
        imu_gyro=Stamped(value=Vec3(0, 0, 0), t=t, seq=0),
        baro=Stamped(value=pos[1], t=t, seq=0),
        gps=Stamped(value=pos, t=t, seq=0),
        velocity=Stamped(value=vel, t=t, seq=0),
        orientation=Stamped(
            value=Orientation.from_quaternion(1, 0, 0, 0), t=t, seq=0,
        ),
    )


def make_vehicle() -> VehicleParams:
    return VehicleParams(
        dry_mass=60.0, full_fuel_mass=80.0, inertia_dry=Vec3(10, 360, 360),
        length=4.0, max_thrust=2000.0, max_gimbal_angle=0.349, max_gimbal_rate=2.0,
    )


def test_initial_setpoint_targets_first_waypoint():
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0.0, 10.0, 0.0), hold_time=1.0),
        Waypoint(position=Vec3(0.0,  1.0, 0.0), hold_time=0.0),
    ])
    g = WaypointGuidance()
    g.reset(mission, make_vehicle())
    sp = g.update(make_frame(t=0.0, pos=Vec3(0, 0, 0)), t_now=0.0)
    assert sp.position[1] == pytest.approx(10.0)


def test_arrival_within_radius_does_not_advance_before_hold_time():
    mission = WaypointMission(waypoints=[
        Waypoint(
            position=Vec3(0.0, 10.0, 0.0), hold_time=1.0,
            acceptance_radius=0.5,
        ),
        Waypoint(position=Vec3(0.0, 20.0, 0.0), hold_time=0.0),
    ])
    g = WaypointGuidance()
    g.reset(mission, make_vehicle())
    # Arrive at waypoint 0 at t=2.0
    sp = g.update(make_frame(t=2.0, pos=Vec3(0, 10.0, 0)), t_now=2.0)
    assert sp.position[1] == pytest.approx(10.0)   # still tracking wp 0
    # 0.5 s later, still holding (hold_time is 1.0)
    sp = g.update(make_frame(t=2.5, pos=Vec3(0, 10.0, 0)), t_now=2.5)
    assert sp.position[1] == pytest.approx(10.0)


def test_after_hold_time_advances_to_next_waypoint():
    mission = WaypointMission(waypoints=[
        Waypoint(
            position=Vec3(0.0, 10.0, 0.0), hold_time=1.0,
            acceptance_radius=0.5,
        ),
        Waypoint(position=Vec3(0.0, 20.0, 0.0), hold_time=0.0),
    ])
    g = WaypointGuidance()
    g.reset(mission, make_vehicle())
    g.update(make_frame(t=2.0, pos=Vec3(0, 10.0, 0)), t_now=2.0)   # arrive
    g.update(make_frame(t=2.5, pos=Vec3(0, 10.0, 0)), t_now=2.5)   # holding
    sp = g.update(make_frame(t=3.5, pos=Vec3(0, 10.0, 0)), t_now=3.5)
    # 1.5 s after arrival > 1.0 hold_time → advance
    assert sp.position[1] == pytest.approx(20.0)


def test_final_waypoint_does_not_advance_past_end():
    mission = WaypointMission(waypoints=[
        Waypoint(
            position=Vec3(0.0, 10.0, 0.0), hold_time=1.0,
            acceptance_radius=0.5,
        ),
    ])
    g = WaypointGuidance()
    g.reset(mission, make_vehicle())
    # Arrive and hold past hold_time
    g.update(make_frame(t=2.0, pos=Vec3(0, 10.0, 0)), t_now=2.0)
    sp = g.update(make_frame(t=10.0, pos=Vec3(0, 10.0, 0)), t_now=10.0)
    # No next waypoint — keep targeting the last one
    assert sp.position[1] == pytest.approx(10.0)


def test_setpoint_velocity_zero_for_simple_position_target():
    """v0.1 WaypointGuidance emits position-only setpoint; velocity is zero."""
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0.0, 10.0, 0.0), hold_time=0.0),
    ])
    g = WaypointGuidance()
    g.reset(mission, make_vehicle())
    sp = g.update(make_frame(t=0.0, pos=Vec3(0, 0, 0)), t_now=0.0)
    assert sp.velocity[0] == 0.0 and sp.velocity[1] == 0.0 and sp.velocity[2] == 0.0
    assert sp.accel_ff[0] == 0.0 and sp.accel_ff[1] == 0.0 and sp.accel_ff[2] == 0.0
```

- [ ] **Step 6.2: Run tests to verify they fail**

```bash
uv run pytest tests/onboard/test_waypoint_guidance.py -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'WaypointGuidance'`.

- [ ] **Step 6.3: Write minimal implementation**

Append to `src/simulacra/onboard/guidance.py`:

```python
class WaypointGuidance:
    """Sequence a WaypointMission into position setpoints.

    Detects arrival from sensed position (uses GPS if available, falls
    back to baro for altitude only). Advances after the active waypoint's
    hold_time elapses since arrival.

    v0.1 emits position-only setpoints (zero velocity and accel_ff).
    Higher-order trajectory shaping is a known evolution noted in the
    spec.
    """

    def __init__(self) -> None:
        self._mission: WaypointMission | None = None
        self._idx: int = 0
        self._arrival_t: float | None = None
        # Default attitude: upright (nose along +Y in world frame).
        self._upright = Orientation.from_quaternion(
            math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4),
        )

    def reset(self, mission: WaypointMission, vehicle: VehicleParams) -> None:
        if not isinstance(mission, WaypointMission):
            raise TypeError(
                f"WaypointGuidance requires WaypointMission, got {type(mission).__name__}"
            )
        self._mission = mission
        self._idx = 0
        self._arrival_t = None

    def update(self, frame: SensorFrame, t_now: float) -> Setpoint:
        if self._mission is None:
            raise RuntimeError("WaypointGuidance.update called before reset()")
        wp = self._mission.waypoints[self._idx]

        # Sensed position: GPS if available, else infer from baro (altitude only).
        sensed_pos = (
            frame.gps.value if frame.gps is not None
            else Vec3(0.0, frame.baro.value, 0.0)
        )

        # Arrival detection (Euclidean).
        dx = float(sensed_pos[0] - wp.position[0])
        dy = float(sensed_pos[1] - wp.position[1])
        dz = float(sensed_pos[2] - wp.position[2])
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist <= wp.acceptance_radius:
            if self._arrival_t is None:
                self._arrival_t = t_now
            elif (t_now - self._arrival_t) >= wp.hold_time:
                # Hold satisfied — try to advance.
                if self._idx + 1 < len(self._mission.waypoints):
                    self._idx += 1
                    self._arrival_t = None
                    wp = self._mission.waypoints[self._idx]
        else:
            # Drifted out of the acceptance radius — reset arrival timer.
            self._arrival_t = None

        return Setpoint(
            position=wp.position,
            velocity=Vec3(0.0, 0.0, 0.0),
            accel_ff=Vec3(0.0, 0.0, 0.0),
            attitude=self._upright,
        )
```

Also add `import math` near the top of `guidance.py` if not already present.

- [ ] **Step 6.4: Run tests to verify they pass**

```bash
uv run pytest tests/onboard/test_waypoint_guidance.py -v 2>&1 | tail -10
```

Expected: all 5 tests pass.

- [ ] **Step 6.5: Commit**

```bash
git add src/simulacra/onboard/guidance.py tests/onboard/test_waypoint_guidance.py
git commit -m "$(cat <<'EOF'
sdk: add WaypointGuidance with arrival detection and sequencing

Closed-loop on sensed position (GPS preferred, baro fallback for
altitude). Advances after hold_time elapses since arrival; resets the
arrival timer if the rocket drifts back out of acceptance_radius.
Emits position-only setpoints in v0.1 (velocity and accel_ff zero).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: NDJSON writer in `runtime/ndjson.py`

**Files:**
- Create: `src/simulacra/runtime/__init__.py`
- Create: `src/simulacra/runtime/ndjson.py`
- Create: `tests/runtime/__init__.py`
- Create: `tests/runtime/test_ndjson.py`

The NDJSON contract is unchanged — the v0.1 runtime emits the same `MetadataMessage`, `ConstantMessage`, `VariableMessage` shapes as today via `simulacra.schema`. The new `NDJSONWriter` is just a thin file-handle wrapper that lets `Scenario` write messages without each scenario script having to import `write_message` directly.

- [ ] **Step 7.1: Write failing tests**

Create `tests/runtime/__init__.py` (empty) and `tests/runtime/test_ndjson.py`:

```python
"""Tests for runtime.ndjson — thin wrapper over simulacra.schema messages."""
from __future__ import annotations

import io
import json

import pytest

from simulacra.runtime.ndjson import NDJSONWriter


def test_writer_writes_one_message_per_line():
    buf = io.StringIO()
    w = NDJSONWriter(buf)
    w.write({"message_type": "metadata", "schema_version": "0.3.0", "dt": 0.01, "duration": 1.0})
    w.write({"message_type": "variable", "timestamp": 0.0, "x": 1.0})
    lines = buf.getvalue().rstrip("\n").split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["message_type"] == "metadata"
    assert json.loads(lines[1])["x"] == 1.0


def test_writer_close_does_not_close_underlying_stream_by_default():
    buf = io.StringIO()
    w = NDJSONWriter(buf)
    w.close()
    # Should still be writeable
    buf.write("after close\n")
    assert "after close" in buf.getvalue()


def test_writer_as_context_manager():
    buf = io.StringIO()
    with NDJSONWriter(buf) as w:
        w.write({"a": 1})
    assert json.loads(buf.getvalue().rstrip())["a"] == 1
```

- [ ] **Step 7.2: Run tests to verify they fail**

```bash
uv run pytest tests/runtime/test_ndjson.py -v 2>&1 | tail -3
```

Expected: `ModuleNotFoundError: No module named 'simulacra.runtime'`.

- [ ] **Step 7.3: Write minimal implementation**

Create `src/simulacra/runtime/__init__.py`:

```python
"""Runtime: scenario runner and I/O glue.

`Scenario` ticks the physics kernel, pushes synthetic sensor samples
to the bus, calls the controller, and writes NDJSON. `NDJSONWriter`
is a thin wrapper around a text file handle.
"""
from .ndjson import NDJSONWriter
from .scenario import Scenario  # noqa: F401 — re-export

__all__ = ["NDJSONWriter", "Scenario"]
```

Create `src/simulacra/runtime/ndjson.py`:

```python
"""NDJSON writer: one JSON object per line."""
from __future__ import annotations

import json
from typing import Any, TextIO


class NDJSONWriter:
    """Writes one JSON-encoded message per line to a text stream.

    Does not close the underlying stream on close() by default — that's
    the caller's responsibility (the stream is often sys.stdout or an
    externally-managed file handle).
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, message: dict[str, Any]) -> None:
        self._stream.write(json.dumps(message, separators=(",", ":")) + "\n")

    def close(self) -> None:
        # No-op for the default case; subclasses may override.
        return None

    def __enter__(self) -> "NDJSONWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
```

Note: `Scenario` is referenced in `__init__.py` but doesn't exist yet (Task 9). To avoid an import error at this checkpoint, leave the `from .scenario import Scenario` line *out* of `__init__.py` until Task 9 — replace the file with:

```python
"""Runtime: scenario runner and I/O glue."""
from .ndjson import NDJSONWriter

__all__ = ["NDJSONWriter"]
```

We'll add the Scenario import in Task 9.

- [ ] **Step 7.4: Run tests to verify they pass**

```bash
uv run pytest tests/runtime/test_ndjson.py -v 2>&1 | tail -5
```

Expected: 3 tests pass.

- [ ] **Step 7.5: Commit**

```bash
git add src/simulacra/runtime/__init__.py src/simulacra/runtime/ndjson.py \
        tests/runtime/__init__.py tests/runtime/test_ndjson.py
git commit -m "$(cat <<'EOF'
sdk: add runtime.NDJSONWriter (thin per-line JSON writer)

Underlying stream is caller-owned — Scenario will pass sys.stdout or
an opened file. Schema dictionaries are constructed by Scenario using
the existing simulacra.schema message types.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `sensors_for` / `actuators_for` helpers wiring `phenomena` to `SensorBus`

**Files:**
- Create: `src/simulacra/runtime/helpers.py`
- Create: `tests/runtime/test_helpers.py`
- Modify: `src/simulacra/runtime/__init__.py` (re-export `sensors_for`, `actuators_for`)

These helpers exist so scenario scripts can write `sensors = sensors_for(rocket, bus)` and `actuators = actuators_for(rocket)` instead of constructing each `GPS`, `Barometer`, `IMU` themselves. They use the **existing** `phenomena` classes (no behaviour change) and wire each one's output into a bus push at the sensor's configured rate.

For Task 8 we implement a **minimal** version that constructs the default sensor suite (GPS, baro, IMU, velocity, orientation) at default rates, and the default actuator suite (throttle + gimbal). Per-sensor noise / rate customisation is a per-scenario concern handled by passing custom `sensors=` / `actuators=` to `Scenario`.

- [ ] **Step 8.1: Inspect existing phenomena to understand the shape**

```bash
head -80 src/simulacra/phenomena/sensors.py
head -80 src/simulacra/phenomena/actuators.py
```

Note: `Sensors` is a container of named sensor instances (`GPS`, `IMU`, `Barometer`, `VelocitySensor`, `OrientationSensor`). `Actuators` is similar for `ThrottleActuator` + `GimbalActuator`. Each sensor has a `.read()` method that returns the latest noisy value given the noumenal state.

- [ ] **Step 8.2: Write failing tests**

Create `tests/runtime/test_helpers.py`:

```python
"""Tests for runtime helpers: sensors_for, actuators_for."""
from __future__ import annotations

import pytest

from simulacra.onboard.bus import SensorBus
from simulacra.runtime.helpers import sensors_for, actuators_for


@pytest.fixture
def minimal_rocket():
    """Build a minimum-viable rocket from noumena to feed the helpers."""
    from simulacra.noumena import (
        Rocket, RocketState, get_upright_orientation,
        MassModel, MassComponent, FuelComponent,
    )
    from simulacra.physics.vec3 import RocketVec3, WorldVec3
    mass = MassModel(
        components=[MassComponent(name="dry", mass=60.0, position=RocketVec3(2.0, 0, 0))],
        fuel=FuelComponent(
            name="fuel", mass=80.0, position=RocketVec3(1.5, 0, 0),
            burn_rate=1.0,
        ),
    )
    state = RocketState(
        position=WorldVec3(0, 0, 0),
        velocity=WorldVec3(0, 0, 0),
        orientation=get_upright_orientation(),
        angular_velocity=RocketVec3(0, 0, 0),
        time=0.0,
    )
    # The exact Rocket constructor signature is in noumena/rocket.py;
    # the fixture is best-effort — adjust if the constructor differs.
    return Rocket(mass_model=mass, state=state, length=4.0, max_thrust=2000.0)


def test_sensors_for_returns_named_sensor_container(minimal_rocket):
    bus = SensorBus()
    sensors = sensors_for(minimal_rocket, bus)
    # Just assert the expected attributes exist.
    assert hasattr(sensors, "GPS")
    assert hasattr(sensors, "Barometer")
    assert hasattr(sensors, "IMU")
    assert hasattr(sensors, "VelocitySensor")
    assert hasattr(sensors, "OrientationSensor")


def test_actuators_for_returns_named_actuator_container(minimal_rocket):
    actuators = actuators_for(minimal_rocket)
    assert hasattr(actuators, "ThrottleActuator")
    assert hasattr(actuators, "GimbalActuator")
```

- [ ] **Step 8.3: Run tests to verify they fail**

```bash
uv run pytest tests/runtime/test_helpers.py -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'simulacra.runtime.helpers'` (or one of the noumena imports — adapt the fixture to match the real `Rocket` constructor as documented in `src/simulacra/noumena/rocket.py`).

- [ ] **Step 8.4: Write minimal implementation**

Create `src/simulacra/runtime/helpers.py`:

```python
"""Default sensor and actuator constructors for scenarios.

These keep scenario scripts terse: instead of constructing every sensor
and actuator individually, scenarios can write:

    sensors   = sensors_for(kernel)
    actuators = actuators_for(rocket)

Custom noise profiles or non-default rates are scenario-specific and
should be set up by constructing sensors directly, not via this helper.
"""
from __future__ import annotations

from simulacra.noumena import Kernel
from simulacra.phenomena.sensors import Sensors
from simulacra.phenomena.actuators import (
    Actuators, GimbalActuator, ThrottleActuator,
)


def sensors_for(kernel: Kernel) -> Sensors:
    """Default sensor suite wired to `kernel`.

    The Scenario tick loop reads each sensor and pushes into the bus.
    Sensors take the kernel (not the rocket) per the phenomena API.
    """
    return Sensors(kernel)


def actuators_for(rocket) -> Actuators:
    """Default actuator suite (throttle + 20° gimbal) for the given rocket."""
    return Actuators(
        throttle=ThrottleActuator(rocket),
        gimbal=GimbalActuator(rocket, max_outer_angle=0.349, max_inner_angle=0.349),
    )
```

Update `src/simulacra/runtime/__init__.py`:

```python
"""Runtime: scenario runner and I/O glue."""
from .helpers import sensors_for, actuators_for
from .ndjson import NDJSONWriter

__all__ = ["NDJSONWriter", "sensors_for", "actuators_for"]
```

- [ ] **Step 8.5: Run tests to verify they pass**

```bash
uv run pytest tests/runtime/test_helpers.py -v 2>&1 | tail -5
```

Expected: 2 tests pass. If the `minimal_rocket` fixture's constructor call doesn't match the real `Rocket` signature, adjust the fixture to match `src/simulacra/noumena/rocket.py`'s actual constructor and re-run.

- [ ] **Step 8.6: Commit**

```bash
git add src/simulacra/runtime/helpers.py src/simulacra/runtime/__init__.py \
        tests/runtime/test_helpers.py
git commit -m "$(cat <<'EOF'
sdk: add runtime helpers sensors_for(rocket, bus) and actuators_for(rocket)

Thin wrappers over phenomena.Sensors / phenomena.Actuators that keep
scenario scripts terse and give us a stable hook point for future
per-rocket sensor customization.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `Scenario` runner in `runtime/scenario.py`

**Files:**
- Create: `src/simulacra/runtime/scenario.py`
- Create: `tests/runtime/test_scenario.py`
- Modify: `src/simulacra/runtime/__init__.py` (re-export `Scenario`)

The new `Scenario` ties everything together: it owns the kernel, the bus, guidance, controller, and the NDJSON writer. Its `run(out)` method runs the simulation to either `duration` seconds or until an early-stop condition (controller crashes, rocket touches ground). At each tick it:

1. Steps the kernel by physics dt.
2. Reads each sensor at its configured rate and pushes into the bus.
3. If a controller tick is due (`1/rate_hz`), snapshots the bus, calls `guidance.update(frame)` → setpoint, calls `controller.step(frame, setpoint, dt)` → command, applies command to actuators.
4. Writes per-tick NDJSON variable messages.

For v0.1 we keep the simulation deterministic given a seed.

- [ ] **Step 9.1: Write failing tests**

Create `tests/runtime/test_scenario.py`:

```python
"""End-to-end Scenario tests using IdentityController."""
from __future__ import annotations

import io
import json

import pytest

from simulacra.physics.vec3 import Vec3, WorldVec3
from simulacra.onboard.controller import IdentityController
from simulacra.onboard.guidance import (
    Waypoint, WaypointMission, WaypointGuidance,
)
from simulacra.runtime import Scenario


@pytest.fixture
def smoke_rocket_and_env():
    """Tiny rocket + ambient env for plumbing tests."""
    from simulacra.noumena import (
        Rocket, RocketState, get_upright_orientation,
        MassModel, MassComponent, FuelComponent, Environment,
    )
    from simulacra.physics.vec3 import RocketVec3, WorldVec3
    mass = MassModel(
        components=[MassComponent(name="dry", mass=60.0, position=RocketVec3(2.0, 0, 0))],
        fuel=FuelComponent(
            name="fuel", mass=80.0, position=RocketVec3(1.5, 0, 0),
            burn_rate=1.0,
        ),
    )
    state = RocketState(
        position=WorldVec3(0, 1.0, 0),  # 1m up so we don't ground-contact instantly
        velocity=WorldVec3(0, 0, 0),
        orientation=get_upright_orientation(),
        angular_velocity=RocketVec3(0, 0, 0),
        time=0.0,
    )
    rocket = Rocket(mass_model=mass, state=state, length=4.0, max_thrust=2000.0)
    env = Environment(gravity=Vec3(0, -9.81, 0))
    return rocket, env


def test_scenario_runs_with_identity_controller(smoke_rocket_and_env):
    rocket, env = smoke_rocket_and_env
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0, 5, 0), hold_time=0.0),
    ])
    out = io.StringIO()
    sc = Scenario(
        rocket=rocket, env=env,
        mission=mission,
        controller=IdentityController(),
        duration=0.5, seed=42,
    )
    sc.run(out)
    text = out.getvalue()
    assert len(text) > 0
    lines = text.rstrip("\n").split("\n")
    # First line is metadata
    assert json.loads(lines[0])["message_type"] == "metadata"
    # At least one variable message
    assert any(json.loads(L).get("message_type") == "variable" for L in lines[1:])


def test_scenario_uses_provided_guidance(smoke_rocket_and_env):
    rocket, env = smoke_rocket_and_env
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0, 5, 0), hold_time=0.0),
    ])
    out = io.StringIO()
    sc = Scenario(
        rocket=rocket, env=env,
        mission=mission,
        controller=IdentityController(),
        guidance=WaypointGuidance(),
        duration=0.5, seed=42,
    )
    sc.run(out)
    assert len(out.getvalue()) > 0


def test_two_runs_with_same_seed_produce_identical_output(smoke_rocket_and_env):
    rocket1, env1 = smoke_rocket_and_env
    rocket2, env2 = smoke_rocket_and_env
    mission = WaypointMission(waypoints=[
        Waypoint(position=Vec3(0, 5, 0), hold_time=0.0),
    ])

    def run_one(rocket, env):
        out = io.StringIO()
        sc = Scenario(
            rocket=rocket, env=env, mission=mission,
            controller=IdentityController(),
            duration=0.5, seed=42,
        )
        sc.run(out)
        return out.getvalue()

    text1 = run_one(rocket1, env1)
    text2 = run_one(rocket2, env2)
    assert text1 == text2, "Same seed must yield identical NDJSON"
```

- [ ] **Step 9.2: Run tests to verify they fail**

```bash
uv run pytest tests/runtime/test_scenario.py -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'Scenario' from 'simulacra.runtime'`.

- [ ] **Step 9.3: Write minimal implementation**

Create `src/simulacra/runtime/scenario.py`:

```python
"""New Scenario runner.

Owns the kernel, bus, guidance, controller, sensors, actuators, and
NDJSON output. Tick loop:

1. Step kernel by dt.
2. For each sensor due to sample at this t, push its reading to the bus.
3. If a controller tick is due (1/rate_hz), snapshot bus,
   guidance.update(frame) -> setpoint, controller.step(frame, setpoint, dt)
   -> command, apply command to actuators.
4. Emit one variable NDJSON message per kernel step.

Deterministic given a seed (numpy and Python RNG both seeded).
"""
from __future__ import annotations

import random
from typing import Any, TextIO

import numpy as np

from simulacra.noumena import Kernel
from simulacra.onboard.bus import SensorBus
from simulacra.onboard.controller import (
    ActuatorCommand, Controller, SensorFrame, Stamped, VehicleParams,
)
from simulacra.onboard.guidance import Guidance, Mission, WaypointGuidance

from .helpers import sensors_for, actuators_for
from .ndjson import NDJSONWriter


class Scenario:
    def __init__(
        self,
        *,
        rocket,
        env,
        mission: Mission,
        controller: Controller,
        guidance: Guidance | None = None,
        sensors: Any = None,
        actuators: Any = None,
        duration: float,
        seed: int = 0,
        dt: float = 0.01,
    ) -> None:
        self.rocket = rocket
        self.env = env
        self.mission = mission
        self.controller = controller
        self.guidance = guidance if guidance is not None else WaypointGuidance()
        self.bus = SensorBus()
        # Kernel must be constructed before sensors (sensors take the kernel).
        self._kernel = Kernel(rocket=rocket, environment=env)
        self.sensors = sensors if sensors is not None else sensors_for(self._kernel)
        self.actuators = actuators if actuators is not None else actuators_for(rocket)
        self.duration = duration
        self.seed = seed
        self.dt = dt

        # Seed RNGs deterministically.
        random.seed(seed)
        np.random.seed(seed)

    def _vehicle_params(self) -> VehicleParams:
        """Derive design-time vehicle params from the rocket.

        v0.1 uses defaults for fields that aren't in RocketProps:
        max_thrust, max_gimbal_*. These can be made per-scenario
        overrides in v0.2.
        """
        from simulacra.physics.vec3 import Vec3
        props = self.rocket.props
        return VehicleParams(
            dry_mass=float(props.mass) - 80.0,           # approx; mass-model truth in v0.2
            full_fuel_mass=80.0,                          # default; per-scenario in v0.2
            inertia_dry=Vec3(*props.inertia_diag),
            length=float(props.length),
            max_thrust=2000.0,                            # v0.1 default
            max_gimbal_angle=0.349,                       # 20°
            max_gimbal_rate=2.0,
        )

    def _push_all_sensors(self, t: float) -> None:
        """Read each sensor (which adds noise) and push to the bus.

        For v0.1, every sensor pushes every kernel tick (i.e. the bus
        runs at physics rate). Per-sensor rate skipping is a v0.2
        feature.
        """
        from simulacra.physics.vec3 import Vec3
        # IMU
        a, w = self.sensors.IMU.read()
        self.bus.push_imu(t=t, accel=Vec3(*a), gyro=Vec3(*w))
        # Baro
        self.bus.push_baro(t=t, altitude=float(self.sensors.Barometer.read()))
        # GPS
        gps = self.sensors.GPS.read()
        self.bus.push_gps(t=t, position=Vec3(*gps))
        # Velocity
        v = self.sensors.VelocitySensor.read()
        self.bus.push_velocity(t=t, velocity=Vec3(*v))
        # Orientation
        q = self.sensors.OrientationSensor.read()
        self.bus.push_orientation(t=t, orientation=q)

    def run(self, out: TextIO) -> None:
        kernel = self._kernel
        writer = NDJSONWriter(out)
        # Metadata message.
        writer.write({
            "message_type": "metadata",
            "schema_version": "0.3.0",
            "dt": self.dt,
            "duration": self.duration,
        })

        vehicle = self._vehicle_params()
        self.guidance.reset(self.mission, vehicle)
        self.controller.reset(self.mission, vehicle)

        controller_period = 1.0 / self.controller.rate_hz
        next_ctrl_t = 0.0
        last_cmd = ActuatorCommand(throttle=0.0, gimbal_pitch=0.0, gimbal_yaw=0.0)

        t = 0.0
        while t < self.duration:
            self._push_all_sensors(t)
            if t >= next_ctrl_t:
                frame = self.bus.snapshot(t_now=t)
                setpoint = self.guidance.update(frame, t_now=t)
                last_cmd = self.controller.step(frame, setpoint, self.dt)
                next_ctrl_t += controller_period
                # Apply.
                from simulacra.phenomena.actuators import (
                    ThrottleCommand, GimbalCommand,
                )
                self.actuators.ThrottleActuator.apply(
                    ThrottleCommand(thrust_N=last_cmd.throttle * self.rocket.max_thrust),
                )
                self.actuators.GimbalActuator.apply(
                    GimbalCommand(
                        pitch=last_cmd.gimbal_pitch, yaw=last_cmd.gimbal_yaw,
                    ),
                )
            # Step physics by dt.
            kernel.step(self.dt)
            t = self.rocket.state.time
            # Per-step variable message.
            writer.write({
                "message_type": "variable",
                "timestamp": t,
                "position": list(map(float, self.rocket.state.position)),
                "velocity": list(map(float, self.rocket.state.velocity)),
                "throttle_cmd": last_cmd.throttle,
                "gimbal_cmd": [last_cmd.gimbal_pitch, last_cmd.gimbal_yaw],
            })

            # Early stop on ground contact.
            if self.rocket.state.position[1] <= 0.0:
                break

        writer.close()
```

Update `src/simulacra/runtime/__init__.py`:

```python
"""Runtime: scenario runner and I/O glue."""
from .helpers import sensors_for, actuators_for
from .ndjson import NDJSONWriter
from .scenario import Scenario

__all__ = ["NDJSONWriter", "Scenario", "sensors_for", "actuators_for"]
```

- [ ] **Step 9.4: Run tests to verify they pass**

```bash
uv run pytest tests/runtime/test_scenario.py -v 2>&1 | tail -15
```

Expected: 3 tests pass. If the test fails because of `Rocket`/`Environment` constructor mismatches, inspect `src/simulacra/noumena/rocket.py` and `src/simulacra/noumena/world.py` to find the real signatures and adjust the test fixture.

- [ ] **Step 9.5: Commit**

```bash
git add src/simulacra/runtime/scenario.py src/simulacra/runtime/__init__.py \
        tests/runtime/test_scenario.py
git commit -m "$(cat <<'EOF'
sdk: add runtime.Scenario — kernel/bus/guidance/controller tick loop

Reset both guidance and controller with mission + vehicle params. Each
tick: push sensors to bus, snapshot+guidance+controller if due, apply
actuator commands, step kernel, emit variable NDJSON message. Seeded
for determinism. Early-stops on ground contact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: End-to-end smoke run with `IdentityController`

**Files:**
- Create: `tests/runtime/test_end_to_end_smoke.py`

This test exercises the full stack — physics, sensors with noise, bus, guidance, identity controller, NDJSON output — and verifies the output is a parseable, non-trivial NDJSON file that the frontend would accept.

- [ ] **Step 10.1: Write failing test**

Create `tests/runtime/test_end_to_end_smoke.py`:

```python
"""End-to-end smoke: full stack produces parseable NDJSON."""
from __future__ import annotations

import io
import json

import pytest

from simulacra.physics.vec3 import Vec3
from simulacra.onboard.controller import IdentityController
from simulacra.onboard.guidance import Waypoint, WaypointMission
from simulacra.runtime import Scenario


@pytest.fixture
def smoke_setup():
    """Reuse the same fixture as test_scenario."""
    from simulacra.noumena import (
        Rocket, RocketState, get_upright_orientation,
        MassModel, MassComponent, FuelComponent, Environment,
    )
    from simulacra.physics.vec3 import RocketVec3, WorldVec3
    mass = MassModel(
        components=[MassComponent(name="dry", mass=60.0, position=RocketVec3(2, 0, 0))],
        fuel=FuelComponent(
            name="fuel", mass=80.0, position=RocketVec3(1.5, 0, 0),
            burn_rate=1.0,
        ),
    )
    state = RocketState(
        position=WorldVec3(0, 10.0, 0),
        velocity=WorldVec3(0, 0, 0),
        orientation=get_upright_orientation(),
        angular_velocity=RocketVec3(0, 0, 0),
        time=0.0,
    )
    rocket = Rocket(mass_model=mass, state=state, length=4.0, max_thrust=2000.0)
    env = Environment(gravity=Vec3(0, -9.81, 0))
    mission = WaypointMission(waypoints=[Waypoint(position=Vec3(0, 5, 0), hold_time=0.0)])
    return rocket, env, mission


def test_smoke_produces_well_formed_ndjson(smoke_setup):
    rocket, env, mission = smoke_setup
    out = io.StringIO()
    Scenario(
        rocket=rocket, env=env, mission=mission,
        controller=IdentityController(),
        duration=1.0, seed=42,
    ).run(out)
    lines = out.getvalue().rstrip("\n").split("\n")
    # Every line is valid JSON.
    parsed = [json.loads(L) for L in lines]
    # First is metadata, rest contain at least one variable message.
    assert parsed[0]["message_type"] == "metadata"
    n_var = sum(1 for p in parsed if p.get("message_type") == "variable")
    assert n_var > 0


def test_smoke_rocket_falls_without_thrust(smoke_setup):
    """Sanity: IdentityController gives zero thrust → rocket descends."""
    rocket, env, mission = smoke_setup
    out = io.StringIO()
    Scenario(
        rocket=rocket, env=env, mission=mission,
        controller=IdentityController(),
        duration=1.0, seed=42,
    ).run(out)
    # Final position should be lower than initial (10 m).
    lines = out.getvalue().rstrip("\n").split("\n")
    variables = [json.loads(L) for L in lines if json.loads(L).get("message_type") == "variable"]
    final_y = variables[-1]["position"][1]
    assert final_y < 10.0
```

- [ ] **Step 10.2: Run tests to verify they pass**

```bash
uv run pytest tests/runtime/test_end_to_end_smoke.py -v 2>&1 | tail -5
```

Expected: 2 tests pass.

- [ ] **Step 10.3: Commit**

```bash
git add tests/runtime/test_end_to_end_smoke.py
git commit -m "$(cat <<'EOF'
sdk: end-to-end smoke test — IdentityController produces parseable NDJSON

Proves the full stack composes correctly: physics + sensors + bus +
guidance + controller + NDJSON writer all wire together and the
rocket actually falls when commanded zero thrust.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Full regression — old tests still pass

**Files:**
- Verify only.

After all new code is in, every existing test must still pass at the same count as the Plan 1 baseline. The new tests should add to (not replace) the existing coverage.

- [ ] **Step 11.1: Run full suite**

```bash
uv run pytest -q --tb=no 2>&1 | tail -3 | tee /tmp/simulacra_plan2_post.txt
```

Expected: all original 297 tests still pass, **plus** the new tests added in this plan (roughly 7 + 1 + 4 + 3 + 1 + 5 + 3 + 2 + 3 + 2 ≈ 31 new tests). Final count: **≈328 passed, 1 warning, 1 error** (the error is still the pre-existing `scripts/test_landing_controller.py`).

If the count of original tests drops, that's a regression — investigate before continuing.

- [ ] **Step 11.2: Quick frontend sanity (manual)**

Spin up a smoke run and view it in the frontend:

```bash
cat > /tmp/v01_smoke.py <<'EOF'
import io
from simulacra.physics.vec3 import Vec3, RocketVec3, WorldVec3
from simulacra.noumena import (
    Rocket, RocketState, get_upright_orientation,
    MassModel, MassComponent, FuelComponent, Environment,
)
from simulacra.onboard.controller import IdentityController
from simulacra.onboard.guidance import Waypoint, WaypointMission
from simulacra.runtime import Scenario

mass = MassModel(
    components=[MassComponent(name="dry", mass=60.0, position=RocketVec3(2, 0, 0))],
    fuel=FuelComponent(name="fuel", mass=80.0, position=RocketVec3(1.5, 0, 0), burn_rate=1.0),
)
state = RocketState(
    position=WorldVec3(0, 50.0, 0), velocity=WorldVec3(0, 0, 0),
    orientation=get_upright_orientation(),
    angular_velocity=RocketVec3(0, 0, 0), time=0.0,
)
rocket = Rocket(mass_model=mass, state=state, length=4.0, max_thrust=2000.0)
env = Environment(gravity=Vec3(0, -9.81, 0))
mission = WaypointMission(waypoints=[Waypoint(position=Vec3(0, 30, 0), hold_time=0.0)])

import sys
Scenario(
    rocket=rocket, env=env, mission=mission,
    controller=IdentityController(), duration=3.0, seed=42,
).run(sys.stdout)
EOF
uv run /tmp/v01_smoke.py > /tmp/v01_smoke.ndjson
wc -l /tmp/v01_smoke.ndjson
```

Expected: a non-empty NDJSON file. The user should be able to open `localhost:5173/?file=/tmp/v01_smoke.ndjson` in the frontend and see the rocket falling.

- [ ] **Step 11.3: Tag and push**

```bash
git tag -a v0.1-api-primitives -m "Plan 2 complete: API primitives + runtime; IdentityController smoke passes"
git push rocketry feature/v0.1-sdk
git push rocketry v0.1-api-primitives
```

---

## Done criteria for Plan 2

All boxes above ticked, and:

1. `src/simulacra/onboard/controller.py` exports `Stamped`, `SensorFrame`, `ActuatorCommand`, `VehicleParams`, `Controller`, `IdentityController`.
2. `src/simulacra/onboard/bus.py` exports `SensorBus` with push_imu / push_baro / push_gps / push_velocity / push_orientation / snapshot.
3. `src/simulacra/onboard/guidance.py` exports `Waypoint`, `WaypointMission`, `Mission`, `Setpoint`, `Guidance`, `WaypointGuidance`.
4. `src/simulacra/runtime/` exports `Scenario`, `NDJSONWriter`, `sensors_for`, `actuators_for`.
5. The end-to-end smoke (Task 10) passes: `Scenario` ticks the kernel, snapshots the bus, calls guidance + controller, applies actuator commands, writes NDJSON.
6. All ~31 new tests pass; original 297 tests still pass.
7. The old `Controller(ABC)` and `IdentityController` continue to exist at `simulacra.onboard.controller_legacy` so existing scenarios and reference controllers still work — they migrate in Plan 3.
8. `v0.1-api-primitives` tag exists and is pushed.

When all of the above are true, Plan 3 (reference migration) can begin.
