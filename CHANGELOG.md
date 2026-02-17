# Simulacra Development Changelog

This document summarizes the recent development work on the Simulacra rocket flight simulator, focusing on making the system RL-ready with realistic sensor/actuator models and a baseline TVC controller.

---

## Overview

The work followed a phased roadmap to transform Simulacra from a basic physics simulator into an RL-ready training environment. Key accomplishments:

1. **Sensor error pipeline** - Sensors now lie realistically
2. **Actuator dynamics** - Thrust misalignment, gimbal rate limits
3. **Full telemetry logging** - All sensor readings and actuator commands logged
4. **TVC hover controller** - Baseline controller for RL comparison
5. **Frontend improvements** - URL preloading, new charts, transparency

---

## Phase 0: Sensor Error Pipeline

**Goal**: Make sensors imperfect, like real hardware.

### Error Types (`phenomena/errors.py`)

| Error Type | Description |
|------------|-------------|
| `GaussianNoise` | Random noise with configurable std, seeded RNG |
| `Bias` | Constant offset |
| `ScaleBias` | Multiplicative error (e.g., 1.02x) |
| `Quantization` | ADC-style stepping |
| `Saturation` | Min/max clamping |
| `RandomWalk` | Accumulating drift over time |
| `CompositeError` | Chain multiple errors together |

### Per-Channel Sensor Errors (`phenomena/sensors.py`)

Each sensor accepts error configs per channel:

```python
sensors = Sensors(
    kernel,
    gps_errors={"y": GaussianNoise(std=0.5, seed=42)},
    velocity_errors={"y": GaussianNoise(std=0.2, seed=43)},
    orientation_errors={
        "roll": GaussianNoise(std=math.radians(0.5)),
        "pitch": GaussianNoise(std=math.radians(0.5)),
    },
    gyro_errors={
        "x": GaussianNoise(std=math.radians(1.0)),
    },
)
```

### Sensors Available

- `GPS` - Position in world frame (x, y, z channels)
- `VelocitySensor` - Velocity in world frame
- `OrientationSensor` - Quaternion attitude (roll, pitch, yaw perturbations)
- `GyroSensor` - Angular velocity in body frame
- `ThrustSensor` - Thrust magnitude

---

## Phase 1: Full Telemetry Logging

**Goal**: Log everything needed to understand what the controller saw and did.

### New Fields in `VariableMessage` (`schema.py`)

```python
# IMU sensor readings
orientation_sensor: [w, x, y, z]  # quaternion from sensor
gyro_sensor: {x, y, z}            # angular velocity from sensor

# Actuator state
gimbal_outer_angle: float         # yaw gimbal (radians)
gimbal_inner_angle: float         # pitch gimbal (radians)
throttle_command: float           # commanded thrust (N)
```

### Frontend Types Updated (`frontend/src/types.ts`)

`FlightFrame` now includes all sensor readings and actuator state for visualization.

---

## Phase 2: Actuator Dynamics

**Goal**: Make actuators imperfect - rate limits, lag, misalignment.

### Thrust Misalignment (`phenomena/actuators.py`)

Real rocket engines have combustion instabilities that cause thrust direction jitter. The `ThrottleActuator` now accepts alignment errors:

```python
throttle = ThrottleActuator(
    rocket,
    alignment_errors={
        "yaw": CompositeError([
            Bias(offset=math.radians(0.5)),      # 0.5° constant bias
            GaussianNoise(std=math.radians(0.2)), # 0.2° noise
        ]),
        "pitch": GaussianNoise(std=math.radians(0.2)),
    }
)
```

**Key fix**: Errors are applied fresh each step to the *nominal* direction, not compounded on the previous perturbed direction (which caused exponential divergence).

### Gimbal Actuator

The `GimbalActuator` supports:
- `max_outer_angle`, `max_inner_angle` - deflection limits
- `max_rate` - servo slew rate (rad/s)
- `tau` - first-order lag time constant

---

## Phase 4: TVC Hover Controller

**Goal**: Baseline controller with both altitude and attitude control for RL comparison.

### Architecture (`onboard/hover_controller.py`)

Cascade control:
1. **Outer loop**: PD altitude control → thrust magnitude
2. **Inner loop**: Quaternion PD + gravity feedforward → gimbal angles

### The Critical Fix: Gravity Feedforward

The original pure PD controller failed because it didn't account for the static torque gravity exerts on a tilted rocket.

**Problem**: A 15° tilt creates ~508 N·m of gravity torque. Pure PD with reasonable gains produced only ~13 N·m of corrective torque.

**Solution**: Compute gravity torque in body frame and command gimbal to counter it directly:

```python
# Gravity in body frame
gravity_world = np.array([0.0, -weight, 0.0])
gravity_body = self._rotate_world_to_body(quat, gravity_world)

# CoM offset from thrust point
r_com = np.array([thrust_moment_arm, 0.0, 0.0])

# Gravity torque about thrust point
tau_gravity = np.cross(r_com, gravity_body)

# Total desired torque = feedforward + feedback
tau_desired = -tau_gravity + (Kp * error - Kd * omega)
```

### Configuration

```python
@dataclass
class HoverControllerConfig:
    target_alt: float = 10.0      # meters
    weight: float = 981.0         # N (mass * g)
    max_thrust: float = 2000.0    # N
    gravity: float = 9.81         # m/s^2

    # Altitude control (outer loop)
    alt_Kp: float = 100.0         # N/m
    alt_Kd: float = 200.0         # N/(m/s)

    # Attitude control (inner loop)
    att_Kp: float = 500.0         # N·m/rad
    att_Kd: float = 200.0         # N·m/(rad/s)

    max_gimbal_angle: float = 15°  # radians
    thrust_moment_arm: float = 2.0 # meters
```

### Performance

| Scenario | Initial Tilt | Final Tilt | Max Tilt | Settling Time |
|----------|--------------|------------|----------|---------------|
| clean | 0° | 0° | 0° | - |
| tilted | 14° | 0° | 14° | ~0.5s |
| disturbed | 0° | 0.9° | 1.1° | steady-state |
| noisy | 0° | 0.2° | 0.5° | steady-state |

---

## Frontend Improvements

### URL Parameter File Loading (`app.ts`)

Load scenarios via URL instead of drag-and-drop:

```
http://localhost:5174/?file=hover_tvc_tilted.ndjson
http://localhost:5174/?file=scenario1.ndjson&file=scenario2.ndjson
```

Files are served from `/tmp/simulacra/` via Vite proxy.

### New Charts (`types.ts`)

| Chart | Label | Description |
|-------|-------|-------------|
| `gimbal` | GIMBAL | Yaw/pitch gimbal angles (°) |
| `ctrl-thr` | CTRL | Commanded vs actual thrust (N) |
| `omega-mag` | \|ω\| | Angular velocity magnitude (rad/s) |
| `tilt` | TILT | Attitude angle from vertical (°) |

### Rocket Transparency (`scene.ts`)

Rocket bodies are now semi-transparent (`opacity: 0.7`) so overlapping scenarios are visible.

### Fuel Display Fix

`MassModel.fuel_fraction` now returns `None` when no fuel capacity (instead of `0.0`). Frontend hides fuel display when null.

---

## Test Scenarios

### `scenarios/hover_tvc_test.py`

```bash
uv run scenarios/hover_tvc_test.py              # clean
uv run scenarios/hover_tvc_test.py --tilted     # 15° initial tilt
uv run scenarios/hover_tvc_test.py --disturbed  # thrust misalignment
uv run scenarios/hover_tvc_test.py --noisy      # sensor noise
```

Output: `/tmp/simulacra/hover_tvc_{variant}.ndjson`

### `scenarios/pid_hover_test.py`

Altitude-only PID controller (no attitude control). Demonstrates instability without TVC.

```bash
uv run scenarios/pid_hover_test.py              # clean
uv run scenarios/pid_hover_test.py --noisy      # with sensor noise
```

---

## File Summary

### Backend Changes

| File | Changes |
|------|---------|
| `phenomena/errors.py` | Added `CompositeError`, `RandomWalk`, `Quantization`, `Saturation`, `ScaleBias` |
| `phenomena/sensors.py` | Per-channel errors, split IMU into `OrientationSensor` + `GyroSensor` |
| `phenomena/actuators.py` | Thrust alignment errors, gimbal dynamics, `Actuators` bundle |
| `onboard/hover_controller.py` | **New** - Full TVC controller with gravity feedforward |
| `onboard/pid_hover.py` | Altitude-only PID controller |
| `noumena/scenario.py` | Fixed mutable defaults bug, added `output` parameter |
| `noumena/mass_model.py` | `fuel_fraction` returns `None` when no fuel |
| `schema.py` | Added IMU, gimbal, throttle fields to `VariableMessage` |

### Frontend Changes

| File | Changes |
|------|---------|
| `src/types.ts` | New chart defs (GIMBAL, CTRL, \|ω\|), nullable `fuelFraction` |
| `src/parser.ts` | Parse new telemetry fields |
| `src/app.ts` | URL parameter file loading |
| `src/scene.ts` | Transparent rocket bodies |
| `vite.config.ts` | **New** - Proxy `/output/` to `/tmp/simulacra/` |

### New Scenarios

| File | Description |
|------|-------------|
| `scenarios/hover_tvc_test.py` | Full TVC hover test with variants |
| `scenarios/pid_hover_test.py` | Altitude-only hover test |

---

## Next Steps (Roadmap)

- **Phase 5**: Scenario parameterization and batch generation
- **Phase 6**: Reference trajectories and scoring
- **Phase 7**: Gymnasium wrapper for RL training
- **Phase 8**: Interpretability tooling

---

## Quick Start

```bash
# Generate test data
cd backend
uv run scenarios/hover_tvc_test.py --tilted

# Start frontend
cd frontend
npm run dev

# View in browser
open "http://localhost:5174/?file=hover_tvc_tilted.ndjson"
```
