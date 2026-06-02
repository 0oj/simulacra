# simulacra C SDK — Engineer's Manual

This manual walks you through writing a flight-controller in C, testing it inside simulacra's Python simulator, and preparing it for the flight computer. If you want the dry reference for every type and function, read `c-sdk-spec.md`. This document is the *guided tour*.

**You should have**:
- A clone of `Rocketry-IN/simulacra`, on the `feature/v0.1-sdk` branch (or wherever `csdk/include/simulacra.h` exists).
- `uv`, `cc`/`clang`/`gcc`, `make` installed.
- Python 3.13 available to `uv`.

If `make` works in `csdk/` and `uv run pytest tests/onboard/test_c_controller.py` passes, you have everything you need.

---

## 1. Mental model

There are three programs in the picture:

1. **The simulator**, written in Python. It owns the world (rocket, gravity, wind, ground), the noisy sensors, and the rate-limited actuators. It calls your controller every 10 ms (or whatever rate you choose).
2. **Your controller**, written in C. It receives a `sim_sensor_frame_t` and a `sim_setpoint_t`, and writes a `sim_actuator_command_t`.
3. **The flight computer**, eventually. The exact same `.c` file you write today will be cross-compiled for the MCU. Sensor ISRs will push into a bus; a timer ISR will fire `controller_step`; the actuator command will drive PWM peripherals.

The simulator's job is to make (1) behave enough like (3) that a controller that flies in sim has a fighting chance of flying on hardware. That means noisy GPS, baro that lags, gimbals with rate limits and lag, gravity, wind, ground contact, fuel burn — all the things you'd otherwise discover the hard way.

You write your controller against `simulacra.h`. **You do not read or write any simulacra source code**; the header is the boundary.

---

## 2. Hello, controller — your first 50 lines of C

We'll re-implement the reference PD altitude controller. The result is the same file that ships in `csdk/examples/pid_hover.c` — read along, or copy it into a new folder and use it as scaffolding.

### 2.1 Create the file

In a fresh project (anywhere on your machine — doesn't need to be inside the simulacra repo):

```
my_controller/
├── simulacra.h          (copied from csdk/include/simulacra.h)
└── my_controller.c
```

Open `my_controller.c`:

```c
#include "simulacra.h"

/* Module state. */
static float s_weight_N    = 0.0f;
static float s_max_thrust  = 1.0f;
static const float s_Kp    = 10.0f;
static const float s_Kd    = 5.0f;
static const float s_g     = 9.81f;

float controller_rate_hz(void) { return 100.0f; }

void controller_reset(const sim_mission_t* mission,
                      const sim_vehicle_params_t* vehicle)
{
    (void)mission;
    s_weight_N   = (vehicle->dry_mass + vehicle->full_fuel_mass) * s_g;
    s_max_thrust = vehicle->max_thrust;
}

void controller_step(const sim_sensor_frame_t* frame,
                     const sim_setpoint_t* setpoint,
                     float dt,
                     sim_actuator_command_t* out)
{
    (void)dt;
    float target_alt = setpoint->position.y;
    float alt        = frame->baro.value;
    float vy         = frame->velocity.value.y;

    float error    = target_alt - alt;
    float thrust_N = s_weight_N + s_Kp * error - s_Kd * vy;
    if (thrust_N < 0.0f)         thrust_N = 0.0f;
    if (thrust_N > s_max_thrust) thrust_N = s_max_thrust;

    out->throttle     = thrust_N / s_max_thrust;
    out->gimbal_pitch = 0.0f;
    out->gimbal_yaw   = 0.0f;
}
```

Three functions. Module-static state. No headers other than `simulacra.h`. That's the whole API.

### 2.2 Build it

On macOS:
```bash
clang -O2 -fPIC -dynamiclib -o libmy_controller.dylib my_controller.c
```

On Linux:
```bash
gcc -O2 -fPIC -shared -o libmy_controller.so my_controller.c
```

You'll get a shared library next to your source.

### 2.3 Run it in the simulator

In a Python script (anywhere):

```python
from simulacra.onboard.c_controller import CController
from simulacra.scenarios import hopper_short

ctrl = CController("/full/path/to/libmy_controller.dylib")
hopper_short.run(ctrl)
```

Pipe stdout to an NDJSON file:
```bash
uv run my_script.py > /tmp/run.ndjson
```

Open the result in the frontend (see `ONBOARDING.md` § 2 for setup):
```
localhost:5173/?file=/tmp/run.ndjson
```

You should see the rocket hovering at the waypoint altitudes. Congratulations — you wrote a C flight controller and watched it fly.

---

## 3. Anatomy of a controller

The reference PD is intentionally minimal. A realistic controller has three additional concerns.

### 3.1 Internal state across ticks

For PID you need to remember the integral and the previous error:

```c
static float s_integral = 0.0f;
static float s_prev_error = 0.0f;

void controller_reset(/* ... */) {
    s_integral = 0.0f;
    s_prev_error = 0.0f;
    /* compute s_weight_N, s_max_thrust */
}

void controller_step(/* ... */, float dt, /* ... */) {
    float error = /* ... */;
    s_integral += error * dt;
    /* anti-windup: clamp s_integral */
    if (s_integral > 5.0f) s_integral = 5.0f;
    if (s_integral < -5.0f) s_integral = -5.0f;
    float deriv = (error - s_prev_error) / dt;
    s_prev_error = error;
    /* use error, s_integral, deriv */
}
```

Always reset internal state in `controller_reset`. The simulator may run your controller many times in a row (different seeds, different scenarios); state from the previous run must not leak.

### 3.2 Stale-sensor handling

GPS updates at 5 Hz; your controller runs at 100 Hz. Most ticks have the *same* GPS sample as the previous tick. That's fine — but if GPS hasn't updated for *seconds*, you should distrust it:

```c
if (frame->gps_valid == 0
    || frame->t_now - frame->gps.t > 0.5f) {
    /* GPS lost or stale — use only baro + IMU */
    fall_back_to_dead_reckoning();
}
```

Use `seq` to tell "is this a new sample?" without timestamp comparison:

```c
static uint32_t s_last_gps_seq = 0;
if (frame->gps.seq != s_last_gps_seq) {
    /* new GPS sample arrived since last tick */
    s_last_gps_seq = frame->gps.seq;
    on_fresh_gps(&frame->gps);
}
```

### 3.3 Saturating clean

Throttle saturated at 1.0 means you have no headroom; gimbal saturated at `max_gimbal_angle` means you've run out of authority. Always clamp:

```c
if (out->throttle > 1.0f) out->throttle = 1.0f;
if (out->throttle < 0.0f) out->throttle = 0.0f;
const float MAX = 0.349f;          /* ~20 degrees */
if (out->gimbal_pitch >  MAX) out->gimbal_pitch =  MAX;
if (out->gimbal_pitch < -MAX) out->gimbal_pitch = -MAX;
```

The simulator will clamp anyway, but doing it inside your controller keeps your PID integrators from accumulating noise that can never produce action.

---

## 4. Coordinate conventions cheat sheet

You will get this wrong on the first try. Everyone does.

- **Right-handed.** World `+X`, `+Y`, `+Z` form `x × y = z`.
- **`+Y` is up.** Gravity is `(0, -9.81, 0)` m/s².
- **Body `+X` is along the nose.** A vertical rocket has its body `+X` parallel to world `+Y`.
- **Quaternions map body → world.** Apply the quaternion to a body-frame vector to get the world-frame vector. The identity quaternion `(1, 0, 0, 0)` corresponds to body axes aligned with world axes (rocket lying on its side along world `+X` — not upright).
- **Upright orientation** is a 90° rotation around world `+Z`: quaternion `(cos π/4, 0, 0, sin π/4)`.
- **`imu_accel` and `imu_gyro` are body frame.** Everything else (`gps`, `velocity`, setpoint position) is world frame.

To rotate a body-frame vector into the world:
```c
/* v_world = q * v_body * q^(-1) */
sim_vec3_t rotate_b2w(sim_quat_t q, sim_vec3_t v) {
    /* shortcut: v + 2*w*(q_v × v) + 2*(q_v × (q_v × v)) */
    sim_vec3_t qv = { q.x, q.y, q.z };
    sim_vec3_t c1 = cross(qv, v);
    sim_vec3_t c2 = cross(qv, c1);
    return (sim_vec3_t){
        v.x + 2.0f*q.w*c1.x + 2.0f*c2.x,
        v.y + 2.0f*q.w*c1.y + 2.0f*c2.y,
        v.z + 2.0f*q.w*c1.z + 2.0f*c2.z,
    };
}
```

For world→body, negate `qv` before the cross products.

---

## 5. Testing your controller

There are three levels of test.

### 5.1 Unit (no sim)

Construct a `sim_sensor_frame_t` and `sim_setpoint_t` by hand, call `controller_step`, assert on the output:

```c
/* In a test runner you write (e.g. CMocka, Unity, or hand-rolled main). */
sim_vehicle_params_t v = { .dry_mass=60, .full_fuel_mass=80,
                            .inertia_dry={10,360,360}, .length=4,
                            .max_thrust=2000, .max_gimbal_angle=0.349,
                            .max_gimbal_rate=2 };
controller_reset(NULL, &v);

sim_sensor_frame_t frame = { /* fill in */ };
sim_setpoint_t sp = { .position={0, 20, 0} };
sim_actuator_command_t cmd;
controller_step(&frame, &sp, 0.01f, &cmd);

assert(cmd.throttle > 0.5f);    /* should be commanding lots of thrust */
```

### 5.2 Integration (the sim)

This is the test the SDK gives you for free. Run your controller against a scenario, watch the NDJSON, watch the frontend. The `hopper_short` scenario is a good standard exam.

### 5.3 Regression (across versions of your controller)

Capture the NDJSON of a known-good run and diff against new runs. The sim is deterministic given a seed, so any difference in NDJSON is a difference in your controller's behaviour. This catches gain-tuning mistakes faster than reading a trajectory.

```bash
# Baseline (known-good controller)
uv run scripts/run_with.py libmy_v1.dylib > /tmp/v1.ndjson

# After a change
uv run scripts/run_with.py libmy_v2.dylib > /tmp/v2.ndjson

# Frontend comparison
# localhost:5173/?file=/tmp/v1.ndjson&file=/tmp/v2.ndjson
```

---

## 6. Going to flight

The Python sim never runs on the flight computer. What does run is your `.c` file, cross-compiled. The interfaces between your controller and "everything else" change in two ways:

### 6.1 Who builds the `sim_sensor_frame_t`

- **In sim:** The simulator polls each in-silico sensor at its rate and pushes into a Python-side bus; at every controller tick it builds a `sim_sensor_frame_t` and hands it to your C step. (This is `src/simulacra/onboard/c_controller.py`.)
- **On the flight computer:** Sensor ISRs (one per chip) push their samples into a C-side sensor bus. The controller-tick timer ISR snapshots the bus into a `sim_sensor_frame_t` and calls `controller_step`. The bus + timer code is not your concern — it'll ship as part of the flight HAL alongside this SDK.

Your `controller_step` is identical in both cases.

### 6.2 What `controller_rate_hz` actually controls

- **In sim:** The simulator calls your step every `1/rate_hz` seconds of simulated time, regardless of how long the Python tick loop actually takes.
- **On the flight computer:** A hardware timer fires the step ISR at this rate. If your step takes longer than `1/rate_hz` it will overrun. The flight HAL will detect overruns and refuse to arm.

Aim for ≪`1/rate_hz` compute time per step. A 100 Hz controller has 10 ms; a target of ≤1 ms leaves plenty of margin for sensor ISRs.

---

## 7. Common mistakes

| Symptom | Cause |
|---|---|
| Rocket falls straight down even though throttle is 1.0 | Forgot to `controller_step`-write all three fields of `out_command`. (`out_command` is not zero-initialised by the caller.) |
| Rocket oscillates wildly | Sign error on derivative term, or wrong frame on a sensor read (used IMU accel as world frame). |
| Rocket drifts in one direction | Integral wind-up — clamp it. |
| Rocket flies upside down | Confused `(w,x,y,z)` for `(x,y,z,w)` somewhere. The SDK convention is `(w,x,y,z)`. |
| Throttle stuck at 0 even when commanded > 0 | Throttle deadband: the actuator snaps thrust below `min_thrust` to zero. Command more thrust. |
| `make` fails with "no such file or directory `simulacra.h`" | Pass `-I/path/to/csdk/include` to your compiler. The Makefile in `csdk/` does this; if you copied files elsewhere, you need to do it yourself. |

---

## 8. Quick reference card

```c
/* Your controller, in three functions: */
float controller_rate_hz(void);
void  controller_reset(const sim_mission_t*, const sim_vehicle_params_t*);
void  controller_step (const sim_sensor_frame_t*, const sim_setpoint_t*,
                       float dt, sim_actuator_command_t*);

/* Read from frame: */
frame->t_now                    /* sim time, seconds                       */
frame->imu_accel.value.{x,y,z}  /* body frame, m/s^2                       */
frame->imu_gyro.value.{x,y,z}   /* body frame, rad/s                       */
frame->baro.value               /* altitude, m                             */
frame->gps_valid                /* 1 if GPS lock, else 0                   */
frame->gps.value.{x,y,z}        /* world frame, m (if gps_valid == 1)      */
frame->velocity.value.{x,y,z}   /* world frame, m/s                        */
frame->orientation.value.{w,x,y,z}  /* unit quat, body -> world           */

/* Read from setpoint: */
setpoint->position.{x,y,z}      /* where guidance wants you                */
setpoint->velocity.{x,y,z}      /* how fast guidance wants you             */
setpoint->accel_ff.{x,y,z}      /* feed-forward acceleration               */
setpoint->attitude.{w,x,y,z}    /* nominal attitude                        */

/* Write to out_command: */
out_command->throttle           /* [0, 1]                                  */
out_command->gimbal_pitch       /* rad                                     */
out_command->gimbal_yaw         /* rad                                     */
```

That's the whole interface. Everything else — physics, sensors, mission, NDJSON — happens on the other side of the boundary.

Welcome to the SDK. Go fly something.
