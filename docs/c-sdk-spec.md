# simulacra C SDK — Specification Sheet

**ABI version:** 0.1.0
**Header:** `csdk/include/simulacra.h`
**Compiles against:** C11 or later
**Float width:** 32-bit (single precision)
**Endianness:** native (the sim and the controller share an address space)

This document is the canonical reference for every type and function in `simulacra.h`. The manual (`c-sdk-manual.md`) explains *how* to use it; this document explains *what* it is.

---

## 1. Scope and intent

The simulacra C SDK lets an engineer write a flight controller in C, compile it to a shared library, and run it inside the simulacra Python simulator without any glue code. **The same `controller_step` source that runs in sim runs on the flight computer later** — only the toolchain differs.

The SDK exposes:

- **Plain-data structs** mirroring the Python SDK 1:1.
- **Three engineer-implemented functions** (`controller_rate_hz`, `controller_reset`, `controller_step`).
- **No simulacra-side functions called from C.** The C controller is purely a callee; sim hosts it.

There is intentionally no `controller_create` / `controller_destroy`. A flight computer hosts exactly one controller instance, allocated statically. Use `static` module variables for state.

---

## 2. Primitive types

### `sim_vec3_t`
```c
typedef struct {
    float x;
    float y;
    float z;
} sim_vec3_t;
```

**Size:** 12 bytes. **Alignment:** 4 bytes (natural float).

A right-handed 3-D vector. World frame unless the field name says otherwise (`imu_*` are body frame). World `+Y` is up; gravity points along `-Y`.

### `sim_quat_t`
```c
typedef struct {
    float w;
    float x;
    float y;
    float z;
} sim_quat_t;
```

**Size:** 16 bytes. **Alignment:** 4 bytes.

Unit quaternion stored as `(w, x, y, z)`. Maps **body → world**: applying this rotation to a body-frame vector yields the same vector expressed in the world frame.

---

## 3. Stamped sensor samples

### `sim_stamped_vec3_t`, `sim_stamped_float_t`, `sim_stamped_quat_t`
```c
typedef struct {
    sim_vec3_t value;
    float      t;
    uint32_t   seq;
} sim_stamped_vec3_t;          /* size = 20 bytes */

typedef struct {
    float    value;
    float    t;
    uint32_t seq;
} sim_stamped_float_t;          /* size = 12 bytes */

typedef struct {
    sim_quat_t value;
    float      t;
    uint32_t   seq;
} sim_stamped_quat_t;           /* size = 24 bytes */
```

Each sensor sample carries:

- `value` — the reading itself.
- `t` — production timestamp, seconds since boot.
- `seq` — monotonic counter incremented on every push. Use `seq` to detect "did this sensor update since my last tick?" without comparing floats.

**Staleness detection** is done by the controller:
```c
if (frame->t_now - frame->gps.t > 0.5f) {
    /* GPS hasn't been updated in 500 ms — trust IMU only */
}
```

---

## 4. Sensor frame

```c
typedef struct {
    float               t_now;
    sim_stamped_vec3_t  imu_accel;
    sim_stamped_vec3_t  imu_gyro;
    sim_stamped_float_t baro;
    sim_stamped_vec3_t  gps;
    uint8_t             gps_valid;
    sim_stamped_vec3_t  velocity;
    sim_stamped_quat_t  orientation;
} sim_sensor_frame_t;
```

**Approximate size:** ~112 bytes (depends on padding).

| Field | Frame | Units | Typical rate |
|---|---|---|---|
| `t_now` | — | s, since boot | controller rate |
| `imu_accel` | body | m/s² | 100–1000 Hz |
| `imu_gyro` | body | rad/s | 100–1000 Hz |
| `baro` | world (altitude only) | m | 50–100 Hz |
| `gps` | world | m | 1–10 Hz |
| `velocity` | world | m/s | 50–100 Hz |
| `orientation` | body → world | unit quaternion | 50–200 Hz |

**`gps_valid` semantics.** When `gps_valid == 0`, the GPS slot has either never been pushed or the lock has been lost. The contents of `gps.value` are undefined in this state and must not be read. When `gps_valid == 1`, `gps.value` is valid and `gps.t` records the last update time.

---

## 5. Actuator command (controller output)

```c
typedef struct {
    float throttle;       /* [0, 1]   */
    float gimbal_pitch;   /* rad      */
    float gimbal_yaw;     /* rad      */
} sim_actuator_command_t;
```

**Size:** 12 bytes.

- `throttle` — fractional thrust, clamped to `[0, 1]` by the simulator. `1.0` corresponds to `vehicle->max_thrust` Newtons.
- `gimbal_pitch` / `gimbal_yaw` — outer / inner gimbal angles, radians. Clamped to `[-max_gimbal_angle, +max_gimbal_angle]` by the actuator. Rate-limited to `max_gimbal_rate` rad/s.

A controller that doesn't gimbal must still write `0.0f` to these fields — they are not zero-initialised by the caller.

---

## 6. Vehicle params (design-time)

```c
typedef struct {
    float      dry_mass;
    float      full_fuel_mass;
    sim_vec3_t inertia_dry;
    float      length;
    float      max_thrust;
    float      max_gimbal_angle;
    float      max_gimbal_rate;
} sim_vehicle_params_t;
```

**Size:** ~32 bytes.

Passed once to `controller_reset`. **All values are design-time constants** — they do not change during flight. In particular, current mass (which decreases as fuel burns) is *not* here; it must be sensed or estimated. Controllers that need current mass should use the fuel sensor or include an estimator.

`inertia_dry` is the principal-axis diagonal of the inertia tensor with empty tanks. It changes as fuel burns, so it's also a design-time approximation. For control gains that depend on inertia, recompute online if precision matters; for v0.1 most controllers use it as a constant.

---

## 7. Mission

```c
typedef struct {
    sim_vec3_t position;
    float      hold_time;
    float      acceptance_radius;
    float      max_speed;
} sim_waypoint_t;

typedef struct {
    const sim_waypoint_t* waypoints;
    uint32_t              n_waypoints;
} sim_mission_t;
```

A waypoint sequence. Storage for `waypoints` is owned by the *caller* (the simulator or, in flight, the mission-loader); the controller may read but must not write or free.

`controller_reset` receives a pointer to `sim_mission_t`. **The controller does not have to track waypoint progress** — guidance produces the setpoint stream automatically. The mission is exposed in case the controller wants to:

- Precompute trajectory invariants (e.g. "fit a polynomial through these waypoints").
- Switch gains based on mission phase.
- Decline to fly if the mission is malformed (return early in `reset`, log via your own means).

`mission` may be `NULL` if the host doesn't have one.

---

## 8. Setpoint (from guidance)

```c
typedef struct {
    sim_vec3_t position;
    sim_vec3_t velocity;
    sim_vec3_t accel_ff;
    sim_quat_t attitude;
} sim_setpoint_t;
```

**Size:** ~52 bytes.

What guidance wants the rocket to be doing *right now*. Controllers ignore individual fields freely:

- A pure-position tracker: `position` only.
- A trajectory tracker: `position` + `velocity` (+ `accel_ff` for feed-forward).
- An attitude-only controller: `attitude` only.

`accel_ff` is the acceleration guidance *expects* you to apply (often gravity-canceled plus desired trajectory acceleration). Controllers may add this to their feedback command for cleaner tracking.

---

## 9. Engineer-implemented functions

### `controller_rate_hz`

```c
float controller_rate_hz(void);
```

Returns the tick rate in Hz. Called once before `controller_reset`. Must return a positive finite number. Typical values: `50.0f` for outer-loop-dominant controllers, `100.0f` for cascaded, `200.0f` or higher for attitude-only.

The simulator obeys this rate regardless of physics step size. On the flight computer, this is the rate at which the controller-step timer ISR fires.

### `controller_reset`

```c
void controller_reset(
    const sim_mission_t*        mission,
    const sim_vehicle_params_t* vehicle
);
```

Called once before the scenario starts. Use this to:

- Cache vehicle properties into static variables (so `controller_step` doesn't dereference `vehicle` repeatedly).
- Reset PID integrals.
- Compute design-time gains (e.g. solve a Riccati equation for LQR).
- Pre-plan a trajectory from `mission` if your design wants it.

May be called multiple times in a single sim session if scenarios chain (rare in v0.1). Each call must restore the controller to a known state.

`mission` may be `NULL`. `vehicle` is never `NULL`.

### `controller_step`

```c
void controller_step(
    const sim_sensor_frame_t*   frame,
    const sim_setpoint_t*       setpoint,
    float                       dt,
    sim_actuator_command_t*     out_command
);
```

Called every `1 / controller_rate_hz()` seconds.

Inputs:
- `frame` — latest sensor snapshot.
- `setpoint` — current target from guidance.
- `dt` — seconds since the last call. In the sim this is always `1 / rate_hz`; in flight it may drift slightly under ISR jitter.

Output:
- `out_command` — write throttle and gimbal angles here. The caller does *not* zero this struct before calling — write every field.

**Must be deterministic.** Same inputs → same outputs (given the same internal state). This is what makes hardware-in-the-loop testing meaningful.

**Must not call `malloc`, `free`, `printf`, file I/O, or anything that touches the OS.** On the flight computer there is no OS, no heap, no stdout.

**Must finish well within `dt`.** A 100 Hz controller has 10 ms per tick. Target ≤1 ms of compute to leave headroom for sensor ISRs.

---

## 10. ABI stability

Version `0.1.0` is the first published ABI. Subsequent `0.1.x` releases are guaranteed to be **forward-binary-compatible**: a controller compiled against `0.1.0` will run against `0.1.x` for `x ≥ 0`.

Breaking changes will only happen at minor version bumps (`0.2.0`, etc.) and will be announced with at least one release of lead time.

### What this means
- We will not change the order or type of fields in existing structs.
- We may *append* fields to structs (engineers' code must continue to work because their controllers don't read the new fields).
- We may *add* new functions to the API.
- We will not remove or rename existing types or functions within a minor version.

### What this excludes
- The Python ctypes mirrors are part of the same ABI promise — if the C struct changes, the Python mirror in `src/simulacra/onboard/c_controller.py` changes in lockstep.
- The build system (`csdk/Makefile`) is not part of the ABI.

---

## 11. Reference implementations

### `csdk/examples/pid_hover.c`
Minimal PD altitude-hold controller. Mirrors `src/simulacra/onboard/reference/pid_hover.py`. ~50 lines including comments.

Build:
```bash
cd csdk && make
```

Use:
```python
from simulacra.onboard.c_controller import CController
ctrl = CController("csdk/examples/libpid_hover.dylib")
from simulacra.scenarios import hopper_short
hopper_short.run(ctrl)
```

### Future
- `pid_hopper.c` — C twin of the cascaded PID hopper (not yet shipped).
- `lqr_skeleton.c` — empty LQR controller with TODO markers, for engineers to fill in.

---

## 12. Layout sanity checks

For each major struct, the expected `sizeof` on a 32-bit-float, 4-byte-aligned platform:

| Type | bytes |
|---|---|
| `sim_vec3_t` | 12 |
| `sim_quat_t` | 16 |
| `sim_stamped_vec3_t` | 20 |
| `sim_stamped_float_t` | 12 |
| `sim_stamped_quat_t` | 24 |
| `sim_actuator_command_t` | 12 |
| `sim_vehicle_params_t` | 32 |
| `sim_waypoint_t` | 24 |
| `sim_setpoint_t` | 52 |
| `sim_sensor_frame_t` | ~112 |
| `sim_mission_t` | ~16 (pointer + count + padding) |

The Python test `test_c_controller_python_struct_layout_matches_header_size` asserts `sizeof(sim_actuator_command_t) == 12`. Additional layout assertions can be added as needed.

---

## 13. Threading model

The simulator calls `controller_step` from a single thread (the simulator's main loop) — there is no concurrency from the C controller's perspective during sim runs.

On the flight computer, the typical layout is:
- Sensor ISRs (e.g. SPI completion) push into a sensor bus. These run on hardware interrupt priority.
- A timer ISR fires `controller_step` at `controller_rate_hz`. This runs on a lower priority than sensor ISRs (so sensors don't miss samples while the controller is computing).
- The controller may need to mask interrupts briefly when reading the latest sample, to avoid torn reads on multi-word fields.

The current `csdk` does not provide a sensor bus on the C side — sim builds the `sim_sensor_frame_t` directly. A C-side sensor bus (with proper memory barriers) will ship in a later release alongside the flight HAL.
