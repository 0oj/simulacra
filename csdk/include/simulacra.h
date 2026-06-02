/*
 * simulacra.h — public C interface for engineer-written controllers.
 *
 * To write a controller in C:
 *   1. #include "simulacra.h" in your .c file.
 *   2. Implement controller_reset, controller_step, controller_rate_hz.
 *   3. Build as a shared library (.so / .dylib / .dll).
 *   4. Load and run it inside simulacra:
 *        from simulacra.onboard.c_controller import CController
 *        ctrl = CController("path/to/libmy_controller.so")
 *        from simulacra.scenarios import hopper_short
 *        hopper_short.run(ctrl)
 *
 * The same controller_step you write here is the function that will
 * run on the microcontroller. The sim and the flight build call into
 * the same C source.
 *
 * ABI version: 0.1.0
 *
 * Conventions
 * -----------
 *  - Floats are float32 (single precision). Sufficient for everything
 *    in the control loop; matches MCU FPU width on Cortex-M4F+.
 *  - All vectors are world-frame unless their name says otherwise
 *    (imu_accel and imu_gyro are body-frame).
 *  - All angles are radians, all times are seconds.
 *  - Right-handed coordinate system. World +Y is up.
 *  - Quaternions are stored as (w, x, y, z) and map body→world.
 */
#ifndef SIMULACRA_H
#define SIMULACRA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIM_ABI_VERSION_MAJOR  0
#define SIM_ABI_VERSION_MINOR  1
#define SIM_ABI_VERSION_PATCH  0


/* ───────────────────────── Primitive types ───────────────────────── */

typedef struct {
    float x;
    float y;
    float z;
} sim_vec3_t;

typedef struct {
    float w;
    float x;
    float y;
    float z;
} sim_quat_t;


/* ─────────────────────── Stamped sensor samples ────────────────────
 * Each sensor sample carries its production timestamp and a sequence
 * counter so the controller can detect "did this update since I last
 * looked?" without comparing timestamps directly.
 */
typedef struct {
    sim_vec3_t value;
    float      t;
    uint32_t   seq;
} sim_stamped_vec3_t;

typedef struct {
    float    value;
    float    t;
    uint32_t seq;
} sim_stamped_float_t;

typedef struct {
    sim_quat_t value;
    float      t;
    uint32_t   seq;
} sim_stamped_quat_t;


/* ──────────────────────────── Sensor frame ────────────────────────
 * Snapshot of all sensor topics at one controller tick.
 *
 * gps_valid is 1 when GPS lock is good; 0 when lock is lost. When
 * 0, gps.value is undefined and must not be read.
 */
typedef struct {
    float               t_now;
    sim_stamped_vec3_t  imu_accel;      /* body frame, m/s^2 */
    sim_stamped_vec3_t  imu_gyro;       /* body frame, rad/s */
    sim_stamped_float_t baro;           /* altitude m */
    sim_stamped_vec3_t  gps;            /* world frame position, m */
    uint8_t             gps_valid;
    sim_stamped_vec3_t  velocity;       /* world frame, m/s */
    sim_stamped_quat_t  orientation;    /* body -> world */
} sim_sensor_frame_t;


/* ─────────────────────────── Actuator command ─────────────────────
 * What the controller emits each tick.
 * throttle is fractional [0..1]; gimbal angles are in radians.
 */
typedef struct {
    float throttle;
    float gimbal_pitch;
    float gimbal_yaw;
} sim_actuator_command_t;


/* ────────────────────────── Vehicle params ────────────────────────
 * Design-time vehicle properties. Runtime mass changes as fuel burns
 * and must be sensed, not read from here.
 */
typedef struct {
    float      dry_mass;            /* kg, structure + payload */
    float      full_fuel_mass;      /* kg, fuel at t=0 */
    sim_vec3_t inertia_dry;         /* kg·m^2, principal diag, dry */
    float      length;              /* m, nose to base */
    float      max_thrust;          /* N */
    float      max_gimbal_angle;    /* rad */
    float      max_gimbal_rate;     /* rad/s */
} sim_vehicle_params_t;


/* ───────────────────────────── Mission ────────────────────────────
 * A waypoint sequence the rocket should visit.
 * `waypoints` points to a contiguous array of `n_waypoints` waypoints.
 * The controller does not have to track which waypoint is current —
 * guidance produces the setpoint stream. Controllers that want to
 * preview the full mission may walk this array.
 */
typedef struct {
    sim_vec3_t position;
    float      hold_time;        /* s, hover duration after arrival */
    float      acceptance_radius;/* m, "arrived" distance */
    float      max_speed;        /* m/s, approach speed limit */
} sim_waypoint_t;

typedef struct {
    const sim_waypoint_t* waypoints;
    uint32_t              n_waypoints;
} sim_mission_t;


/* ───────────────────────── Setpoint (from guidance) ──────────────
 * What guidance wants the rocket to do RIGHT NOW.
 * Controllers may ignore individual fields:
 *   - Pure-position tracker:   uses position, ignores velocity, accel_ff
 *   - Attitude-only controller: uses attitude, ignores position
 *
 * Controllers MAY look at `frame->t_now` to know the wall time, but
 * the dt is provided so controllers can be tested with synthetic
 * frames at non-monotone t values.
 */
typedef struct {
    sim_vec3_t position;
    sim_vec3_t velocity;
    sim_vec3_t accel_ff;
    sim_quat_t attitude;
} sim_setpoint_t;


/* ───────────────────────────── Controller API ────────────────────
 * Implement these three functions in your controller's .c file.
 * They will be called from a Python simulator (today) or from a
 * timer ISR on the flight computer (later).
 */

/*
 * Returns the controller's tick rate in Hz. Called once before
 * controller_reset. Must return a positive number.
 */
float controller_rate_hz(void);

/*
 * Called once before the scenario starts. Use this to:
 *  - Cache vehicle params (mass, max_thrust, inertia).
 *  - Reset internal PID integrals to zero.
 *  - Compute design-time control gains (e.g. solve LQR Riccati).
 *
 * `mission` may be NULL if no mission is associated with this run.
 */
void controller_reset(
    const sim_mission_t*        mission,
    const sim_vehicle_params_t* vehicle
);

/*
 * Called every 1/controller_rate_hz seconds.
 *
 * Inputs:
 *   frame    — latest sensor snapshot; check Stamped.seq to detect updates.
 *   setpoint — what guidance wants right now (position/velocity/etc).
 *   dt       — seconds since the previous step (always 1/rate_hz in sim).
 *
 * Output:
 *   out_command — write throttle and gimbal angles here.
 *
 * MUST be deterministic. MUST NOT call malloc, file I/O, or any
 * non-real-time-safe function. Aim for << dt of compute time.
 */
void controller_step(
    const sim_sensor_frame_t*   frame,
    const sim_setpoint_t*       setpoint,
    float                       dt,
    sim_actuator_command_t*     out_command
);


#ifdef __cplusplus
}
#endif

#endif /* SIMULACRA_H */
