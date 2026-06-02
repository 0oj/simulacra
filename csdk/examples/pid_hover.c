/*
 * pid_hover.c — reference PD altitude-hold controller in C.
 *
 * Holds the rocket at the altitude given by setpoint.position.y.
 * No gimbal control — assumes vertical flight only.
 *
 * This is the C twin of src/simulacra/onboard/reference/pid_hover.py.
 * The math is identical; the only differences are:
 *   - State is module-static instead of instance fields.
 *   - No allocations.
 *
 * Build (macOS):
 *     clang -O2 -fPIC -shared -I../include -o libpid_hover.dylib pid_hover.c
 * Build (Linux):
 *     gcc -O2 -fPIC -shared -I../include -o libpid_hover.so pid_hover.c
 */
#include "simulacra.h"

/* ────────── module state (in flight: one chip, one instance) ────────── */
static float s_weight_N    = 0.0f;
static float s_max_thrust  = 1.0f;
static const float s_Kp    = 10.0f;
static const float s_Kd    =  5.0f;
static const float s_g     = 9.81f;


/* ────────── Controller API implementation ────────── */

float controller_rate_hz(void)
{
    return 100.0f;
}

void controller_reset(
    const sim_mission_t*        mission,
    const sim_vehicle_params_t* vehicle)
{
    (void)mission;   /* not used by this controller */
    s_weight_N   = (vehicle->dry_mass + vehicle->full_fuel_mass) * s_g;
    s_max_thrust = vehicle->max_thrust;
}

void controller_step(
    const sim_sensor_frame_t*   frame,
    const sim_setpoint_t*       setpoint,
    float                       dt,
    sim_actuator_command_t*     out_command)
{
    (void)dt;

    float target_alt = setpoint->position.y;
    float alt        = frame->baro.value;
    float vy         = frame->velocity.value.y;

    float error    = target_alt - alt;
    float thrust_N = s_weight_N + s_Kp * error - s_Kd * vy;

    if (thrust_N < 0.0f)          thrust_N = 0.0f;
    if (thrust_N > s_max_thrust)  thrust_N = s_max_thrust;

    out_command->throttle     = (s_max_thrust > 0.0f) ? thrust_N / s_max_thrust : 0.0f;
    out_command->gimbal_pitch = 0.0f;
    out_command->gimbal_yaw   = 0.0f;
}
