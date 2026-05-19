"""Step-by-step debug of gimbal attitude control."""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from simulacra.gym_env import RocketLandingEnvV2
from simulacra.physics import RocketVec3


def rotate_body_to_world(quat, vec):
    w, x, y, z = quat
    qv = np.array([x, y, z])
    v = np.array(vec, dtype=float)
    cross1 = np.cross(qv, v)
    cross2 = np.cross(qv, cross1)
    return v + 2.0 * w * cross1 + 2.0 * cross2


def rotate_world_to_body(quat, vec):
    w, x, y, z = quat
    qv = np.array([-x, -y, -z])
    v = np.array(vec, dtype=float)
    cross1 = np.cross(qv, v)
    cross2 = np.cross(qv, cross1)
    return v + 2.0 * w * cross1 + 2.0 * cross2


env = RocketLandingEnvV2(
    init_altitude_range=(30., 31.),
    init_velocity_range=(0., 0.),
    init_horizontal_range=(0., 0.),
    init_tilt_range=(0.05, 0.05),  # ~3 degrees
)

obs, _ = env.reset(seed=42)

# Also get the TRUE state for comparison
true_orient = env._kernel.rocket.state.orientation
true_quat = true_orient.as_array()
nose_true = RocketVec3(1, 0, 0).to_world(true_orient)

obs_quat = (float(obs[6]), float(obs[7]), float(obs[8]), float(obs[9]))
nose_obs = rotate_body_to_world(obs_quat, [1, 0, 0])

print(f"Initial state:")
print(f"  True quat: {true_quat}")
print(f"  Obs quat:  {obs_quat}")
print(f"  Nose (true): {nose_true.x:.4f}, {nose_true.y:.4f}, {nose_true.z:.4f}")
print(f"  Nose (obs):  {nose_obs[0]:.4f}, {nose_obs[1]:.4f}, {nose_obs[2]:.4f}")
print(f"  True tilt: {math.degrees(env._get_tilt_angle()):.2f}°")
print(f"  Obs omega: wx={obs[10]:.4f}, wy={obs[11]:.4f}, wz={obs[12]:.4f}")
print(f"  Gimbal state: outer={env._actuators.gimbal_outer_angle:.4f}, inner={env._actuators.gimbal_inner_angle:.4f}")

# Now step with the LandingController's exact attitude control
att_Kp = 500.0
att_Kd = 200.0
max_thrust = 2000.0
moment_arm = 2.0
max_gimbal = 0.26
throttle = 0.55

print(f"\nStepping with LandingController attitude law (Kp={att_Kp}, Kd={att_Kd}):")
print(f"{'step':>4} {'tilt':>6} {'omega_z':>8} {'cmd_outer':>10} {'act_outer':>10} {'nose_x':>7} {'nose_y':>7}")

for step in range(30):
    # Compute gimbal from current obs (like LandingController)
    qw, qx, qy, qz = float(obs[6]), float(obs[7]), float(obs[8]), float(obs[9])
    wx, wy, wz = float(obs[10]), float(obs[11]), float(obs[12])
    quat = (qw, qx, qy, qz)

    nose_world = rotate_body_to_world(quat, [1.0, 0.0, 0.0])
    desired = np.array([0.0, 1.0, 0.0])
    error_world = np.cross(nose_world, desired)
    error_body = rotate_world_to_body(quat, error_world)
    omega = np.array([wx, wy, wz])

    tau_feedback = att_Kp * error_body - att_Kd * omega

    thrust_n = throttle * max_thrust
    if thrust_n > 100:
        scale = 1.0 / (thrust_n * moment_arm)
        outer_cmd = float(np.clip(-tau_feedback[2] * scale, -max_gimbal, max_gimbal))
        inner_cmd = float(np.clip(-tau_feedback[1] * scale, -max_gimbal, max_gimbal))
    else:
        outer_cmd = 0.0
        inner_cmd = 0.0

    # Normalize to [-1, 1] for env
    outer_norm = outer_cmd / max_gimbal
    inner_norm = inner_cmd / max_gimbal

    # Step env
    action = np.array([throttle, outer_norm, inner_norm], dtype=np.float32)
    obs, _, t, tr, info = env.step(action)

    tilt_deg = info.get("tilt_deg", 0)
    actual_outer = env._actuators.gimbal_outer_angle
    actual_inner = env._actuators.gimbal_inner_angle

    # Get actual nose from obs
    nose_obs = rotate_body_to_world(
        (float(obs[6]), float(obs[7]), float(obs[8]), float(obs[9])),
        [1, 0, 0]
    )

    print(f"{step:4d} {tilt_deg:6.2f}° {wz:8.4f} {outer_cmd:10.5f} {actual_outer:10.5f} "
          f"{nose_obs[0]:7.4f} {nose_obs[1]:7.4f}")

    if t or tr:
        print(f"  Episode terminated at step {step}")
        break

# Also compare: what does the LandingController scenario do with the same init?
print(f"\n{'='*60}")
print("For reference: running LandingController on same scenario")
print(f"{'='*60}")

from simulacra.noumena import Kernel, Rocket, RocketProps, RocketState, Environment, MassModel, get_upright_orientation
from simulacra.noumena.mass_model import MassComponent, FuelComponent
from simulacra.phenomena.actuators import Actuators, GimbalActuator, ThrottleActuator
from simulacra.phenomena.sensors import Sensors
from simulacra.onboard.landing_controller import LandingController
from simulacra.physics import Orientation, WorldVec3
from scipy.spatial.transform import Rotation as R

# Create same initial conditions
init_tilt = 0.05
base_rot = get_upright_orientation()._as_scipy_rotation()
perturb = R.from_euler("y", init_tilt)
combined = perturb * base_rot
x, y, z, w = combined.as_quat()
init_orient = Orientation.from_wxyz(w, x, y, z)

init_state = RocketState(
    position=WorldVec3(0.0, 30.0, 0.0),
    velocity=WorldVec3(0.0, 0.0, 0.0),
    orientation=init_orient,
    angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
    thrust_body=RocketVec3(0.0, 0.0, 0.0),
)

mass_model = MassModel(components=[
    MassComponent("dry", 60.0, RocketVec3(2.0, 0, 0)),
    FuelComponent("fuel", mass=80.0, position=RocketVec3(1.2, 0, 0), full_mass=80.0, burn_rate=4.0),
])
props = mass_model.to_rocket_props(length=4.0, thrust_application_offset=RocketVec3(0, 0, 0))
rocket = Rocket(props=props, state=init_state, mass_model=mass_model)
environment = Environment(gravity=WorldVec3(0.0, -9.81, 0.0), wind=WorldVec3(0.0, 0.0, 0.0), air_density=1.225)

kernel = Kernel(rocket, environment)
sensors = Sensors(kernel)
gimbal_act = GimbalActuator(rocket=rocket, max_outer_angle=0.26, max_inner_angle=0.26, max_rate=1.0, tau=0.05)
throttle_act = ThrottleActuator(rocket=rocket, max_thrust_rate=4000.0, min_thrust=0.0)
actuators = Actuators(gimbal=gimbal_act, throttle=throttle_act)

controller = LandingController(throttle=throttle_act, gimbal=gimbal_act)
controller = controller.with_sensors(sensors)

print(f"{'step':>4} {'tilt':>6} {'omega_z':>8} {'gimbal_outer':>12} {'alt':>6}")

for step in range(30):
    sensors.set_time(step * 0.02)
    actuators.set_time(step * 0.02)

    # Run the landing controller
    controller.run()

    # Step actuators and physics
    actuators.step(0.02)
    kernel.step(0.02)
    sensors.update_attitude_estimator(0.02)

    tilt = math.degrees(math.acos(max(-1.0, min(1.0,
        RocketVec3(1, 0, 0).to_world(kernel.rocket.state.orientation).y))))
    omega = kernel.rocket.state.angular_velocity_body
    alt = kernel.rocket.state.position.y

    print(f"{step:4d} {tilt:6.2f}° {omega.z:8.4f} {gimbal_act.outer_angle:12.5f} {alt:6.1f}m")
