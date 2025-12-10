import numpy as np

from simulacra.noumena.rocket import (
    Rocket3D,
    RocketProps3D,
    RocketState3D,
)
from simulacra.phenomena.actuators import (
    GimbalActuator3D,
    GimbalCommand3D,
)
from simulacra.physics import Orientation, RocketVec3, WorldVec3


def make_rocket3d_for_gimbal(thrust_magnitude: float = 10.0) -> Rocket3D:
    props = RocketProps3D(
        length=1.0,
        mass=10.0,
        center_of_mass_offset=RocketVec3(0.0, 0.0, 0.0),
        inertia_diag=(1.0, 1.0, 1.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )

    state = RocketState3D(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=Orientation.identity(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_magnitude=thrust_magnitude,
        thrust_direction=RocketVec3(thrust_magnitude, 0.0, 0.0),
    )

    return Rocket3D(props=props, state=state)


def test_gimbal_actuator_3d_zero_angles_points_along_body_x() -> None:
    rocket = make_rocket3d_for_gimbal()
    actuator = GimbalActuator3D(
        rocket, max_outer_angle=np.pi / 2, max_inner_angle=np.pi / 2
    )

    actuator.apply(GimbalCommand3D(0.0, 0.0))

    thrust = rocket.state.thrust_direction
    np.testing.assert_allclose(
        [thrust.x, thrust.y, thrust.z],
        [rocket.state.thrust_magnitude, 0.0, 0.0],
        atol=1e-12,
    )


def test_gimbal_actuator_3d_outer_yaw_only() -> None:
    """
    With outer yaw of +90 deg about body +Z and inner angle 0,
    nozzle should point along body +Y.
    """
    rocket = make_rocket3d_for_gimbal()
    actuator = GimbalActuator3D(
        rocket, max_outer_angle=np.pi, max_inner_angle=np.pi / 2
    )

    actuator.apply(GimbalCommand3D(np.pi / 2.0, 0.0))

    thrust = rocket.state.thrust_direction
    np.testing.assert_allclose(
        [thrust.x, thrust.y, thrust.z],
        [0.0, rocket.state.thrust_magnitude, 0.0],
        atol=1e-12,
    )


def test_gimbal_actuator_3d_nested_pitch_after_yaw() -> None:
    """
    After a small yaw, a pitch about inner +Y should tilt thrust out of the plane.

    This verifies we are rotating about the inner frame's axis, not the original body axis.
    """
    rocket = make_rocket3d_for_gimbal()
    actuator = GimbalActuator3D(
        rocket, max_outer_angle=np.pi / 2, max_inner_angle=np.pi / 2
    )

    yaw = np.deg2rad(30.0)
    pitch = np.deg2rad(20.0)
    actuator.apply(GimbalCommand3D(yaw, pitch))

    thrust = rocket.state.thrust_direction
    # Magnitude should equal thrust_magnitude.
    np.testing.assert_allclose(
        np.linalg.norm([thrust.x, thrust.y, thrust.z]),
        rocket.state.thrust_magnitude,
        atol=1e-12,
    )

    # Compare direction against SciPy composition done independently.
    from scipy.spatial.transform import Rotation as R

    r_outer = R.from_rotvec(yaw * np.array([0.0, 0.0, 1.0]))
    axis_inner_body = r_outer.apply([0.0, 1.0, 0.0])
    r_inner = R.from_rotvec(pitch * np.asarray(axis_inner_body, dtype=float))
    r_total = r_inner * r_outer

    expected_dir = r_total.apply([1.0, 0.0, 0.0])
    expected = rocket.state.thrust_magnitude * np.asarray(expected_dir, dtype=float)

    np.testing.assert_allclose(
        [thrust.x, thrust.y, thrust.z],
        expected,
        atol=1e-12,
    )
