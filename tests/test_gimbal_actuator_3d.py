import numpy as np

from simulacra.noumena.rocket import (
    Rocket,
    RocketProps,
    RocketState,
)
from simulacra.phenomena.actuators import (
    GimbalActuator,
    GimbalCommand,
)
from simulacra.physics import Orientation, RocketVec3, WorldVec3


THRUST = 10.0


def make_rocket_for_gimbal(thrust_magnitude: float = THRUST) -> Rocket:
    props = RocketProps(
        length=1.0,
        mass=10.0,
        center_of_mass_offset=RocketVec3(0.0, 0.0, 0.0),
        inertia_diag=(1.0, 1.0, 1.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )

    state = RocketState(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=Orientation.identity(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(thrust_magnitude, 0.0, 0.0),
    )

    return Rocket(props=props, state=state)


def test_gimbal_actuator_3d_zero_angles_points_along_body_x() -> None:
    rocket = make_rocket_for_gimbal()
    actuator = GimbalActuator(
        rocket, max_outer_angle=np.pi / 2, max_inner_angle=np.pi / 2
    )

    actuator.apply(GimbalCommand(0.0, 0.0))
    actuator.step(0.01)  # Advance dynamics to apply commanded angles

    thrust = rocket.state.thrust_body
    np.testing.assert_allclose(
        [thrust.x, thrust.y, thrust.z],
        [THRUST, 0.0, 0.0],
        atol=1e-12,
    )


def test_gimbal_actuator_3d_outer_yaw_only() -> None:
    """
    With outer yaw of +90 deg about body +Z and inner angle 0,
    nozzle should point along body +Y.
    """
    rocket = make_rocket_for_gimbal()
    actuator = GimbalActuator(
        rocket, max_outer_angle=np.pi, max_inner_angle=np.pi / 2
    )

    actuator.apply(GimbalCommand(np.pi / 2.0, 0.0))
    actuator.step(0.01)  # Advance dynamics to apply commanded angles

    thrust = rocket.state.thrust_body
    np.testing.assert_allclose(
        [thrust.x, thrust.y, thrust.z],
        [0.0, THRUST, 0.0],
        atol=1e-12,
    )


def test_gimbal_actuator_3d_nested_pitch_after_yaw() -> None:
    """
    After a small yaw, a pitch about inner +Y should tilt thrust out of the plane.

    This verifies we are rotating about the inner frame's axis, not the original body axis.
    """
    rocket = make_rocket_for_gimbal()
    actuator = GimbalActuator(
        rocket, max_outer_angle=np.pi / 2, max_inner_angle=np.pi / 2
    )

    yaw = np.deg2rad(30.0)
    pitch = np.deg2rad(20.0)
    actuator.apply(GimbalCommand(yaw, pitch))
    actuator.step(0.01)  # Advance dynamics to apply commanded angles

    thrust = rocket.state.thrust_body
    # Magnitude should be preserved by the actuator.
    np.testing.assert_allclose(
        np.linalg.norm([thrust.x, thrust.y, thrust.z]),
        THRUST,
        atol=1e-12,
    )

    # Compare direction against SciPy composition done independently.
    from scipy.spatial.transform import Rotation as R

    r_outer = R.from_rotvec(yaw * np.array([0.0, 0.0, 1.0]))
    axis_inner_body = r_outer.apply([0.0, 1.0, 0.0])
    r_inner = R.from_rotvec(pitch * np.asarray(axis_inner_body, dtype=float))
    r_total = r_inner * r_outer

    expected_dir = r_total.apply([1.0, 0.0, 0.0])
    expected = THRUST * np.asarray(expected_dir, dtype=float)

    np.testing.assert_allclose(
        [thrust.x, thrust.y, thrust.z],
        expected,
        atol=1e-12,
    )
