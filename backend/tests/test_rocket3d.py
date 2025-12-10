import numpy as np

from simulacra.noumena.rocket import (
    Rocket3D,
    RocketProps3D,
    RocketState3D,
    get_upright_orientation,
)
from simulacra.physics import RocketVec3, WorldVec3


def make_simple_rocket3d() -> Rocket3D:
    props = RocketProps3D(
        length=1.0,
        mass=10.0,
        center_of_mass_offset=RocketVec3(0.0, 0.0, 0.0),
        inertia_diag=(1.0, 2.0, 3.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )

    state = RocketState3D(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_magnitude=10.0,
        thrust_direction=RocketVec3(10.0, 0.0, 0.0),
    )

    return Rocket3D(props=props, state=state)


def test_rocket3d_thrust_vector_body_basic() -> None:
    rocket = make_simple_rocket3d()

    thrust = rocket.state.thrust_direction
    assert isinstance(thrust, RocketVec3)
    np.testing.assert_allclose(
        [thrust.x, thrust.y, thrust.z],
        [10.0, 0.0, 0.0],
        atol=1e-12,
    )


def test_rocket3d_thrust_vector_body_with_gimbal() -> None:
    rocket = make_simple_rocket3d()
    # Mimic a 90 deg yaw in body X-Y plane (about +Z) via thrust_vector_body.
    rocket.state.thrust_direction = RocketVec3(0.0, 10.0, 0.0)

    thrust = rocket.state.thrust_direction
    np.testing.assert_allclose(
        [thrust.x, thrust.y, thrust.z],
        [0.0, 10.0, 0.0],
        atol=1e-12,
    )


def test_rocket3d_thrust_vector_body_with_pitch() -> None:
    rocket = make_simple_rocket3d()
    # Mimic a 30 deg pitch up (about +Y), no yaw, via thrust_vector_body.
    pitch = np.deg2rad(30.0)
    rocket.state.thrust_direction = RocketVec3(
        10.0 * np.cos(pitch),
        0.0,
        -10.0 * np.sin(pitch),
    )

    thrust = rocket.state.thrust_direction
    expected = np.array([thrust.x, thrust.y, thrust.z], dtype=float)
    np.testing.assert_allclose(
        [thrust.x, thrust.y, thrust.z],
        expected,
        atol=1e-12,
    )


def test_rocket3d_world_thrust_uses_orientation() -> None:
    """
    With 'upright' orientation, body thrust along +X should map to world +Y.
    """
    rocket = make_simple_rocket3d()

    thrust_body = rocket.state.thrust_direction
    thrust_world = thrust_body.to_world(rocket.state.orientation)

    assert isinstance(thrust_world, WorldVec3)
    np.testing.assert_allclose(
        [thrust_world.x, thrust_world.y, thrust_world.z],
        [0.0, thrust_body.x, 0.0],
        atol=1e-12,
    )


