import numpy as np

from simulacra.physics import Orientation, RocketVec3, WorldVec3


def test_rocketvec3_to_world_identity() -> None:
    """
    With identity orientation, body and world frames coincide.
    """
    o = Orientation.identity()
    v_body = RocketVec3(1.0, 2.0, 3.0)

    v_world = v_body.to_world(o)

    assert isinstance(v_world, WorldVec3)
    np.testing.assert_allclose(
        [v_world.x, v_world.y, v_world.z],
        [v_body.x, v_body.y, v_body.z],
        atol=1e-12,
    )


def test_rocketvec3_to_world_rotate_about_z() -> None:
    """
    90-degree rotation about +Z should map body X to world Y, etc.
    """
    angle = np.pi / 2.0  # 90 deg
    w = np.cos(angle / 2.0)
    z = np.sin(angle / 2.0)
    o = Orientation.from_wxyz(w, 0.0, 0.0, z)

    # Body X axis should go to world Y axis
    v_body = RocketVec3(1.0, 0.0, 0.0)
    v_world = v_body.to_world(o)
    np.testing.assert_allclose(
        [v_world.x, v_world.y, v_world.z],
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_rocketvec3_to_world_pitch_about_y() -> None:
    """
    Pitching the rocket up by +theta about body/world Y axis:
    body +X (nose) should get a +Z component in world.
    """
    theta = np.deg2rad(30.0)  # 30 deg pitch up
    w = np.cos(theta / 2.0)
    y = np.sin(theta / 2.0)
    o = Orientation.from_wxyz(w, 0.0, y, 0.0)

    thrust_body = RocketVec3(1.0, 0.0, 0.0)  # along rocket nose (+X in body)
    thrust_world = thrust_body.to_world(o)

    # Right-hand rotation about +Y: X rotates toward -Z.
    expected = np.array([np.cos(theta), 0.0, -np.sin(theta)], dtype=float)

    np.testing.assert_allclose(
        [thrust_world.x, thrust_world.y, thrust_world.z],
        expected,
        atol=1e-12,
    )


def test_rocketvec3_to_world_roll_about_x() -> None:
    """
    Rolling the rocket 90 deg about +X should map body +Y to body/world +Z.

    Interpreting +X as nose, +Y as right wing, +Z as down:
    a vector pointing to the right in body frame should end up pointing down
    after a right-roll.
    """
    phi = np.pi / 2.0  # 90 deg roll about X
    w = np.cos(phi / 2.0)
    x = np.sin(phi / 2.0)
    o = Orientation.from_wxyz(w, x, 0.0, 0.0)

    v_body = RocketVec3(0.0, 1.0, 0.0)  # +Y in body frame
    v_world = v_body.to_world(o)

    np.testing.assert_allclose(
        [v_world.x, v_world.y, v_world.z],
        [0.0, 0.0, 1.0],
        atol=1e-12,
    )

