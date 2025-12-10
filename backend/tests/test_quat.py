import numpy as np

from simulacra.physics.quat import Orientation


def test_identity_has_identity_matrix() -> None:
    o = Orientation.identity()
    R = o.rotation_matrix()
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_from_wxyz_normalizes() -> None:
    # 90 deg about Z: base quaternion
    base = Orientation.from_wxyz(
        np.cos(np.pi / 4),
        0.0,
        0.0,
        np.sin(np.pi / 4),
    )
    # Same rotation but scaled
    scaled = Orientation.from_wxyz(
        10.0 * np.cos(np.pi / 4),
        0.0,
        0.0,
        10.0 * np.sin(np.pi / 4),
    )
    np.testing.assert_allclose(base.as_array(), scaled.as_array(), atol=1e-12)


def test_conjugate_inverts_rotation_matrix() -> None:
    # 30 deg about X, 20 about Y, 10 about Z (arbitrary non-axis-aligned rotation)
    w = np.cos(0.3)
    xyz = np.array([0.1, 0.2, 0.3], dtype=float)
    xyz /= np.linalg.norm(xyz)
    x, y, z = np.sin(0.3) * xyz

    o = Orientation.from_wxyz(w, x, y, z)
    R = o.rotation_matrix()
    o_inv = o.conjugate()
    R_inv = o_inv.rotation_matrix()

    # R_inv should be inverse of R
    I = R @ R_inv
    np.testing.assert_allclose(I, np.eye(3), atol=1e-12)


def test_rotation_matrix_rotates_vector_about_z() -> None:
    # 90 degrees about Z should send X to Y
    angle = np.pi / 2.0
    w = np.cos(angle / 2.0)
    z = np.sin(angle / 2.0)
    o = Orientation.from_wxyz(w, 0.0, 0.0, z)
    R = o.rotation_matrix()

    v_body = np.array([1.0, 0.0, 0.0])
    v_world = R @ v_body
    np.testing.assert_allclose(v_world, np.array([0.0, 1.0, 0.0]), atol=1e-12)
