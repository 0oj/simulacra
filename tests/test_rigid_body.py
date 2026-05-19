import numpy as np

from simulacra.physics.quat import Orientation
from simulacra.physics.rigid_body import integrate_orientation_body_omega


def test_integrate_orientation_body_omega_yaw_matches_axis_angle() -> None:
    """
    Integrating a constant yaw rate for dt seconds should match an axis-angle
    rotation of omega_z * dt about +Z.
    """
    omega_z = 0.5  # rad/s
    dt = 2.0  # seconds
    angle = omega_z * dt

    o0 = Orientation.identity()
    o1 = integrate_orientation_body_omega(o0, [0.0, 0.0, omega_z], dt)

    o_expected = Orientation.from_axis_angle([0.0, 0.0, 1.0], angle)

    # Compare rotation matrices for robustness to quaternion sign.
    R1 = o1.rotation_matrix()
    R_expected = o_expected.rotation_matrix()
    np.testing.assert_allclose(R1, R_expected, atol=1e-12)


