"""
Tests for physics/inertia.py — pure math on plain floats and tuples.
"""

import pytest

from simulacra.physics.inertia import total_mass, center_of_mass, inertia_diag


# --- total_mass ---


def test_total_mass_single():
    assert total_mass([5.0]) == pytest.approx(5.0)


def test_total_mass_multiple():
    assert total_mass([10.0, 20.0, 5.0]) == pytest.approx(35.0)


def test_total_mass_empty():
    assert total_mass([]) == pytest.approx(0.0)


# --- center_of_mass ---


def test_com_single_mass():
    com = center_of_mass([10.0], [(3.0, 4.0, 5.0)])
    assert com == pytest.approx((3.0, 4.0, 5.0))


def test_com_symmetric():
    com = center_of_mass([10.0, 10.0], [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)])
    assert com == pytest.approx((2.0, 0.0, 0.0))


def test_com_weighted():
    # 30 kg at origin, 10 kg at (4, 0, 0) -> CoM at (1, 0, 0)
    com = center_of_mass([30.0, 10.0], [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)])
    assert com == pytest.approx((1.0, 0.0, 0.0))


def test_com_empty():
    com = center_of_mass([], [])
    assert com == pytest.approx((0.0, 0.0, 0.0))


def test_com_3d():
    com = center_of_mass(
        [1.0, 1.0],
        [(0.0, 0.0, 0.0), (2.0, 4.0, 6.0)],
    )
    assert com == pytest.approx((1.0, 2.0, 3.0))


# --- inertia_diag ---


def test_inertia_two_point_masses_on_x_axis():
    """
    Two 10 kg masses at x=0 and x=4, CoM at (2,0,0).
    Each is dx=2 from CoM, dy=dz=0.
    Iy = 2*10*4 = 80, Iz = 80, Ix floored.
    """
    masses = [10.0, 10.0]
    positions = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
    com = (2.0, 0.0, 0.0)
    Ix, Iy, Iz = inertia_diag(masses, positions, com)

    assert Iy == pytest.approx(80.0)
    assert Iz == pytest.approx(80.0)
    # Ix floored to 0.01 * (80+80)/2 = 0.8
    assert Ix == pytest.approx(0.8)


def test_inertia_off_axis():
    """Masses with non-zero y offset contribute to Ix."""
    masses = [5.0, 5.0]
    positions = [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]
    com = (0.0, 0.0, 0.0)
    Ix, Iy, Iz = inertia_diag(masses, positions, com)

    # Ix = 2 * 5 * 1^2 = 10
    assert Ix == pytest.approx(10.0)


def test_inertia_custom_min_fraction():
    """min_ix_fraction controls the floor on Ix."""
    masses = [10.0, 10.0]
    positions = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
    com = (2.0, 0.0, 0.0)

    # With fraction=0.1, floor = 0.1 * (80+80)/2 = 8.0
    Ix, _, _ = inertia_diag(masses, positions, com, min_ix_fraction=0.1)
    assert Ix == pytest.approx(8.0)


def test_inertia_single_mass_at_com():
    """A single mass at the CoM has zero distance, all inertias floored to minimum."""
    masses = [5.0]
    positions = [(1.0, 2.0, 3.0)]
    com = (1.0, 2.0, 3.0)
    Ix, Iy, Iz = inertia_diag(masses, positions, com)

    # All inertias are floored to a minimum value to prevent singularities
    # in Euler's equations. The minimum is based on min_ix_fraction * total_m * max_dist_sq.
    # With all masses at CoM, max_dist_sq is clamped to 1.0, so min = 0.01 * 5.0 * 1.0 = 0.05
    assert Ix == pytest.approx(0.05)
    assert Iy == pytest.approx(0.05)
    assert Iz == pytest.approx(0.05)
