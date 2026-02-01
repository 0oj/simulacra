"""
Point-mass inertia calculations.

Pure math operating on plain floats and tuples — no domain types,
no numpy. Follows the same pattern as rigid_body.py.
"""

from __future__ import annotations

from typing import Sequence


def total_mass(masses: Sequence[float]) -> float:
    """Sum of all component masses."""
    return sum(masses)


def center_of_mass(
    masses: Sequence[float],
    positions: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """
    Mass-weighted average position.

    Returns (0, 0, 0) when total mass is zero.
    """
    total = sum(masses)
    if total == 0.0:
        return (0.0, 0.0, 0.0)
    wx = sum(m * p[0] for m, p in zip(masses, positions))
    wy = sum(m * p[1] for m, p in zip(masses, positions))
    wz = sum(m * p[2] for m, p in zip(masses, positions))
    return (wx / total, wy / total, wz / total)


def inertia_diag(
    masses: Sequence[float],
    positions: Sequence[tuple[float, float, float]],
    com: tuple[float, float, float],
    min_ix_fraction: float = 0.01,
) -> tuple[float, float, float]:
    """
    Diagonal inertia tensor about a given centre of mass, via the
    parallel-axis theorem applied to point masses.

    For a point mass at offset (dx, dy, dz) from the CoM:
        Ix += m * (dy² + dz²)
        Iy += m * (dx² + dz²)
        Iz += m * (dx² + dy²)

    With all components on the X-axis (y=z=0), roll inertia Ix would
    be exactly zero from point masses alone. We floor Ix to
    min_ix_fraction * (Iy + Iz) / 2 to avoid division-by-zero in
    Euler's equations.
    """
    Ix = 0.0
    Iy = 0.0
    Iz = 0.0
    for m, pos in zip(masses, positions):
        dx = pos[0] - com[0]
        dy = pos[1] - com[1]
        dz = pos[2] - com[2]
        Ix += m * (dy**2 + dz**2)
        Iy += m * (dx**2 + dz**2)
        Iz += m * (dx**2 + dy**2)

    # Floor roll inertia to avoid singularity
    min_Ix = min_ix_fraction * (Iy + Iz) / 2.0
    Ix = max(Ix, min_Ix)

    return (Ix, Iy, Iz)
