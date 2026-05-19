"""
Mass distribution model for a rocket.

A rocket is a collection of named mass components (nose cone, engine, fuel
tanks, etc.). Total mass, centre of mass, and inertia are computed from
the components. Fuel components deplete over time, shifting the CoM and
reducing mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ..physics import RocketVec3
from ..physics.inertia import (
    total_mass as _total_mass,
    center_of_mass as _center_of_mass,
    inertia_diag as _inertia_diag,
)


@dataclass
class MassComponent:
    """A single point-mass element in the rocket body frame."""

    name: str
    mass: float  # kg
    position: RocketVec3  # body-frame, meters from base


@dataclass
class FuelComponent(MassComponent):
    """A mass component that depletes when the engine fires."""

    full_mass: float = 0.0  # kg at 100%
    burn_rate: float = 0.0  # kg/s when engine fires


@dataclass
class MassModel:
    """
    Compositional mass model: a list of named point-mass components.

    Computes total mass, centre of mass, and diagonal inertia from the
    component list. Fuel components can be depleted via burn_fuel().
    """

    components: list[MassComponent] = field(default_factory=list)

    @classmethod
    def constant(cls, mass: float, position: RocketVec3) -> MassModel:
        """
        Create a single-component constant-mass model (no fuel).

        burn_fuel() is a no-op because there are no FuelComponents.
        """
        return cls(components=[MassComponent("body", mass, position)])

    def _mass_position_lists(
        self,
    ) -> tuple[list[float], list[tuple[float, float, float]]]:
        """Extract parallel (mass, position-tuple) lists from components."""
        masses = [c.mass for c in self.components]
        positions = [(c.position.x, c.position.y, c.position.z) for c in self.components]
        return masses, positions

    @property
    def total_mass(self) -> float:
        """Sum of all component masses."""
        masses, _ = self._mass_position_lists()
        return _total_mass(masses)

    @property
    def center_of_mass(self) -> RocketVec3:
        """Mass-weighted average position in body frame."""
        masses, positions = self._mass_position_lists()
        x, y, z = _center_of_mass(masses, positions)
        return RocketVec3(x, y, z)

    @property
    def inertia_diag(self) -> Tuple[float, float, float]:
        """
        Diagonal inertia tensor about the centre of mass, via the
        parallel-axis theorem applied to point masses.
        """
        masses, positions = self._mass_position_lists()
        com = _center_of_mass(masses, positions)
        return _inertia_diag(masses, positions, com)

    def set_fuel_fraction(self, fraction: float) -> None:
        """
        Set all fuel components to the given fraction (0.0 = empty, 1.0 = full).
        """
        fraction = max(0.0, min(1.0, fraction))
        for c in self.components:
            if isinstance(c, FuelComponent):
                c.mass = c.full_mass * fraction

    def burn_fuel(self, dt: float) -> float:
        """
        Deplete fuel components for dt seconds at their burn rates.

        Each FuelComponent loses burn_rate * dt kg, clamped at zero.
        The burn is distributed proportionally if multiple fuel components
        exist (each burns independently at its own rate).

        Returns the total mass burned (kg).
        """
        total_burned = 0.0
        for c in self.components:
            if isinstance(c, FuelComponent) and c.mass > 0.0:
                burn_amount = min(c.mass, c.burn_rate * dt)
                c.mass -= burn_amount
                total_burned += burn_amount
        return total_burned

    @property
    def fuel_mass(self) -> float:
        """Current total fuel mass."""
        return sum(c.mass for c in self.components if isinstance(c, FuelComponent))

    @property
    def fuel_capacity(self) -> float:
        """Total fuel capacity (sum of full_mass)."""
        return sum(
            c.full_mass for c in self.components if isinstance(c, FuelComponent)
        )

    @property
    def fuel_fraction(self) -> float | None:
        """Current fuel level as a fraction 0..1, or None if no fuel system."""
        cap = self.fuel_capacity
        if cap == 0.0:
            return None
        return self.fuel_mass / cap

    def to_rocket_props(
        self,
        length: float,
        thrust_application_offset: RocketVec3,
        drag_coefficient: float = 0.5,
        reference_area: float = 0.031416,
    ) -> "RocketProps":
        """
        Build a RocketProps snapshot from the current mass distribution.
        """
        from .rocket import RocketProps

        return RocketProps(
            length=length,
            mass=self.total_mass,
            center_of_mass_offset=self.center_of_mass,
            inertia_diag=self.inertia_diag,
            thrust_application_offset=thrust_application_offset,
            drag_coefficient=drag_coefficient,
            reference_area=reference_area,
        )


def default_sounding_rocket() -> MassModel:
    """
    Factory: a default sounding rocket mass model based on real
    sounding-rocket proportions. 120 kg total, 6 m length, ~42% propellant.

    Component positions are along body X (nose direction), measured from
    the base (nozzle end = 0 m, nose = 6 m).
    """
    return MassModel(
        components=[
            MassComponent("Nose cone", 5.0, RocketVec3(5.5, 0.0, 0.0)),
            MassComponent("Avionics", 8.0, RocketVec3(4.5, 0.0, 0.0)),
            MassComponent("Payload", 10.0, RocketVec3(3.8, 0.0, 0.0)),
            FuelComponent(
                "Oxidizer tank", 20.0, RocketVec3(3.0, 0.0, 0.0),
                full_mass=20.0, burn_rate=4.0,
            ),
            FuelComponent(
                "Fuel tank", 30.0, RocketVec3(2.0, 0.0, 0.0),
                full_mass=30.0, burn_rate=6.0,
            ),
            MassComponent("Structure", 15.0, RocketVec3(2.5, 0.0, 0.0)),
            MassComponent("Engine", 25.0, RocketVec3(0.3, 0.0, 0.0)),
            MassComponent("Nozzle", 7.0, RocketVec3(0.0, 0.0, 0.0)),
        ]
    )
