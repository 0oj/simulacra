import numpy as np
import pytest

from simulacra.noumena.mass_model import (
    FuelComponent,
    MassComponent,
    MassModel,
    default_sounding_rocket,
)
from simulacra.noumena.rocket import RocketProps
from simulacra.physics import RocketVec3


def make_symmetric_model() -> MassModel:
    """Two equal masses at symmetric positions about the midpoint."""
    return MassModel(
        components=[
            MassComponent("left", 10.0, RocketVec3(0.0, 0.0, 0.0)),
            MassComponent("right", 10.0, RocketVec3(4.0, 0.0, 0.0)),
        ]
    )


def make_fuel_model() -> MassModel:
    """Model with one fixed and one fuel component."""
    return MassModel(
        components=[
            MassComponent("engine", 10.0, RocketVec3(0.0, 0.0, 0.0)),
            FuelComponent(
                "fuel", 20.0, RocketVec3(2.0, 0.0, 0.0),
                full_mass=20.0, burn_rate=5.0,
            ),
        ]
    )


# --- Total mass ---


def test_total_mass_sum_of_components():
    model = make_symmetric_model()
    assert model.total_mass == pytest.approx(20.0)


def test_total_mass_with_fuel():
    model = make_fuel_model()
    assert model.total_mass == pytest.approx(30.0)


# --- Centre of mass ---


def test_com_symmetric_at_midpoint():
    model = make_symmetric_model()
    com = model.center_of_mass
    np.testing.assert_allclose(
        [com.x, com.y, com.z], [2.0, 0.0, 0.0], atol=1e-12
    )


def test_com_shifts_toward_engine_as_fuel_depletes():
    model = make_fuel_model()
    com_full = model.center_of_mass.x

    model.set_fuel_fraction(0.0)
    com_empty = model.center_of_mass.x

    # Engine is at x=0, fuel is at x=2. With fuel gone, CoM moves toward engine.
    assert com_empty < com_full


def test_com_with_unequal_masses():
    model = MassModel(
        components=[
            MassComponent("heavy", 30.0, RocketVec3(0.0, 0.0, 0.0)),
            MassComponent("light", 10.0, RocketVec3(4.0, 0.0, 0.0)),
        ]
    )
    com = model.center_of_mass
    # Weighted average: (30*0 + 10*4) / 40 = 1.0
    np.testing.assert_allclose(com.x, 1.0, atol=1e-12)


# --- Inertia ---


def test_inertia_two_point_masses():
    """
    Two 10 kg masses at x=0 and x=4. CoM at x=2.
    Each is dx=2 from CoM, dy=dz=0.
    Ix = 0 (floored), Iy = 2*10*4 = 80, Iz = 2*10*4 = 80.
    """
    model = make_symmetric_model()
    Ix, Iy, Iz = model.inertia_diag

    np.testing.assert_allclose(Iy, 80.0, atol=1e-12)
    np.testing.assert_allclose(Iz, 80.0, atol=1e-12)
    # Ix is floored to 0.01 * (Iy + Iz) / 2
    assert Ix == pytest.approx(0.01 * (80.0 + 80.0) / 2.0)


def test_inertia_off_axis():
    """Mass with non-zero y offset contributes to Ix."""
    model = MassModel(
        components=[
            MassComponent("a", 5.0, RocketVec3(0.0, 1.0, 0.0)),
            MassComponent("b", 5.0, RocketVec3(0.0, -1.0, 0.0)),
        ]
    )
    Ix, Iy, Iz = model.inertia_diag
    # Each mass: dx=0, dy=1, dz=0 from CoM (0,0,0).
    # Ix = 2 * 5 * 1^2 = 10
    np.testing.assert_allclose(Ix, 10.0, atol=1e-12)


# --- Fuel operations ---


def test_set_fuel_fraction_full():
    model = make_fuel_model()
    model.set_fuel_fraction(0.5)
    fuel = [c for c in model.components if isinstance(c, FuelComponent)]
    for c in fuel:
        assert c.mass == pytest.approx(c.full_mass * 0.5)


def test_set_fuel_fraction_clamps():
    model = make_fuel_model()
    model.set_fuel_fraction(1.5)
    fuel = [c for c in model.components if isinstance(c, FuelComponent)]
    for c in fuel:
        assert c.mass == pytest.approx(c.full_mass)


def test_burn_fuel_reduces_mass():
    model = make_fuel_model()
    initial = model.total_mass
    burned = model.burn_fuel(1.0)
    assert burned == pytest.approx(5.0)
    assert model.total_mass == pytest.approx(initial - 5.0)


def test_burn_fuel_clamps_at_zero():
    model = make_fuel_model()
    # Burn way more than available
    model.burn_fuel(100.0)
    fuel = [c for c in model.components if isinstance(c, FuelComponent)]
    for c in fuel:
        assert c.mass == pytest.approx(0.0)


def test_fuel_fraction_property():
    model = make_fuel_model()
    assert model.fuel_fraction == pytest.approx(1.0)
    model.set_fuel_fraction(0.5)
    assert model.fuel_fraction == pytest.approx(0.5)


# --- to_rocket_props ---


def test_to_rocket_props_produces_valid_props():
    model = make_fuel_model()
    props = model.to_rocket_props(
        length=6.0,
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )
    assert isinstance(props, RocketProps)
    assert props.length == 6.0
    assert props.mass == pytest.approx(30.0)
    assert isinstance(props.center_of_mass_offset, RocketVec3)
    assert len(props.inertia_diag) == 3


# --- Default sounding rocket ---


def test_default_sounding_rocket_total_mass():
    model = default_sounding_rocket()
    assert model.total_mass == pytest.approx(120.0)


def test_default_sounding_rocket_fuel_fraction():
    model = default_sounding_rocket()
    assert model.fuel_fraction == pytest.approx(1.0)
    # 50 kg of propellant out of 120 kg total
    assert model.fuel_mass == pytest.approx(50.0)


def test_default_sounding_rocket_com_in_range():
    model = default_sounding_rocket()
    com = model.center_of_mass
    # CoM should be somewhere in the middle of the 6m rocket
    assert 1.0 < com.x < 4.0


def test_empty_model_com_zero():
    model = MassModel(components=[])
    com = model.center_of_mass
    np.testing.assert_allclose([com.x, com.y, com.z], [0.0, 0.0, 0.0])
