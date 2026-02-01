"""Tests for translational aerodynamic drag."""

import numpy as np

from simulacra.noumena.kernel import Kernel
from simulacra.noumena.rocket import Rocket, RocketProps, RocketState
from simulacra.noumena.world import Environment
from simulacra.physics import Orientation, RocketVec3, WorldVec3


def _make_rocket(
    velocity: WorldVec3,
    drag_coefficient: float = 0.5,
    reference_area: float = 0.031416,
    mass: float = 10.0,
) -> tuple[Rocket, Environment]:
    """Rocket in zero-g with the given initial velocity and drag params."""
    props = RocketProps(
        length=1.0,
        mass=mass,
        center_of_mass_offset=RocketVec3(0.5, 0.0, 0.0),
        inertia_diag=(1.0, 1.0, 1.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        drag_coefficient=drag_coefficient,
        reference_area=reference_area,
    )
    state = RocketState(
        position=WorldVec3(0.0, 100.0, 0.0),
        velocity=velocity,
        orientation=Orientation.identity(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),
    )
    env = Environment(
        gravity=WorldVec3(0.0, 0.0, 0.0),  # zero-g to isolate drag
        wind=WorldVec3(0.0, 0.0, 0.0),
    )
    return Rocket(props=props, state=state), env


def test_drag_reduces_speed():
    """A rocket moving horizontally in zero-g should lose speed due to drag."""
    v0 = WorldVec3(50.0, 0.0, 0.0)
    rocket, env = _make_rocket(velocity=v0)
    k = Kernel(rocket, env)

    dt = 0.01
    for _ in range(100):
        k.step(dt)

    speed_after = rocket.state.velocity.norm
    assert speed_after < 50.0, f"Speed should decrease, got {speed_after}"


def test_drag_zero_when_stationary():
    """No drag force when v=0."""
    rocket, env = _make_rocket(velocity=WorldVec3(0.0, 0.0, 0.0))
    k = Kernel(rocket, env)

    k.step(0.01)

    # In zero-g with no thrust and no velocity, the rocket shouldn't move.
    np.testing.assert_allclose(rocket.state.velocity.norm, 0.0, atol=1e-15)


def test_drag_scales_with_v_squared():
    """Drag at 2v should produce ~4× the deceleration vs drag at v (single step)."""
    dt = 0.001  # small step so velocity barely changes

    # Run at speed v
    v = 10.0
    rocket1, env1 = _make_rocket(velocity=WorldVec3(v, 0.0, 0.0))
    k1 = Kernel(rocket1, env1)
    k1.step(dt)
    dv1 = v - rocket1.state.velocity.x  # deceleration in one step

    # Run at speed 2v
    rocket2, env2 = _make_rocket(velocity=WorldVec3(2 * v, 0.0, 0.0))
    k2 = Kernel(rocket2, env2)
    k2.step(dt)
    dv2 = 2 * v - rocket2.state.velocity.x

    # dv2 / dv1 ≈ 4 (drag ∝ v²)
    ratio = dv2 / dv1
    np.testing.assert_allclose(ratio, 4.0, rtol=0.05)


def test_drag_zero_when_cd_zero():
    """Rockets with Cd=0 experience no drag."""
    v0 = WorldVec3(50.0, 0.0, 0.0)
    rocket, env = _make_rocket(velocity=v0, drag_coefficient=0.0)
    k = Kernel(rocket, env)

    dt = 0.01
    for _ in range(100):
        k.step(dt)

    # With zero drag in zero-g, speed should be unchanged.
    np.testing.assert_allclose(rocket.state.velocity.x, 50.0, atol=1e-12)
