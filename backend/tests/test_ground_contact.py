"""Tests for ground-contact tip-over physics."""

import math

import numpy as np

from simulacra.noumena.kernel import Kernel, GroundContact
from simulacra.noumena.rocket import Rocket, RocketProps, RocketState, get_upright_orientation
from simulacra.noumena.world import Environment
from simulacra.physics import Orientation, RocketVec3, WorldVec3


def _make_grounded_rocket(
    orientation: Orientation,
    angular_velocity: RocketVec3 | None = None,
) -> tuple[Rocket, Environment]:
    """
    A rocket sitting on the ground (y=0) with the given orientation.

    Uses zero drag so we can isolate the ground-contact behaviour.
    """
    props = RocketProps(
        length=2.0,
        mass=10.0,
        center_of_mass_offset=RocketVec3(1.0, 0.0, 0.0),
        inertia_diag=(1.0, 10.0, 10.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        drag_coefficient=0.0,
        reference_area=0.0,
    )
    state = RocketState(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=orientation,
        angular_velocity_body=angular_velocity if angular_velocity is not None else RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),
    )
    env = Environment(
        gravity=WorldVec3(0.0, -9.81, 0.0),
        wind=WorldVec3(0.0, 0.0, 0.0),
    )
    return Rocket(props=props, state=state), env


def test_rocket_tips_over_when_tilted():
    """A tilted rocket on the ground acquires angular velocity and eventually lies flat."""
    # Tilt 10 degrees from upright (upright = +90° about Z, so tilt = 80° about Z)
    tilt_angle = np.radians(80.0)
    orientation = Orientation.from_axis_angle([0.0, 0.0, 1.0], tilt_angle)
    rocket, env = _make_grounded_rocket(orientation)
    k = Kernel(rocket, env)

    dt = 0.01
    tipped = False
    for _ in range(500):
        k.step(dt)
        # Check if nose has gone below horizontal
        nose_world = RocketVec3(1.0, 0.0, 0.0).to_world(rocket.state.orientation)
        if nose_world.y <= 0.0:
            tipped = True
            break

    assert tipped, "Tilted rocket should tip over and lie flat"


def test_upright_rocket_stays_upright():
    """A perfectly vertical rocket on the ground doesn't tip (torque ~= 0)."""
    orientation = get_upright_orientation()
    rocket, env = _make_grounded_rocket(orientation)
    k = Kernel(rocket, env)

    dt = 0.01
    for _ in range(100):
        k.step(dt)

    # Nose should still point nearly straight up.
    nose_world = RocketVec3(1.0, 0.0, 0.0).to_world(rocket.state.orientation)
    np.testing.assert_allclose(nose_world.y, 1.0, atol=1e-6,
                               err_msg="Upright rocket should remain upright")


def test_lying_down_freezes_state():
    """Once tipped past horizontal, velocity and angular velocity are zero."""
    # Orientation: body +X points along world -Y (nose pointing down).
    orientation = Orientation.from_axis_angle([0.0, 0.0, 1.0], np.radians(-90.0))
    rocket, env = _make_grounded_rocket(
        orientation,
        angular_velocity=RocketVec3(0.0, 0.0, 1.0),
    )
    # Give it some velocity too
    rocket.state.velocity = WorldVec3(5.0, 0.0, 0.0)

    k = Kernel(rocket, env)
    k.step(0.01)

    # Should be frozen
    np.testing.assert_allclose(rocket.state.velocity.norm, 0.0, atol=1e-12)
    np.testing.assert_allclose(rocket.state.angular_velocity_body.norm, 0.0, atol=1e-12)


# ============================================================================
# New tests: ground contact physics + landing metrics
# ============================================================================


def _make_falling_rocket(
    orientation: Orientation,
    velocity: WorldVec3,
    height: float = 0.5,
) -> tuple[Rocket, Environment]:
    """A rocket falling toward the ground from the given height."""
    props = RocketProps(
        length=2.0,
        mass=10.0,
        center_of_mass_offset=RocketVec3(1.0, 0.0, 0.0),
        inertia_diag=(1.0, 10.0, 10.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        drag_coefficient=0.0,
        reference_area=0.0,
    )
    state = RocketState(
        position=WorldVec3(0.0, height, 0.0),
        velocity=velocity,
        orientation=orientation,
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_body=RocketVec3(0.0, 0.0, 0.0),
    )
    env = Environment(
        gravity=WorldVec3(0.0, -9.81, 0.0),
        wind=WorldVec3(0.0, 0.0, 0.0),
    )
    return Rocket(props=props, state=state), env


def test_grounded_rocket_lifts_off_with_sufficient_thrust():
    """Thrust > weight → on_ground = False, position rises."""
    orientation = get_upright_orientation()
    rocket, env = _make_grounded_rocket(orientation)
    k = Kernel(rocket, env)

    # Apply thrust exceeding weight (weight = 10 * 9.81 = 98.1 N)
    # Body +X maps to world +Y when upright, so thrust along body +X = thrust up.
    rocket.state.thrust_body = RocketVec3(200.0, 0.0, 0.0)

    dt = 0.01
    for _ in range(10):
        k.step(dt)

    assert not k.ground_contact.on_ground, "Rocket should have lifted off"
    assert rocket.state.position.y > 0.0, "Position should be above ground"


def test_thrust_reduces_tipping_rate():
    """A tilted rocket with near-hover thrust tips slower than one with no thrust."""
    tilt_angle = np.radians(80.0)  # 10 degrees from upright
    orientation = Orientation.from_axis_angle([0.0, 0.0, 1.0], tilt_angle)

    # No-thrust case
    rocket_no_thrust, env = _make_grounded_rocket(orientation)
    k_no_thrust = Kernel(rocket_no_thrust, env)

    # Near-hover thrust case (90% of weight along body +X)
    rocket_thrust, env2 = _make_grounded_rocket(orientation)
    # Weight = 10 * 9.81 = 98.1 N. Apply 90 N along body +X.
    rocket_thrust.state.thrust_body = RocketVec3(90.0, 0.0, 0.0)
    k_thrust = Kernel(rocket_thrust, env2)

    dt = 0.01
    steps = 50
    for _ in range(steps):
        k_no_thrust.step(dt)
        k_thrust.step(dt)

    omega_no_thrust = rocket_no_thrust.state.angular_velocity_body.norm
    omega_thrust = rocket_thrust.state.angular_velocity_body.norm

    assert omega_thrust < omega_no_thrust, (
        f"Near-hover thrust should reduce tipping rate: "
        f"no_thrust={omega_no_thrust:.4f}, thrust={omega_thrust:.4f}"
    )


def test_impact_metrics_latched_on_touchdown():
    """Drop a rocket → impact_speed matches velocity at contact."""
    orientation = get_upright_orientation()
    rocket, env = _make_falling_rocket(
        orientation,
        velocity=WorldVec3(0.0, -5.0, 0.0),
        height=0.01,  # Very close to ground, hits next step
    )
    k = Kernel(rocket, env)

    dt = 0.01
    # Step until grounded
    for _ in range(100):
        k.step(dt)
        if k.ground_contact.on_ground:
            break

    assert k.ground_contact.on_ground
    assert k.ground_contact.impact_speed > 0.0, "Impact speed should be recorded"
    assert k.ground_contact.impact_vertical_speed > 0.0, "Vertical speed should be recorded"


def test_landing_angle_upright():
    """Upright drop → landing_angle ≈ 0."""
    orientation = get_upright_orientation()
    rocket, env = _make_falling_rocket(
        orientation,
        velocity=WorldVec3(0.0, -2.0, 0.0),
        height=0.01,
    )
    k = Kernel(rocket, env)

    dt = 0.01
    for _ in range(100):
        k.step(dt)
        if k.ground_contact.on_ground:
            break

    assert k.ground_contact.on_ground
    np.testing.assert_allclose(
        k.ground_contact.landing_angle, 0.0, atol=0.01,
        err_msg="Upright rocket should have near-zero landing angle",
    )


def test_landing_angle_tilted():
    """Tilted drop → landing_angle ≈ tilt_angle."""
    tilt_from_vertical = np.radians(30.0)
    # Upright is 90° about Z; tilting 30° from vertical means 60° about Z.
    rot_angle = np.radians(90.0) - tilt_from_vertical
    orientation = Orientation.from_axis_angle([0.0, 0.0, 1.0], rot_angle)

    rocket, env = _make_falling_rocket(
        orientation,
        velocity=WorldVec3(0.0, -2.0, 0.0),
        height=0.01,
    )
    k = Kernel(rocket, env)

    dt = 0.01
    for _ in range(100):
        k.step(dt)
        if k.ground_contact.on_ground:
            break

    assert k.ground_contact.on_ground
    np.testing.assert_allclose(
        k.ground_contact.landing_angle, tilt_from_vertical, atol=0.05,
        err_msg="Landing angle should match the tilt from vertical",
    )


def test_metrics_reset_on_liftoff():
    """Touch down, then thrust up → metrics reset."""
    orientation = get_upright_orientation()
    rocket, env = _make_falling_rocket(
        orientation,
        velocity=WorldVec3(0.0, -2.0, 0.0),
        height=0.01,
    )
    k = Kernel(rocket, env)

    dt = 0.01
    # Drop until grounded
    for _ in range(100):
        k.step(dt)
        if k.ground_contact.on_ground:
            break

    assert k.ground_contact.on_ground
    assert k.ground_contact.impact_speed > 0.0

    # Now apply heavy thrust to lift off
    rocket.state.thrust_body = RocketVec3(200.0, 0.0, 0.0)
    for _ in range(20):
        k.step(dt)

    assert not k.ground_contact.on_ground, "Should have lifted off"
    assert k.ground_contact.impact_speed == 0.0, "Impact speed should be reset"
    assert k.ground_contact.landing_angle == 0.0, "Landing angle should be reset"


def test_on_ground_and_tipped_flags():
    """Verify flag transitions through a full tip-over sequence."""
    # Start tilted 10° from upright
    tilt_angle = np.radians(80.0)
    orientation = Orientation.from_axis_angle([0.0, 0.0, 1.0], tilt_angle)
    rocket, env = _make_grounded_rocket(orientation)
    k = Kernel(rocket, env)

    dt = 0.01

    # Initially not on_ground (first step hasn't run)
    assert not k.ground_contact.on_ground
    assert not k.ground_contact.tipped_over

    # After first step, should be on ground
    k.step(dt)
    assert k.ground_contact.on_ground
    assert not k.ground_contact.tipped_over

    # Run until tipped
    for _ in range(500):
        k.step(dt)
        if k.ground_contact.tipped_over:
            break

    assert k.ground_contact.on_ground
    assert k.ground_contact.tipped_over


def test_lateral_velocity_zeroed_on_ground():
    """Rocket touching down with horizontal velocity → lateral velocity zeroed."""
    orientation = get_upright_orientation()
    rocket, env = _make_falling_rocket(
        orientation,
        velocity=WorldVec3(10.0, -2.0, 5.0),
        height=0.01,
    )
    k = Kernel(rocket, env)

    dt = 0.01
    for _ in range(100):
        k.step(dt)
        if k.ground_contact.on_ground:
            break

    assert k.ground_contact.on_ground
    np.testing.assert_allclose(rocket.state.velocity.x, 0.0, atol=1e-12,
                               err_msg="Lateral X velocity should be zeroed on ground")
    np.testing.assert_allclose(rocket.state.velocity.z, 0.0, atol=1e-12,
                               err_msg="Lateral Z velocity should be zeroed on ground")
