import numpy as np

from simulacra.noumena.kernel import Kernel3D
from simulacra.noumena.rocket import Rocket3D, RocketProps3D, RocketState3D, get_upright_orientation
from simulacra.noumena.world import Environment3D
from simulacra.physics import RocketVec3, WorldVec3


def make_simple_rocket3d(mass: float = 10.0, thrust: float = 0.0) -> tuple[Rocket3D, Environment3D]:
    props = RocketProps3D(
        length=1.0,
        mass=mass,
        center_of_mass_offset=RocketVec3(0.0, 0.0, 0.0),
        inertia_diag=(1.0, 1.0, 1.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )

    state = RocketState3D(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_magnitude=thrust,
        thrust_direction=RocketVec3(thrust, 0.0, 0.0),
    )

    rocket = Rocket3D(props=props, state=state)
    env = Environment3D()
    return rocket, env


def test_kernel3d_free_fall_matches_gravity() -> None:
    mass = 10.0
    rocket, env = make_simple_rocket3d(mass=mass, thrust=0.0)
    k = Kernel3D(rocket, env)

    dt = 0.1
    k.step(dt)

    # Expect v_y = g * dt, with gravity along -Y.
    vy = rocket.state.velocity.y
    np.testing.assert_allclose(vy, env.gravity.y * dt, atol=1e-12)


def test_kernel3d_hover_when_thrust_equals_weight() -> None:
    mass = 10.0
    g = -9.81
    thrust = mass * (-g)

    rocket, env = make_simple_rocket3d(mass=mass, thrust=thrust)
    k = Kernel3D(rocket, env)

    dt = 0.1
    k.step(dt)

    # With upright orientation, thrust along body +X becomes world +Y.
    # If thrust == weight, net acceleration should be ~0.
    vy = rocket.state.velocity.y
    np.testing.assert_allclose(vy, 0.0, atol=1e-12)


