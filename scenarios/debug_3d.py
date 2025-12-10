from simulacra.noumena import (
    Environment3D,
    Kernel3D,
    Rocket3D,
    RocketProps3D,
    RocketState3D,
    Scenario3D,
    get_upright_orientation,
)
from simulacra.onboard.controller import Controller, IdentityController
from simulacra.physics import RocketVec3, WorldVec3


def make_rocket() -> Rocket3D:
    props = RocketProps3D(
        length=6.0,
        mass=120.0,
        center_of_mass_offset=RocketVec3(3.0, 0.0, 0.0),
        inertia_diag=(50.0, 50.0, 10.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )

    state = RocketState3D(
        position=WorldVec3(0.0, 0.0, 0.0),
        velocity=WorldVec3(0.0, 0.0, 0.0),
        orientation=get_upright_orientation(),
        angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
        thrust_magnitude=0.0,
        thrust_direction=RocketVec3(1.0, 0.0, 0.0),
    )

    return Rocket3D(props=props, state=state)


rocket = make_rocket()
environment = Environment3D(gravity=WorldVec3(0.0, -9.81, 0.0))

kernel = Kernel3D(rocket=rocket, environment=environment)
controller = IdentityController()


def callback(t: float, kernel: Kernel3D, controller: Controller) -> None:
    """
    Placeholder callback for external analysis / future hooks.
    Currently unused.
    """
    ...


scenario = Scenario3D(
    kernel=kernel,
    controller=controller,
    callback=callback,
)


def throttle(to: float):
    def _throttle(t: float, k: Kernel3D, c: Controller) -> None:
        k.rocket.state.thrust_magnitude = to
        # Keep thrust pointing along body +X; controller/actuators can change this later.
        # k.rocket.state.thrust_direction = RocketVec3(1.0, 0.0, 0.0)

    return _throttle

def set_thrust_direction(direction: RocketVec3):
    def _set_thrust_direction(t: float, k: Kernel3D, c: Controller) -> None:
        k.rocket.state.thrust_direction = direction
    return _set_thrust_direction


# Simple thrust program: pulse the engine a few times.
scenario.add_event(1.0, throttle(4000.0))
# scenario.add_event(3.0, throttle(0.0))
scenario.add_event(9.0, set_thrust_direction(RocketVec3(1.0, 0.00051, 0.0).direction))
# scenario.add_event(5.0, set_thrust_direction(RocketVec3(1.0, -0.00051, 0.0).direction))
scenario.add_event(10.0, set_thrust_direction(RocketVec3(1.0, 0.0, 0.0).direction))
# scenario.add_event(8.0, throttle(0.0))

if __name__ == "__main__":
    scenario.run(duration=16.0)
