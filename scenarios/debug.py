import numpy as np
from simulacra.noumena import Environment, Kernel, Rocket, Scenario
from simulacra.noumena.rocket import RocketProperties, RocketState
from simulacra.onboard.controller import Controller, IdentityController
from simulacra.physics import RocketVec2, WorldVec2

rocket = Rocket(
    properties=RocketProperties(
        length=6.0,
        mass=120.0,
        center_of_mass_offset=RocketVec2(1.0, 0.0),
        moment_of_inertia=1.0,
        thrust_application_offset=RocketVec2(0.0, 0.0),
    ),
    state=RocketState(
        position=WorldVec2(0.0, 0.0),
        velocity=WorldVec2(0.0, 0.0),
        orientation=np.radians(90.0),
        angular_velocity=0.0,
        gimbal_angle=0.0,
        thrust_magnitude=0.0,
    ),
)

environment = Environment(
    # wind=WorldVec2(-14.0, 0.0),
)

kernel = Kernel(rocket=rocket, environment=environment)
controller = IdentityController()

def callback(t: float, kernel: Kernel, controller: Controller):
    ...


scenario = Scenario(
    kernel=kernel,
    controller=controller,
    callback=callback,
)

def throttle(to: float):
    return lambda t, k, c: setattr(k.rocket.state, 'thrust_magnitude', to)

scenario.add_event(1.0, throttle(100.0))
scenario.add_event(2.0, throttle(0.0))
scenario.add_event(3.0, throttle(100.0))
scenario.add_event(4.0, throttle(0.0))

scenario.run(duration=12.0)
