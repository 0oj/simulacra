import numpy as np
from simulacra.noumena import Environment, Kernel, Rocket, Scenario
from simulacra.noumena.rocket import RocketProperties, RocketState
from simulacra.onboard.controller import Controller, IdentityController
from simulacra.physics import RocketVec2, WorldVec2

rocket = Rocket(
    properties=RocketProperties(
        length=2.0,
        mass=10.0,
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
    wind=WorldVec2(0.0, 0.0),
)

kernel = Kernel(rocket=rocket, environment=environment)
controller = IdentityController()

def callback(t: float, kernel: Kernel, controller: Controller):
    kernel.rocket.state.thrust_magnitude = 600.0

scenario = Scenario(
    kernel=kernel,
    controller=controller,
    callback=callback,
)

scenario.run(duration=10.0)
