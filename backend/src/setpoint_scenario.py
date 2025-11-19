from scenario import Scenario
from noumena import Rocket, Environment, Kernel
from noumena.rocket import RocketProperties, RocketState
from physics import RocketVec2, WorldVec2
from onboard.controller import SimpleSetpointGimbalController, Controller
from phenomena.sensors import Sensors
import numpy as np
from phenomena.actuators import GimbalActuator

rocket = Rocket(
    properties=RocketProperties(
        length=2.0,
        mass=10.0,
        center_of_mass_offset=RocketVec2(1.0, 0.0),
        moment_of_inertia=1.0,
        thrust_application_point=RocketVec2(0.0, 0.0),
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
controller = SimpleSetpointGimbalController(
    sensors=Sensors(kernel=kernel),
    gimbal_actuator=GimbalActuator(kernel=kernel, max_gimbal_angle=np.radians(15.0)),
    setpoint=WorldVec2(10.0, 10.0),
)

def callback(t: float, kernel: Kernel, controller: Controller):
    kernel.rocket.state.thrust_magnitude = 600.0

scenario = Scenario(
    kernel=kernel,
    sensors=Sensors(kernel=kernel),
    controller=controller,
    callback=callback,
)

scenario.run(duration=10.0)
