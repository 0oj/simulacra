from scenario import Scenario
from noumena import Rocket, Environment, Kernel
from noumena.rocket import RocketProperties, RocketState
from physics import RocketVec2, WorldVec2
from onboard.controller import SimpleSetpointGimbalController, Controller, IdentityController
from phenomena.actuators import GimbalActuator, ThrustActuator
from phenomena.sensors import GPS, IMU, Sensors
import numpy as np

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
    gravity=WorldVec2(0.0, -1),
    wind=WorldVec2(-14.0, 0.0),
)

kernel = Kernel(rocket=rocket, environment=environment)
controller = SimpleSetpointGimbalController(
    sensors=Sensors(kernel=kernel),
    gimbal_actuator=GimbalActuator(kernel=kernel, max_gimbal_angle=np.radians(5)),
    thrust_actuator=ThrustActuator(kernel=kernel, max_thrust_magnitude=100.0),
    setpoint=WorldVec2(15.0, 180.0),
)
# controller = IdentityController(sensors=Sensors(kernel=kernel))

def callback(t: float, kernel: Kernel, controller: Controller):
    ...
    # Define a schedule of (trigger_time, actions...) in order
    # schedule = [
    #     (0.0, lambda k: (
    #         setattr(k.rocket.state, 'thrust_magnitude', 200.0),
    #         # setattr(k.rocket.state, 'gimbal_angle', np.radians(0.01)),
    #         # setattr(k.rocket.state, 'thrust_magnitude', 100.0),
    #     )),
    #     (2.0, lambda k: (
    #         # setattr(k.rocket.state, 'orientation', np.radians(20.0)),
    #     )),
    #     (3.0, lambda k: (
    #         # setattr(k.rocket.state, 'position', WorldVec2(100.0, 100.0)),
    #         # setattr(k.rocket.state, 'orientation', np.radians(60.0)),
    #         setattr(k.rocket.state, 'thrust_magnitude', 100.0),
    #     )),
    #     # (4.0, lambda k: setattr(k.rocket.state, 'position', WorldVec2(0.0, 100.0))),
    #     (8.0, lambda k: setattr(k.rocket.state, 'thrust_magnitude', 100.0)),
    #     # (6.0, lambda k: setattr(k.rocket.state, 'position', WorldVec2(100.0, 0.0))),
    # ]

    # # Initialize once: list to track which steps have fired
    # if not hasattr(callback, "_done_indices"):
    #     callback._done_indices = set()
    # done = callback._done_indices

    # for i, (trigger, action) in enumerate(schedule):
    #     if t >= trigger and i not in done:
    #         action(kernel)
    #         done.add(i)

scenario = Scenario(
    kernel=kernel,
    controller=controller,
    sensors=Sensors(kernel=kernel),
    callback=callback,
)

scenario.run(duration=12.0)
