"""
The physics engine. God's hand that moves the rocket through time,
applying forces and updating state according to the laws of motion.
"""

import numpy as np

from ..noumena.rocket import Rocket
from ..noumena.world import Environment


class Kernel:
    """
    The kernel is the core of the simulation.
    It advances the physics of the rocket and the environment by dt seconds.
    """

    def __init__(self, rocket: Rocket, environment: Environment):
        self.rocket = rocket
        self.environment = environment

    def step(self, dt: float) -> None:
        """
        Advance the physics by dt seconds.

        Args:
            externals: The externals to applied by the onboard computer (and scenario's events).
            dt: The time step to advance the physics by.
        """

        # TODO: Much of this should be in the physics module.

        # Extract thrust (in the rocket frame) from effects
        # TODO: Couldn't this be done in the noumena.rocket.RocketState class? as a property?
        thrust_vector = self.rocket.state.thrust_vector
        # Compute the torque in the rocket frame (thrust offset × thrust vector)
        thrust_offset = (
            self.rocket.properties.thrust_application_offset
            - self.rocket.properties.center_of_mass_offset
        )
        thrust_torque = np.cross(thrust_offset, thrust_vector)
        # Compute the angular acceleration (torque / moment of inertia)
        angular_acceleration = thrust_torque / self.rocket.properties.moment_of_inertia
        # Update the angular velocity and orientation
        self.rocket.state.angular_velocity += angular_acceleration * dt
        self.rocket.state.orientation += self.rocket.state.angular_velocity * dt
        # Compute the world thrust (in the world frame)
        world_thrust = thrust_vector.to_world(self.rocket.state.orientation)
        # Compute the gravity force (in the world frame)
        gravity_force = self.environment.gravity * self.rocket.properties.mass
        # Compute the net force (in the world frame)
        net_force = world_thrust + gravity_force + self.environment.wind
        linear_acceleration = net_force / self.rocket.properties.mass
        # Update the linear velocity and position
        self.rocket.state.velocity += linear_acceleration * dt
        self.rocket.state.position += self.rocket.state.velocity * dt

        # SPECIAL CASE: Ground collision
        if self.rocket.state.position.y < 0.0:
            self.rocket.state.position.y = 0.0
            if self.rocket.state.velocity.y < 0.0:
                self.rocket.state.velocity.y = 0.0
                self.rocket.state.angular_velocity = 0.0
        # END SPECIAL CASE: Ground collision
