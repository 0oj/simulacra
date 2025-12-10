"""
The physics engine. God's hand that moves the rocket through time,
applying forces and updating state according to the laws of motion.
"""

import numpy as np

from ..noumena.rocket import Rocket2D, Rocket3D
from ..noumena.world import Environment2D, Environment3D
from ..physics.rigid_body import integrate_orientation_body_omega


class Kernel2D:
    """
    2D physics kernel (legacy).

    It advances the physics of the 2D rocket and the environment by dt seconds.
    """

    def __init__(self, rocket: Rocket2D, environment: Environment2D):
        self.rocket = rocket
        self.environment = environment

    def step(self, dt: float) -> None:
        """
        Advance the 2D physics by dt seconds.
        """

        # Extract thrust (in the rocket frame) from effects
        thrust_vector = self.rocket.state.thrust_vector
        # Compute the torque in the rocket frame (thrust offset × thrust vector)
        thrust_offset = (
            self.rocket.props.thrust_application_offset
            - self.rocket.props.center_of_mass_offset
        )
        thrust_torque = np.cross(thrust_offset, thrust_vector)
        # Compute the angular acceleration (torque / moment of inertia)
        angular_acceleration = thrust_torque / self.rocket.props.moment_of_inertia
        # Update the angular velocity and orientation
        self.rocket.state.angular_velocity += angular_acceleration * dt
        self.rocket.state.orientation += self.rocket.state.angular_velocity * dt
        # Compute the world thrust (in the world frame)
        world_thrust = thrust_vector.to_world(self.rocket.state.orientation)
        # Compute the gravity force (in the world frame)
        gravity_force = self.environment.gravity * self.rocket.props.mass
        # Compute the net force (in the world frame)
        net_force = world_thrust + gravity_force + self.environment.wind
        linear_acceleration = net_force / self.rocket.props.mass
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


class Kernel3D:
    """
    3D physics kernel.

    Advances a Rocket3D in a 3D Environment3D using simple rigid-body
    dynamics with a diagonal inertia tensor in the body frame.
    """

    def __init__(self, rocket: Rocket3D, environment: Environment3D):
        self.rocket = rocket
        self.environment = environment

    def step(self, dt: float) -> None:
        """
        Advance the 3D physics by dt seconds.
        """

        r = self.rocket
        s = r.state
        p = r.props
        env = self.environment

        # ---------- Translational dynamics (world frame) ----------

        # Thrust in world frame: magnitude * direction (direction is body-frame unit vector)
        thrust_body = s.thrust_direction * s.thrust_magnitude
        thrust_world = thrust_body.to_world(s.orientation)

        # Gravity force
        gravity_force = env.gravity * p.mass

        # For now: mimic original kernel and treat wind as an extra force term.
        net_force = thrust_world + gravity_force + env.wind
        linear_accel = net_force / p.mass

        s.velocity += linear_accel * dt
        s.position += s.velocity * dt

        # ---------- Rotational dynamics (body frame) ----------

        # Torque in body frame: tau = r x F, with F = thrust_magnitude * thrust_direction
        thrust_offset = p.thrust_application_offset - p.center_of_mass_offset
        thrust_force_body = s.thrust_direction * s.thrust_magnitude
        torque_vec = thrust_offset.cross(thrust_force_body)  # Vec3

        tau_x = float(torque_vec[0])
        tau_y = float(torque_vec[1])
        tau_z = float(torque_vec[2])

        Ix, Iy, Iz = p.inertia_diag
        wx = float(s.angular_velocity_body.x)
        wy = float(s.angular_velocity_body.y)
        wz = float(s.angular_velocity_body.z)

        # Rigid-body equations for diagonal inertia tensor:
        # Ix * wx_dot = tau_x + (Iy - Iz) * wy * wz, etc.
        wx_dot = (tau_x + (Iy - Iz) * wy * wz) / Ix
        wy_dot = (tau_y + (Iz - Ix) * wz * wx) / Iy
        wz_dot = (tau_z + (Ix - Iy) * wx * wy) / Iz

        wx_new = wx + wx_dot * dt
        wy_new = wy + wy_dot * dt
        wz_new = wz + wz_dot * dt

        from ..physics import RocketVec3

        s.angular_velocity_body = RocketVec3(wx_new, wy_new, wz_new)

        # Update orientation using the new angular velocity (body frame).
        s.orientation = integrate_orientation_body_omega(
            s.orientation,
            [wx_new, wy_new, wz_new],
            dt,
        )

        # Special case: ground collision
        if s.position.y < 0.0:
            s.position.y = 0.0
            if s.velocity.y < 0.0:
                s.velocity.y = 0.0
                # Zero out angular velocity while preserving RocketVec3 type
                s.angular_velocity_body = RocketVec3(0.0, 0.0, 0.0)
        # End special case: ground collision
