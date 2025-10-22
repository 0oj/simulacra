from dataclasses import dataclass, field
from random import random
import time
import math

from .physics import Vector
from .world import Frame


@dataclass
class Rocket:
    mass: float

    position: Vector
    velocity: Vector

    # Physical body properties
    length: float = 4.0  # Length of the rocket body
    moment_of_inertia: float = 1.0  # Moment of inertia around center of mass

    # Orientation (attitude)
    orientation: float = 0.0  # Angle in radians (0 = vertical, π/2 = horizontal)
    angular_velocity: float = 0.0  # Angular velocity in rad/s

    # Drag properties
    drag_factor: float = 0.1  # Linear drag coefficient
    angular_drag_factor: float = 0.05  # Angular drag coefficient

    @property
    def frame(self) -> Frame:
        """Get the rocket's body frame."""
        return Frame(position=self.position, orientation=self.orientation)

    def apply(self, force: Vector) -> None:
        self.velocity += force / self.mass

    def apply_torque(self, torque: float) -> None:
        """Apply torque to the rocket body."""
        self.angular_velocity += torque / self.moment_of_inertia

    @property
    def drag(self) -> Vector:
        return -self.velocity * self.drag_factor * self.mass

    @property
    def angular_drag(self) -> float:
        """Angular drag torque."""
        return (
            -self.angular_velocity * self.angular_drag_factor * self.moment_of_inertia
        )

    @property
    def body_thrust(self) -> Vector:
        """Thrust vector in the rocket's body frame."""
        if time.time() % 2 < 1:
            # Steady thrust in body frame (along Y axis of rocket)
            return Vector(x=-0.02, y=2.0, z=0.0)  # Reduced thrust for testing
        else:
            # Variable thrust with some 3D variation in body frame
            return Vector(
                x=(random() - 0.5 - 0.02) * 0.5,  # Reduced side thrust
                y=3.0,  # Reduced main thrust
                z=(random() - 0.5) * 0.1,  # Small Z component for 3D effect
            )

    @property
    def thrust(self) -> Vector:
        """Thrust vector in world coordinates, applied at the base of the rocket."""
        # Transform from body frame to world frame
        return self.frame.to_world(self.body_thrust)

    @property
    def thrust_torque(self) -> float:
        """Torque created by thrust misalignment with center of mass."""
        # The thrust is applied at the base of the rocket (distance from center: length/2)
        thrust_arm = self.length / 2

        # Get thrust directly in body frame
        body_thrust = self.body_thrust

        # body_thrust.x is the *side* force, which causes torque
        # Torque = side force × lever arm (moment arm)
        return body_thrust.x * thrust_arm

    @property
    def orientation_vector(self) -> Vector:
        """Get the rocket's orientation as a unit vector pointing along its body."""
        return Vector(x=math.sin(self.orientation), y=math.cos(self.orientation), z=0.0)

    def step(self, dt: float) -> None:
        """Update rocket physics for one timestep."""
        # Linear dynamics
        self.position += self.velocity * dt
        self.apply(self.drag)
        self.apply(self.thrust)

        # Angular dynamics
        self.orientation += self.angular_velocity * dt
        self.apply_torque(self.angular_drag)
        self.apply_torque(self.thrust_torque)
