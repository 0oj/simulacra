"""
The physics engine. God's hand that moves the rocket through time,
applying forces and updating state according to the laws of motion.
"""

import math
from dataclasses import dataclass

from ..noumena.rocket import Rocket
from ..noumena.world import Environment
from ..physics import RocketVec3, WorldVec3
from ..physics.rigid_body import integrate_orientation_body_omega


@dataclass
class GroundContact:
    """Landing metrics computed by the kernel during ground contact."""
    on_ground: bool = False
    tipped_over: bool = False
    crashed: bool = False                 # impact too hard
    impact_speed: float = 0.0            # m/s, total speed at touchdown
    impact_vertical_speed: float = 0.0   # m/s, downward component at touchdown
    landing_angle: float = 0.0           # radians from vertical (0 = upright)

    # Landing thresholds
    MAX_LANDING_SPEED: float = 3.0       # m/s - above this is a crash
    MAX_LANDING_TILT: float = 0.26       # radians (~15°) - above this tips over


class Kernel:
    """
    Physics kernel.

    Advances a Rocket in an Environment using simple rigid-body
    dynamics with a diagonal inertia tensor in the body frame.
    """

    def __init__(self, rocket: Rocket, environment: Environment):
        self.rocket = rocket
        self.environment = environment
        self.ground_contact = GroundContact()

    def step(self, dt: float) -> None:
        """
        Advance the physics by dt seconds.
        """

        r = self.rocket
        s = r.state
        env = self.environment

        # Burn fuel when thrusting; only rebuild props if mass actually changed.
        if s.thrust_body.norm > 0.0:
            # Check if we have fuel to burn
            if r.mass_model.fuel_fraction == 0.0:
                # Out of fuel - cut thrust
                s.thrust_body = RocketVec3(0.0, 0.0, 0.0)
            else:
                burned = r.mass_model.burn_fuel(dt)
                if burned > 0.0:
                    r.props = r.mass_model.to_rocket_props(
                        length=r.props.length,
                        thrust_application_offset=r.props.thrust_application_offset,
                        drag_coefficient=r.props.drag_coefficient,
                        reference_area=r.props.reference_area,
                    )

        p = r.props

        # ---------- Translational dynamics (world frame) ----------

        # Thrust in world frame
        thrust_world = s.thrust_body.to_world(s.orientation)

        # Gravity force
        gravity_force = env.gravity * p.mass

        # Aerodynamic drag: F = -½ ρ Cd A |v| v
        speed = s.velocity.norm
        if speed > 0.0:
            drag_force = s.velocity * (-0.5 * env.air_density * p.drag_coefficient * p.reference_area * speed)
        else:
            drag_force = WorldVec3(0.0, 0.0, 0.0)

        # For now: mimic original kernel and treat wind as an extra force term.
        net_force = thrust_world + gravity_force + env.wind + drag_force
        linear_accel = net_force / p.mass

        s.velocity += linear_accel * dt
        s.position += s.velocity * dt

        # ---------- Ground contact detection ----------

        on_ground = s.position.y <= 0.0
        was_airborne = not self.ground_contact.on_ground

        # No torque from ground - we just detect landing and freeze

        # ---------- Rotational dynamics (body frame) ----------

        # Torque from thrust about CoM
        thrust_offset = p.thrust_application_offset - p.center_of_mass_offset
        thrust_torque = thrust_offset.cross(s.thrust_body)

        tau_x = float(thrust_torque[0])
        tau_y = float(thrust_torque[1])
        tau_z = float(thrust_torque[2])

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

        s.angular_velocity_body = RocketVec3(wx_new, wy_new, wz_new)

        # Update orientation using the new angular velocity (body frame).
        s.orientation = integrate_orientation_body_omega(
            s.orientation,
            [wx_new, wy_new, wz_new],
            dt,
        )

        # ---------- Ground constraints ----------

        # Clamp position: rocket cannot go through ground
        if s.position.y < 0.0:
            s.position.y = 0.0

        if on_ground:
            # Latch impact metrics on airborne → grounded transition
            if was_airborne:
                self._latch_impact_metrics()

                # Evaluate landing outcome against thresholds
                gc = self.ground_contact
                impact_speed = s.velocity.norm
                nose_world = RocketVec3(1.0, 0.0, 0.0).to_world(s.orientation)
                cos_tilt = max(-1.0, min(1.0, nose_world.y))
                tilt_angle = math.acos(cos_tilt)

                # Check failure conditions
                if impact_speed > gc.MAX_LANDING_SPEED:
                    gc.crashed = True
                if tilt_angle > gc.MAX_LANDING_TILT:
                    gc.tipped_over = True

            # Check if thrust exceeds weight → allow liftoff
            thrust_world = s.thrust_body.to_world(s.orientation)
            thrust_up = thrust_world.y
            weight = p.mass * abs(env.gravity.y)
            if thrust_up > weight:
                # Liftoff: rocket has enough thrust to leave the ground
                s.position.y = 0.0  # clamp to ground plane
                # Keep velocity (thrust built it up) but zero downward component
                if s.velocity.y < 0.0:
                    s.velocity.y = 0.0
                self.ground_contact.on_ground = False
                if not was_airborne:
                    pass  # was already on ground, now lifting off
            else:
                # Stay on ground: freeze state
                s.position.y = 0.0
                s.velocity = WorldVec3(0.0, 0.0, 0.0)
                s.angular_velocity_body = RocketVec3(0.0, 0.0, 0.0)
                self.ground_contact.on_ground = True
        else:
            # Airborne — reset metrics if transitioning from ground
            if self.ground_contact.on_ground:
                self._reset_on_liftoff()
            self.ground_contact.on_ground = False

    def _latch_impact_metrics(self) -> None:
        """Record impact metrics at the moment of touchdown."""
        s = self.rocket.state
        self.ground_contact.impact_speed = s.velocity.norm
        self.ground_contact.impact_vertical_speed = abs(min(0.0, s.velocity.y))
        nose_world = RocketVec3(1.0, 0.0, 0.0).to_world(s.orientation)
        cos_angle = max(-1.0, min(1.0, nose_world.y))
        self.ground_contact.landing_angle = math.acos(cos_angle)

    def _reset_on_liftoff(self) -> None:
        """Reset metrics when the rocket lifts off (for re-landing)."""
        self.ground_contact.impact_speed = 0.0
        self.ground_contact.impact_vertical_speed = 0.0
        self.ground_contact.landing_angle = 0.0
        self.ground_contact.tipped_over = False
        self.ground_contact.crashed = False
