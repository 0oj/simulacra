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
    impact_speed: float = 0.0            # m/s, total speed at touchdown
    impact_vertical_speed: float = 0.0   # m/s, downward component at touchdown
    landing_angle: float = 0.0           # radians from vertical (0 = upright)


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

        # ---------- Normal force torque (if grounded) ----------

        normal_torque = RocketVec3(0.0, 0.0, 0.0)
        lifting_off = False

        if on_ground:
            g_mag = abs(env.gravity.y)
            weight = p.mass * g_mag
            thrust_y = thrust_world.y
            normal_force_mag = max(0.0, weight - thrust_y)

            if normal_force_mag == 0.0:
                lifting_off = True
            else:
                # Normal force acts upward at the base contact point.
                # Torque about CoM = (base - CoM) × N, computed in body frame.
                N_world = WorldVec3(0.0, normal_force_mag, 0.0)
                N_body = N_world.to_body(s.orientation)
                lever = p.thrust_application_offset - p.center_of_mass_offset
                normal_torque = lever.cross(N_body)

        # ---------- Rotational dynamics (body frame, ONE path) ----------

        # Torque from thrust about CoM
        thrust_offset = p.thrust_application_offset - p.center_of_mass_offset
        thrust_torque = thrust_offset.cross(s.thrust_body)

        # Combined torque
        total_torque = thrust_torque + normal_torque

        tau_x = float(total_torque[0])
        tau_y = float(total_torque[1])
        tau_z = float(total_torque[2])

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

        if on_ground and not lifting_off:
            # Latch impact metrics on airborne → grounded transition
            if was_airborne:
                self._latch_impact_metrics()
                self._apply_impact_angular_impulse()

            # Clamp position and velocity
            s.position.y = 0.0
            if s.velocity.y < 0.0:
                s.velocity.y = 0.0

            # Ground friction: zero lateral velocity
            s.velocity.x = 0.0
            s.velocity.z = 0.0

            # Tip-over check: nose pointing below horizontal → freeze
            nose_world = RocketVec3(1.0, 0.0, 0.0).to_world(s.orientation)
            if nose_world.y <= 0.0:
                s.velocity = WorldVec3(0.0, 0.0, 0.0)
                s.angular_velocity_body = RocketVec3(0.0, 0.0, 0.0)
                s.thrust_body = RocketVec3(0.0, 0.0, 0.0)
                self.ground_contact.tipped_over = True

            self.ground_contact.on_ground = True
        else:
            # Airborne or lifting off — reset metrics on any grounded→airborne transition
            if self.ground_contact.on_ground:
                self._reset_on_liftoff()
            self.ground_contact.on_ground = False

    def _apply_impact_angular_impulse(self) -> None:
        """
        Convert linear momentum into angular momentum at ground impact.

        When the base hits the ground, the ground impulse acts at the
        contact point.  Since the impulse passes through the contact
        point it has zero moment about it, so angular momentum about
        the base is conserved:

            L_base_before  =  m (r × v) + I_world · ω
            L_base_after   =  I_base · ω_after

        where r = base-to-CoM vector and I_base uses the parallel-axis
        theorem.  Solving for ω_after gives the correct post-impact
        angular velocity.
        """
        import numpy as np

        s = self.rocket.state
        p = self.rocket.props

        R_mat = s.orientation.rotation_matrix()

        # r = vector from base to CoM in world frame
        r_body = np.array([
            p.center_of_mass_offset.x,
            p.center_of_mass_offset.y,
            p.center_of_mass_offset.z,
        ])
        r = R_mat @ r_body

        # Pre-impact velocity and angular velocity in world frame
        v = np.array([s.velocity.x, s.velocity.y, s.velocity.z])
        omega_body = np.array([
            s.angular_velocity_body.x,
            s.angular_velocity_body.y,
            s.angular_velocity_body.z,
        ])
        omega_world = R_mat @ omega_body

        # World-frame inertia about CoM: I_world = R · diag(Ix,Iy,Iz) · R^T
        I_body = np.diag(p.inertia_diag)
        I_cm_world = R_mat @ I_body @ R_mat.T

        # Angular momentum about the base contact point (conserved)
        L_base = p.mass * np.cross(r, v) + I_cm_world @ omega_world

        # Parallel-axis theorem: I_base = I_cm + m·(|r|²·I - r⊗r)
        I_base = I_cm_world + p.mass * (
            np.dot(r, r) * np.eye(3) - np.outer(r, r)
        )

        # Solve I_base · ω_after = L_base
        omega_after_world = np.linalg.solve(I_base, L_base)

        # Convert back to body frame
        omega_after_body = R_mat.T @ omega_after_world
        s.angular_velocity_body = RocketVec3(
            float(omega_after_body[0]),
            float(omega_after_body[1]),
            float(omega_after_body[2]),
        )

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
