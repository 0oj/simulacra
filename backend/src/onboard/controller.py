from abc import ABC, abstractmethod
import numpy as np
from phenomena.actuators import GimbalCommand, GimbalActuator, ThrustCommand, ThrustActuator
from phenomena.sensors import Sensors
from physics import WorldVec2


class Controller(ABC):
    @abstractmethod
    def run(self) -> None: ...


class SimpleSetpointGimbalController(Controller):
    """
    A comprehensive PID controller that:
    1. Keeps the rocket upright (orientation = 90°) using full PID control
    2. Controls thrust using PID based on altitude/velocity errors
    3. Optionally guides toward a setpoint, but above the setpoint only aligns X (ignores Y)
    4. Uses velocity feedback to improve control (damping and feedforward)

    Gains have been chosen for:
        - Tall, light rocket
        - Strong thrust, limited gimbal authority (~15 deg)
    """
    def __init__(
        self,
        sensors: Sensors,
        gimbal_actuator: GimbalActuator,
        thrust_actuator: ThrustActuator,
        setpoint: WorldVec2 = None,
        target_orientation: float = np.pi / 2.0,
        target_altitude: float = None,  # Target altitude (Y coordinate) in meters
        target_velocity_y: float = 0.0,  # Target vertical velocity (m/s)
        # Orientation PID gains
        kp_orientation: float = 1.5,  # Increased for better recovery
        ki_orientation: float = 0.02,  # Reduced to prevent windup
        kd_angular_velocity: float = 0.4,  # Increased for better damping
        # Thrust PID gains (for altitude/velocity control)
        kp_altitude: float = 2.0,  # Much reduced - was causing huge overshoot
        ki_altitude: float = 0.1,  # Much reduced
        kd_velocity_y: float = 5.0,  # Reduced
        # Control parameters
        setpoint_weight: float = 0.3,
        velocity_damping: float = 0.1,  # Damping based on velocity magnitude
        lookahead_time: float = 0.5,  # Time to look ahead for setpoint guidance (seconds)
        base_thrust: float = 200.0,  # Base thrust level (Newtons) - hover thrust
        dt: float = 0.01,  # Timestep for integral calculation (should match simulation dt)
    ):
        self.sensors = sensors
        self.gimbal_actuator = gimbal_actuator
        self.thrust_actuator = thrust_actuator
        self.setpoint = setpoint
        self.target_orientation = target_orientation
        self.target_altitude = target_altitude
        self.target_velocity_y = target_velocity_y
        self.kp_orientation = kp_orientation
        self.ki_orientation = ki_orientation
        self.kd_angular_velocity = kd_angular_velocity
        self.kp_altitude = kp_altitude
        self.ki_altitude = ki_altitude
        self.kd_velocity_y = kd_velocity_y
        self.setpoint_weight = setpoint_weight
        self.velocity_damping = velocity_damping
        self.lookahead_time = lookahead_time
        self.base_thrust = base_thrust
        self.dt = dt

        # PID state variables
        self.orientation_error_integral = 0.0
        self.last_orientation_error = 0.0
        self.altitude_error_integral = 0.0

    def set_setpoint(self, setpoint: WorldVec2) -> None:
        self.setpoint = setpoint

    def run(self) -> None:
        if self.gimbal_actuator is None or self.thrust_actuator is None:
            return

        orientation, angular_velocity = self.sensors.IMU.read()
        rocket_position = self.sensors.GPS.read()
        rocket_velocity = self.sensors.VelocitySensor.read()

        # ========== ORIENTATION CONTROL (Full PID) ==========
        # Calculate orientation error (normalize to [-π, π])
        orientation_error = orientation - self.target_orientation
        orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))
        
        # Disable integral when error is large (prevents windup during recovery)
        abs_error = abs(orientation_error)
        if abs_error < np.pi / 4:  # Only use integral when error is small (< 45°)
            # Update integral term (with anti-windup: limit integral accumulation)
            self.orientation_error_integral += orientation_error * self.dt
            # Anti-windup: limit integral to prevent excessive accumulation
            max_integral = np.pi / 2.0 / (self.ki_orientation + 1e-6)  # More conservative limit
            self.orientation_error_integral = np.clip(self.orientation_error_integral, -max_integral, max_integral)
            integral_term = -self.ki_orientation * self.orientation_error_integral
        else:
            # Reset integral when error is large (prevents windup)
            self.orientation_error_integral = 0.0
            integral_term = 0.0
        
        # Full PID control for orientation
        # P: proportional to error
        proportional_term = -self.kp_orientation * orientation_error
        # I: integral of error (eliminates steady-state error, only when error is small)
        # D: derivative (angular velocity is the rate of change of orientation)
        derivative_term = -self.kd_angular_velocity * angular_velocity
        
        upright_gimbal = proportional_term + integral_term + derivative_term
        
        # Add velocity-based damping: if moving fast, reduce control authority to prevent overshoot
        velocity_magnitude = np.linalg.norm([rocket_velocity.x, rocket_velocity.y])
        velocity_damping_factor = 1.0 / (1.0 + self.velocity_damping * velocity_magnitude)
        desired_gimbal = upright_gimbal * velocity_damping_factor

        # ========== SETPOINT GUIDANCE (if setpoint provided) ==========
        if self.setpoint is not None:
            # Use lookahead: predict where rocket will be based on current velocity
            predicted_position = WorldVec2(
                rocket_position.x + rocket_velocity.x * self.lookahead_time,
                rocket_position.y + rocket_velocity.y * self.lookahead_time
            )
            
            # If rocket is above the setpoint in Y, only align X, keep upright
            if rocket_position.y > self.setpoint.y:
                # Only align X, ignore Y. Direction is solely in X axis.
                # Use predicted position for better guidance
                dx = self.setpoint.x - predicted_position.x
                if abs(dx) > 1e-6:
                    xnorm = np.sign(dx)
                else:
                    xnorm = 0.0
                # Rocket frame: x is forward, y is right
                # "Desired" is the sideways component (how much to lean left/right)
                # We want to point "toward" the setpoint X, so we just use sign(dx)
                setpoint_gimbal = np.arctan2(0.0, xnorm)
                # Blend: if dx is large, more weight, else just keep upright
                desired_gimbal = (1.0 - self.setpoint_weight) * desired_gimbal + self.setpoint_weight * setpoint_gimbal
            else:
                # Normal 2D setpoint guidance with lookahead
                direction_world = self.setpoint - predicted_position
                norm = np.linalg.norm([direction_world.x, direction_world.y])
                if norm > 1e-6:
                    direction_world_normalized = WorldVec2(
                        direction_world.x / norm, direction_world.y / norm
                    )

                    # Convert to rocket frame
                    cos_orient = np.cos(-orientation)
                    sin_orient = np.sin(-orientation)
                    direction_rocket_x = (
                        direction_world_normalized.x * cos_orient
                        - direction_world_normalized.y * sin_orient
                    )
                    direction_rocket_y = (
                        direction_world_normalized.x * sin_orient
                        + direction_world_normalized.y * cos_orient
                    )

                    setpoint_gimbal = np.arctan2(direction_rocket_y, direction_rocket_x)
                    desired_gimbal = (1.0 - self.setpoint_weight) * desired_gimbal + self.setpoint_weight * setpoint_gimbal

        # ========== THRUST CONTROL (Full PID for altitude/velocity) ==========
        desired_thrust = self.base_thrust  # Start with base thrust
        
        if self.target_altitude is not None:
            # Altitude control using PID
            altitude_error = self.target_altitude - rocket_position.y
            
            # Only update integral when error is reasonable (prevents windup)
            abs_alt_error = abs(altitude_error)
            if abs_alt_error < 50.0:  # Only use integral when error is small (< 50m)
                # Update integral term (with anti-windup)
                self.altitude_error_integral += altitude_error * self.dt
                max_altitude_integral = 20.0 / (self.ki_altitude + 1e-6)  # More conservative limit
                self.altitude_error_integral = np.clip(self.altitude_error_integral, -max_altitude_integral, max_altitude_integral)
                thrust_integral = self.ki_altitude * self.altitude_error_integral
            else:
                # Reset integral when error is large
                self.altitude_error_integral = 0.0
                thrust_integral = 0.0
            
            # Velocity error (vertical velocity control) - this is the derivative term
            velocity_y_error = self.target_velocity_y - rocket_velocity.y
            
            # Full PID control for thrust
            # P: proportional to altitude error
            thrust_proportional = self.kp_altitude * altitude_error
            # I: integral of altitude error (eliminates steady-state error, only when error is small)
            # D: derivative (vertical velocity is the rate of change of altitude)
            thrust_derivative = self.kd_velocity_y * velocity_y_error
            
            desired_thrust = self.base_thrust + thrust_proportional + thrust_integral + thrust_derivative
            
            # Safety: ensure thrust is always positive and reasonable
            # Don't let it go below a minimum (e.g., 50N) unless we're way above target
            if desired_thrust < 50.0 and altitude_error < -100.0:
                desired_thrust = 50.0  # Minimum thrust
            elif desired_thrust < 0.0:
                desired_thrust = 0.0  # Can't have negative thrust

        # ========== APPLY COMMANDS ==========
        gimbal_command = GimbalCommand(desired_gimbal_angle=-desired_gimbal)
        self.gimbal_actuator.apply(gimbal_command)
        
        thrust_command = ThrustCommand(desired_thrust_magnitude=desired_thrust)
        self.thrust_actuator.apply(thrust_command)

        # Update last orientation error for next derivative calculation
        self.last_orientation_error = orientation_error


class IdentityController(Controller):
    def __init__(self, sensors: Sensors = None):
        self.sensors = sensors

    def run(self) -> None: ...
