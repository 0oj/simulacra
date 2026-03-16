"""
The rocket's eyes. Sensors that peek into god's truth and report back,
sometimes accurately, sometimes with error, but always limited.

Each sensor accepts optional per-channel error configurations. Channels are
named consistently: 'x', 'y', 'z' for vector quantities; 'roll', 'pitch', 'yaw'
for Euler perturbations on orientation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Optional, Protocol, Tuple, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..physics import Orientation, RocketVec3, WorldVec3
from .errors import ChannelErrors, IdentityError, SensorError


class GPSReading(NamedTuple):
    """
    GPS horizontal position reading.

    GPS provides good horizontal accuracy (~2-5m) but poor vertical (~10-15m).
    Use Barometer for altitude instead.
    """
    x: float  # World-frame horizontal position (meters)
    z: float  # World-frame horizontal position (meters)

if TYPE_CHECKING:
    from ..noumena import Kernel


class Sensor(ABC):
    """Base class for all sensors."""

    @abstractmethod
    def read(self) -> Any:
        """Read the sensor value, applying any configured errors."""
        ...


class SensorsInterface(Protocol):
    """
    Minimal interface that sensor bundles satisfy.

    The controller only relies on these attributes existing, not on their
    exact return types.
    """

    GPS: Sensor
    OrientationSensor: Sensor
    GyroSensor: Sensor
    ThrustSensor: Sensor
    VelocitySensor: Sensor

    # Legacy alias for backward compatibility
    IMU: Sensor


def _identity_error() -> SensorError:
    """Factory for default identity error."""
    return IdentityError()


def _default_errors(*channels: str) -> ChannelErrors:
    """Create default error dict with IdentityError for each channel."""
    return {ch: _identity_error() for ch in channels}


class GPS(Sensor):
    """
    Global Positioning System.

    Returns horizontal position only. GPS vertical accuracy is poor (~10-15m);
    use Barometer for altitude instead.

    Channels: 'x', 'z' (world-frame horizontal position components).
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("x", "z")
        self._time: float = 0.0

    def read(self) -> GPSReading:
        """Read horizontal position with per-channel errors applied."""
        truth = self._kernel.rocket.state.position
        t = self._time
        return GPSReading(
            x=self._errors.get("x", _identity_error()).apply(truth.x, t),
            z=self._errors.get("z", _identity_error()).apply(truth.z, t),
        )


class OrientationSensor(Sensor):
    """
    Orientation sensor (attitude reference).

    Channels: 'roll', 'pitch', 'yaw' for Euler perturbations (radians).
    Errors are additive perturbations to the true orientation.
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("roll", "pitch", "yaw")
        self._time: float = 0.0

    def read(self) -> Orientation:
        """Read orientation with Euler perturbations applied."""
        truth = self._kernel.rocket.state.orientation
        t = self._time

        # Get perturbations in roll/pitch/yaw
        droll = self._errors.get("roll", _identity_error()).apply(0.0, t)
        dpitch = self._errors.get("pitch", _identity_error()).apply(0.0, t)
        dyaw = self._errors.get("yaw", _identity_error()).apply(0.0, t)

        if droll == 0.0 and dpitch == 0.0 and dyaw == 0.0:
            return truth

        # Convert true orientation to scipy Rotation
        w, x, y, z = truth.as_array()
        r_true = R.from_quat([x, y, z, w])

        # Create perturbation rotation (ZYX intrinsic = extrinsic XYZ)
        r_perturb = R.from_euler("xyz", [droll, dpitch, dyaw])

        # Compose: perturbed = true * perturb (body-frame perturbation)
        r_result = r_true * r_perturb

        # Convert back to our quaternion format
        x, y, z, w = r_result.as_quat()
        return Orientation.from_wxyz(w, x, y, z)


class GyroSensor(Sensor):
    """
    Gyroscope sensor (angular velocity).

    Channels: 'x', 'y', 'z' (body-frame angular velocity components, rad/s).
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("x", "y", "z")
        self._time: float = 0.0

    def read(self) -> RocketVec3:
        """Read angular velocity with per-channel errors applied."""
        truth = self._kernel.rocket.state.angular_velocity_body
        t = self._time
        return RocketVec3(
            self._errors.get("x", _identity_error()).apply(truth.x, t),
            self._errors.get("y", _identity_error()).apply(truth.y, t),
            self._errors.get("z", _identity_error()).apply(truth.z, t),
        )


class IMU(Sensor):
    """
    Inertial Measurement Unit (legacy combined sensor).

    Wraps OrientationSensor and GyroSensor for backward compatibility.
    """

    def __init__(
        self,
        kernel: Kernel,
        orientation_errors: Optional[ChannelErrors] = None,
        gyro_errors: Optional[ChannelErrors] = None,
    ):
        self._orientation = OrientationSensor(kernel, orientation_errors)
        self._gyro = GyroSensor(kernel, gyro_errors)
        self._time: float = 0.0

    def read(self) -> Tuple[Orientation, RocketVec3]:
        """
        Reads orientation and body-frame angular velocity.

        Returns:
            (Orientation, RocketVec3): Attitude and angular velocity.
        """
        return (self._orientation.read(), self._gyro.read())

    def _set_time(self, t: float) -> None:
        """Update time for child sensors."""
        self._time = t
        self._orientation._time = t
        self._gyro._time = t


class ThrustSensor(Sensor):
    """
    Sensor that reads the thrust magnitude of the rocket.

    Channels: 'magnitude' (scalar thrust in Newtons).
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("magnitude")
        self._time: float = 0.0

    def read(self) -> float:
        """Read thrust magnitude with error applied."""
        truth = self._kernel.rocket.state.thrust_body.norm
        t = self._time
        return self._errors.get("magnitude", _identity_error()).apply(truth, t)


class VelocitySensor(Sensor):
    """
    Sensor that reads the linear velocity of the rocket.

    Channels: 'x', 'y', 'z' (world-frame velocity components, m/s).
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("x", "y", "z")
        self._time: float = 0.0

    def read(self) -> WorldVec3:
        """Read velocity with per-channel errors applied."""
        truth = self._kernel.rocket.state.velocity
        t = self._time
        return WorldVec3(
            self._errors.get("x", _identity_error()).apply(truth.x, t),
            self._errors.get("y", _identity_error()).apply(truth.y, t),
            self._errors.get("z", _identity_error()).apply(truth.z, t),
        )


class FuelSensor(Sensor):
    """
    Sensor that reads the remaining fuel fraction.

    Channels: 'fraction' (0.0 = empty, 1.0 = full).
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("fraction")
        self._time: float = 0.0

    def read(self) -> float:
        """Read fuel fraction with error applied."""
        truth = self._kernel.rocket.mass_model.fuel_fraction
        t = self._time
        return self._errors.get("fraction", _identity_error()).apply(truth, t)


class Accelerometer(Sensor):
    """
    Accelerometer sensor (BMI088, MPU6050 style).

    Measures specific force (non-gravitational acceleration) in the body frame.
    When stationary on ground, reads +g along the vertical body axis.
    During free fall, reads zero.

    Channels: 'x', 'y', 'z' (body-frame acceleration components, m/s²).
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("x", "y", "z")
        self._time: float = 0.0

    def read(self) -> RocketVec3:
        """
        Read body-frame acceleration (specific force).

        The accelerometer measures all forces except gravity, divided by mass.
        This equals: (thrust + drag + wind) / mass, expressed in body frame.
        """
        r = self._kernel.rocket
        env = self._kernel.environment
        s = r.state
        p = r.props

        # Thrust contribution (already in body frame)
        thrust_accel_body = s.thrust_body / p.mass

        # Drag in world frame: F = -½ ρ Cd A |v| v
        speed = s.velocity.norm
        if speed > 0.0:
            drag_world = s.velocity * (
                -0.5 * env.air_density * p.drag_coefficient * p.reference_area * speed
            )
        else:
            drag_world = WorldVec3(0.0, 0.0, 0.0)

        # Wind as force (simplified model from kernel)
        wind_world = env.wind

        # Non-gravitational world-frame acceleration
        non_grav_world = (drag_world + wind_world) / p.mass

        # Rotate to body frame (world → body is inverse of orientation)
        non_grav_body = non_grav_world.to_body(s.orientation)

        # Total specific force in body frame
        truth = RocketVec3(
            thrust_accel_body.x + non_grav_body.x,
            thrust_accel_body.y + non_grav_body.y,
            thrust_accel_body.z + non_grav_body.z,
        )

        # When on ground and not thrusting, we should read gravity (normal force).
        # The kernel zeroes velocity on ground, but we need to add the normal force.
        if self._kernel.ground_contact.on_ground and s.thrust_body.norm == 0.0:
            # Normal force = -gravity, rotated to body frame
            gravity_world = env.gravity  # Points down (negative Y)
            normal_world = WorldVec3(-gravity_world.x, -gravity_world.y, -gravity_world.z)
            normal_body = normal_world.to_body(s.orientation)
            truth = RocketVec3(
                normal_body.x + non_grav_body.x,
                normal_body.y + non_grav_body.y,
                normal_body.z + non_grav_body.z,
            )

        t = self._time
        return RocketVec3(
            self._errors.get("x", _identity_error()).apply(truth.x, t),
            self._errors.get("y", _identity_error()).apply(truth.y, t),
            self._errors.get("z", _identity_error()).apply(truth.z, t),
        )


class Magnetometer(Sensor):
    """
    Magnetometer sensor (HMC5883L, QMC5883L, BMM150 style).

    Measures Earth's magnetic field direction in the body frame.
    Combined with accelerometer, allows absolute orientation determination.

    Channels: 'x', 'y', 'z' (body-frame magnetic field components, normalized).
    """

    # Default Earth magnetic field: points north (-Z) and down (-Y)
    # ~30° inclination typical for mid-latitudes
    DEFAULT_FIELD = WorldVec3(0.0, -0.5, -0.866)  # normalized

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
        world_field: Optional[WorldVec3] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("x", "y", "z")
        self._world_field = world_field if world_field is not None else self.DEFAULT_FIELD
        self._time: float = 0.0

    def read(self) -> RocketVec3:
        """
        Read magnetic field in body frame.

        Returns normalized magnetic field direction rotated to body frame.
        """
        s = self._kernel.rocket.state

        # Rotate world field to body frame
        truth_body = self._world_field.to_body(s.orientation)

        t = self._time
        return RocketVec3(
            self._errors.get("x", _identity_error()).apply(truth_body.x, t),
            self._errors.get("y", _identity_error()).apply(truth_body.y, t),
            self._errors.get("z", _identity_error()).apply(truth_body.z, t),
        )


class Barometer(Sensor):
    """
    Barometric pressure sensor for altitude measurement.

    More accurate than GPS for vertical position, especially at low altitudes.
    Real sensors measure pressure and convert to altitude using the barometric
    formula; we simplify by reading altitude directly with appropriate errors.

    Channels: 'altitude' (meters above ground level).
    """

    def __init__(
        self,
        kernel: Kernel,
        errors: Optional[ChannelErrors] = None,
    ):
        self._kernel = kernel
        self._errors = errors or _default_errors("altitude")
        self._time: float = 0.0

    def read(self) -> float:
        """Read altitude with error applied."""
        truth = self._kernel.rocket.state.position.y
        t = self._time
        return self._errors.get("altitude", _identity_error()).apply(truth, t)


class AttitudeEstimator:
    """
    Onboard attitude estimation (AHRS) using complementary filter.

    Fuses accelerometer, gyroscope, and magnetometer to estimate orientation.
    This is what real flight computers do - the RL agent sees the estimated
    orientation, not ground truth.

    Algorithm:
        1. Gyro integration: fast, smooth, but drifts over time
        2. Accel+Mag reference: noisy but absolute (no drift)
        3. Complementary filter blends them: trust gyro short-term, accel+mag long-term

    The alpha parameter controls the blend:
        - alpha=0.98: trust gyro 98%, correct 2% toward accel+mag each step
        - Higher alpha = smoother but more drift
        - Lower alpha = less drift but noisier
    """

    def __init__(
        self,
        accelerometer: Accelerometer,
        gyro: GyroSensor,
        magnetometer: Magnetometer,
        alpha: float = 0.98,
        initial_orientation: Optional[Orientation] = None,
    ):
        self._accel = accelerometer
        self._gyro = gyro
        self._mag = magnetometer
        self._alpha = alpha

        # Current estimated orientation
        self._orientation = initial_orientation or Orientation.identity()
        self._initialized = initial_orientation is not None

    def update(self, dt: float) -> Orientation:
        """
        Update attitude estimate and return current orientation.

        Call this once per timestep with the simulation dt.
        """
        accel = self._accel.read()
        gyro = self._gyro.read()
        mag = self._mag.read()

        # --- Gyro prediction: integrate angular velocity ---
        # Convert current orientation to scipy Rotation
        w, x, y, z = self._orientation.as_array()
        r_current = R.from_quat([x, y, z, w])  # scipy uses [x,y,z,w]

        # Create rotation delta from gyro (body-frame angular velocity)
        omega = np.array([gyro.x, gyro.y, gyro.z])
        angle = np.linalg.norm(omega) * dt
        if angle > 1e-10:
            axis = omega / np.linalg.norm(omega)
            r_delta = R.from_rotvec(axis * angle)
            r_gyro_pred = r_current * r_delta  # body-frame rotation
        else:
            r_gyro_pred = r_current

        # --- Accel+Mag reference orientation ---
        r_ref = self._orientation_from_accel_mag(accel, mag)

        if not self._initialized:
            # First update: use reference directly
            self._orientation = r_ref
            self._initialized = True
            return self._orientation

        # --- Complementary filter: blend gyro prediction with reference ---
        # SLERP between gyro prediction and accel+mag reference
        r_gyro_pred_quat = r_gyro_pred.as_quat()  # [x,y,z,w]
        r_ref_scipy = r_ref._as_scipy_rotation()
        r_ref_quat = r_ref_scipy.as_quat()

        # Spherical linear interpolation
        r_blended = R.from_quat(self._slerp(r_gyro_pred_quat, r_ref_quat, 1.0 - self._alpha))

        # Convert back to our Orientation type
        x, y, z, w = r_blended.as_quat()
        self._orientation = Orientation.from_wxyz(w, x, y, z)

        return self._orientation

    def _orientation_from_accel_mag(self, accel: RocketVec3, mag: RocketVec3) -> Orientation:
        """
        Compute orientation from accelerometer and magnetometer readings.

        Uses TRIAD-like algorithm:
        - Accelerometer gives "down" direction (gravity) in body frame
        - Magnetometer gives "north" direction in body frame
        - Together they define a full 3D reference frame
        """
        # Normalize accelerometer (points opposite to gravity when stationary)
        accel_vec = np.array([accel.x, accel.y, accel.z])
        accel_norm = np.linalg.norm(accel_vec)
        if accel_norm < 0.1:
            # Free fall or very low acceleration - can't determine down
            return self._orientation  # Keep previous estimate

        # "Down" in body frame is opposite to accelerometer reading
        down_body = -accel_vec / accel_norm

        # Normalize magnetometer
        mag_vec = np.array([mag.x, mag.y, mag.z])
        mag_norm = np.linalg.norm(mag_vec)
        if mag_norm < 0.01:
            return self._orientation  # Invalid mag reading

        mag_body = mag_vec / mag_norm

        # "East" is perpendicular to both down and north (cross product)
        east_body = np.cross(down_body, mag_body)
        east_norm = np.linalg.norm(east_body)
        if east_norm < 0.01:
            return self._orientation  # Degenerate case
        east_body = east_body / east_norm

        # "North" is perpendicular to down and east (in horizontal plane)
        north_body = np.cross(east_body, down_body)

        # Build rotation matrix: columns are body-frame axes in world frame
        # Actually we need the inverse: world axes in body frame
        # R_body_to_world has columns [north_body, up_body, east_body] mapped to [X, Y, Z]
        # Wait - our convention: world +Y is up, body +X is nose (forward)

        # Let's build this carefully:
        # World frame: +Y is up, let's say +X is "forward/north", +Z is "east/right"
        # We have in body frame: down_body (opposite of world +Y), north-ish from mag

        up_body = -down_body  # World +Y direction in body frame

        # Build rotation matrix (body to world)
        # Each column is where body axis goes in world frame
        # We need: what is body +X (nose) in world frame?

        # Actually easier: build world-to-body matrix, then invert
        # World axes expressed in body frame:
        # world_Y (up) in body = up_body
        # world_X (north) in body ≈ north_body (horizontal component of mag)
        # world_Z (east) in body = east_body

        # Rotation matrix R where R @ world_vec = body_vec
        # Columns of R^T are world axes in body frame
        R_world_to_body = np.column_stack([north_body, up_body, east_body]).T
        R_body_to_world = R_world_to_body.T

        # Convert to quaternion
        r = R.from_matrix(R_body_to_world)
        x, y, z, w = r.as_quat()
        return Orientation.from_wxyz(w, x, y, z)

    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions."""
        # Ensure we take the short path
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot

        if dot > 0.9995:
            # Very close - use linear interpolation
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        s1 = np.sin((1 - t) * theta) / sin_theta
        s2 = np.sin(t * theta) / sin_theta
        return s1 * q1 + s2 * q2

    def read(self) -> Orientation:
        """Read current orientation estimate (call update() first each timestep)."""
        return self._orientation

    def reset(self, orientation: Optional[Orientation] = None) -> None:
        """Reset estimator state."""
        self._orientation = orientation or Orientation.identity()
        self._initialized = orientation is not None


class Sensors:
    """
    Bundle of sensors attached to a kernel.

    Provides unified time management for all sensors. The Scenario sets
    _time each step, which propagates to all sensors.

    Realistic sensor suite:
        - GPS: World-frame horizontal position (x, z)
        - Barometer: Altitude (better vertical accuracy than GPS)
        - Accelerometer: Body-frame specific force (what BMI088/MPU6050 output)
        - GyroSensor: Body-frame angular velocity (what BMI088/MPU6050 output)
        - Magnetometer: Body-frame magnetic field (for absolute heading)
        - FuelSensor: Propellant fraction remaining
        - AttitudeEstimator: Fused orientation from accel+gyro+mag (complementary filter)

    Legacy/convenience sensors (may use these for simpler RL setups):
        - OrientationSensor: Direct quaternion (abstracts sensor fusion)
        - VelocitySensor: World-frame velocity (abstracts GPS+accel fusion)
        - ThrustSensor: Thrust magnitude (abstracts engine telemetry)
        - IMU: Combined orientation + angular velocity
    """

    def __init__(
        self,
        kernel: Kernel,
        gps_errors: Optional[ChannelErrors] = None,
        orientation_errors: Optional[ChannelErrors] = None,
        gyro_errors: Optional[ChannelErrors] = None,
        thrust_errors: Optional[ChannelErrors] = None,
        velocity_errors: Optional[ChannelErrors] = None,
        fuel_errors: Optional[ChannelErrors] = None,
        accel_errors: Optional[ChannelErrors] = None,
        mag_errors: Optional[ChannelErrors] = None,
        baro_errors: Optional[ChannelErrors] = None,
        attitude_alpha: float = 0.98,
    ):
        self._kernel = kernel
        self._time: float = 0.0

        # Realistic sensors (what actual flight hardware provides)
        self.GPS = GPS(self._kernel, gps_errors)
        self.Barometer = Barometer(self._kernel, baro_errors)
        self.Accelerometer = Accelerometer(self._kernel, accel_errors)
        self.GyroSensor = GyroSensor(self._kernel, gyro_errors)
        self.Magnetometer = Magnetometer(self._kernel, mag_errors)
        self.FuelSensor = FuelSensor(self._kernel, fuel_errors)

        # Attitude estimator (fuses accel + gyro + mag using complementary filter)
        # Initialize with ground truth orientation to simulate pre-launch calibration.
        # Without this, the estimator starts at identity and takes seconds to converge
        # because the accelerometer reads ~0 during initial free-fall.
        initial_orientation = kernel.rocket.state.orientation
        self.AttitudeEstimator = AttitudeEstimator(
            accelerometer=self.Accelerometer,
            gyro=self.GyroSensor,
            magnetometer=self.Magnetometer,
            alpha=attitude_alpha,
            initial_orientation=initial_orientation,
        )

        # Legacy/convenience sensors (abstract away sensor fusion)
        self.OrientationSensor = OrientationSensor(self._kernel, orientation_errors)
        self.VelocitySensor = VelocitySensor(self._kernel, velocity_errors)
        self.ThrustSensor = ThrustSensor(self._kernel, thrust_errors)
        self.IMU = IMU(self._kernel, orientation_errors, gyro_errors)

    def set_time(self, t: float) -> None:
        """
        Set the current simulation time for all sensors.

        Called by Scenario each step to enable time-dependent errors.
        """
        self._time = t
        # Realistic sensors
        self.GPS._time = t
        self.Barometer._time = t
        self.Accelerometer._time = t
        self.GyroSensor._time = t
        self.Magnetometer._time = t
        self.FuelSensor._time = t
        # Legacy sensors
        self.OrientationSensor._time = t
        self.VelocitySensor._time = t
        self.ThrustSensor._time = t
        self.IMU._set_time(t)

    def update_attitude_estimator(self, dt: float) -> None:
        """
        Update the attitude estimator with the current timestep.

        Should be called once per simulation step after set_time().
        The estimator fuses accelerometer, gyroscope, and magnetometer
        readings to produce an orientation estimate.
        """
        self.AttitudeEstimator.update(dt)
