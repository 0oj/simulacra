"""
Tests for the sensor error pipeline.

Verifies that:
- Default sensors return truth (IdentityError)
- Noisy sensors corrupt truth with reproducible randomness
- Per-channel errors apply independently
- Time-dependent errors (drift) work correctly
"""

import pytest
import numpy as np

from simulacra.noumena import Kernel, Environment
from simulacra.noumena.rocket import Rocket, RocketProps, RocketState
from simulacra.phenomena.sensors import (
    GPS,
    GyroSensor,
    OrientationSensor,
    Sensors,
    ThrustSensor,
    VelocitySensor,
    Accelerometer,
    Magnetometer,
    Barometer,
)
from simulacra.phenomena.errors import (
    Bias,
    CompositeError,
    GaussianNoise,
    IdentityError,
    RandomWalk,
    Quantization,
    Saturation,
    ScaleBias,
)
from simulacra.physics import Orientation, RocketVec3, WorldVec3


@pytest.fixture
def kernel():
    """Create a kernel with known state for testing."""
    props = RocketProps(
        length=10.0,
        mass=1000.0,
        center_of_mass_offset=RocketVec3(5.0, 0.0, 0.0),
        inertia_diag=(100.0, 1000.0, 1000.0),
        thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
    )
    state = RocketState(
        position=WorldVec3(100.0, 200.0, 300.0),
        velocity=WorldVec3(10.0, 20.0, 30.0),
        orientation=Orientation.identity(),
        angular_velocity_body=RocketVec3(0.1, 0.2, 0.3),
        thrust_body=RocketVec3(5000.0, 0.0, 0.0),
    )
    rocket = Rocket(props, state)
    env = Environment()
    return Kernel(rocket, env)


class TestIdentityError:
    """Tests for IdentityError (default sensor behavior)."""

    def test_returns_truth(self):
        error = IdentityError()
        assert error.apply(42.0, 0.0) == 42.0
        assert error.apply(-3.14, 1.0) == -3.14


class TestGaussianNoise:
    """Tests for GaussianNoise error."""

    def test_zero_std_returns_truth(self):
        error = GaussianNoise(mean=0.0, std=0.0)
        assert error.apply(42.0, 0.0) == 42.0

    def test_seeded_reproducibility(self):
        """Same seed produces same sequence."""
        error1 = GaussianNoise(mean=0.0, std=1.0, seed=12345)
        error2 = GaussianNoise(mean=0.0, std=1.0, seed=12345)

        results1 = [error1.apply(0.0, float(i)) for i in range(10)]
        results2 = [error2.apply(0.0, float(i)) for i in range(10)]

        assert results1 == results2

    def test_different_seeds_differ(self):
        """Different seeds produce different sequences."""
        error1 = GaussianNoise(mean=0.0, std=1.0, seed=12345)
        error2 = GaussianNoise(mean=0.0, std=1.0, seed=54321)

        results1 = [error1.apply(0.0, float(i)) for i in range(10)]
        results2 = [error2.apply(0.0, float(i)) for i in range(10)]

        assert results1 != results2

    def test_mean_offset(self):
        """Mean shifts the distribution."""
        error = GaussianNoise(mean=100.0, std=0.0, seed=42)
        assert error.apply(0.0, 0.0) == 100.0


class TestBias:
    """Tests for Bias error."""

    def test_constant_offset(self):
        error = Bias(offset=5.0)
        assert error.apply(10.0, 0.0) == 15.0
        assert error.apply(10.0, 1.0) == 15.0

    def test_negative_offset(self):
        error = Bias(offset=-3.0)
        assert error.apply(10.0, 0.0) == 7.0


class TestScaleBias:
    """Tests for ScaleBias error."""

    def test_scale_factor(self):
        error = ScaleBias(scale=1.5)
        assert error.apply(10.0, 0.0) == 15.0

    def test_identity_scale(self):
        error = ScaleBias(scale=1.0)
        assert error.apply(42.0, 0.0) == 42.0


class TestCompositeError:
    """Tests for CompositeError chaining."""

    def test_empty_chain_returns_truth(self):
        error = CompositeError(errors=[])
        assert error.apply(42.0, 0.0) == 42.0

    def test_single_error(self):
        error = CompositeError(errors=[Bias(offset=5.0)])
        assert error.apply(10.0, 0.0) == 15.0

    def test_chain_order(self):
        """Errors apply left-to-right."""
        # Bias first, then scale: (10 + 5) * 2 = 30
        error1 = CompositeError(errors=[Bias(offset=5.0), ScaleBias(scale=2.0)])
        assert error1.apply(10.0, 0.0) == 30.0

        # Scale first, then bias: (10 * 2) + 5 = 25
        error2 = CompositeError(errors=[ScaleBias(scale=2.0), Bias(offset=5.0)])
        assert error2.apply(10.0, 0.0) == 25.0

    def test_noise_plus_bias(self):
        """Typical sensor model: noise + bias."""
        error = CompositeError(
            errors=[
                GaussianNoise(mean=0.0, std=1.0, seed=42),
                Bias(offset=10.0),
            ]
        )
        # Result should be around 10 (bias) + noise
        result = error.apply(0.0, 0.0)
        assert 5.0 < result < 15.0  # Reasonable bounds


class TestQuantization:
    """Tests for Quantization error."""

    def test_rounds_to_step(self):
        error = Quantization(step_size=0.5)
        assert error.apply(1.2, 0.0) == 1.0
        assert error.apply(1.3, 0.0) == 1.5
        # Python uses banker's rounding (round half to even)
        assert error.apply(1.75, 0.0) == 2.0

    def test_zero_step_returns_truth(self):
        error = Quantization(step_size=0.0)
        assert error.apply(1.234, 0.0) == 1.234


class TestSaturation:
    """Tests for Saturation error."""

    def test_clamps_to_range(self):
        error = Saturation(min_val=0.0, max_val=10.0)
        assert error.apply(5.0, 0.0) == 5.0
        assert error.apply(-5.0, 0.0) == 0.0
        assert error.apply(15.0, 0.0) == 10.0


class TestRandomWalk:
    """Tests for RandomWalk drift error."""

    def test_accumulates_drift(self):
        """Drift increases over time."""
        error = RandomWalk(sigma=1.0, seed=42)

        # First reading establishes time baseline
        r0 = error.apply(0.0, 0.0)
        assert r0 == 0.0  # No drift on first read

        # Subsequent readings accumulate drift
        r1 = error.apply(0.0, 1.0)
        r2 = error.apply(0.0, 2.0)

        # Drift should be non-zero (with high probability given sigma=1)
        assert r1 != 0.0 or r2 != 0.0

    def test_seeded_reproducibility(self):
        """Same seed produces same drift sequence."""
        error1 = RandomWalk(sigma=1.0, seed=12345)
        error2 = RandomWalk(sigma=1.0, seed=12345)

        results1 = [error1.apply(0.0, float(i)) for i in range(10)]
        results2 = [error2.apply(0.0, float(i)) for i in range(10)]

        assert results1 == results2


class TestGPSSensor:
    """Tests for GPS sensor (horizontal position only)."""

    def test_default_returns_truth(self, kernel):
        """GPS returns horizontal position (x, z), not altitude."""
        gps = GPS(kernel)
        reading = gps.read()

        assert reading.x == 100.0
        assert reading.z == 300.0
        # GPS doesn't have .y - use Barometer for altitude
        assert not hasattr(reading, 'y')

    def test_per_channel_bias(self, kernel):
        """Each channel can have different errors."""
        gps = GPS(
            kernel,
            errors={
                "x": Bias(offset=1.0),
                "z": Bias(offset=3.0),
            },
        )
        reading = gps.read()

        assert reading.x == 101.0
        assert reading.z == 303.0

    def test_partial_channel_errors(self, kernel):
        """Only specified channels get errors."""
        gps = GPS(kernel, errors={"x": Bias(offset=10.0)})
        reading = gps.read()

        assert reading.x == 110.0
        assert reading.z == 300.0  # No error on z

    def test_noisy_gps_reproducible(self, kernel):
        """Seeded noise produces reproducible readings."""
        gps1 = GPS(kernel, errors={"x": GaussianNoise(std=1.0, seed=42)})
        gps2 = GPS(kernel, errors={"x": GaussianNoise(std=1.0, seed=42)})

        r1 = gps1.read()
        r2 = gps2.read()

        assert r1.x == r2.x


class TestGyroSensor:
    """Tests for GyroSensor with per-channel errors."""

    def test_default_returns_truth(self, kernel):
        gyro = GyroSensor(kernel)
        reading = gyro.read()

        assert reading.x == pytest.approx(0.1)
        assert reading.y == pytest.approx(0.2)
        assert reading.z == pytest.approx(0.3)

    def test_per_channel_bias(self, kernel):
        gyro = GyroSensor(
            kernel,
            errors={
                "x": Bias(offset=0.01),
                "y": Bias(offset=0.02),
                "z": Bias(offset=0.03),
            },
        )
        reading = gyro.read()

        assert reading.x == pytest.approx(0.11)
        assert reading.y == pytest.approx(0.22)
        assert reading.z == pytest.approx(0.33)


class TestOrientationSensor:
    """Tests for OrientationSensor with Euler perturbations."""

    def test_default_returns_truth(self, kernel):
        sensor = OrientationSensor(kernel)
        reading = sensor.read()

        # Identity orientation
        arr = reading.as_array()
        assert arr[0] == pytest.approx(1.0)  # w
        assert arr[1] == pytest.approx(0.0)  # x
        assert arr[2] == pytest.approx(0.0)  # y
        assert arr[3] == pytest.approx(0.0)  # z

    def test_roll_perturbation(self, kernel):
        """Small roll perturbation changes orientation."""
        sensor = OrientationSensor(
            kernel,
            errors={"roll": Bias(offset=0.1)},  # 0.1 rad roll error
        )
        reading = sensor.read()

        # Should no longer be identity
        arr = reading.as_array()
        assert not np.allclose(arr, [1.0, 0.0, 0.0, 0.0])


class TestVelocitySensor:
    """Tests for VelocitySensor with per-channel errors."""

    def test_default_returns_truth(self, kernel):
        sensor = VelocitySensor(kernel)
        reading = sensor.read()

        assert reading.x == 10.0
        assert reading.y == 20.0
        assert reading.z == 30.0

    def test_per_channel_noise(self, kernel):
        sensor = VelocitySensor(
            kernel,
            errors={
                "x": GaussianNoise(std=1.0, seed=42),
                "y": GaussianNoise(std=1.0, seed=43),
                "z": GaussianNoise(std=1.0, seed=44),
            },
        )
        reading = sensor.read()

        # Should be close to truth but not exact
        assert reading.x != 10.0
        assert reading.y != 20.0
        assert reading.z != 30.0


class TestThrustSensor:
    """Tests for ThrustSensor."""

    def test_default_returns_truth(self, kernel):
        sensor = ThrustSensor(kernel)
        reading = sensor.read()

        assert reading == pytest.approx(5000.0)

    def test_with_scale_error(self, kernel):
        sensor = ThrustSensor(
            kernel,
            errors={"magnitude": ScaleBias(scale=1.02)},  # 2% overread
        )
        reading = sensor.read()

        assert reading == pytest.approx(5100.0)


class TestSensorsBundle:
    """Tests for the unified Sensors bundle."""

    def test_default_all_truth(self, kernel):
        """Default sensors return truth."""
        sensors = Sensors(kernel)

        gps = sensors.GPS.read()
        assert gps.x == 100.0

        velocity = sensors.VelocitySensor.read()
        assert velocity.x == 10.0

    def test_time_propagation(self, kernel):
        """set_time propagates to all sensors."""
        sensors = Sensors(kernel)
        sensors.set_time(5.0)

        assert sensors._time == 5.0
        assert sensors.GPS._time == 5.0
        assert sensors.GyroSensor._time == 5.0
        assert sensors.OrientationSensor._time == 5.0
        assert sensors.ThrustSensor._time == 5.0
        assert sensors.VelocitySensor._time == 5.0

    def test_custom_errors_per_sensor(self, kernel):
        """Each sensor type can have custom errors."""
        sensors = Sensors(
            kernel,
            gps_errors={"x": Bias(offset=1.0)},
            velocity_errors={"y": Bias(offset=2.0)},
        )

        gps = sensors.GPS.read()
        assert gps.x == 101.0

        velocity = sensors.VelocitySensor.read()
        assert velocity.y == 22.0

    def test_imu_backward_compatibility(self, kernel):
        """IMU still works as combined sensor."""
        sensors = Sensors(kernel)
        orientation, angular_velocity = sensors.IMU.read()

        # Should return valid types
        assert isinstance(orientation, Orientation)
        assert isinstance(angular_velocity, RocketVec3)


class TestSensorErrorsWithScenario:
    """Integration tests verifying errors work in Scenario context."""

    def test_time_dependent_errors(self, kernel):
        """Time-dependent errors (like drift) receive correct time."""
        from simulacra.noumena.scenario import Scenario
        from simulacra.onboard.controller_legacy import IdentityController

        # Create sensors with drift error on GPS x-channel
        sensors = Sensors(
            kernel,
            gps_errors={"x": RandomWalk(sigma=1.0, seed=42)},
        )

        scenario = Scenario(
            kernel=kernel,
            controller=IdentityController(),
            sensors=sensors,
            dt=0.1,
        )

        # Read GPS before any steps (time=0)
        initial_x = sensors.GPS.read().x

        # Step the scenario to advance time
        # set_time is called at beginning of step, before time increments
        scenario.step()  # set_time(0.0), then _time -> 0.1
        scenario.step()  # set_time(0.1), then _time -> 0.2

        # Drift should have accumulated
        later_x = sensors.GPS.read().x

        # Verify time was propagated (sensors._time is the time at last set_time call)
        assert sensors._time == pytest.approx(0.1)
        # Scenario._time has advanced past the sensor time
        assert scenario._time == pytest.approx(0.2)


class TestAccelerometer:
    """Tests for Accelerometer sensor (BMI088/MPU6050 style)."""

    def test_measures_thrust_when_thrusting(self, kernel):
        """When thrusting, accelerometer reads thrust/mass in body frame."""
        accel = Accelerometer(kernel)
        reading = accel.read()

        # Kernel has thrust_body = (5000, 0, 0), mass = 1000
        # Expected acceleration ≈ 5 m/s² along body X
        assert reading.x == pytest.approx(5.0, rel=0.1)
        assert reading.y == pytest.approx(0.0, abs=0.1)
        assert reading.z == pytest.approx(0.0, abs=0.1)

    def test_reads_gravity_when_stationary_on_ground(self):
        """On ground with no thrust, reads normal force (opposite gravity)."""
        from simulacra.noumena.rocket import get_upright_orientation

        props = RocketProps(
            length=10.0,
            mass=1000.0,
            center_of_mass_offset=RocketVec3(5.0, 0.0, 0.0),
            inertia_diag=(100.0, 1000.0, 1000.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        )
        state = RocketState(
            position=WorldVec3(0.0, 0.0, 0.0),  # On ground
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=get_upright_orientation(),  # Upright
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(0.0, 0.0, 0.0),  # No thrust
        )
        rocket = Rocket(props, state)
        env = Environment()
        kernel = Kernel(rocket, env)
        kernel.ground_contact.on_ground = True  # Simulate ground contact

        accel = Accelerometer(kernel)
        reading = accel.read()

        # Upright rocket: body +X points world +Y (up)
        # Normal force is +9.81 in world +Y, which maps to body +X
        assert reading.x == pytest.approx(9.81, rel=0.01)
        assert reading.y == pytest.approx(0.0, abs=0.1)
        assert reading.z == pytest.approx(0.0, abs=0.1)

    def test_per_channel_bias(self, kernel):
        """Per-channel errors apply correctly."""
        accel = Accelerometer(
            kernel,
            errors={
                "x": Bias(offset=0.5),
                "y": Bias(offset=-0.3),
            },
        )
        reading = accel.read()

        # Expect thrust/mass + bias on x
        assert reading.x == pytest.approx(5.5, rel=0.1)
        assert reading.y == pytest.approx(-0.3, abs=0.1)


class TestMagnetometer:
    """Tests for Magnetometer sensor."""

    def test_default_field_identity_orientation(self, kernel):
        """With identity orientation, body frame == world frame."""
        # Kernel fixture has identity orientation
        mag = Magnetometer(kernel)
        reading = mag.read()

        # Default field is (0, -0.5, -0.866) in world frame
        # With identity orientation, body frame equals world frame
        assert reading.x == pytest.approx(0.0, abs=0.01)
        assert reading.y == pytest.approx(-0.5, abs=0.01)
        assert reading.z == pytest.approx(-0.866, abs=0.01)

    def test_field_rotates_with_orientation(self):
        """Magnetic field reading changes with rocket orientation."""
        from simulacra.noumena.rocket import get_upright_orientation

        props = RocketProps(
            length=10.0,
            mass=1000.0,
            center_of_mass_offset=RocketVec3(5.0, 0.0, 0.0),
            inertia_diag=(100.0, 1000.0, 1000.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        )
        state = RocketState(
            position=WorldVec3(0.0, 50.0, 0.0),
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=get_upright_orientation(),
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(0.0, 0.0, 0.0),
        )
        rocket = Rocket(props, state)
        env = Environment()
        kernel = Kernel(rocket, env)

        mag = Magnetometer(kernel)
        reading = mag.read()

        # Upright: body +X points world +Y, body +Y points world -X
        # World field (0, -0.5, -0.866) should map differently
        # Body X ← World Y component: -0.5
        # Body Y ← World -X component: 0
        # Body Z ← World Z component: -0.866
        assert reading.x == pytest.approx(-0.5, abs=0.01)
        assert reading.z == pytest.approx(-0.866, abs=0.01)

    def test_custom_world_field(self, kernel):
        """Can configure different magnetic field."""
        custom_field = WorldVec3(1.0, 0.0, 0.0)  # Pure X
        mag = Magnetometer(kernel, world_field=custom_field)
        reading = mag.read()

        # Identity orientation: body == world
        assert reading.x == pytest.approx(1.0, abs=0.01)
        assert reading.y == pytest.approx(0.0, abs=0.01)
        assert reading.z == pytest.approx(0.0, abs=0.01)


class TestBarometer:
    """Tests for Barometer altitude sensor."""

    def test_default_returns_altitude(self, kernel):
        """Reads altitude (y position) by default."""
        baro = Barometer(kernel)
        reading = baro.read()

        # Kernel fixture has position.y = 200.0
        assert reading == pytest.approx(200.0)

    def test_with_bias(self, kernel):
        """Altitude bias (e.g., pressure offset) applies."""
        baro = Barometer(
            kernel,
            errors={"altitude": Bias(offset=-5.0)},  # Reads 5m low
        )
        reading = baro.read()

        assert reading == pytest.approx(195.0)

    def test_with_noise(self, kernel):
        """Altitude noise (e.g., pressure fluctuation) applies."""
        baro = Barometer(
            kernel,
            errors={"altitude": GaussianNoise(std=0.5, seed=42)},
        )
        reading = baro.read()

        # Should be close to 200 but not exact
        assert reading != 200.0
        assert abs(reading - 200.0) < 3.0  # Within 3 sigma


class TestRealisticSensorBundle:
    """Tests for realistic sensor integration in the Sensors bundle."""

    def test_realistic_sensors_available(self, kernel):
        """All realistic sensors are accessible."""
        sensors = Sensors(kernel)

        # Should have all realistic sensors
        assert hasattr(sensors, 'Accelerometer')
        assert hasattr(sensors, 'Magnetometer')
        assert hasattr(sensors, 'Barometer')

        # And they should be readable
        accel = sensors.Accelerometer.read()
        mag = sensors.Magnetometer.read()
        baro = sensors.Barometer.read()

        assert isinstance(accel, RocketVec3)
        assert isinstance(mag, RocketVec3)
        assert isinstance(baro, float)

    def test_time_propagates_to_new_sensors(self, kernel):
        """set_time updates all sensors including new ones."""
        sensors = Sensors(kernel)
        sensors.set_time(3.14)

        assert sensors.Accelerometer._time == 3.14
        assert sensors.Magnetometer._time == 3.14
        assert sensors.Barometer._time == 3.14


class TestAttitudeEstimator:
    """Tests for the AttitudeEstimator (complementary filter)."""

    def test_estimator_exists_in_bundle(self, kernel):
        """Sensors bundle includes AttitudeEstimator."""
        sensors = Sensors(kernel)
        assert hasattr(sensors, 'AttitudeEstimator')

    def test_initial_update_uses_reference(self):
        """First update initializes from accelerometer + magnetometer."""
        from simulacra.noumena.rocket import get_upright_orientation

        # Create a rocket that's upright
        props = RocketProps(
            length=10.0,
            mass=1000.0,
            center_of_mass_offset=RocketVec3(5.0, 0.0, 0.0),
            inertia_diag=(100.0, 1000.0, 1000.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        )
        state = RocketState(
            position=WorldVec3(0.0, 50.0, 0.0),
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=get_upright_orientation(),
            angular_velocity_body=RocketVec3(0.0, 0.0, 0.0),
            thrust_body=RocketVec3(0.0, 0.0, 0.0),
        )
        rocket = Rocket(props, state)
        env = Environment()
        kernel = Kernel(rocket, env)
        kernel.ground_contact.on_ground = True

        sensors = Sensors(kernel)

        # First update should initialize from accel+mag
        sensors.update_attitude_estimator(dt=0.02)
        estimated = sensors.AttitudeEstimator.read()

        # Should be approximately upright (nose pointing up = body +X to world +Y)
        # The exact value depends on the mag field, but it should be reasonable
        assert isinstance(estimated, Orientation)

    def test_gyro_integration_short_term(self, kernel):
        """Estimator follows gyro for short-term changes."""
        sensors = Sensors(kernel)

        # Initialize
        sensors.update_attitude_estimator(dt=0.02)
        initial = sensors.AttitudeEstimator.read()

        # With alpha=0.98, the estimator heavily trusts gyro short-term
        # The kernel fixture has angular_velocity_body = (0.1, 0.2, 0.3)
        # After one step, orientation should change slightly

        sensors.update_attitude_estimator(dt=0.02)
        after_step = sensors.AttitudeEstimator.read()

        # Orientation should have changed (due to angular velocity)
        initial_arr = initial.as_array()
        after_arr = after_step.as_array()
        assert not np.allclose(initial_arr, after_arr, atol=1e-6)

    def test_estimator_reset(self, kernel):
        """Estimator can be reset to a known orientation."""
        sensors = Sensors(kernel)

        # Do some updates
        sensors.update_attitude_estimator(dt=0.02)
        sensors.update_attitude_estimator(dt=0.02)

        # Reset to identity
        sensors.AttitudeEstimator.reset(Orientation.identity())
        reset_orientation = sensors.AttitudeEstimator.read()

        # Should be exactly identity
        arr = reset_orientation.as_array()
        assert arr[0] == pytest.approx(1.0)  # w
        assert arr[1] == pytest.approx(0.0)  # x
        assert arr[2] == pytest.approx(0.0)  # y
        assert arr[3] == pytest.approx(0.0)  # z

    def test_alpha_parameter_affects_blend(self):
        """Different alpha values change gyro vs reference trust."""
        from simulacra.phenomena.sensors import AttitudeEstimator, Accelerometer, GyroSensor, Magnetometer

        # Create two estimators with different alpha
        props = RocketProps(
            length=10.0,
            mass=1000.0,
            center_of_mass_offset=RocketVec3(5.0, 0.0, 0.0),
            inertia_diag=(100.0, 1000.0, 1000.0),
            thrust_application_offset=RocketVec3(0.0, 0.0, 0.0),
        )
        state = RocketState(
            position=WorldVec3(0.0, 50.0, 0.0),
            velocity=WorldVec3(0.0, 0.0, 0.0),
            orientation=Orientation.identity(),
            angular_velocity_body=RocketVec3(0.5, 0.0, 0.0),  # Rotating
            thrust_body=RocketVec3(0.0, 0.0, 0.0),
        )
        rocket = Rocket(props, state)
        env = Environment()
        kernel = Kernel(rocket, env)

        accel = Accelerometer(kernel)
        gyro = GyroSensor(kernel)
        mag = Magnetometer(kernel)

        # High alpha = trust gyro more (more drift, smoother)
        estimator_high = AttitudeEstimator(accel, gyro, mag, alpha=0.99)
        # Low alpha = trust reference more (less drift, noisier)
        estimator_low = AttitudeEstimator(accel, gyro, mag, alpha=0.90)

        # Run several updates
        for _ in range(10):
            estimator_high.update(dt=0.02)
            estimator_low.update(dt=0.02)

        # They should differ (different blend of gyro vs reference)
        high_arr = estimator_high.read().as_array()
        low_arr = estimator_low.read().as_array()
        assert not np.allclose(high_arr, low_arr, atol=1e-3)
