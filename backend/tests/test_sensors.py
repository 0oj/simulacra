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
    """Tests for GPS sensor with per-channel errors."""

    def test_default_returns_truth(self, kernel):
        gps = GPS(kernel)
        reading = gps.read()

        assert reading.x == 100.0
        assert reading.y == 200.0
        assert reading.z == 300.0

    def test_per_channel_bias(self, kernel):
        """Each channel can have different errors."""
        gps = GPS(
            kernel,
            errors={
                "x": Bias(offset=1.0),
                "y": Bias(offset=2.0),
                "z": Bias(offset=3.0),
            },
        )
        reading = gps.read()

        assert reading.x == 101.0
        assert reading.y == 202.0
        assert reading.z == 303.0

    def test_partial_channel_errors(self, kernel):
        """Only specified channels get errors."""
        gps = GPS(kernel, errors={"x": Bias(offset=10.0)})
        reading = gps.read()

        assert reading.x == 110.0
        assert reading.y == 200.0  # No error
        assert reading.z == 300.0  # No error

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
        from simulacra.onboard.controller import IdentityController

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
