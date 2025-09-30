import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from simulacra.core.utils import clamp


@dataclass
class State:
    """
    1D vertical state for the minimal sim.

    Attributes:
        z: altitude in metres (upward positive)
        v: vertical velocity in metres/second (upward positive)
    """

    z: float
    v: float

    def as_array(self) -> np.ndarray:
        """Return state as a float ndarray [z, v]."""
        return np.array([self.z, self.v], dtype=float)

    @staticmethod
    def from_array(a: np.ndarray) -> "State":
        """Create State from ndarray-like [z, v]."""
        if not isinstance(a, np.ndarray):
            a = np.array(a, dtype=float)
        if a.shape[0] < 2:
            raise ValueError("array must have length >= 2")
        return State(float(a[0]), float(a[1]))

    def __post_init__(self):
        if not math.isfinite(self.z) or not math.isfinite(self.v):
            raise ValueError("State values must be finite numbers")


class Rocket:
    """
    Minimal point-mass rocket in 1D vertical axis.

    Parameters
    ----------
    mass_kg : positive float
        Mass in kilograms.
    area : float
        Reference area for drag (m^2).
    cd : float
        Drag coefficient (dimensionless).
    g : float
        Gravitational acceleration (m/s^2), positive value (9.81).
    rho0 : float
        Air density at sea level (kg/m^3) used in a simple drag model.
    """

    def __init__(self, mass_kg=10.0, area=0.05, cd=0.6, g=9.81, rho0=1.225):
        if mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")

        self.m = float(mass_kg)
        self.area = float(area)
        self.cd = float(cd)
        self.g = float(g)
        self.rho0 = float(rho0)

    def drag(self, v: float) -> float:
        """Quadratic drag opposing motion: 0.5 * rho * Cd * A * v * |v|."""
        # Works whether v positive or negative: sign preserved by v*abs(v)
        return 0.5 * self.rho0 * self.cd * self.area * v * abs(v)

    def derivatives(self, state: State, thrust: float) -> np.ndarray:
        """
        Compute state derivatives [dz/dt, dv/dt].

        thrust:
            Upward positive Newtons (float).
            Can be zero or positive for our simple model.
        """
        # simple validation
        if not math.isfinite(thrust):
            raise ValueError("thrust must be finite")
        v = state.v
        dzdt = v
        D = self.drag(v)
        dvdt = (thrust - self.m * self.g - D) / self.m
        return np.array([dzdt, dvdt], dtype=float)


class PID:
    """
    Simple PID controller.
    error = setpoint - measurement
    output in [u_min, u_max]
    """

    def __init__(
        self,
        kp: float = 2.0,  # proportional gain
        ki: float = 0.8,  # integral gain
        kd: float = 0.6,  # derivative gain
        u_min: float = 0.0,
        u_max: float = 1.0,
        integrator_limit: Optional[float] = None,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.integrator_limit = (
            None if integrator_limit is None else abs(float(integrator_limit))
        )

        self.integrator = 0.0
        self.last_error: Optional[float] = None
        self.last_output: float = 0.0

    def reset(self) -> None:
        """Reset integrator and history."""
        self.integrator = 0.0
        self.last_error = None
        self.last_output = 0.0

    def _clamp_integrator(self):
        if self.integrator_limit is not None:
            self.integrator = clamp(
                self.integrator, -self.integrator_limit, self.integrator_limit
            )

    def __call__(self, error: Optional[float], dt: float) -> float:
        """Compute control output for given error and timestep."""
        if error is None:
            return self.last_output  # hold

        if dt <= 0.0:
            u_unsat = self.kp * error
            u = clamp(u_unsat, self.u_min, self.u_max)
            self.last_output = u
            return u

        # terms
        deriv = 0.0 if self.last_error is None else (error - self.last_error) / dt
        self.integrator += error * dt
        self._clamp_integrator()

        u_unsat = self.kp * error + self.ki * self.integrator + self.kd * deriv
        u = clamp(u_unsat, self.u_min, self.u_max)

        self.last_error = error
        self.last_output = u

        return u
