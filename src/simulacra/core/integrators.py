from typing import Callable, TypeAlias

import numpy as np

StateVec: TypeAlias = np.ndarray
ODE_Func: TypeAlias = Callable[[StateVec], StateVec]
TimeStep: TypeAlias = float


def rk4_step(ode_f: ODE_Func, state_vec: StateVec, dt: TimeStep) -> StateVec:
    """
    Generic RK4 integrator for vector ODEs.

    Args:
        f: The ODE function.
            Must take a single argument (the state vector) and return a vector of the same shape.
        state_vec: The state vector.
        dt: The time step.

    Returns:
        The updated state vector.
        Of the same shape as input `state_vec`.
    """

    k1 = ode_f(state_vec)
    k2 = ode_f(state_vec + 0.5 * dt * k1)
    k3 = ode_f(state_vec + 0.5 * dt * k2)
    k4 = ode_f(state_vec + dt * k3)

    return state_vec + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
