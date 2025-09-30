from math import exp

import numpy as np

from simulacra.core.integrators import rk4_step


def test_rk4_scalar_exponential():
    """
    Integrate dy/dt = -lambda * y with RK4 and compare to analytical solution y(t)=y0*exp(-lambda*t).
    """

    lam = 1.5

    def f(y):
        # y is a 1-element ndarray
        return -lam * y

    y0 = np.array([1.0])
    dt = 0.01
    T = 1.0
    steps = int(T / dt)
    y = y0.copy()
    for _ in range(steps):
        y = rk4_step(f, y, dt)
    y_exact = np.array([y0[0] * exp(-lam * T)])
    err = np.abs(y - y_exact)
    assert err < 2e-6, f"rk4 scalar error too large: {err}"


def test_rk4_vector_decoupled():
    """
    Integrate a tiny 2D linear system:
      dy0/dt = -1.0 * y0
      dy1/dt = -2.0 * y1
    Solutions are decoupled exponentials.
    """

    lambdas = np.array([1.0, 2.0])

    def f(y):
        return -lambdas * y

    y0 = np.array([2.0, 1.0])
    dt = 0.02
    T = 0.6
    steps = int(T / dt)
    y = y0.copy()
    for _ in range(steps):
        y = rk4_step(f, y, dt)

    y_exact = y0 * np.exp(-lambdas * T)
    max_err = np.max(np.abs(y - y_exact))
    assert max_err < 1e-5, f"rk4 vector error too large: {max_err}"
