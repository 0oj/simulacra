from simulacra.steps.step01_vertical_hop.sim import PID


def test_pid_basic_response():
    p = PID(kp=1.0, ki=0.0, kd=0.0, u_min=0.0, u_max=10.0)
    u = p(error=2.0, dt=0.1)
    assert 1.9 < u < 2.1  # about 2.0


def test_pid_integrator_limit():
    p = PID(kp=0.0, ki=10.0, kd=0.0, integrator_limit=1.0, u_min=-10, u_max=10)
    for _ in range(100):
        p(error=1.0, dt=0.1)
    assert abs(p.integrator) <= 1.01


def test_pid_holds_on_none():
    p = PID(kp=1.0)
    out1 = p(error=1.0, dt=0.1)
    out2 = p(error=None, dt=0.1)
    assert out2 == out1


def test_pid_dt_zero_safe():
    p = PID(kp=2.0)
    u = p(error=1.0, dt=0.0)
    assert 0.0 <= u <= 1.0
