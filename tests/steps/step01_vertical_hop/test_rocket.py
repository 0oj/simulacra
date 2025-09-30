import numpy as np

from simulacra.steps.step01_vertical_hop.sim import Rocket, State


def test_derivatives_freefall_no_drag():
    r = Rocket(mass_kg=5.0, area=0.0, cd=0.0)  # zero drag
    s = State(z=10.0, v=0.0)
    d = r.derivatives(s, thrust=0.0)
    assert np.allclose(d, np.array([0.0, -r.g]), atol=1e-12)


def test_drag_opposes_motion():
    r = Rocket(mass_kg=5.0, area=0.1, cd=1.0)
    # upward velocity positive -> drag should be positive (subtracting in dvdt)
    d_up = r.drag(2.0)
    # downward velocity negative -> drag should be negative magnitude (opposes motion)
    d_down = r.drag(-2.0)
    assert d_up == -d_down
    assert abs(d_up) > 0.0
