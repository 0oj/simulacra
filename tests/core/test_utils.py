import pytest

from simulacra.core.utils import clamp


def test_clamp_basic():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0
    assert clamp(2, 0, 3) == 2
    assert clamp(0, 0, 3) == 0
    assert clamp(3, 0, 3) == 3


def test_clamp_none_raises():
    with pytest.raises(TypeError):
        clamp(None, 0, 1)
