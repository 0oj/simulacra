from typing import TypeVar

T = TypeVar("T", bound="SupportsComparison")

class SupportsComparison:
    def __lt__(self, other): ...
    def __gt__(self, other): ...

def clamp(x: T, lo: T, hi: T) -> T:
    """
    Clamp a value between a lower and upper bound.

    Args:
        x: The value to clamp.
        lo: The lower bound.
        hi: The upper bound.

    Returns:
        The clamped value.
    """

    return lo if x < lo else (hi if x > hi else x)
