"""
Utility functions. Small tools that serve both god and prisoner,
like clamping values to safe bounds.
"""
from typing import TypeVar, Optional

T = TypeVar("T", bound=float)

def clamp(x: Optional[T], lo: T, hi: T) -> Optional[T]:
    if x is None:
        return None
    return lo if x < lo else (hi if x > hi else x)
