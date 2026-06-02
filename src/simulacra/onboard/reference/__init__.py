"""Reference controllers shipped with simulacra.

Engineers writing their own controllers can study these for working
examples on the new Controller interface. They are also importable
from scenarios as defaults.
"""
from .pid_hover import PIDHoverController
from .pid_hopper import HopperConfig, PIDHopperController

__all__ = ["HopperConfig", "PIDHoverController", "PIDHopperController"]
