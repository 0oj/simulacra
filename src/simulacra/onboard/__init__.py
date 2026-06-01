"""
The rocket's flight computer. This is the prisoner's mind,
making decisions based on imperfect sensor readings.
"""

from ..onboard.controller_legacy import Controller
from ..onboard.pid_hover import PIDHoverController
from ..onboard.hover_controller import HoverController, HoverControllerConfig
from ..onboard.landing_controller import LandingController, LandingControllerConfig
from ..onboard.wasd_controller import WASDController, WASDControllerConfig

__all__ = [
    "Controller",
    "PIDHoverController",
    "HoverController",
    "HoverControllerConfig",
    "LandingController",
    "LandingControllerConfig",
    "WASDController",
    "WASDControllerConfig",
]
