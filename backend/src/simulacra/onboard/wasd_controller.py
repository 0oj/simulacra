import sys
import select
import tty
import termios
from .controller import Controller
from ..phenomena.sensors import Sensors
from ..phenomena.actuators import GimbalActuator3D, GimbalCommand3D


class WASDController(Controller):
    """
    Controller that listens for WASD keypresses each run() to control the gimbal/thrust vector.

    By convention:
      - W/S: pitch up/down (inner gimbal: +Y axis, green)
      - A/D: yaw left/right (outer gimbal: +Z axis, blue)
    """

    def __init__(self, gimbal_actuator: GimbalActuator3D):
        self.sensors: Sensors | None = None
        self.gimbal_actuator = gimbal_actuator

        # Gimbal angles (radians)
        self.outer_angle: float = 0.0  # yaw (about Z/blue), A/D keys
        self.inner_angle: float = 0.0  # pitch (about Y/green), W/S keys

        # Increment per keypress (radians)
        self.step: float = 0.02

    def with_sensors(self, sensors: Sensors) -> "WASDController":
        self.sensors = sensors
        return self

    def _get_keyboard_input(self, prompt: str = "WASD> ") -> str:
        """
        Non-blocking, prompt for one line of input.
        If running in a TTY, disables line buffering to allow immediate key read.
        """
        sys.stdout.write(prompt)
        sys.stdout.flush()
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            # Wait for input (but non-blocking)
            r, _, _ = select.select([fd], [], [], 0.1)
            if r:
                ch = sys.stdin.read(1)
                sys.stdout.write("\n")
                sys.stdout.flush()
                return ch
            else:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return ""
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def run(self) -> None:
        if self.sensors is None:
            raise ValueError("Sensors not attached to controller")

        # Prompt user each timestep for WASD input
        key = self._get_keyboard_input(
            "WASD (a/d: yaw, w/s: pitch, q: quit, Enter: noop)> "
        ).lower()

        if "q" in key:
            print("Quitting by user command (q pressed).")
            sys.exit(0)

        # Apply controls
        if "w" in key:
            self.inner_angle += self.step
        if "s" in key:
            self.inner_angle -= self.step
        if "a" in key:
            self.outer_angle += self.step
        if "d" in key:
            self.outer_angle -= self.step

        # Clamp to actuator mechanical limits
        max_outer = getattr(self.gimbal_actuator, "max_outer_angle", 0.3)
        max_inner = getattr(self.gimbal_actuator, "max_inner_angle", 0.3)
        self.outer_angle = max(-max_outer, min(self.outer_angle, max_outer))
        self.inner_angle = max(-max_inner, min(self.inner_angle, max_inner))

        # Command the actuator
        self.gimbal_actuator.apply(
            GimbalCommand3D(
                desired_outer_angle=self.outer_angle,
                desired_inner_angle=self.inner_angle,
            )
        )
