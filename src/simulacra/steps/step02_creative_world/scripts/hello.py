from ..scenario import Scenario
from ..world import NormalWorld
from ..rocket import Rocket
from ..physics import Vector
from ..graphics import run_scenario_with_graphics


def main() -> None:
    """High-level example: Simple simulation with defaults."""

    scenario = Scenario(
        world=NormalWorld,
        rocket=Rocket(
            mass=1.0,
            position=Vector(x=50, y=50),
            velocity=Vector(x=0, y=0),
            drag_factor=0.1,
        ),
        dt=0.01,
    )

    run_scenario_with_graphics(
        scenario,
        max_time=50.0,
        setup_callback=lambda renderer: renderer.customise(
            set_window_title="Hello, world!"
        ),
        show_thrust_window=True,
    )


if __name__ == "__main__":
    main()
