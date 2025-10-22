from ..scenario import Scenario
from ..world import NormalWorld, WindyWorld, GustyWorld, TurbulentWorld, ChaoticWorld
from ..rocket import Rocket
from ..physics import Vector
from ..graphics_3d import run_scenario_with_graphics_3d


def main() -> None:
    """3D rocket simulation with dynamic wind patterns."""

    # Try different wind patterns by changing the world:
    # - NormalWorld: No wind
    # - WindyWorld: Steady wind
    # - GustyWorld: Wind with periodic gusts
    # - TurbulentWorld: Smooth turbulent wind
    # - ChaoticWorld: Complex wind with gusts, turbulence, and direction changes

    scenario = Scenario(
        world=NormalWorld,  # Try different wind patterns here!
        rocket=Rocket(
            mass=1.0,
            position=Vector(x=50, y=50, z=25),  # Start in 3D space
            velocity=Vector(x=0, y=0, z=0),
            drag_factor=0.1,
        ),
        dt=0.001,  # 10x slower motion
    )

    run_scenario_with_graphics_3d(
        scenario,
        max_time=50.0,
        fps=10.0,  # Much slower visualization (default is 30 FPS)
        setup_callback=lambda renderer: renderer.customise(
            set_window_title="3D Normal World"
        ),
        show_thrust_window=False,
    )


if __name__ == "__main__":
    main()
