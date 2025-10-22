import time
from dataclasses import dataclass
from typing import Optional, Callable

from .rocket import Rocket
from .world import World


@dataclass
class Scenario:
    world: World
    rocket: Rocket
    dt: float
    _should_stop: bool = False

    def step(self) -> "Scenario":
        """Step the simulation forward by one timestep.

        Returns:
            Self for method chaining.
        """
        self.rocket.apply(self.world.gravity)
        self.rocket.apply(self.world.current_wind)

        self.rocket.step(self.dt)
        self.world.step(self.dt)

        return self

    def run(
        self,
        max_time: Optional[float] = None,
        on_step: Optional[Callable[["Scenario"], None]] = None,
        on_complete: Optional[Callable[["Scenario"], None]] = None,
    ) -> None:
        """Run the simulation headlessly.

        Args:
            max_time: Maximum simulation time in seconds
            on_step: Callback called after each step with scenario as argument
            on_complete: Callback called when simulation completes

        Examples:
            # High-level: Simple run
            scenario.run(max_time=10.0)

            # Mid-level: With callbacks
            scenario.run(
                max_time=10.0,
                on_step=lambda s: print(f"t={s.world.current_time:.2f}s"),
                on_complete=lambda s: save_results(s)
            )
        """
        print(f"Starting simulation (dt={self.dt:.3f}s). Use scenario.stop() to stop.")

        # Run headless simulation
        self._run_simulation(max_time, on_step, on_complete)

    def _run_simulation(
        self,
        max_time: Optional[float] = None,
        on_step: Optional[Callable[["Scenario"], None]] = None,
        on_complete: Optional[Callable[["Scenario"], None]] = None,
    ) -> None:
        """Internal simulation loop."""
        while not self._should_stop:
            self.step()

            # Call step callback if provided
            if on_step:
                on_step(self)

            # Check if we've reached the maximum simulation time
            if max_time is not None and self.world.current_time >= max_time:
                print(f"Simulation completed at t={self.world.current_time:.2f}s")
                if on_complete:
                    on_complete(self)
                break

            time.sleep(self.dt)

        if self._should_stop:
            print(f"Simulation stopped at t={self.world.current_time:.2f}s")
            if on_complete:
                on_complete(self)

    def run_until(self, condition: Callable[["Scenario"], bool]) -> None:
        """Run simulation until a custom condition is met.

        Args:
            condition: Function that takes scenario and returns True to stop

        Examples:
            # Run until rocket reaches altitude 50
            scenario.run_until(lambda s: s.rocket.position.y >= 50)

            # Run until velocity is negative (falling)
            scenario.run_until(lambda s: s.rocket.velocity.y < 0)
        """
        print(f"Starting simulation (dt={self.dt:.3f}s). Use scenario.stop() to stop.")

        while not self._should_stop:
            self.step()

            if condition(self):
                print(f"Condition met at t={self.world.current_time:.2f}s")
                break

            time.sleep(self.dt)

        if self._should_stop:
            print(f"Simulation stopped at t={self.world.current_time:.2f}s")

    def run_headless(self, max_time: Optional[float] = None) -> None:
        """Run simulation headlessly (public version of _run_simulation).

        This is the low-level method for manual control.

        Args:
            max_time: Maximum simulation time in seconds

        Examples:
            # Low-level: Manual control
            while scenario.world.current_time < 10.0:
                scenario.step()
                my_custom_logic()
        """
        self._run_simulation(max_time)

    def stop(self) -> None:
        """Stop the simulation gracefully."""
        self._should_stop = True
