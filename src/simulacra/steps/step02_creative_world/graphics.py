import matplotlib
from typing import Optional, Callable

matplotlib.use("TkAgg")  # Use TkAgg backend for better window management
import matplotlib.pyplot as plt
import time
import signal

from .scenario import Scenario
from .world import World
from .rocket import Rocket


class Renderer:
    def __init__(
        self,
        scenario: Scenario,
        fps: float = 30.0,
        show_plot: bool = True,
        show_thrust_window: bool = True,
    ):
        self.scenario = scenario
        self.fps = fps
        self.show_plot = show_plot
        self.show_thrust_window = show_thrust_window
        self.frame_time = 1.0 / fps
        self._should_stop = False
        self._custom_elements = []  # Store custom elements to redraw

        # Data storage for thrust history
        self._thrust_history = {
            "time": [],
            "magnitude": [],
            "direction": [],
            "x": [],
            "y": [],
        }

        if self.show_plot:
            plt.ion()  # Interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 8))

            # Configure window to not stay on top
            self.fig.canvas.manager.window.wm_attributes("-topmost", False)

            # Set up graceful exit on window close
            self.fig.canvas.mpl_connect("close_event", self._on_close)

            # Set up Ctrl+C handler
            signal.signal(signal.SIGINT, self._on_interrupt)

        # Create separate thrust window if requested
        if self.show_thrust_window and self.show_plot:
            self._create_thrust_window()

    def render_frame(self, frame_callback=None) -> None:
        """Render a single frame of the scenario.

        Args:
            frame_callback: Optional callback called after standard drawing,
                          receives (renderer, scenario) as arguments

        Examples:
            # Standard rendering
            renderer.render_frame()

            # With custom frame callback
            def my_frame_callback(renderer, scenario):
                renderer.ax.text(10, 10, f"Speed: {scenario.rocket.velocity.magnitude:.1f}")

            renderer.render_frame(frame_callback=my_frame_callback)
        """
        if not self.show_plot:
            return

        self.ax.clear()

        # Draw world bounds
        self._draw_world_bounds(self.scenario.world)

        # Draw wind visualization
        self._draw_wind(self.scenario.world)

        # Draw rocket
        self._draw_rocket(self.scenario.rocket)

        # Draw thrust exhaust
        self._draw_thrust_exhaust(self.scenario.rocket)

        # Redraw custom elements
        self._redraw_custom_elements()

        # Call frame callback if provided
        if frame_callback:
            frame_callback(self, self.scenario)

        # Set plot properties
        x0, y0 = self.scenario.world.bounds[0].x, self.scenario.world.bounds[0].y
        x1, y1 = self.scenario.world.bounds[1].x, self.scenario.world.bounds[1].y
        self.ax.set_xlim(x0 - 5, x1 + 5)
        self.ax.set_ylim(y0 - 5, y1 + 10)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title(
            f"Rocket at ({self.scenario.rocket.position.x:.2f}, {self.scenario.rocket.position.y:.2f})"
        )

        # Use draw_idle() instead of pause() to avoid window focus issues
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Update thrust window if it exists
        if hasattr(self, "thrust_fig") and self.thrust_fig:
            self._update_thrust_window()

    def render_animation(self, duration: float) -> None:
        """Render animation for specified duration by polling the scenario state."""
        start_time = time.time()
        end_time = start_time + duration

        last_render_time = start_time
        print(
            f"Starting animation for {duration:.1f} seconds. Press Ctrl+C or close window to stop early."
        )

        while time.time() < end_time and not self._should_stop:
            current_time = time.time()

            # Render at target FPS (polling the scenario for current state)
            if current_time - last_render_time >= self.frame_time:
                self.render_frame()
                last_render_time = current_time

            # Small sleep to prevent excessive CPU usage
            time.sleep(0.001)

        if self._should_stop:
            print("Animation stopped by user.")
        else:
            print("Animation completed.")

    def customise(
        self,
        set_window_title: str = "Generic Title",
    ) -> None:
        """This API is here to allow for quick/common customisations."""

        self.fig.canvas.manager.set_window_title(set_window_title)

    def _draw_world_bounds(self, world: World) -> None:
        """Draw world bounds as a rectangle.

        Override this method to customize world bounds rendering.
        """
        x0, y0 = world.bounds[0].x, world.bounds[0].y
        x1, y1 = world.bounds[1].x, world.bounds[1].y
        self.ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "b--", linewidth=1)

    def _draw_wind(self, world: World) -> None:
        """Visualize wind as thin gray line segments moving across the grid.

        Override this method to customize wind visualization.
        """
        wind = world.wind
        if wind.x == 0 and wind.y == 0:
            return

        # Get world bounds
        x0, y0 = world.bounds[0].x, world.bounds[0].y
        x1, y1 = world.bounds[1].x, world.bounds[1].y

        # Create a grid of wind lines
        num_x = 12
        num_y = 8

        # Calculate line length based on wind magnitude
        wind_magnitude = (wind.x**2 + wind.y**2) ** 0.5
        line_length = min(3.0, max(1.0, wind_magnitude * 2.0))

        # Add subtle animation offset based on time
        time_offset = world.current_time * 50

        for i in range(num_x):
            for j in range(num_y):
                # Base position in grid
                px = x0 + (x1 - x0) * (i + 0.5) / num_x
                py = y0 + (y1 - y0) * (j + 0.5) / num_y

                # Add subtle movement animation
                anim_offset = 0.3 * (i + j) * 0.1
                px += wind.x * (time_offset + anim_offset) * 0.1
                py += wind.y * (time_offset + anim_offset) * 0.1

                # Wrap around world bounds for continuous flow
                if px < x0:
                    px += x1 - x0
                elif px > x1:
                    px -= x1 - x0
                if py < y0:
                    py += y1 - y0
                elif py > y1:
                    py -= y1 - y0

                # Calculate line endpoints
                if wind_magnitude > 0:
                    # Normalize wind direction
                    wind_norm_x = wind.x / wind_magnitude
                    wind_norm_y = wind.y / wind_magnitude

                    # Line endpoints
                    start_x = px - wind_norm_x * line_length * 0.5
                    start_y = py - wind_norm_y * line_length * 0.5
                    end_x = px + wind_norm_x * line_length * 0.5
                    end_y = py + wind_norm_y * line_length * 0.5

                    # Draw thin gray line segment
                    self.ax.plot(
                        [start_x, end_x],
                        [start_y, end_y],
                        color="gray",
                        linewidth=0.8,
                        alpha=0.6,
                        zorder=1,
                    )

    def _draw_rocket(self, rocket: Rocket) -> None:
        """Draw a simple rocket: body as a vertical rectangle, nose as a triangle.

        Override this method to customize rocket rendering.
        """
        rx, ry = rocket.position.x, rocket.position.y
        body_height = 4.0
        body_width = 0.7

        # Body rectangle
        body_x = [
            rx - body_width / 2,
            rx + body_width / 2,
            rx + body_width / 2,
            rx - body_width / 2,
            rx - body_width / 2,
        ]
        body_y = [ry, ry, ry + body_height, ry + body_height, ry]
        self.ax.fill(body_x, body_y, color="gray", zorder=2)

        # Nose cone (triangle)
        nose_x = [rx - body_width / 2, rx, rx + body_width / 2]
        nose_y = [ry + body_height, ry + body_height + 1.0, ry + body_height]
        self.ax.fill(nose_x, nose_y, color="red", zorder=3)

        # Fins (optional, for realism)
        fin_span = 0.5
        fin_height = 0.7
        left_fin_x = [
            rx - body_width / 2,
            rx - body_width / 2 - fin_span,
            rx - body_width / 2,
        ]
        left_fin_y = [ry, ry - fin_height, ry + 0.5]
        right_fin_x = [
            rx + body_width / 2,
            rx + body_width / 2 + fin_span,
            rx + body_width / 2,
        ]
        right_fin_y = [ry, ry - fin_height, ry + 0.5]
        self.ax.fill(left_fin_x, left_fin_y, color="blue", zorder=2)
        self.ax.fill(right_fin_x, right_fin_y, color="blue", zorder=2)

    def _draw_thrust_exhaust(self, rocket: Rocket) -> None:
        """Draw thrust exhaust coming out of the rocket's base.

        Shows the actual thrust vector (opposite to rocket movement direction)
        with accurate magnitude and direction.
        """
        thrust = rocket.thrust
        if thrust.magnitude == 0:
            return

        # Get rocket position
        rx, ry = rocket.position.x, rocket.position.y

        # Calculate thrust exhaust properties
        thrust_magnitude = thrust.magnitude
        thrust_scale = 0.2  # Scale factor for visibility in world coordinates

        # Only draw if thrust is significant
        if thrust_magnitude < 0.1:
            return

        # Calculate exhaust direction (opposite to thrust vector)
        # The exhaust points in the opposite direction of the thrust vector
        exhaust_norm_x = -thrust.x / thrust_magnitude
        exhaust_norm_y = -thrust.y / thrust_magnitude

        # Exhaust starts from rocket base (bottom of rocket)
        body_height = 4.0
        exhaust_start_x = rx
        exhaust_start_y = ry  # Bottom of rocket

        # Calculate exhaust length based on thrust magnitude
        exhaust_length = thrust_magnitude * thrust_scale

        # Exhaust ends in the opposite direction of thrust (where exhaust goes)
        exhaust_end_x = exhaust_start_x + exhaust_norm_x * exhaust_length
        exhaust_end_y = exhaust_start_y + exhaust_norm_y * exhaust_length

        # Draw thrust exhaust as a thick orange/red line
        self.ax.plot(
            [exhaust_start_x, exhaust_end_x],
            [exhaust_start_y, exhaust_end_y],
            color="orange",
            linewidth=max(
                2.0, min(8.0, thrust_magnitude * 0.5)
            ),  # Line width based on thrust
            alpha=0.8,
            zorder=1,
        )

        # Add a small arrow head to show direction
        if exhaust_length > 1.0:  # Only add arrow if exhaust is long enough
            arrow_size = min(1.0, exhaust_length * 0.3)
            self.ax.arrow(
                exhaust_end_x - exhaust_norm_x * arrow_size * 0.5,
                exhaust_end_y - exhaust_norm_y * arrow_size * 0.5,
                exhaust_norm_x * arrow_size,
                exhaust_norm_y * arrow_size,
                head_width=0.5,
                head_length=0.8,
                fc="red",
                ec="darkred",
                alpha=0.9,
                zorder=2,
            )

    def _create_thrust_window(self) -> None:
        """Create a separate window for thrust visualization with time-series plots."""
        self.thrust_fig = plt.figure(figsize=(12, 8))
        self.thrust_fig.canvas.manager.set_window_title("Rocket Thrust Monitor")

        # Create subplots: vector plot (top-left) and time-series plots
        self.thrust_ax = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self.magnitude_ax = plt.subplot2grid((3, 3), (0, 2))
        self.direction_ax = plt.subplot2grid((3, 3), (1, 2))
        self.components_ax = plt.subplot2grid((3, 3), (2, 0), colspan=3)

        # Configure vector plot
        self.thrust_ax.set_xlim(-15, 15)
        self.thrust_ax.set_ylim(-15, 15)
        self.thrust_ax.set_aspect("equal", adjustable="box")
        self.thrust_ax.grid(True, alpha=0.3)
        self.thrust_ax.set_title("Thrust Vector")
        self.thrust_ax.set_xlabel("Thrust X")
        self.thrust_ax.set_ylabel("Thrust Y")

        # Configure time-series plots
        self.magnitude_ax.set_title("Magnitude")
        self.magnitude_ax.set_ylabel("Magnitude")
        self.magnitude_ax.grid(True, alpha=0.3)

        self.direction_ax.set_title("Direction")
        self.direction_ax.set_ylabel("Degrees")
        self.direction_ax.grid(True, alpha=0.3)

        self.components_ax.set_title("Components Over Time")
        self.components_ax.set_xlabel("Time (s)")
        self.components_ax.set_ylabel("Thrust")
        self.components_ax.grid(True, alpha=0.3)

        # Initialize empty plots for time-series
        (self.magnitude_line,) = self.magnitude_ax.plot([], [], "b-", linewidth=2)
        (self.direction_line,) = self.direction_ax.plot([], [], "g-", linewidth=2)
        (self.x_line,) = self.components_ax.plot([], [], "r-", linewidth=2, label="X")
        (self.y_line,) = self.components_ax.plot([], [], "b-", linewidth=2, label="Y")
        self.components_ax.legend()

        plt.tight_layout()

    def _update_thrust_window(self) -> None:
        """Update the thrust visualization window with time-series data."""
        if not hasattr(self, "thrust_fig") or not self.thrust_fig:
            return

        # Get current thrust and time
        thrust = self.scenario.rocket.thrust
        current_time = self.scenario.world.current_time

        # Store thrust data
        self._thrust_history["time"].append(current_time)
        self._thrust_history["magnitude"].append(thrust.magnitude)
        self._thrust_history["direction"].append(
            thrust.direction * 180 / 3.14159
        )  # Convert to degrees
        self._thrust_history["x"].append(thrust.x)
        self._thrust_history["y"].append(thrust.y)

        # Keep only last 1000 points to avoid memory issues
        max_points = 1000
        if len(self._thrust_history["time"]) > max_points:
            for key in self._thrust_history:
                self._thrust_history[key] = self._thrust_history[key][-max_points:]

        # Update vector plot
        self.thrust_ax.clear()
        self.thrust_ax.set_xlim(-15, 15)
        self.thrust_ax.set_ylim(-15, 15)
        self.thrust_ax.set_aspect("equal", adjustable="box")
        self.thrust_ax.grid(True, alpha=0.3)
        self.thrust_ax.set_title("Thrust Vector")
        self.thrust_ax.set_xlabel("Thrust X")
        self.thrust_ax.set_ylabel("Thrust Y")

        if thrust.magnitude > 0:
            # Draw thrust vector from origin
            self.thrust_ax.arrow(
                0,
                0,
                thrust.x,
                thrust.y,
                head_width=0.8,
                head_length=1.0,
                fc="orange",
                ec="red",
                alpha=0.8,
                linewidth=3.0,
                zorder=2,
            )

            # Add current values as text
            self.thrust_ax.text(
                0,
                -12,
                f"Magnitude: {thrust.magnitude:.2f}",
                fontsize=12,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            )

            direction_deg = thrust.direction * 180 / 3.14159
            self.thrust_ax.text(
                0,
                -14,
                f"Direction: {direction_deg:.1f}°",
                fontsize=10,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.8),
            )

            self.thrust_ax.text(
                0,
                -16,
                f"X: {thrust.x:.2f}, Y: {thrust.y:.2f}",
                fontsize=10,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8),
            )
        else:
            self.thrust_ax.text(
                0,
                0,
                "No Thrust",
                fontsize=14,
                ha="center",
                color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            )

        # Update time-series plots
        if len(self._thrust_history["time"]) > 1:
            # Update magnitude plot
            self.magnitude_line.set_data(
                self._thrust_history["time"], self._thrust_history["magnitude"]
            )
            self.magnitude_ax.relim()
            self.magnitude_ax.autoscale_view()

            # Update direction plot
            self.direction_line.set_data(
                self._thrust_history["time"], self._thrust_history["direction"]
            )
            self.direction_ax.relim()
            self.direction_ax.autoscale_view()

            # Update components plot
            self.x_line.set_data(
                self._thrust_history["time"], self._thrust_history["x"]
            )
            self.y_line.set_data(
                self._thrust_history["time"], self._thrust_history["y"]
            )
            self.components_ax.relim()
            self.components_ax.autoscale_view()

        # Update the thrust window
        self.thrust_fig.canvas.draw_idle()
        self.thrust_fig.canvas.flush_events()

    def stop(self) -> None:
        """Stop the renderer gracefully."""
        self._should_stop = True

    def _on_close(self, event) -> None:
        """Handle window close event."""
        print("\nWindow closed. Stopping animation...")
        self._should_stop = True

    def _on_interrupt(self, signum, frame) -> None:
        """Handle Ctrl+C interrupt."""
        print("\nInterrupt received. Stopping animation...")
        self._should_stop = True

    def add_custom_element(self, element_type: str, **kwargs):
        """Add a custom element that will be redrawn on every frame."""
        self._custom_elements.append((element_type, kwargs))

    def _redraw_custom_elements(self):
        """Redraw all custom elements."""
        for element_type, kwargs in self._custom_elements:
            if element_type == "hline":
                self.ax.axhline(**kwargs)
            elif element_type == "vline":
                self.ax.axvline(**kwargs)
            elif element_type == "text":
                self.ax.text(**kwargs)
            elif element_type == "arrow":
                self.ax.arrow(**kwargs)
            elif element_type == "plot":
                self.ax.plot(**kwargs)
            elif element_type == "scatter":
                self.ax.scatter(**kwargs)
            elif element_type == "fill":
                self.ax.fill(**kwargs)


def run_scenario_with_graphics(
    scenario: Scenario,
    max_time: float,
    fps: float = 30.0,
    show_thrust_window: bool = True,
    setup_callback: Optional[Callable[[Renderer], None]] = None,
    frame_callback: Optional[Callable[[Renderer, Scenario], None]] = None,
) -> None:
    """
    Run a scenario with graphics.

    Args:
        scenario: The scenario to run
        max_time: Maximum simulation time
        fps: Frames per second for visualization
        show_thrust_window: Whether to show separate thrust visualization window
        setup_callback: Optional function to customize graphics before running
                        Called with (renderer) as arguments
        frame_callback: Optional function called on each frame
                        Called with (renderer, scenario) as arguments

    Examples:
        # High-level: Simple run
        run_scenario_with_graphics(scenario, max_time=10.0)

        # Mid-level: With callbacks
        def setup_graphics(renderer):
            renderer.add_custom_element("hline", y=50, color="red")

        def frame_update(renderer, scenario):
            renderer.ax.text(10, 10, f"Speed: {scenario.rocket.velocity.magnitude:.1f}")

        run_scenario_with_graphics(
            scenario,
            max_time=10.0,
            setup_callback=setup_graphics,
            frame_callback=frame_update
        )
    """
    import signal
    import threading

    # Create renderer
    renderer = Renderer(
        scenario, fps=fps, show_plot=True, show_thrust_window=show_thrust_window
    )

    # Allow customization of graphics
    if setup_callback:
        setup_callback(renderer)

    # Set up signal handler in main thread
    def signal_handler(signum, frame):
        print("\nInterrupt received. Shutting down gracefully...")
        scenario.stop()
        renderer.stop()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run scenario in background thread
    sim_thread = threading.Thread(target=scenario.run, kwargs={"max_time": max_time})
    sim_thread.daemon = True
    sim_thread.start()

    # Run graphics in main thread
    try:
        # Create custom render_frame that includes frame_callback
        if frame_callback:
            original_render_frame = renderer.render_frame

            def custom_render_frame():
                original_render_frame(frame_callback=frame_callback)

            renderer.render_frame = custom_render_frame

        renderer.render_animation(duration=max_time)
    except Exception as e:
        print(f"Graphics error: {e}")
    finally:
        scenario.stop()
        renderer.stop()
        sim_thread.join()
