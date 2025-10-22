import matplotlib
from typing import Optional, Callable
import numpy as np

matplotlib.use("TkAgg")  # Use TkAgg backend for better window management
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import signal

from .scenario import Scenario
from .world import World
from .rocket import Rocket


class Renderer3D:
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
            "elevation": [],
            "x": [],
            "y": [],
            "z": [],
        }

        # Trail data for rocket path
        self._trail_data = {
            "x": [],
            "y": [],
            "z": [],
        }

        if self.show_plot:
            plt.ion()  # Interactive mode
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111, projection="3d")

            # Set the default view to show X-Z as horizontal plane, Y going up
            self.ax.view_init(elev=20, azim=45)

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
        """Render a single frame of the 3D scenario."""
        if not self.show_plot:
            return

        # Only clear and setup on first frame
        if not hasattr(self, "_first_frame_drawn"):
            self.ax.clear()

            # Draw world bounds (static)
            self._draw_world_bounds(self.scenario.world)

            # Set plot properties
            x0, y0, z0 = (
                self.scenario.world.bounds[0].x,
                self.scenario.world.bounds[0].y,
                self.scenario.world.bounds[0].z,
            )
            x1, y1, z1 = (
                self.scenario.world.bounds[1].x,
                self.scenario.world.bounds[1].y,
                self.scenario.world.bounds[1].z,
            )

            self.ax.set_xlim(x0 - 5, x1 + 5)
            self.ax.set_ylim(y0 - 5, y1 + 5)
            self.ax.set_zlim(z0 - 5, z1 + 5)
            self.ax.set_xlabel("X (Horizontal)")
            self.ax.set_ylabel("Z (Depth)")
            self.ax.set_zlabel("Y (Altitude)")

            self._first_frame_drawn = True
        else:
            # Clear only the dynamic elements, not the entire plot
            # This preserves the world bounds and zoom settings
            self._clear_dynamic_elements()

        # Draw dynamic elements (these change every frame)
        self._draw_wind(self.scenario.world)
        self._draw_rocket_trail()
        self._draw_rocket(self.scenario.rocket)
        self._draw_thrust_exhaust(self.scenario.rocket)
        self._redraw_custom_elements()

        # Call frame callback if provided
        if frame_callback:
            frame_callback(self, self.scenario)

        # Update title (lightweight)
        self.ax.set_title(
            f"Rocket at ({self.scenario.rocket.position.x:.2f}, {self.scenario.rocket.position.y:.2f}, {self.scenario.rocket.position.z:.2f})"
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
            f"Starting 3D animation for {duration:.1f} seconds. Press Ctrl+C or close window to stop early."
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
        set_window_title: str = "3D Rocket Simulation",
    ) -> None:
        """This API is here to allow for quick/common customisations."""
        self.fig.canvas.manager.set_window_title(set_window_title)

    def _clear_dynamic_elements(self) -> None:
        """Clear only the dynamic elements (rocket, trail, wind, thrust) while preserving static elements."""
        # Get all current lines and collections (convert to list to avoid modification during iteration)
        lines = list(self.ax.lines)
        collections = list(self.ax.collections)

        # Remove dynamic elements (keep world bounds which are static)
        for line in lines:
            # Remove lines that are not world bounds (world bounds are blue dashed)
            if not (
                hasattr(line, "_color")
                and line._color == "b"
                and line._linestyle == "--"
            ):
                line.remove()

        for collection in collections:
            # Remove collections that are not world bounds
            collection.remove()

    def _draw_world_bounds(self, world: World) -> None:
        """Draw world bounds as a 3D box."""
        x0, y0, z0 = world.bounds[0].x, world.bounds[0].y, world.bounds[0].z
        x1, y1, z1 = world.bounds[1].x, world.bounds[1].y, world.bounds[1].z

        # Draw the 12 edges of the box
        # Bottom face (at y0) - swap Y and Z so Y (altitude) goes to Z axis
        self.ax.plot(
            [x0, x1, x1, x0, x0],
            [z0, z0, z1, z1, z0],
            [y0, y0, y0, y0, y0],
            "b--",
            alpha=0.5,
        )
        # Top face (at y1)
        self.ax.plot(
            [x0, x1, x1, x0, x0],
            [z0, z0, z1, z1, z0],
            [y1, y1, y1, y1, y1],
            "b--",
            alpha=0.5,
        )
        # Vertical edges
        for i, j in [(0, 0), (1, 0), (1, 1), (0, 1)]:
            x = x0 if i == 0 else x1
            z = z0 if j == 0 else z1
            self.ax.plot([x, x], [z, z], [y0, y1], "b--", alpha=0.5)

    def _draw_wind(self, world: World) -> None:
        """Visualize wind as a 3D vector arrow."""
        wind = world.current_wind
        if wind.x == 0 and wind.y == 0 and wind.z == 0:
            return

        # Get world bounds for positioning
        x0, y0, z0 = world.bounds[0].x, world.bounds[0].y, world.bounds[0].z
        x1, y1, z1 = world.bounds[1].x, world.bounds[1].y, world.bounds[1].z

        # Position wind indicator in top right corner
        wind_x = x1 - 8  # 8 units from right edge
        wind_y = y1 - 2  # 2 units from top edge (altitude)
        wind_z = z1 - 5  # 5 units from front edge

        # Calculate wind properties
        wind_magnitude = wind.magnitude
        if wind_magnitude == 0:
            return

        # Scale wind vector for visibility (max length of 3 units)
        scale = min(3.0, wind_magnitude * 2.0)
        wind_norm_x = wind.x / wind_magnitude
        wind_norm_y = wind.y / wind_magnitude
        wind_norm_z = wind.z / wind_magnitude

        # Draw wind vector as 3D arrow - swap Y and Z coordinates
        self.ax.quiver(
            wind_x,
            wind_z,
            wind_y,
            wind_norm_x * scale,
            wind_norm_z * scale,
            wind_norm_y * scale,
            color="gray",
            alpha=0.8,
            linewidth=2.0,
        )

    def _draw_rocket_trail(self) -> None:
        """Draw the rocket's path trail."""
        if len(self._trail_data["x"]) > 1:
            self.ax.plot(
                self._trail_data["x"],
                self._trail_data["z"],
                self._trail_data["y"],
                "g-",
                alpha=0.6,
                linewidth=1,
            )

    def _draw_rocket(self, rocket: Rocket) -> None:
        """Draw a 3D rocket oriented according to its attitude."""
        rx, ry, rz = rocket.position.x, rocket.position.y, rocket.position.z
        body_height = rocket.length
        body_width = 0.7

        # Update trail
        self._trail_data["x"].append(rx)
        self._trail_data["y"].append(ry)
        self._trail_data["z"].append(rz)

        # Keep only last 100 points to avoid memory issues
        max_trail_points = 100
        if len(self._trail_data["x"]) > max_trail_points:
            for key in self._trail_data:
                self._trail_data[key] = self._trail_data[key][-max_trail_points:]

        # Get rocket orientation vector (2D orientation in XY plane)
        orientation_vec = rocket.orientation_vector

        # Calculate rocket body corners in world coordinates
        # Body extends along the orientation vector in XY plane
        half_height = body_height / 2
        half_width = body_width / 2

        # Perpendicular vector for width (in XY plane)
        perp_x = -orientation_vec.y
        perp_y = orientation_vec.x

        # Calculate body corners
        # Bottom center (base of rocket)
        base_x = rx - orientation_vec.x * half_height
        base_y = ry - orientation_vec.y * half_height

        # Top center (nose of rocket)
        nose_x = rx + orientation_vec.x * half_height
        nose_y = ry + orientation_vec.y * half_height

        # Body rectangle corners (in XY plane, Z extends vertically)
        body_corners_x = [
            base_x - perp_x * half_width,
            base_x + perp_x * half_width,
            nose_x + perp_x * half_width,
            nose_x - perp_x * half_width,
            base_x - perp_x * half_width,
        ]
        body_corners_y = [
            base_y - perp_y * half_width,
            base_y + perp_y * half_width,
            nose_y + perp_y * half_width,
            nose_y - perp_y * half_width,
            base_y - perp_y * half_width,
        ]

        # Draw body as wireframe - swap Y and Z coordinates for 3D display
        # Bottom face (at rz)
        self.ax.plot(
            body_corners_x,
            [rz - half_width] * 5,
            body_corners_y,
            "k-",
            linewidth=2,
        )
        # Top face (at rz + half_width)
        self.ax.plot(
            body_corners_x,
            [rz + half_width] * 5,
            body_corners_y,
            "k-",
            linewidth=2,
        )
        # Vertical edges
        for i in range(4):
            self.ax.plot(
                [body_corners_x[i], body_corners_x[i]],
                [rz - half_width, rz + half_width],
                [body_corners_y[i], body_corners_y[i]],
                "k-",
                linewidth=2,
            )

        # Nose cone (triangle pointing in direction of orientation)
        nose_tip_x = nose_x + orientation_vec.x * 1.0
        nose_tip_y = nose_y + orientation_vec.y * 1.0

        nose_x_coords = [
            nose_x - perp_x * half_width,
            nose_tip_x,
            nose_x + perp_x * half_width,
        ]
        nose_y_coords = [
            nose_y - perp_y * half_width,
            nose_tip_y,
            nose_y + perp_y * half_width,
        ]

        # Draw nose cone - swap Y and Z coordinates
        self.ax.plot(
            nose_x_coords,
            [rz + half_width] * 3,
            nose_y_coords,
            "r-",
            linewidth=3,
        )

        # Fins at the base of the rocket
        fin_span = 0.5
        fin_height = 0.7

        # Left fin
        left_fin_base_x = base_x - perp_x * half_width
        left_fin_base_y = base_y - perp_y * half_width
        left_fin_tip_x = left_fin_base_x - perp_x * fin_span
        left_fin_tip_y = left_fin_base_y - perp_y * fin_span
        left_fin_mid_x = left_fin_base_x - orientation_vec.x * fin_height
        left_fin_mid_y = left_fin_base_y - orientation_vec.y * fin_height

        left_fin_x = [left_fin_base_x, left_fin_tip_x, left_fin_mid_x]
        left_fin_y = [left_fin_base_y, left_fin_tip_y, left_fin_mid_y]

        # Right fin
        right_fin_base_x = base_x + perp_x * half_width
        right_fin_base_y = base_y + perp_y * half_width
        right_fin_tip_x = right_fin_base_x + perp_x * fin_span
        right_fin_tip_y = right_fin_base_y + perp_y * fin_span
        right_fin_mid_x = right_fin_base_x - orientation_vec.x * fin_height
        right_fin_mid_y = right_fin_base_y - orientation_vec.y * fin_height

        right_fin_x = [right_fin_base_x, right_fin_tip_x, right_fin_mid_x]
        right_fin_y = [right_fin_base_y, right_fin_tip_y, right_fin_mid_y]

        # Draw fins - swap Y and Z coordinates
        self.ax.plot(
            left_fin_x,
            [rz - half_width] * 3,
            left_fin_y,
            "b-",
            linewidth=2,
        )
        self.ax.plot(
            right_fin_x,
            [rz - half_width] * 3,
            right_fin_y,
            "b-",
            linewidth=2,
        )

    def _draw_thrust_exhaust(self, rocket: Rocket) -> None:
        """Draw thrust exhaust in 3D."""
        thrust = rocket.thrust
        if thrust.magnitude == 0:
            return

        # Get rocket position
        rx, ry, rz = rocket.position.x, rocket.position.y, rocket.position.z

        # Calculate thrust exhaust properties
        thrust_magnitude = thrust.magnitude
        thrust_scale = 0.2  # Scale factor for visibility

        # Only draw if thrust is significant
        if thrust_magnitude < 0.1:
            return

        # Calculate exhaust direction (opposite to thrust vector)
        exhaust_norm_x = -thrust.x / thrust_magnitude
        exhaust_norm_y = -thrust.y / thrust_magnitude
        exhaust_norm_z = -thrust.z / thrust_magnitude

        # Exhaust starts from rocket base
        exhaust_start_x = rx
        exhaust_start_y = ry
        exhaust_start_z = rz

        # Calculate exhaust length based on thrust magnitude
        exhaust_length = thrust_magnitude * thrust_scale

        # Exhaust ends in the opposite direction of thrust
        exhaust_end_x = exhaust_start_x + exhaust_norm_x * exhaust_length
        exhaust_end_y = exhaust_start_y + exhaust_norm_y * exhaust_length
        exhaust_end_z = exhaust_start_z + exhaust_norm_z * exhaust_length

        # Draw thrust exhaust as a thick orange line - swap Y and Z coordinates
        self.ax.plot(
            [exhaust_start_x, exhaust_end_x],
            [exhaust_start_z, exhaust_end_z],
            [exhaust_start_y, exhaust_end_y],
            color="orange",
            linewidth=max(2.0, min(8.0, thrust_magnitude * 0.5)),
            alpha=0.8,
        )

    def _create_thrust_window(self) -> None:
        """Create a separate window for thrust visualization with time-series plots."""
        self.thrust_fig = plt.figure(figsize=(15, 10))
        self.thrust_fig.canvas.manager.set_window_title("3D Rocket Thrust Monitor")

        # Create subplots: 3D vector plot and time-series plots
        self.thrust_ax = plt.subplot2grid(
            (3, 4), (0, 0), colspan=2, rowspan=2, projection="3d"
        )
        self.magnitude_ax = plt.subplot2grid((3, 4), (0, 2))
        self.direction_ax = plt.subplot2grid((3, 4), (0, 3))
        self.elevation_ax = plt.subplot2grid((3, 4), (1, 2))
        self.components_ax = plt.subplot2grid((3, 4), (2, 0), colspan=4)

        # Configure 3D vector plot
        self.thrust_ax.set_xlim(-15, 15)
        self.thrust_ax.set_ylim(-15, 15)
        self.thrust_ax.set_zlim(-15, 15)
        self.thrust_ax.set_xlabel("Thrust X (Horizontal)")
        self.thrust_ax.set_ylabel("Thrust Y (Altitude)")
        self.thrust_ax.set_zlabel("Thrust Z (Depth)")
        self.thrust_ax.set_title("3D Thrust Vector")

        # Configure time-series plots
        self.magnitude_ax.set_title("Magnitude")
        self.magnitude_ax.set_ylabel("Magnitude")
        self.magnitude_ax.grid(True, alpha=0.3)

        self.direction_ax.set_title("Azimuth")
        self.direction_ax.set_ylabel("Degrees")
        self.direction_ax.grid(True, alpha=0.3)

        self.elevation_ax.set_title("Elevation")
        self.elevation_ax.set_ylabel("Degrees")
        self.elevation_ax.grid(True, alpha=0.3)

        self.components_ax.set_title("Components Over Time")
        self.components_ax.set_xlabel("Time (s)")
        self.components_ax.set_ylabel("Thrust")
        self.components_ax.grid(True, alpha=0.3)

        # Initialize empty plots for time-series
        (self.magnitude_line,) = self.magnitude_ax.plot([], [], "b-", linewidth=2)
        (self.direction_line,) = self.direction_ax.plot([], [], "g-", linewidth=2)
        (self.elevation_line,) = self.elevation_ax.plot([], [], "r-", linewidth=2)
        (self.x_line,) = self.components_ax.plot([], [], "r-", linewidth=2, label="X")
        (self.y_line,) = self.components_ax.plot([], [], "g-", linewidth=2, label="Y")
        (self.z_line,) = self.components_ax.plot([], [], "b-", linewidth=2, label="Z")
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
            thrust.azimuth * 180 / 3.14159
        )  # Convert to degrees
        self._thrust_history["elevation"].append(
            thrust.elevation * 180 / 3.14159
        )  # Convert to degrees
        self._thrust_history["x"].append(thrust.x)
        self._thrust_history["y"].append(thrust.y)
        self._thrust_history["z"].append(thrust.z)

        # Keep only last 1000 points to avoid memory issues
        max_points = 1000
        if len(self._thrust_history["time"]) > max_points:
            for key in self._thrust_history:
                self._thrust_history[key] = self._thrust_history[key][-max_points:]

        # Update 3D vector plot
        self.thrust_ax.clear()
        self.thrust_ax.set_xlim(-15, 15)
        self.thrust_ax.set_ylim(-15, 15)
        self.thrust_ax.set_zlim(-15, 15)
        self.thrust_ax.set_xlabel("Thrust X (Horizontal)")
        self.thrust_ax.set_ylabel("Thrust Y (Altitude)")
        self.thrust_ax.set_zlabel("Thrust Z (Depth)")
        self.thrust_ax.set_title("3D Thrust Vector")

        if thrust.magnitude > 0:
            # Draw thrust vector from origin - swap Y and Z coordinates
            self.thrust_ax.quiver(
                0,
                0,
                0,
                thrust.x,
                thrust.z,
                thrust.y,
                color="orange",
                alpha=0.8,
                linewidth=3.0,
            )

            # Add current values as text
            self.thrust_ax.text2D(
                0.02,
                0.98,
                f"Magnitude: {thrust.magnitude:.2f}",
                transform=self.thrust_ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            )

            azimuth_deg = thrust.azimuth * 180 / 3.14159
            elevation_deg = thrust.elevation * 180 / 3.14159
            self.thrust_ax.text2D(
                0.02,
                0.90,
                f"Azimuth: {azimuth_deg:.1f}°",
                transform=self.thrust_ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.8),
            )

            self.thrust_ax.text2D(
                0.02,
                0.82,
                f"Elevation: {elevation_deg:.1f}°",
                transform=self.thrust_ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8),
            )

            self.thrust_ax.text2D(
                0.02,
                0.74,
                f"X: {thrust.x:.2f}, Y: {thrust.y:.2f}, Z: {thrust.z:.2f}",
                transform=self.thrust_ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.8),
            )
        else:
            self.thrust_ax.text2D(
                0.5,
                0.5,
                "No Thrust",
                transform=self.thrust_ax.transAxes,
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

            # Update elevation plot
            self.elevation_line.set_data(
                self._thrust_history["time"], self._thrust_history["elevation"]
            )
            self.elevation_ax.relim()
            self.elevation_ax.autoscale_view()

            # Update components plot
            self.x_line.set_data(
                self._thrust_history["time"], self._thrust_history["x"]
            )
            self.y_line.set_data(
                self._thrust_history["time"], self._thrust_history["y"]
            )
            self.z_line.set_data(
                self._thrust_history["time"], self._thrust_history["z"]
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
            if element_type == "plot":
                self.ax.plot(**kwargs)
            elif element_type == "scatter":
                self.ax.scatter(**kwargs)
            elif element_type == "text":
                self.ax.text(**kwargs)


def run_scenario_with_graphics_3d(
    scenario: Scenario,
    max_time: float,
    fps: float = 30.0,
    show_thrust_window: bool = True,
    setup_callback: Optional[Callable[[Renderer3D], None]] = None,
    frame_callback: Optional[Callable[[Renderer3D, Scenario], None]] = None,
) -> None:
    """
    Run a 3D scenario with graphics.

    Args:
        scenario: The scenario to run
        max_time: Maximum simulation time
        fps: Frames per second for visualization
        show_thrust_window: Whether to show separate thrust visualization window
        setup_callback: Optional function to customize graphics before running
        frame_callback: Optional function called on each frame

    Examples:
        # High-level: Simple run
        run_scenario_with_graphics_3d(scenario, max_time=10.0)

        # Mid-level: With callbacks
        def setup_graphics(renderer):
            renderer.add_custom_element("plot", xs=[0, 100], ys=[0, 100], zs=[0, 100], color="red")

        def frame_update(renderer, scenario):
            renderer.ax.text2D(0.02, 0.98, f"Speed: {scenario.rocket.velocity.magnitude:.1f}",
                              transform=renderer.ax.transAxes)

        run_scenario_with_graphics_3d(
            scenario,
            max_time=10.0,
            setup_callback=setup_graphics,
            frame_callback=frame_update
        )
    """
    import signal
    import threading

    # Create renderer
    renderer = Renderer3D(
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
