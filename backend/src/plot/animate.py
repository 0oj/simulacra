"""
Animate simulation data - position trajectory with rocket orientation arrow.
Reads from stdin (NDJSON format) in real-time streaming mode.
"""

import sys
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import numpy as np
from schema import read_sim_state
from typing import Optional


def stream_states_from_stdin(state_queue: queue.Queue, done_event: threading.Event):
    """
    Read simulation states from stdin in a background thread.
    Puts states into queue as they arrive.
    
    Args:
        state_queue: Queue to put states into
        done_event: Event to signal when reading is done
    """
    try:
        while True:
            try:
                state = read_sim_state()
                state_queue.put(state)
            except EOFError:
                # Normal end - simulation finished writing
                break
    except Exception as e:
        print(f"Error reading from stdin: {e}", file=sys.stderr)
    finally:
        done_event.set()


def animate_simulation_streaming(output_file: Optional[str] = None, interval: int = 50, final_mode: bool = False):
    """
    Animate position trajectory with rocket orientation arrow in real-time.
    Reads states from stdin as they arrive and updates animation.
    
    Args:
        output_file: Optional path to save animation (GIF or MP4)
        interval: Animation interval in milliseconds (default: 50ms)
        final_mode: If True, wait for all data and show final frame statically
    """
    # Setup streaming
    state_queue = queue.Queue()
    done_event = threading.Event()
    
    # Start background thread to read from stdin
    reader_thread = threading.Thread(
        target=stream_states_from_stdin,
        args=(state_queue, done_event),
        daemon=True
    )
    reader_thread.start()
    
    # Collect states as they arrive
    states = []
    
    # Wait for first state to initialize plots
    try:
        first_state = state_queue.get(timeout=5.0)
        states.append(first_state)
    except queue.Empty:
        print("No simulation data received. Make sure to pipe simulation output:", file=sys.stderr)
        print("  python -m src.setpoint_scenario | python -m src.plot.animate", file=sys.stderr)
        return
    
    # Initialize data arrays (will grow as data arrives)
    t = [first_state.t]
    pos_x = [first_state.pos_x]
    pos_y = [first_state.pos_y]
    orientation = [first_state.orientation]
    gimbal_angle = [first_state.gimbal_angle]
    angular_velocity = [first_state.angular_velocity]
    thrust_magnitude = [first_state.thrust_magnitude]
    
    # Sensor data arrays (optional - only populated if sensor data is present)
    # Initialize with first state's sensor data if available, otherwise empty list
    gps_x = []
    gps_y = []
    imu_orientation = []
    imu_angular_velocity = []
    thrust_sensor = []
    
    if first_state.gps_x is not None:
        gps_x.append(first_state.gps_x)
        gps_y.append(first_state.gps_y)
    if first_state.imu_orientation is not None:
        imu_orientation.append(first_state.imu_orientation)
        if first_state.imu_angular_velocity is not None:
            imu_angular_velocity.append(first_state.imu_angular_velocity)
    if first_state.thrust_sensor is not None:
        thrust_sensor.append(first_state.thrust_sensor)
    
    # Create figure with optimized layout: trajectory on left, all other plots on right and bottom
    fig = plt.figure(figsize=(18, 10), facecolor='#1e1e1e')
    # Use a 3x3 grid: left 2 columns for trajectory, right column for compasses/time plots
    # Bottom row for angular velocity and thrust plots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25, left=0.05, right=0.98, top=0.95, bottom=0.12)
    
    # Left side: Large trajectory plot (takes top 2 rows, 2 columns)
    ax = fig.add_subplot(gs[:2, :2])  # Main trajectory plot (rows 0-1, columns 0-1)
    
    # Right side: Stacked plots (1 column, top 2 rows)
    ax_orient = fig.add_subplot(gs[0, 2])  # Orientation compass (top right)
    ax_gimbal = fig.add_subplot(gs[1, 2])  # Gimbal compass (middle right)
    
    # Right side bottom: Time plots (row 2, column 2)
    ax_orient_time = fig.add_subplot(gs[2, 2])  # Orientation vs time (bottom right)
    
    # Bottom: Angular velocity and thrust plots (row 2, columns 0-1)
    ax_angular_vel_time = fig.add_subplot(gs[2, 0])  # Angular velocity vs time (bottom left)
    ax_thrust_time = fig.add_subplot(gs[2, 1])  # Thrust vs time (bottom middle)
    
    # Gimbal time plot - combine with orientation time plot using dual y-axis
    ax_gimbal_time = ax_orient_time.twinx()  # Share x-axis with orientation plot
    
    # Dark mode styling for all axes
    for axes in [ax, ax_orient_time, ax_angular_vel_time, ax_thrust_time]:
        axes.set_facecolor('#1e1e1e')
        axes.tick_params(colors='white')
        axes.xaxis.label.set_color('white')
        axes.yaxis.label.set_color('white')
        axes.title.set_color('white')
        axes.spines['bottom'].set_color('white')
        axes.spines['top'].set_color('white')
        axes.spines['left'].set_color('white')
        axes.spines['right'].set_color('white')
    
    # Setup plot
    ax.set_xlabel('Position X (m)', color='white')
    ax.set_ylabel('Position Y (m)', color='white')
    ax.set_title('Rocket Position Trajectory', color='white')
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_aspect('auto')
    
    # Set initial axis limits (will update dynamically)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    # Initialize plot elements
    trajectory_line, = ax.plot([], [], 'cyan', linewidth=2, alpha=0.8, label='Trajectory (True)')
    gps_line, = ax.plot([], [], 'yellow', linewidth=1.5, alpha=0.6, linestyle='--', label='GPS', marker='o', markersize=3)
    start_point, = ax.plot([], [], 'go', markersize=10, label='Start')
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12, color='white',
                       bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.8, edgecolor='white'))
    
    # Rocket rod (will be updated each frame)
    rocket_rod = None
    # Thrust/gimbal indicator line (will be updated each frame)
    thrust_line = None
    
    # Setup compass plots (smaller, more compact)
    def setup_compass(ax_compass, title):
        """Setup a compass-style plot."""
        ax_compass.set_xlim(-1.2, 1.2)
        ax_compass.set_ylim(-1.2, 1.2)
        ax_compass.set_aspect('equal')
        ax_compass.set_title(title, fontsize=9, fontweight='bold', color='white', pad=3)
        ax_compass.grid(True, alpha=0.3, color='gray')
        ax_compass.set_facecolor('#1e1e1e')
        
        # Draw circle
        circle = plt.Circle((0, 0), 1.0, fill=False, color='white', linewidth=1.5)
        ax_compass.add_patch(circle)
        
        # Draw reference axes (no labels, smaller)
        ax_compass.arrow(0, 0, 0, 0.9, head_width=0.04, head_length=0.04, 
                        fc='gray', ec='gray', alpha=0.5, linewidth=0.8)
        ax_compass.arrow(0, 0, 0.9, 0, head_width=0.04, head_length=0.04, 
                        fc='gray', ec='gray', alpha=0.5, linewidth=0.8)
        
        # Remove ticks
        ax_compass.set_xticks([])
        ax_compass.set_yticks([])
    
    setup_compass(ax_orient, 'Orientation')
    setup_compass(ax_gimbal, 'Gimbal')
    
    # Setup time plots (compact, on right side)
    ax_orient_time.set_xlabel('Time (s)', color='white', fontsize=8)
    ax_orient_time.set_ylabel('Orient (deg)', color='white', fontsize=8)
    ax_orient_time.set_title('Orientation', color='white', fontsize=9, pad=3)
    ax_orient_time.grid(True, alpha=0.3, color='gray')
    ax_orient_time.set_xlim(0, 10)  # Will update dynamically
    orient_time_line, = ax_orient_time.plot([], [], 'cyan', linewidth=1.5, label='True')
    imu_orient_time_line, = ax_orient_time.plot([], [], 'yellow', linewidth=1, alpha=0.7, linestyle='--', label='IMU', marker='o', markersize=2)
    orient_time_point, = ax_orient_time.plot([], [], 'co', markersize=4)
    ax_orient_time.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=7, loc='upper right')
    ax_orient_time.tick_params(labelsize=7)
    
    # Setup gimbal time plot (dual y-axis with orientation)
    ax_gimbal_time.set_ylabel('Gimbal (deg)', color='orange', fontsize=8)
    ax_gimbal_time.tick_params(colors='orange', labelsize=7)
    ax_gimbal_time.set_facecolor('#1e1e1e')
    for spine in ax_gimbal_time.spines.values():
        spine.set_color('orange')
    gimbal_time_line, = ax_gimbal_time.plot([], [], 'orange', linewidth=1.5, label='Gimbal')
    gimbal_time_point, = ax_gimbal_time.plot([], [], 'o', color='orange', markersize=4)
    ax_gimbal_time.axhline(y=0, color='orange', linestyle='--', alpha=0.3)
    ax_gimbal_time.set_xlim(0, 10)  # Will update dynamically
    
    # Setup angular velocity plot (bottom left)
    ax_angular_vel_time.set_xlabel('Time (s)', color='white', fontsize=9)
    ax_angular_vel_time.set_ylabel('Angular Vel (deg/s)', color='white', fontsize=9)
    ax_angular_vel_time.set_title('Angular Velocity', color='white', fontsize=10)
    ax_angular_vel_time.grid(True, alpha=0.3, color='gray')
    ax_angular_vel_time.set_xlim(0, 10)  # Will update dynamically
    angular_vel_time_line, = ax_angular_vel_time.plot([], [], 'magenta', linewidth=2, label='True')
    imu_angular_vel_time_line, = ax_angular_vel_time.plot([], [], 'yellow', linewidth=1.5, alpha=0.7, linestyle='--', label='IMU', marker='o', markersize=3)
    angular_vel_time_point, = ax_angular_vel_time.plot([], [], 'mo', markersize=6)
    ax_angular_vel_time.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    ax_angular_vel_time.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=8)
    
    # Setup thrust plot (bottom right)
    ax_thrust_time.set_xlabel('Time (s)', color='white', fontsize=9)
    ax_thrust_time.set_ylabel('Thrust (N)', color='white', fontsize=9)
    ax_thrust_time.set_title('Thrust', color='white', fontsize=10)
    ax_thrust_time.grid(True, alpha=0.3, color='gray')
    ax_thrust_time.set_xlim(0, 10)  # Will update dynamically
    thrust_time_line, = ax_thrust_time.plot([], [], 'red', linewidth=2, label='True')
    thrust_sensor_time_line, = ax_thrust_time.plot([], [], 'yellow', linewidth=1.5, alpha=0.7, linestyle='--', label='Sensor', marker='o', markersize=3)
    thrust_time_point, = ax_thrust_time.plot([], [], 'ro', markersize=6)
    ax_thrust_time.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    ax_thrust_time.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=8)
    
    # Compass arrows (will be updated each frame)
    orient_arrow = None
    gimbal_arrow = None
    
    # Angle label text objects (smaller, more compact)
    orient_text = ax_orient.text(0, -1.3, '', ha='center', va='top', 
                                 fontsize=8, fontweight='bold', color='white')
    gimbal_text = ax_gimbal.text(0, -1.3, '', ha='center', va='top', 
                                fontsize=8, fontweight='bold', color='white')
    
    # Will convert to degrees on-the-fly
    
    # Track if we're in looping mode (data collection done)
    looping_mode = False
    loop_frame = 0
    
    # If final_mode, wait for all data to be collected first
    if final_mode:
        print("Waiting for all simulation data...", file=sys.stderr)
        # Wait for reader thread to finish
        reader_thread.join(timeout=60.0)  # Wait up to 60 seconds
        # Collect all remaining states from queue
        while True:
            try:
                new_state = state_queue.get_nowait()
                states.append(new_state)
                t.append(new_state.t)
                pos_x.append(new_state.pos_x)
                pos_y.append(new_state.pos_y)
                orientation.append(new_state.orientation)
                gimbal_angle.append(new_state.gimbal_angle)
                angular_velocity.append(new_state.angular_velocity)
                thrust_magnitude.append(new_state.thrust_magnitude)
                
                # Collect sensor data if available
                if new_state.gps_x is not None:
                    gps_x.append(new_state.gps_x)
                    gps_y.append(new_state.gps_y)
                elif len(gps_x) > 0:
                    gps_x.append(None)
                    gps_y.append(None)
                
                if new_state.imu_orientation is not None:
                    imu_orientation.append(new_state.imu_orientation)
                    if new_state.imu_angular_velocity is not None:
                        imu_angular_velocity.append(new_state.imu_angular_velocity)
                    elif len(imu_angular_velocity) > 0:
                        imu_angular_velocity.append(None)
                elif len(imu_orientation) > 0:
                    imu_orientation.append(None)
                    if len(imu_angular_velocity) > 0:
                        imu_angular_velocity.append(None)
                
                if new_state.thrust_sensor is not None:
                    thrust_sensor.append(new_state.thrust_sensor)
                elif len(thrust_sensor) > 0:
                    thrust_sensor.append(None)
            except queue.Empty:
                break
        
        # Set to looping mode and show final frame
        looping_mode = True
        # Set loop_frame to final index (will be used before increment in animate function)
        if len(states) > 0:
            loop_frame = len(states) - 1
            print(f"Collected {len(states)} states. Showing final frame (index {loop_frame}, time={states[loop_frame].t:.2f}s).", file=sys.stderr)
        else:
            loop_frame = 0
            print("Warning: No states collected!", file=sys.stderr)
    
    # Animation control state
    is_paused = final_mode  # Pause initially if in final_mode
    step_frame = False  # True when we want to step one frame
    current_interval = interval
    follow_mode = False  # True when following the rocket
    follow_zoom_factor = 0.05  # Fraction of plot range to show when following (10% zoom)
    gimbal_visual_multiplier = 4.0  # Multiplier to make small gimbal angles more visible
    
    def animate(frame):
        """Update animation for frame - reads new states from queue, then loops."""
        nonlocal rocket_rod, thrust_line, orient_arrow, gimbal_arrow, states, t, pos_x, pos_y, orientation, gimbal_angle
        nonlocal angular_velocity, thrust_magnitude, gps_x, gps_y, imu_orientation, imu_angular_velocity, thrust_sensor
        nonlocal looping_mode, loop_frame, is_paused, step_frame, follow_mode
        
        # Handle pause and step controls
        if is_paused and not step_frame:
            return []  # Don't update if paused (unless stepping)
        step_frame = False  # Reset step flag after use
        
        # If not in looping mode, try to read new states
        if not looping_mode:
            # Read all available states from queue (non-blocking)
            while True:
                try:
                    new_state = state_queue.get_nowait()
                    states.append(new_state)
                    t.append(new_state.t)
                    pos_x.append(new_state.pos_x)
                    pos_y.append(new_state.pos_y)
                    orientation.append(new_state.orientation)
                    gimbal_angle.append(new_state.gimbal_angle)
                    angular_velocity.append(new_state.angular_velocity)
                    thrust_magnitude.append(new_state.thrust_magnitude)
                    
                    # Collect sensor data if available
                    # Keep arrays in sync with main data arrays
                    if new_state.gps_x is not None:
                        gps_x.append(new_state.gps_x)
                        gps_y.append(new_state.gps_y)
                    elif len(gps_x) > 0:
                        # Pad with None to keep arrays aligned
                        gps_x.append(None)
                        gps_y.append(None)
                    
                    if new_state.imu_orientation is not None:
                        imu_orientation.append(new_state.imu_orientation)
                        if new_state.imu_angular_velocity is not None:
                            imu_angular_velocity.append(new_state.imu_angular_velocity)
                        elif len(imu_angular_velocity) > 0:
                            # Keep arrays aligned
                            imu_angular_velocity.append(None)
                    elif len(imu_orientation) > 0:
                        # Pad with None to keep arrays aligned
                        imu_orientation.append(None)
                        if len(imu_angular_velocity) > 0:
                            imu_angular_velocity.append(None)
                    
                    if new_state.thrust_sensor is not None:
                        thrust_sensor.append(new_state.thrust_sensor)
                    elif len(thrust_sensor) > 0:
                        # Pad with None to keep arrays aligned
                        thrust_sensor.append(None)
                except queue.Empty:
                    break
            
            # Check if data collection is done
            if done_event.is_set() and state_queue.empty() and len(states) > 0:
                looping_mode = True
                loop_frame = 0
                print("Data collection complete. Starting loop animation...", file=sys.stderr)
        
        # In looping mode, cycle through collected states
        if looping_mode:
            if len(states) == 0:
                return []
            # Use modulo to loop
            current_idx = loop_frame % len(states)
            loop_frame += 1
        else:
            # Streaming mode: use latest state
            current_idx = len(states) - 1
            if current_idx < 0:
                return []
        
        # Convert to numpy arrays for easier indexing
        t_arr = np.array(t)
        pos_x_arr = np.array(pos_x)
        pos_y_arr = np.array(pos_y)
        orientation_arr = np.array(orientation)
        gimbal_angle_arr = np.array(gimbal_angle)
        angular_velocity_arr = np.array(angular_velocity)
        thrust_magnitude_arr = np.array(thrust_magnitude)
        
        if len(t_arr) == 0:
            return []
        
        # Update axis limits dynamically (only update if we have data)
        if len(pos_x_arr) > 1:
            if follow_mode and current_idx >= 0:
                # Follow mode: center on rocket and zoom in
                # Calculate zoom range based on full data range
                full_x_range = pos_x_arr.max() - pos_x_arr.min()
                full_y_range = pos_y_arr.max() - pos_y_arr.min()
                
                # Use a fraction of the full range for zoom
                zoom_x_range = max(full_x_range * follow_zoom_factor, 10.0)  # At least 10m wide
                zoom_y_range = max(full_y_range * follow_zoom_factor, 10.0)  # At least 10m tall
                
                # Center on current rocket position
                current_x = pos_x_arr[current_idx]
                current_y = pos_y_arr[current_idx]
                
                ax.set_xlim(current_x - zoom_x_range / 2, current_x + zoom_x_range / 2)
                ax.set_ylim(current_y - zoom_y_range / 2, current_y + zoom_y_range / 2)
            else:
                # Normal mode: show all data with margin
                x_margin = (pos_x_arr.max() - pos_x_arr.min()) * 0.25
                y_margin = (pos_y_arr.max() - pos_y_arr.min()) * 0.25
                ax.set_xlim(pos_x_arr.min() - x_margin, pos_x_arr.max() + x_margin)
                ax.set_ylim(pos_y_arr.min() - y_margin, pos_y_arr.max() + y_margin)
        
        if len(t_arr) > 0:
            ax_orient_time.set_xlim(0, max(t_arr.max(), 10))
            ax_gimbal_time.set_xlim(0, max(t_arr.max(), 10))
            ax_angular_vel_time.set_xlim(0, max(t_arr.max(), 10))
            ax_thrust_time.set_xlim(0, max(t_arr.max(), 10))
            
            # Update y-axis limits for time plots
            orientation_deg = np.degrees(orientation_arr)
            gimbal_angle_deg = np.degrees(gimbal_angle_arr)
            angular_velocity_deg_per_s = np.degrees(angular_velocity_arr)
            
            if len(orientation_deg) > 1:
                orient_range = orientation_deg.max() - orientation_deg.min()
                ax_orient_time.set_ylim(orientation_deg.min() - orient_range * 0.1, 
                                        orientation_deg.max() + orient_range * 0.1)
            if len(gimbal_angle_deg) > 1:
                gimbal_range = gimbal_angle_deg.max() - gimbal_angle_deg.min()
                ax_gimbal_time.set_ylim(gimbal_angle_deg.min() - gimbal_range * 0.1, 
                                        gimbal_angle_deg.max() + gimbal_range * 0.1)
            if len(angular_velocity_deg_per_s) > 1:
                ang_vel_range = angular_velocity_deg_per_s.max() - angular_velocity_deg_per_s.min()
                ax_angular_vel_time.set_ylim(angular_velocity_deg_per_s.min() - ang_vel_range * 0.1, 
                                             angular_velocity_deg_per_s.max() + ang_vel_range * 0.1)
            if len(thrust_magnitude_arr) > 1:
                thrust_range = thrust_magnitude_arr.max() - thrust_magnitude_arr.min()
                if thrust_range > 0:
                    ax_thrust_time.set_ylim(thrust_magnitude_arr.min() - thrust_range * 0.1, 
                                            thrust_magnitude_arr.max() + thrust_range * 0.1)
                else:
                    # If all values are the same, set a small range around that value
                    ax_thrust_time.set_ylim(thrust_magnitude_arr[0] - 10, thrust_magnitude_arr[0] + 10)
        
        # Update trajectory
        if looping_mode:
            # In loop mode: show trajectory up to current frame
            # When at the final frame, show complete trajectory
            if current_idx == len(states) - 1:
                # Show complete trajectory for final frame
                trajectory_line.set_data(pos_x_arr, pos_y_arr)
            else:
                trajectory_line.set_data(pos_x_arr[:current_idx+1], pos_y_arr[:current_idx+1])
            # Update GPS line if available
            if len(gps_x) > 0 and len(gps_x) > current_idx:
                # Filter out None values for plotting
                gps_x_valid = [x for x in gps_x[:current_idx+1] if x is not None]
                gps_y_valid = [y for y in gps_y[:current_idx+1] if y is not None]
                if len(gps_x_valid) > 0:
                    gps_line.set_data(gps_x_valid, gps_y_valid)
        else:
            # In streaming mode: show all data collected so far
            trajectory_line.set_data(pos_x_arr, pos_y_arr)
            # Update GPS line if available
            if len(gps_x) > 0:
                # Filter out None values for plotting
                gps_x_valid = [x for x in gps_x if x is not None]
                gps_y_valid = [y for y in gps_y if y is not None]
                if len(gps_x_valid) > 0:
                    gps_line.set_data(gps_x_valid, gps_y_valid)
        
        # Update start point (only show once)
        if len(states) == 1:
            start_point.set_data([pos_x_arr[0]], [pos_y_arr[0]])
        
        # Remove old arrows, rod, and thrust line, reset to None
        if rocket_rod is not None:
            rocket_rod.remove()
            rocket_rod = None
        if thrust_line is not None:
            thrust_line.remove()
            thrust_line = None
        if orient_arrow is not None:
            orient_arrow.remove()
            orient_arrow = None
        if gimbal_arrow is not None:
            gimbal_arrow.remove()
            gimbal_arrow = None
        
        if current_idx >= 0 and len(t_arr) > 0 and current_idx < len(t_arr):
            # Check for valid data (no NaN or inf)
            try:
                pos_x_val = pos_x_arr[current_idx]
                pos_y_val = pos_y_arr[current_idx]
                orient_val = orientation_arr[current_idx]
                gimbal_val = gimbal_angle_arr[current_idx]
                
                if not (np.isnan(pos_x_val) or np.isnan(pos_y_val) or 
                        np.isnan(orient_val) or np.isnan(gimbal_val) or
                        np.isinf(pos_x_val) or np.isinf(pos_y_val) or
                        np.isinf(orient_val) or np.isinf(gimbal_val)):
                    # Draw rocket as a rod centered at center of mass
                    # Get rocket length from state, or use default proportional to plot
                    rocket_length = states[current_idx].rocket_length if hasattr(states[current_idx], 'rocket_length') and states[current_idx].rocket_length is not None else None
                    
                    # Get plot dimensions to account for aspect ratio
                    x_lim = ax.get_xlim()
                    y_lim = ax.get_ylim()
                    plot_width = x_lim[1] - x_lim[0]
                    plot_height = y_lim[1] - y_lim[0]
                    
                    # Get figure dimensions to calculate true aspect ratio
                    fig_width, fig_height = fig.get_size_inches()
                    # Calculate the actual display aspect ratio (accounting for subplot size)
                    # The trajectory plot takes 2 rows and 2 columns out of 3x3 grid
                    subplot_width_ratio = 2.0 / 3.0
                    subplot_height_ratio = 2.0 / 3.0
                    display_width = fig_width * subplot_width_ratio
                    display_height = fig_height * subplot_height_ratio
                    
                    # Calculate how data units map to display units
                    data_to_display_x = display_width / plot_width
                    data_to_display_y = display_height / plot_height
                    
                    if rocket_length is None:
                        # Use 2% of plot height as default rocket length (in data units)
                        rocket_length = plot_height * 0.02
                    
                    # Calculate rod endpoints: half length in each direction along orientation
                    # Orientation is CCW from x-axis, so cos/sin gives direction
                    # To make the rod visually match the compass, we need to compensate for aspect ratio
                    # The compass shows the true direction (aspect='equal'), so we adjust here
                    half_length = rocket_length / 2.0
                    
                    # Calculate direction vector in data coordinates
                    dir_x_data = np.cos(orient_val)
                    dir_y_data = np.sin(orient_val)
                    
                    # Calculate aspect ratio (how much Y is stretched relative to X)
                    aspect_ratio = data_to_display_y / data_to_display_x
                    
                    # To make the visual angle match the compass, we need to adjust the X component
                    # If Y is stretched more (aspect_ratio > 1), we need to stretch X more in data coords
                    # So we multiply dx by aspect_ratio to compensate
                    dx = dir_x_data * half_length * aspect_ratio
                    dy = dir_y_data * half_length
                    
                    # Rod endpoints: center of mass ± half length along orientation
                    rod_x = [pos_x_val - dx, pos_x_val + dx]
                    rod_y = [pos_y_val - dy, pos_y_val + dy]
                    
                    rocket_rod, = ax.plot(rod_x, rod_y, 'r-', linewidth=4, alpha=0.9, label='Rocket')
                    
                    # Draw thrust vector line from rocket tail showing gimbal direction
                    # Gimbal angle is in rocket frame, so we need to add it to orientation to get world frame direction
                    # The thrust comes from the tail (opposite end from nose)
                    # Calculate tail position (opposite end of rod from nose)
                    # Nose is at pos + (dx, dy), tail is at pos - (dx, dy)
                    tail_x = pos_x_val - dx
                    tail_y = pos_y_val - dy
                    
                    # Thrust direction in world frame = rocket orientation + (gimbal angle * multiplier for visibility)
                    # Apply multiplier to make small gimbal angles more visible
                    amplified_gimbal = gimbal_val * gimbal_visual_multiplier
                    thrust_angle_world = orient_val + amplified_gimbal
                    
                    # Calculate thrust vector direction (accounting for aspect ratio like the rod)
                    thrust_dir_x_data = np.cos(thrust_angle_world)
                    thrust_dir_y_data = np.sin(thrust_angle_world)
                    
                    # Use a reasonable length for the thrust line (proportional to rocket length)
                    thrust_line_length = rocket_length * 0.5  # Half the rocket length
                    thrust_half_length = thrust_line_length / 2.0
                    
                    # Account for aspect ratio so it visually matches
                    thrust_dx = thrust_dir_x_data * thrust_half_length * aspect_ratio
                    thrust_dy = thrust_dir_y_data * thrust_half_length
                    
                    # Thrust line endpoints: from tail, extending in thrust direction
                    thrust_line_x = [tail_x, tail_x + thrust_dx * 2]  # *2 because we want full length
                    thrust_line_y = [tail_y, tail_y + thrust_dy * 2]
                    
                    thrust_line, = ax.plot(thrust_line_x, thrust_line_y, 'orange', linewidth=2, alpha=0.8, linestyle='--', label='Thrust')
                    
                    # Orientation compass: Show orientation in world frame
                    # Orientation is CCW from x-axis, so:
                    # - 0° = pointing right (East)
                    # - 90° = pointing up (North) = upright rocket
                    # - 180° = pointing left (West)
                    # - 270° = pointing down (South)
                    orient_dx = np.cos(orient_val)
                    orient_dy = np.sin(orient_val)
                    
                    orient_arrow = ax_orient.annotate(
                        '',
                        xy=(orient_dx * 0.9, orient_dy * 0.9),
                        xytext=(0, 0),
                        arrowprops=dict(
                            arrowstyle='->',
                            lw=3,
                            color='cyan',
                            mutation_scale=25,
                        ),
                    )
                    
                    # Gimbal compass: Show gimbal angle in rocket frame
                    # Gimbal angle is relative to rocket's forward direction (rocket frame)
                    # In rocket frame: 0° = forward (rocket's nose direction, +X axis)
                    # Positive = gimbal right (+Y axis), negative = gimbal left (-Y axis)
                    # The compass shows rocket frame directly, so we use gimbal_angle as-is
                    # In the compass: right = +X, up = +Y (matching rocket frame)
                    gimbal_dx = np.cos(gimbal_val)
                    gimbal_dy = np.sin(gimbal_val)
                    
                    gimbal_arrow = ax_gimbal.annotate(
                        '',
                        xy=(gimbal_dx * 0.9, gimbal_dy * 0.9),
                        xytext=(0, 0),
                        arrowprops=dict(
                            arrowstyle='->',
                            lw=3,
                            color='orange',
                            mutation_scale=25,
                        ),
                    )
            except (IndexError, ValueError, TypeError):
                # If there's any error accessing the data, skip arrow creation
                pass
            
            # Update angle labels
            orient_angle_deg = np.degrees(orientation_arr[current_idx])
            gimbal_angle_deg = np.degrees(gimbal_angle_arr[current_idx])
            orient_text.set_text(f'{orient_angle_deg:.1f}°')
            gimbal_text.set_text(f'{gimbal_angle_deg:.3f}°')
            
            # Update time plots
            orientation_deg = np.degrees(orientation_arr)
            gimbal_angle_deg_arr = np.degrees(gimbal_angle_arr)
            angular_velocity_deg_per_s = np.degrees(angular_velocity_arr)
            
            if looping_mode:
                # In loop mode: show data up to current frame
                orient_time_line.set_data(t_arr[:current_idx+1], orientation_deg[:current_idx+1])
                gimbal_time_line.set_data(t_arr[:current_idx+1], gimbal_angle_deg_arr[:current_idx+1])
                angular_vel_time_line.set_data(t_arr[:current_idx+1], angular_velocity_deg_per_s[:current_idx+1])
                thrust_time_line.set_data(t_arr[:current_idx+1], thrust_magnitude_arr[:current_idx+1])
                
                # Update IMU orientation line if available
                if len(imu_orientation) > 0 and len(imu_orientation) > current_idx:
                    imu_orient_valid_t = [t_arr[i] for i in range(current_idx+1) if i < len(imu_orientation) and imu_orientation[i] is not None]
                    imu_orient_valid = [np.degrees(imu_orientation[i]) for i in range(current_idx+1) if i < len(imu_orientation) and imu_orientation[i] is not None]
                    if len(imu_orient_valid) > 0:
                        imu_orient_time_line.set_data(imu_orient_valid_t, imu_orient_valid)
                
                # Update IMU angular velocity line if available
                if len(imu_angular_velocity) > 0 and len(imu_angular_velocity) > current_idx:
                    imu_angvel_valid_t = [t_arr[i] for i in range(current_idx+1) if i < len(imu_angular_velocity) and imu_angular_velocity[i] is not None]
                    imu_angvel_valid = [np.degrees(imu_angular_velocity[i]) for i in range(current_idx+1) if i < len(imu_angular_velocity) and imu_angular_velocity[i] is not None]
                    if len(imu_angvel_valid) > 0:
                        imu_angular_vel_time_line.set_data(imu_angvel_valid_t, imu_angvel_valid)
                
                # Update thrust sensor line if available
                if len(thrust_sensor) > 0 and len(thrust_sensor) > current_idx:
                    thrust_sensor_valid_t = [t_arr[i] for i in range(current_idx+1) if i < len(thrust_sensor) and thrust_sensor[i] is not None]
                    thrust_sensor_valid = [thrust_sensor[i] for i in range(current_idx+1) if i < len(thrust_sensor) and thrust_sensor[i] is not None]
                    if len(thrust_sensor_valid) > 0:
                        thrust_sensor_time_line.set_data(thrust_sensor_valid_t, thrust_sensor_valid)
            else:
                # In streaming mode: show all data
                orient_time_line.set_data(t_arr, orientation_deg)
                gimbal_time_line.set_data(t_arr, gimbal_angle_deg_arr)
                angular_vel_time_line.set_data(t_arr, angular_velocity_deg_per_s)
                thrust_time_line.set_data(t_arr, thrust_magnitude_arr)
                
                # Update IMU orientation line if available
                if len(imu_orientation) > 0:
                    imu_orient_valid_t = [t_arr[i] for i in range(len(imu_orientation)) if imu_orientation[i] is not None]
                    imu_orient_valid = [np.degrees(imu_orientation[i]) for i in range(len(imu_orientation)) if imu_orientation[i] is not None]
                    if len(imu_orient_valid) > 0:
                        imu_orient_time_line.set_data(imu_orient_valid_t, imu_orient_valid)
                
                # Update IMU angular velocity line if available
                if len(imu_angular_velocity) > 0:
                    imu_angvel_valid_t = [t_arr[i] for i in range(len(imu_angular_velocity)) if imu_angular_velocity[i] is not None]
                    imu_angvel_valid = [np.degrees(imu_angular_velocity[i]) for i in range(len(imu_angular_velocity)) if imu_angular_velocity[i] is not None]
                    if len(imu_angvel_valid) > 0:
                        imu_angular_vel_time_line.set_data(imu_angvel_valid_t, imu_angvel_valid)
                
                # Update thrust sensor line if available
                if len(thrust_sensor) > 0:
                    thrust_sensor_valid_t = [t_arr[i] for i in range(len(thrust_sensor)) if thrust_sensor[i] is not None]
                    thrust_sensor_valid = [thrust_sensor[i] for i in range(len(thrust_sensor)) if thrust_sensor[i] is not None]
                    if len(thrust_sensor_valid) > 0:
                        thrust_sensor_time_line.set_data(thrust_sensor_valid_t, thrust_sensor_valid)
            
            orient_time_point.set_data([t_arr[current_idx]], [orientation_deg[current_idx]])
            gimbal_time_point.set_data([t_arr[current_idx]], [gimbal_angle_deg_arr[current_idx]])
            angular_vel_time_point.set_data([t_arr[current_idx]], [angular_velocity_deg_per_s[current_idx]])
            thrust_time_point.set_data([t_arr[current_idx]], [thrust_magnitude_arr[current_idx]])
        
        # Update time text
        if len(t_arr) > 0:
            time_text.set_text(f'Time: {t_arr[current_idx]:.2f} s')
        
        return [trajectory_line, start_point, time_text, orient_text, gimbal_text,
                orient_time_line, orient_time_point, gimbal_time_line, gimbal_time_point,
                angular_vel_time_line, angular_vel_time_point,
                thrust_time_line, thrust_time_point] + \
               ([gps_line] if len(gps_x) > 0 else []) + \
               ([imu_orient_time_line] if len(imu_orientation) > 0 else []) + \
               ([imu_angular_vel_time_line] if len(imu_angular_velocity) > 0 else []) + \
               ([thrust_sensor_time_line] if len(thrust_sensor) > 0 else []) + \
               ([rocket_rod] if rocket_rod is not None else []) + \
               ([thrust_line] if thrust_line is not None else []) + \
               ([orient_arrow] if orient_arrow is not None else []) + \
               ([gimbal_arrow] if gimbal_arrow is not None else [])
    
    # Create animation - runs continuously, checking for new data each frame
    def animation_generator():
        """Generator that yields frame numbers indefinitely."""
        frame = 0
        while True:
            yield frame
            frame += 1
    
    # Create animation first (needed for callbacks)
    anim = animation.FuncAnimation(
        fig, animate, frames=animation_generator(),
        interval=current_interval, blit=False, repeat=True  # Repeat to allow looping
    )
    
    # If in final_mode, trigger initial frame to show final state
    if final_mode and len(states) > 0:
        # Directly call animate function to show final frame
        # Temporarily set is_paused to False so animate runs
        was_paused = is_paused
        is_paused = False
        # Verify loop_frame is set correctly
        expected_final_idx = len(states) - 1
        if loop_frame != expected_final_idx:
            print(f"Warning: loop_frame={loop_frame}, expected={expected_final_idx}. Correcting...", file=sys.stderr)
            loop_frame = expected_final_idx
        # Call animate - it will use loop_frame which is already set to final index (len(states) - 1)
        animate(0)  # Call animate function directly with frame 0
        # After animate runs, loop_frame will be incremented, so reset it back to final index
        # so that when paused, it stays on the final frame
        loop_frame = expected_final_idx
        print(f"Animate called, loop_frame reset to: {loop_frame}, states length: {len(states)}", file=sys.stderr)
        fig.canvas.draw()  # Force canvas to update
        is_paused = was_paused
    
    # Control callbacks
    def update_speed(val):
        """Update animation speed from slider."""
        nonlocal current_interval
        # Slider value is in range 1-100, convert to interval (1 = fast, 100 = slow)
        # Invert so higher value = faster (lower interval)
        current_interval = int(10 + (100 - val) * 0.9)  # Range: 10-100ms
        anim.event_source.interval = current_interval
    
    def toggle_pause(event):
        """Toggle pause/resume."""
        nonlocal is_paused
        is_paused = not is_paused
        pause_button.label.set_text('Resume' if is_paused else 'Pause')
        fig.canvas.draw_idle()
    
    def step_frame_forward(event):
        """Step one frame forward."""
        nonlocal step_frame, loop_frame, looping_mode
        if len(states) == 0:
            return
        step_frame = True
        if looping_mode:
            loop_frame = (loop_frame + 1) % len(states)
        else:
            # In streaming mode, just show latest frame
            pass
        # Trigger animation update
        anim.event_source.trigger()
    
    def step_frame_backward(event):
        """Step one frame backward."""
        nonlocal step_frame, loop_frame, looping_mode
        if len(states) == 0:
            return
        step_frame = True
        if looping_mode:
            loop_frame = (loop_frame - 1) % len(states)
        else:
            # In streaming mode, can't go backward
            return
        # Trigger animation update
        anim.event_source.trigger()
    
    def reset_animation(event):
        """Reset animation to beginning."""
        nonlocal loop_frame, is_paused, looping_mode
        if len(states) == 0:
            return
        # If in final_mode, start looping from beginning
        if final_mode:
            looping_mode = True
            loop_frame = 0
        else:
            loop_frame = 0
        is_paused = False
        pause_button.label.set_text('Pause')
        # Trigger animation update to show first frame
        anim.event_source.trigger()
    
    def toggle_follow_mode(event):
        """Toggle follow mode when position plot is clicked."""
        nonlocal follow_mode
        # Only toggle if click is on the main trajectory plot
        if event.inaxes == ax:
            follow_mode = not follow_mode
            # Update title to indicate follow mode
            if follow_mode:
                ax.set_title('Rocket Position Trajectory (Following)', color='white')
            else:
                ax.set_title('Rocket Position Trajectory', color='white')
            fig.canvas.draw_idle()
    
    # Create control widgets
    # Layout is already adjusted via gridspec parameters
    
    # Speed slider
    ax_slider = plt.axes([0.2, 0.02, 0.5, 0.03], facecolor='#2d2d2d')
    speed_slider = widgets.Slider(
        ax_slider, 'Speed', 1, 100, valinit=50, valstep=1,
        valfmt='%d'
    )
    speed_slider.on_changed(update_speed)
    # Style slider for dark mode
    speed_slider.label.set_color('white')
    speed_slider.valtext.set_color('white')
    for spine in ax_slider.spines.values():
        spine.set_color('white')
    
    # Control buttons
    ax_pause = plt.axes([0.70, 0.01, 0.07, 0.04])
    pause_button = widgets.Button(ax_pause, 'Resume' if is_paused else 'Pause', color='#2d2d2d', hovercolor='#3d3d3d')
    pause_button.label.set_color('white')
    pause_button.on_clicked(toggle_pause)
    
    ax_step_back = plt.axes([0.78, 0.01, 0.07, 0.04])
    step_back_button = widgets.Button(ax_step_back, '◄ Step', color='#2d2d2d', hovercolor='#3d3d3d')
    step_back_button.label.set_color('white')
    step_back_button.on_clicked(step_frame_backward)
    
    ax_step_forward = plt.axes([0.86, 0.01, 0.07, 0.04])
    step_forward_button = widgets.Button(ax_step_forward, 'Step ►', color='#2d2d2d', hovercolor='#3d3d3d')
    step_forward_button.label.set_color('white')
    step_forward_button.on_clicked(step_frame_forward)
    
    ax_reset = plt.axes([0.94, 0.01, 0.05, 0.04])
    reset_button = widgets.Button(ax_reset, 'Reset', color='#2d2d2d', hovercolor='#3d3d3d')
    reset_button.label.set_color('white')
    reset_button.on_clicked(reset_animation)
    
    # Connect click event to toggle follow mode
    fig.canvas.mpl_connect('button_press_event', toggle_follow_mode)
    
    ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=9)
    # Layout is managed by gridspec, no need for tight_layout
    
    if output_file:
        if output_file.endswith('.gif'):
            try:
                anim.save(output_file, writer='pillow', fps=1000//interval)
                print(f"Animation saved to {output_file}")
            except Exception as e:
                print(f"Error saving GIF: {e}")
                print("Trying with ffmpeg...")
                anim.save(output_file.replace('.gif', '.mp4'), writer='ffmpeg', fps=1000//interval)
        elif output_file.endswith(('.mp4', '.mov')):
            anim.save(output_file, writer='ffmpeg', fps=1000//interval)
            print(f"Animation saved to {output_file}")
        else:
            print("Unsupported format. Use .gif or .mp4")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Animate rocket simulation')
    parser.add_argument('--output', '-o', type=str, help='Output file (GIF or MP4)')
    parser.add_argument('--interval', '-i', type=int, default=50, help='Animation interval in ms (default: 50)')
    parser.add_argument('--final', action='store_true', help='Wait for all data and show final frame statically')
    args = parser.parse_args()
    
    # Stream and animate in real-time
    animate_simulation_streaming(args.output, args.interval, final_mode=args.final)

