#!/usr/bin/env python3
"""
Multi-Hop Vertical Demo: Showcase multiple vertical hops in sequence.

This demo runs a series of vertical hops with different target altitudes,
demonstrating the rocket's ability to perform multiple controlled maneuvers.
"""

from __future__ import annotations
import argparse
import os
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# imports from your package
from simulacra.steps.step01_vertical_hop.sim import State, Rocket, PID
from simulacra.core.integrators import rk4_step


def run_multi_hop_demo(
    rocket: Rocket,
    pid: PID,
    hop_sequence: List[Tuple[float, float]],  # [(setpoint, duration), ...]
    dt: float = 0.001,
    Tmax: float = 300.0,
    initial_z: float = 0.0,
    initial_v: float = 0.0,
):
    """Run a multi-hop simulation with time-varying setpoints."""
    state = State(z=initial_z, v=initial_v)
    t = 0.0
    pid.reset()

    # High-resolution logging
    t_list, z_list, v_list, thrust_list, u_list, sp_list = [], [], [], [], [], []

    # Calculate total simulation time
    total_time = sum(duration for _, duration in hop_sequence)
    steps = int(np.ceil(total_time / dt))

    print(f"Running {steps:,} simulation steps for {len(hop_sequence)} hops...")
    print("Hop sequence:")
    for i, (sp, duration) in enumerate(hop_sequence):
        print(f"  Hop {i + 1}: {sp:.1f}m for {duration:.1f}s")

    current_hop = 0
    hop_start_time = 0.0
    current_setpoint = hop_sequence[0][0] if hop_sequence else 0.0

    for step in range(steps):
        # Check if we need to change setpoint
        if (
            current_hop < len(hop_sequence) - 1
            and t >= hop_start_time + hop_sequence[current_hop][1]
        ):
            current_hop += 1
            hop_start_time = t
            current_setpoint = hop_sequence[current_hop][0]
            print(
                f"  Switching to hop {current_hop + 1}: {current_setpoint:.1f}m at t={t:.1f}s"
            )

        z_meas = state.z  # Perfect measurement
        error = current_setpoint - z_meas
        u = pid(error, dt)
        thrust = u * Tmax

        def f(sv):
            s = State.from_array(sv)
            return rocket.derivatives(s, thrust)

        new_arr = rk4_step(f, state.as_array(), dt)
        new_state = State.from_array(new_arr)

        t_list.append(t)
        z_list.append(state.z)
        v_list.append(state.v)
        thrust_list.append(thrust)
        u_list.append(u)
        sp_list.append(current_setpoint)

        state = new_state
        t += dt

        # Progress indicator
        if step % (steps // 10) == 0:
            print(f"  Progress: {100 * step / steps:.0f}%")

    return {
        "t": np.array(t_list),
        "z": np.array(z_list),
        "v": np.array(v_list),
        "thrust": np.array(thrust_list),
        "u": np.array(u_list),
        "sp": np.array(sp_list),
    }


def create_multi_hop_animation(
    logs: Dict[str, np.ndarray],
    rocket: Rocket,
    hop_sequence: List[Tuple[float, float]],
    title: str = "Multi-Hop Vertical Demo",
    save_path: str = None,
    fps: int = 30,
    max_frames: int = 500,
):
    """Create an enhanced animation for multi-hop demo."""
    t = logs["t"]
    z = logs["z"]
    v = logs["v"]
    thrust = logs["thrust"]
    u = logs["u"]
    sp = logs["sp"]

    # Downsample for animation
    N_total = len(t)
    step = max(1, N_total // max_frames)
    N = min(max_frames, N_total)

    t_anim = t[::step][:N]
    z_anim = z[::step][:N]
    v_anim = v[::step][:N]
    thrust_anim = thrust[::step][:N]
    u_anim = u[::step][:N]
    sp_anim = sp[::step][:N]

    # Create figure with enhanced layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])

    # Main altitude plot
    ax_main = fig.add_subplot(gs[0, :2])
    ax_rocket = fig.add_subplot(gs[0, 2])  # Separate rocket view
    ax_thrust = fig.add_subplot(gs[1, :])
    ax_control = fig.add_subplot(gs[2, :])

    # Initialize plot elements
    (ln_z,) = ax_main.plot([], [], "b-", lw=3, label="Altitude (m)")
    (ln_sp,) = ax_main.plot([], [], "r--", lw=2, label="Setpoint (m)")
    (ln_th,) = ax_thrust.plot([], [], "g-", lw=2, label="Thrust (N)")
    (ln_u,) = ax_control.plot([], [], "m-", lw=2, label="Control Signal")

    # Rocket visualization in separate view
    ax_rocket.set_xlim(-1, 1)
    max_alt = max(np.max(z_anim), max(sp for sp, _ in hop_sequence)) + 2
    ax_rocket.set_ylim(0, max_alt)
    ax_rocket.set_aspect("equal")
    ax_rocket.set_title("Rocket View", fontsize=12, fontweight="bold")
    ax_rocket.grid(True, alpha=0.3)

    # Create rocket body (vertical rectangle) - better proportions
    rocket_body = Rectangle((-0.08, 0), 0.16, 0.4, color="darkblue", alpha=0.8)
    ax_rocket.add_patch(rocket_body)

    # Create rocket nose cone - properly centered triangle
    nose_cone = plt.Polygon(
        [(-0.08, 0.4), (0.08, 0.4), (0, 0.6)], color="darkblue", alpha=0.8
    )
    ax_rocket.add_patch(nose_cone)

    # Add rocket fins for more realistic look
    fin1 = Rectangle((-0.12, 0), 0.04, 0.15, color="darkblue", alpha=0.8)
    fin2 = Rectangle((0.08, 0), 0.04, 0.15, color="darkblue", alpha=0.8)
    ax_rocket.add_patch(fin1)
    ax_rocket.add_patch(fin2)

    # Thrust visualization - simple line
    thrust_line = ax_rocket.plot([], [], "orange", lw=4, alpha=0.8)[0]

    # Ground line
    ax_rocket.axhline(y=0, color="brown", linewidth=3, alpha=0.7, label="Ground")

    # Add setpoint lines for each hop
    hop_colors = ["red", "orange", "green", "blue", "purple", "brown"]
    hop_lines = []
    for i, (sp_val, duration) in enumerate(hop_sequence):
        color = hop_colors[i % len(hop_colors)]
        line = ax_rocket.axhline(
            y=sp_val,
            color=color,
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label=f"Hop {i + 1}: {sp_val:.1f}m",
        )
        hop_lines.append(line)

    # Set up axes
    ax_main.set_ylabel("Altitude (m)", fontsize=12)
    ax_main.set_title(title, fontsize=14, fontweight="bold")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc="upper right")

    ax_rocket.set_ylabel("Altitude (m)", fontsize=12)
    ax_rocket.legend(loc="upper right", fontsize=8)

    ax_thrust.set_ylabel("Thrust (N)", fontsize=12)
    ax_thrust.grid(True, alpha=0.3)
    ax_thrust.legend(loc="upper right")

    ax_control.set_ylabel("Control Signal", fontsize=12)
    ax_control.set_xlabel("Time (s)", fontsize=12)
    ax_control.grid(True, alpha=0.3)
    ax_control.legend(loc="upper right")

    # Set axis limits
    z_min, z_max = np.min(z_anim), np.max(z_anim)
    ax_main.set_xlim(0, t_anim[-1])
    ax_main.set_ylim(max(0, z_min - 1), z_max + 2)

    ax_thrust.set_xlim(0, t_anim[-1])
    ax_thrust.set_ylim(0, np.max(thrust_anim) * 1.1)

    ax_control.set_xlim(0, t_anim[-1])
    ax_control.set_ylim(0, 1.1)

    # Performance metrics text
    metrics_text = ax_main.text(
        0.02,
        0.98,
        "",
        transform=ax_main.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    def init():
        return (
            ln_z,
            ln_sp,
            ln_th,
            ln_u,
            rocket_body,
            nose_cone,
            fin1,
            fin2,
            thrust_line,
            metrics_text,
        )

    def update(i):
        # Show data up to frame i
        end_idx = i + 1
        t_show = t_anim[:end_idx]
        z_show = z_anim[:end_idx]
        thrust_show = thrust_anim[:end_idx]
        u_show = u_anim[:end_idx]
        sp_show = sp_anim[:end_idx]

        # Update lines
        ln_z.set_data(t_show, z_show)
        ln_sp.set_data(t_show, sp_show)  # Show actual setpoint changes
        ln_th.set_data(t_show, thrust_show)
        ln_u.set_data(t_show, u_show)

        # Update rocket position in separate view
        if len(z_show) > 0:
            current_z = z_show[-1]
            current_thrust = thrust_show[-1]
            current_sp = sp_show[-1]

            # Move rocket body vertically
            rocket_body.set_y(current_z)
            # Update nose cone position - properly centered
            nose_cone.set_xy(
                [
                    [-0.08, current_z + 0.4],
                    [0.08, current_z + 0.4],
                    [0, current_z + 0.6],
                ]
            )
            # Update fins position
            fin1.set_y(current_z)
            fin2.set_y(current_z)

            # Update thrust line
            thrust_scale = min(current_thrust / 200, 0.5)  # Scale thrust for visibility
            thrust_line.set_data([0, 0], [current_z, current_z - thrust_scale])

        # Update metrics
        if len(z_show) > 0:
            current_z = z_show[-1]
            current_v = v_anim[:end_idx][-1] if len(v_anim[:end_idx]) > 0 else 0
            current_thrust = thrust_show[-1]
            current_sp = sp_show[-1]
            error = current_sp - current_z

            # Determine current hop
            current_hop = 1
            for j, (sp_val, duration) in enumerate(hop_sequence):
                if t_show[-1] <= sum(hop_sequence[k][1] for k in range(j + 1)):
                    current_hop = j + 1
                    break

            metrics_text.set_text(
                f"Hop: {current_hop}/{len(hop_sequence)}\n"
                f"Altitude: {current_z:.2f}m\n"
                f"Target: {current_sp:.1f}m\n"
                f"Error: {error:.2f}m\n"
                f"Velocity: {current_v:.2f}m/s\n"
                f"Thrust: {current_thrust:.0f}N"
            )

        return (
            ln_z,
            ln_sp,
            ln_th,
            ln_u,
            rocket_body,
            nose_cone,
            fin1,
            fin2,
            thrust_line,
            metrics_text,
        )

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=range(N), init_func=init, blit=False, interval=1000.0 / fps
    )

    plt.tight_layout()

    if save_path:
        print(f"Creating multi-hop animation: {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ext = os.path.splitext(save_path)[1].lower()
        if ext in [".gif", ".gifv"]:
            try:
                from matplotlib.animation import PillowWriter

                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=120)
                print(f"✅ Multi-hop animation saved: {save_path}")
            except ImportError:
                print("❌ PillowWriter not available. Install with: uv add pillow")
        else:
            print("❌ Unsupported format. Use .gif")
    else:
        plt.show()

    return anim


def create_multi_hop_performance_summary(
    logs: Dict[str, np.ndarray],
    rocket: Rocket,
    pid: PID,
    hop_sequence: List[Tuple[float, float]],
):
    """Create a comprehensive performance summary for multi-hop demo."""
    t = logs["t"]
    z = logs["z"]
    v = logs["v"]
    thrust = logs["thrust"]
    u = logs["u"]
    sp = logs["sp"]

    print("\n" + "=" * 70)
    print("🚀 MULTI-HOP VERTICAL DEMO - PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Rocket: {rocket.m:.1f}kg, Cd={rocket.cd:.2f}, Area={rocket.area:.3f}m²")
    print(f"PID: Kp={pid.kp:.1f}, Ki={pid.ki:.1f}, Kd={pid.kd:.1f}")
    print(f"Total Hops: {len(hop_sequence)}")
    print(f"Simulation: {t[-1]:.1f}s, {len(t):,} steps")
    print()

    # Analyze each hop
    hop_start_times = [0]
    for i in range(len(hop_sequence) - 1):
        hop_start_times.append(hop_start_times[-1] + hop_sequence[i][1])

    print("📊 HOP-BY-HOP ANALYSIS:")
    print("-" * 70)

    for i, (target_sp, duration) in enumerate(hop_sequence):
        start_time = hop_start_times[i]
        end_time = start_time + duration

        # Find data for this hop
        hop_mask = (t >= start_time) & (t <= end_time)
        hop_t = t[hop_mask]
        hop_z = z[hop_mask]
        hop_sp = sp[hop_mask]

        if len(hop_z) == 0:
            continue

        # Calculate metrics for this hop
        z_max = np.max(hop_z)
        overshoot = ((z_max - target_sp) / target_sp) * 100.0 if target_sp > 0 else 0
        final_error = target_sp - hop_z[-1]

        # Find rise time (90% of target)
        target_90 = 0.9 * target_sp
        rise_idx = np.where(hop_z >= target_90)[0]
        rise_time = hop_t[rise_idx[0]] - start_time if len(rise_idx) > 0 else None

        print(f"Hop {i + 1}: {target_sp:.1f}m for {duration:.1f}s")
        print(
            f"  🎯 Rise Time:     {rise_time:.3f}s"
            if rise_time
            else "  🎯 Rise Time:     N/A"
        )
        print(f"  📈 Overshoot:     {overshoot:.1f}%")
        print(f"  🎯 Final Error:   {final_error:.3f}m")
        print(f"  ⚡ Max Velocity:  {np.max(np.abs(v[hop_mask])):.2f}m/s")
        print()

    # Overall performance
    total_overshoot = np.mean(
        [
            (
                (
                    np.max(
                        z[
                            (t >= hop_start_times[i])
                            & (t <= hop_start_times[i] + hop_sequence[i][1])
                        ]
                    )
                    - hop_sequence[i][0]
                )
                / hop_sequence[i][0]
            )
            * 100.0
            for i in range(len(hop_sequence))
            if hop_sequence[i][0] > 0
        ]
    )

    print("🏆 OVERALL PERFORMANCE:")
    print(f"  📈 Average Overshoot: {total_overshoot:.1f}%")
    print(f"  🔥 Max Thrust Used:    {np.max(thrust):.0f}N")
    print(f"  ⚡ Max Velocity:       {np.max(np.abs(v)):.2f}m/s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Multi-Hop Vertical Demo")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step (s)")
    parser.add_argument(
        "--save-demo", type=str, default=None, help="Save demo animation"
    )
    parser.add_argument("--fps", type=int, default=30, help="Animation FPS")
    parser.add_argument(
        "--max-frames", type=int, default=500, help="Max animation frames"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="3,4,2,5,1",
        help="Comma-separated altitude sequence (m)",
    )
    args = parser.parse_args()

    # Parse hop sequence
    try:
        hop_altitudes = [float(x.strip()) for x in args.sequence.split(",")]
        hop_duration = 4.0  # 4 seconds per hop
        hop_sequence = [(alt, hop_duration) for alt in hop_altitudes]
    except ValueError:
        print(
            "❌ Invalid sequence format. Use comma-separated numbers like '3,4,2,5,1'"
        )
        return

    # Create rocket with realistic parameters
    rocket = Rocket(
        mass_kg=10.0,  # 10kg rocket
        area=0.05,  # 0.05 m² cross-section
        cd=0.6,  # Drag coefficient
        g=9.81,  # Earth gravity
    )

    # Use the best PID parameters from our grid search
    pid = PID(
        kp=2.2,  # Best proportional gain
        ki=1.0,  # Best integral gain
        kd=0.7,  # Best derivative gain
        u_min=0.0,  # Minimum thrust (0%)
        u_max=1.0,  # Maximum thrust (100%)
        integrator_limit=8.0,  # Prevent windup
    )

    print("🚀 Starting Multi-Hop Vertical Demo...")
    print(f"   Sequence: {hop_sequence}")
    print(f"   PID: Kp={pid.kp}, Ki={pid.ki}, Kd={pid.kd}")

    # Run simulation
    logs = run_multi_hop_demo(
        rocket=rocket,
        pid=pid,
        hop_sequence=hop_sequence,
        dt=args.dt,
        initial_z=0.0,
        initial_v=0.0,
    )

    # Show performance summary
    create_multi_hop_performance_summary(logs, rocket, pid, hop_sequence)

    # Create enhanced animation
    save_path = args.save_demo or "out/multi_hop_demo.gif"
    create_multi_hop_animation(
        logs=logs,
        rocket=rocket,
        hop_sequence=hop_sequence,
        title=f"Multi-Hop Demo - {len(hop_sequence)} Hops",
        save_path=save_path,
        fps=args.fps,
        max_frames=args.max_frames,
    )

    print(f"\n🎬 Multi-hop demo complete! Check out: {save_path}")


if __name__ == "__main__":
    main()
