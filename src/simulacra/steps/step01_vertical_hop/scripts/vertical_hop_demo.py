#!/usr/bin/env python3
"""
Vertical Hop Demo: Showcase the best PID controller in action.

This demo runs a vertical hop simulation with the optimal PID parameters
found from our grid search, creating a beautiful visualization that
demonstrates rocket control fundamentals.
"""

from __future__ import annotations
import argparse
import os
import time
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches

# imports from your package
from simulacra.steps.step01_vertical_hop.sim import State, Rocket, PID
from simulacra.core.integrators import rk4_step


def run_vertical_hop_demo(
    rocket: Rocket,
    pid: PID,
    z_sp: float = 5.0,  # Higher setpoint for more dramatic demo
    t_final: float = 8.0,
    dt: float = 0.001,  # Higher resolution for smooth demo
    Tmax: float = 300.0,
    initial_z: float = 0.0,
    initial_v: float = 0.0,
):
    """Run a high-quality vertical hop simulation for demo purposes."""
    state = State(z=initial_z, v=initial_v)
    t = 0.0
    pid.reset()

    # High-resolution logging
    t_list, z_list, v_list, thrust_list, u_list, sp_list = [], [], [], [], [], []

    steps = int(np.ceil(t_final / dt))
    print(f"Running {steps:,} simulation steps...")

    for step in range(steps):
        z_meas = state.z  # Perfect measurement
        error = z_sp - z_meas
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
        sp_list.append(z_sp)

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


def create_enhanced_animation(
    logs: Dict[str, np.ndarray],
    rocket: Rocket,
    title: str = "Vertical Hop Demo",
    save_path: str = None,
    fps: int = 30,
    max_frames: int = 400,
):
    """Create an enhanced animation with rocket visualization."""
    t = logs["t"]
    z = logs["z"]
    v = logs["v"]
    thrust = logs["thrust"]
    u = logs["u"]
    sp = logs["sp"][0]  # Scalar setpoint

    # Downsample for animation
    N_total = len(t)
    step = max(1, N_total // max_frames)
    N = min(max_frames, N_total)

    t_anim = t[::step][:N]
    z_anim = z[::step][:N]
    v_anim = v[::step][:N]
    thrust_anim = thrust[::step][:N]
    u_anim = u[::step][:N]

    # Create figure with enhanced layout
    fig = plt.figure(figsize=(15, 10))
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
    ax_rocket.set_ylim(0, sp + 2)
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

    # Setpoint line in rocket view
    ax_rocket.axhline(
        y=sp, color="red", linestyle="--", alpha=0.7, linewidth=2, label="Target"
    )

    # Set up axes
    ax_main.set_ylabel("Altitude (m)", fontsize=12)
    ax_main.set_title(title, fontsize=14, fontweight="bold")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc="upper right")

    ax_rocket.set_ylabel("Altitude (m)", fontsize=12)
    ax_rocket.legend(loc="upper right")

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

    # Add setpoint line
    ax_main.axhline(y=sp, color="red", linestyle="--", alpha=0.7, linewidth=2)

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

        # Update lines
        ln_z.set_data(t_show, z_show)
        ln_sp.set_data(t_show, np.full(len(t_show), sp))
        ln_th.set_data(t_show, thrust_show)
        ln_u.set_data(t_show, u_show)

        # Update rocket position in separate view
        if len(z_show) > 0:
            current_z = z_show[-1]
            current_thrust = thrust_show[-1]

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
            error = sp - current_z

            metrics_text.set_text(
                f"Altitude: {current_z:.2f}m\n"
                f"Velocity: {current_v:.2f}m/s\n"
                f"Error: {error:.2f}m\n"
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
        print(f"Creating enhanced animation: {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ext = os.path.splitext(save_path)[1].lower()
        if ext in [".gif", ".gifv"]:
            try:
                from matplotlib.animation import PillowWriter

                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=120)
                print(f"✅ Demo animation saved: {save_path}")
            except ImportError:
                print("❌ PillowWriter not available. Install with: uv add pillow")
        else:
            print("❌ Unsupported format. Use .gif")
    else:
        plt.show()

    return anim


def create_performance_summary(logs: Dict[str, np.ndarray], rocket: Rocket, pid: PID):
    """Create a comprehensive performance summary."""
    t = logs["t"]
    z = logs["z"]
    v = logs["v"]
    thrust = logs["thrust"]
    u = logs["u"]
    sp = logs["sp"][0]

    # Calculate metrics
    z_max = np.max(z)
    overshoot = ((z_max - sp) / sp) * 100.0
    final_error = sp - z[-1]

    # Find rise time (90% of setpoint)
    target_90 = 0.9 * sp
    rise_idx = np.where(z >= target_90)[0]
    rise_time = t[rise_idx[0]] if len(rise_idx) > 0 else None

    # Find settling time (within 2% of setpoint)
    settling_tol = 0.02
    low = (1.0 - settling_tol) * sp
    high = (1.0 + settling_tol) * sp
    outside = np.where((z < low) | (z > high))[0]
    settling_time = (
        t[outside[-1] + 1] if len(outside) > 0 and outside[-1] < len(t) - 1 else None
    )

    print("\n" + "=" * 60)
    print("🚀 VERTICAL HOP DEMO - PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Rocket: {rocket.m:.1f}kg, Cd={rocket.cd:.2f}, Area={rocket.area:.3f}m²")
    print(f"PID: Kp={pid.kp:.1f}, Ki={pid.ki:.1f}, Kd={pid.kd:.1f}")
    print(f"Setpoint: {sp:.1f}m")
    print(f"Simulation: {t[-1]:.1f}s, {len(t):,} steps")
    print()
    print("📊 PERFORMANCE METRICS:")
    print(
        f"  🎯 Rise Time:     {rise_time:.3f}s"
        if rise_time
        else "  🎯 Rise Time:     N/A"
    )
    print(f"  📈 Overshoot:     {overshoot:.1f}%")
    print(
        f"  ⏱️  Settling Time: {settling_time:.3f}s"
        if settling_time
        else "  ⏱️  Settling Time: N/A"
    )
    print(f"  🎯 Steady Error:  {final_error:.3f}m")
    print(f"  🔥 Max Thrust:    {np.max(thrust):.0f}N")
    print(f"  ⚡ Max Velocity:  {np.max(np.abs(v)):.2f}m/s")
    print()

    # Performance rating
    if overshoot < 10:
        rating = "🏆 EXCELLENT"
    elif overshoot < 25:
        rating = "🥈 GOOD"
    elif overshoot < 50:
        rating = "🥉 FAIR"
    else:
        rating = "❌ POOR"

    print(f"🎖️  Overall Rating: {rating}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Vertical Hop Demo")
    parser.add_argument("--z_sp", type=float, default=5.0, help="Target altitude (m)")
    parser.add_argument(
        "--t_final", type=float, default=8.0, help="Simulation time (s)"
    )
    parser.add_argument("--dt", type=float, default=0.001, help="Time step (s)")
    parser.add_argument(
        "--save-demo", type=str, default=None, help="Save demo animation"
    )
    parser.add_argument("--fps", type=int, default=30, help="Animation FPS")
    parser.add_argument(
        "--max-frames", type=int, default=400, help="Max animation frames"
    )
    args = parser.parse_args()

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

    print("🚀 Starting Vertical Hop Demo...")
    print(f"   Target: {args.z_sp}m altitude")
    print(f"   Duration: {args.t_final}s")
    print(f"   PID: Kp={pid.kp}, Ki={pid.ki}, Kd={pid.kd}")

    # Run simulation
    logs = run_vertical_hop_demo(
        rocket=rocket,
        pid=pid,
        z_sp=args.z_sp,
        t_final=args.t_final,
        dt=args.dt,
        initial_z=0.0,
        initial_v=0.0,
    )

    # Show performance summary
    create_performance_summary(logs, rocket, pid)

    # Create enhanced animation
    save_path = args.save_demo or "out/vertical_hop_demo.gif"
    create_enhanced_animation(
        logs=logs,
        rocket=rocket,
        title=f"Vertical Hop Demo - Target: {args.z_sp}m",
        save_path=save_path,
        fps=args.fps,
        max_frames=args.max_frames,
    )

    print(f"\n🎬 Demo complete! Check out: {save_path}")


if __name__ == "__main__":
    main()
