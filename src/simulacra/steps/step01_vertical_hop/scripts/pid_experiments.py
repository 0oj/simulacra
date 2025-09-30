#!/usr/bin/env python3
"""
PID experiments: interactive animation + batch testing.

Usage examples:
  # interactive single run (opens animation window)
  uv run python scripts/pid_experiments.py --controller PID --kp 2.2 --ki 1.0 --kd 0.7 --t_final 6.0

  # batch sweep (two parameter sets)
  uv run python scripts/pid_experiments.py --batch \
    --batch-configs "[{'name':'P','kp':2.0,'ki':0.0,'kd':0.0},{'name':'PID','kp':2.2,'ki':1.0,'kd':0.7}]" \
    --outdir out/batch1

  # save animation gif (requires pillow)
  uv run python scripts/pid_experiments.py --controller PID --save-gif out/anim.gif

Notes:
 - script uses your existing sim building blocks:
    simulacra.steps.step01_vertical_hop.sim: State, Rocket, PID
    simulacra.core.integrators: rk4_step
 - To save GIF: `uv add pillow` then `uv install` (PillowWriter).
 - To save MP4 via ffmpeg: have ffmpeg on PATH; FuncAnimation will write via FFMpegWriter.
"""

from __future__ import annotations
import argparse
import ast
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# imports from your package
from simulacra.steps.step01_vertical_hop.sim import State, Rocket, PID
from simulacra.core.integrators import rk4_step


# ------------------------------
# Small utility: run a simple sim
# ------------------------------
def run_simple_sim(
    rocket: Rocket,
    pid: PID,
    z_sp: float,
    t_final: float = 6.0,
    dt: float = 0.002,
    Tmax: float = 300.0,
    initial_z: float = 0.0,
):
    """Run minimal sim using rk4_step, return logs dict with numpy arrays."""
    state = State(z=initial_z, v=0.0)
    t = 0.0
    pid.reset()
    t_list, z_list, v_list, thrust_list, u_list, sp_list = [], [], [], [], [], []

    steps = int(np.ceil(t_final / dt))
    for _ in range(steps):
        z_meas = state.z  # perfect measurement here; later can inject noise
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

    return {
        "t": np.array(t_list),
        "z": np.array(z_list),
        "v": np.array(v_list),
        "thrust": np.array(thrust_list),
        "u": np.array(u_list),
        "sp": np.array(sp_list),
    }


# ------------------------------
# Performance metrics
# ------------------------------
@dataclass
class PIDMetrics:
    rise_time: Optional[float]  # time to first reach 90% of setpoint (s)
    overshoot: float  # (max - sp) / sp * 100% (negative if undershoot)
    settling_time: Optional[float]  # time after which signal remains within 2% band
    steady_error: float  # final error (sp - final_z)


def compute_metrics(
    logs: Dict[str, np.ndarray],
    sp: float,
    rise_pct: float = 0.9,
    settling_tol: float = 0.02,
) -> PIDMetrics:
    """Compute simple time-domain metrics from logs."""
    t = logs["t"]
    z = logs["z"]
    if len(t) == 0:
        return PIDMetrics(None, 0.0, None, float(sp))
    # Rise time: time to first reach rise_pct * sp
    target = rise_pct * sp
    above = np.where(z >= target)[0]
    rise_time = float(t[above[0]]) if above.size > 0 else None
    # Overshoot: relative to sp
    z_max = np.max(z)
    overshoot = ((z_max - sp) / sp) * 100.0
    # Steady-state error: last value
    final_error = sp - float(z[-1])
    # Settling time: last time after which z stays within (1 +/- settling_tol)*sp
    low = (1.0 - settling_tol) * sp
    high = (1.0 + settling_tol) * sp
    outside = np.where((z < low) | (z > high))[0]
    if outside.size == 0:
        settling_time = 0.0
    else:
        last_outside_idx = outside[-1]
        if last_outside_idx >= len(t) - 1:
            settling_time = None
        else:
            settling_time = float(t[last_outside_idx + 1])
    return PIDMetrics(rise_time, overshoot, settling_time, final_error)


# ------------------------------
# Animation helpers
# ------------------------------
def animate_logs(
    logs: Dict[str, np.ndarray],
    title: str = "Vertical Hop",
    save_path: Optional[str] = None,
    fps: int = 30,
    max_frames: int = 300,  # Limit frames for reasonable file size
):
    """Create a matplotlib animation for a single logs dict.

    Fixed: uses a scalar setpoint value extracted from logs['sp'] to avoid
    array/singleton ambiguity when computing axis limits.
    """
    t = logs["t"]
    z = logs["z"]
    u = logs["u"]
    thrust = logs["thrust"]
    sp_arr = logs["sp"]
    # pick a scalar setpoint for plotting (use first element if array-like)
    if np.size(sp_arr) == 0:
        sp_val = 0.0
    else:
        sp_val = float(np.asarray(sp_arr).flatten()[0])

    # Downsample data for animation to avoid massive files
    N_total = len(t)
    if N_total == 0:
        print("No data to animate.")
        return None

    # Calculate step size to get max_frames
    step = max(1, N_total // max_frames)
    N = min(max_frames, N_total)

    # Downsample all arrays
    t_anim = t[::step][:N]
    z_anim = z[::step][:N]
    u_anim = u[::step][:N]
    thrust_anim = thrust[::step][:N]

    fig, (ax_alt, ax_th) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    (ln_z,) = ax_alt.plot([], [], lw=2, label="altitude (m)")
    (ln_sp,) = ax_alt.plot([], [], "--", lw=1, label="setpoint (m)")
    ax_alt.set_ylabel("m")
    ax_alt.grid(True)
    ax_alt.legend(loc="upper left")

    (ln_th,) = ax_th.plot([], [], lw=2, label="thrust (N)")
    (ln_u,) = ax_th.plot([], [], "--", lw=1, label="u (norm)")
    ax_th.set_ylabel("thrust / u")
    ax_th.grid(True)
    ax_th.legend(loc="upper left")

    ax_th.set_xlabel("time (s)")
    fig.suptitle(title)

    # compute some safe axis extents using scalar values and array min/max
    z_min = float(np.nanmin(z_anim))
    z_max = float(np.nanmax(z_anim))
    th_min = float(np.nanmin(thrust_anim))
    th_max = float(np.nanmax(thrust_anim))

    def init():
        ax_alt.set_xlim(0, max(1.0, float(t_anim[-1])))
        # altitude y-limits: expand slightly around data and setpoint
        alt_lo = min(z_min - 0.2 * max(abs(z_min), 1.0), sp_val * 0.5 - 1.0)
        alt_hi = max(z_max + 0.2 * max(abs(z_max), 1.0), sp_val * 1.5 + 1.0)
        ax_alt.set_ylim(alt_lo, alt_hi)

        ax_th.set_xlim(0, max(1.0, float(t_anim[-1])))
        th_lo = min(th_min - 0.2 * max(abs(th_min), 1.0), -0.1)
        th_hi = max(th_max + 0.2 * max(abs(th_max), 1.0), 1.0)
        ax_th.set_ylim(th_lo, th_hi)

        return ln_z, ln_sp, ln_th, ln_u

    def update(i):
        # Show data up to frame i (no growing buffers!)
        end_idx = i + 1
        t_show = t_anim[:end_idx]
        z_show = z_anim[:end_idx]
        u_show = u_anim[:end_idx]
        thrust_show = thrust_anim[:end_idx]

        # update lines
        ln_z.set_data(t_show, z_show)
        # setpoint plotted as constant scalar line
        ln_sp.set_data(t_show, np.full(len(t_show), sp_val))
        ln_th.set_data(t_show, thrust_show)
        ln_u.set_data(t_show, u_show)

        # No autoscaling - keep fixed limits for performance
        return ln_z, ln_sp, ln_th, ln_u

    anim = animation.FuncAnimation(
        fig, update, frames=range(N), init_func=init, blit=False, interval=1000.0 / fps
    )
    plt.tight_layout()

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        print(
            f"Saving animation to {save_path} ({N} frames at {fps} fps) — this may take a moment."
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if ext in [".gif", ".gifv"]:
            try:
                from matplotlib.animation import PillowWriter

                writer = PillowWriter(fps=fps)
                anim.save(
                    save_path, writer=writer, dpi=100
                )  # Lower DPI for smaller files
                print(f"Successfully saved GIF: {save_path}")
            except ImportError:
                print("PillowWriter not available. Install with: uv add pillow")
            except Exception as e:
                print("Failed to save GIF:", e)
        elif ext in [".mp4", ".mov", ".m4v"]:
            try:
                from matplotlib.animation import FFMpegWriter

                writer = FFMpegWriter(
                    fps=fps, bitrate=1000
                )  # Lower bitrate for smaller files
                anim.save(save_path, writer=writer, dpi=100)
                print(f"Successfully saved MP4: {save_path}")
            except ImportError:
                print("FFMpegWriter not available. Install ffmpeg or use GIF format.")
            except Exception as e:
                print("Failed to save MP4:", e)
        else:
            print("Unknown extension; supported: .gif, .mp4")
    else:
        plt.show()

    return anim


# ------------------------------
# Batch runner: runs multiple configs, writes metrics CSV
# ------------------------------
def run_batch(configs: List[Dict[str, Any]], outdir: str):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for idx, cfg in enumerate(configs):
        name = cfg.get("name", f"run{idx}")
        kp = float(cfg.get("kp", 2.0))
        ki = float(cfg.get("ki", 0.0))
        kd = float(cfg.get("kd", 0.0))
        u_min = float(cfg.get("u_min", 0.0))
        u_max = float(cfg.get("u_max", 1.0))
        integrator_limit = cfg.get("integrator_limit", None)
        Tmax = float(cfg.get("Tmax", 300.0))
        z_sp = float(cfg.get("z_sp", 2.0))
        dt = float(cfg.get("dt", 0.002))
        t_final = float(cfg.get("t_final", 6.0))
        rocket = Rocket(
            mass_kg=cfg.get("mass_kg", 10.0),
            area=cfg.get("area", 0.05),
            cd=cfg.get("cd", 0.6),
            g=cfg.get("g", 9.81),
        )
        pid = PID(
            kp=kp,
            ki=ki,
            kd=kd,
            u_min=u_min,
            u_max=u_max,
            integrator_limit=integrator_limit,
        )
        logs = run_simple_sim(
            rocket,
            pid,
            z_sp=z_sp,
            t_final=t_final,
            dt=dt,
            Tmax=Tmax,
            initial_z=cfg.get("initial_z", 0.0),
        )
        metrics = compute_metrics(logs, sp=z_sp)
        row = {
            "name": name,
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "Tmax": Tmax,
            "rise_time": metrics.rise_time,
            "overshoot_pct": metrics.overshoot,
            "settling_time": metrics.settling_time,
            "steady_error": metrics.steady_error,
        }
        rows.append(row)
        # save a small plot per run
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.plot(logs["t"], logs["z"], label="z")
        plt.plot(logs["t"], logs["sp"], "--", label="sp")
        plt.xlabel("t (s)")
        plt.ylabel("z (m)")
        plt.title(name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        png_path = os.path.join(outdir, f"{name}.png")
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved plot for {name} -> {png_path}")

        # save animation if requested
        save_anim = cfg.get("save_anim")
        if save_anim:
            print(f"Creating animation for {name} -> {save_anim}")
            animate_logs(
                logs,
                title=f"{name} (kp={kp}, ki={ki}, kd={kd})",
                save_path=save_anim,
            )
    # write CSV
    csv_path = os.path.join(outdir, "batch_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Batch complete. Metrics saved to", csv_path)
    return csv_path


# ------------------------------
# CLI parser
# ------------------------------
def parse_batch_configs(arg: Optional[str]) -> List[Dict[str, Any]]:
    if not arg:
        return []
    # allow passing python literal list/dict via CLI (use ast.literal_eval for safety)
    try:
        configs = ast.literal_eval(arg)
    except Exception as e:
        raise ValueError(
            "Could not parse batch configs; provide a Python literal list of dicts."
        ) from e
    if not isinstance(configs, list):
        raise ValueError("Batch configs must be a list of dicts.")
    return configs


def main_cli():
    p = argparse.ArgumentParser(
        description="PID experiments: interactive animation + batch testing."
    )
    p.add_argument("--controller", choices=["P", "PI", "PID"], default="PID")
    p.add_argument("--kp", type=float, default=2.2)
    p.add_argument("--ki", type=float, default=1.0)
    p.add_argument("--kd", type=float, default=0.7)
    p.add_argument("--Tmax", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--t_final", type=float, default=6.0)
    p.add_argument("--z_sp", type=float, default=2.0)
    p.add_argument(
        "--save-gif", type=str, default=None, help="Path to save gif (requires pillow)."
    )
    p.add_argument(
        "--save-mp4", type=str, default=None, help="Path to save mp4 (requires ffmpeg)."
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Run multiple configs passed to --batch-configs.",
    )
    p.add_argument(
        "--batch-configs",
        type=str,
        default=None,
        help="Python literal list of dicts specifying configs.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="out/pid_experiments",
        help="Output directory for batch results.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of animation frames (default: 300).",
    )
    args = p.parse_args()

    if args.batch:
        configs = parse_batch_configs(args.batch_configs)
        if not configs:
            raise SystemExit(
                "No batch configs provided. Pass --batch-configs \"[{'name':'A', 'kp':2.0}, ...]\""
            )
        run_batch(configs, outdir=args.outdir)
        return

    # single-run interactive
    rocket = Rocket(mass_kg=10.0, area=0.05, cd=0.6)
    # choose controller type
    kp = args.kp
    ki = args.ki
    kd = args.kd
    if args.controller == "P":
        pid = PID(kp=kp, ki=0.0, kd=0.0, u_min=0.0, u_max=1.0)
    elif args.controller == "PI":
        pid = PID(kp=kp, ki=ki, kd=0.0, u_min=0.0, u_max=1.0, integrator_limit=8.0)
    else:
        pid = PID(kp=kp, ki=ki, kd=kd, u_min=0.0, u_max=1.0, integrator_limit=8.0)

    logs = run_simple_sim(
        rocket, pid, z_sp=args.z_sp, t_final=args.t_final, dt=args.dt, Tmax=args.Tmax
    )
    # compute metrics and print
    metrics = compute_metrics(logs, sp=args.z_sp)
    print("Metrics:", metrics)

    # animation (show or save)
    save_path = args.save_gif or args.save_mp4
    animate_logs(
        logs,
        title=f"Controller={args.controller} kp={kp} ki={ki} kd={kd}",
        save_path=save_path,
        max_frames=args.max_frames,
    )
    # also save a final static plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 3))
    plt.plot(logs["t"], logs["z"], label="z")
    plt.plot(logs["t"], logs["sp"], "--", label="sp")
    plt.xlabel("t (s)")
    plt.ylabel("z (m)")
    plt.grid(True)
    plt.legend()
    os.makedirs("out", exist_ok=True)
    png = os.path.join("out", f"run_{int(time.time())}.png")
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    print("Saved static plot to", png)


if __name__ == "__main__":
    main_cli()
