#!/usr/bin/env python3
"""
Grid search summary: analyze and visualize PID parameter sweep results.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os


def create_summary_plots(csv_path: str, outdir: str):
    """Create summary plots from grid search results."""
    # Read CSV manually
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "name": row["name"],
                    "kp": float(row["kp"]),
                    "ki": float(row["ki"]),
                    "kd": float(row["kd"]),
                    "rise_time": float(row["rise_time"]) if row["rise_time"] else None,
                    "overshoot_pct": float(row["overshoot_pct"]),
                    "steady_error": float(row["steady_error"]),
                }
            )

    # Convert to numpy arrays for easier handling
    names = [d["name"] for d in data]
    kp = np.array([d["kp"] for d in data])
    ki = np.array([d["ki"] for d in data])
    kd = np.array([d["kd"] for d in data])
    rise_times = np.array(
        [d["rise_time"] if d["rise_time"] is not None else np.nan for d in data]
    )
    overshoots = np.array([d["overshoot_pct"] for d in data])
    steady_errors = np.array([d["steady_error"] for d in data])

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # 1. Performance comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Rise time vs Overshoot
    scatter = ax1.scatter(
        rise_times, overshoots, c=kp, s=100, alpha=0.7, cmap="viridis"
    )
    ax1.set_xlabel("Rise Time (s)")
    ax1.set_ylabel("Overshoot (%)")
    ax1.set_title("Rise Time vs Overshoot (colored by Kp)")
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label="Kp")

    # Add labels for each point
    for i, name in enumerate(names):
        ax1.annotate(
            name,
            (rise_times[i], overshoots[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Steady state error vs Overshoot
    scatter2 = ax2.scatter(
        steady_errors, overshoots, c=ki, s=100, alpha=0.7, cmap="plasma"
    )
    ax2.set_xlabel("Steady State Error (m)")
    ax2.set_ylabel("Overshoot (%)")
    ax2.set_title("Steady State Error vs Overshoot (colored by Ki)")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label="Ki")

    # Add labels for each point
    for i, name in enumerate(names):
        ax2.annotate(
            name,
            (steady_errors[i], overshoots[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Parameter space visualization
    ax3.scatter(kp, ki, c=overshoots, s=100, alpha=0.7, cmap="Reds")
    ax3.set_xlabel("Kp (Proportional)")
    ax3.set_ylabel("Ki (Integral)")
    ax3.set_title("Parameter Space (colored by Overshoot)")
    ax3.grid(True, alpha=0.3)

    # Add labels for each point
    for i, name in enumerate(names):
        ax3.annotate(
            name, (kp[i], ki[i]), xytext=(5, 5), textcoords="offset points", fontsize=8
        )

    # Performance ranking
    sorted_indices = np.argsort(overshoots)
    sorted_names = [names[i] for i in sorted_indices]
    sorted_overshoots = overshoots[sorted_indices]

    bars = ax4.barh(range(len(sorted_names)), sorted_overshoots)
    ax4.set_yticks(range(len(sorted_names)))
    ax4.set_yticklabels(sorted_names, fontsize=8)
    ax4.set_xlabel("Overshoot (%)")
    ax4.set_title("Performance Ranking (Lower Overshoot = Better)")
    ax4.grid(True, alpha=0.3)

    # Color bars by performance
    for i, bar in enumerate(bars):
        if sorted_overshoots[i] < 50:
            bar.set_color("green")
        elif sorted_overshoots[i] < 100:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, "grid_search_summary.png"), dpi=150, bbox_inches="tight"
    )
    plt.show()

    # 2. Detailed metrics table
    print("\n" + "=" * 80)
    print("PID GRID SEARCH RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Name':<25} {'Kp':<6} {'Ki':<6} {'Kd':<6} {'Rise(s)':<8} {'Overshoot%':<12} {'SteadyErr':<10}"
    )
    print("-" * 80)

    for i in sorted_indices:
        rise_str = f"{rise_times[i]:.3f}" if not np.isnan(rise_times[i]) else "N/A"
        print(
            f"{names[i]:<25} {kp[i]:<6.1f} {ki[i]:<6.1f} {kd[i]:<6.1f} "
            f"{rise_str:<8} {overshoots[i]:<12.1f} {steady_errors[i]:<10.3f}"
        )

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)

    # Find best performers
    best_overshoot_idx = np.argmin(overshoots)
    best_steady_idx = np.argmin(np.abs(steady_errors))
    fastest_rise_idx = np.nanargmin(rise_times)

    print(
        f"🏆 Best Overshoot Control: {names[best_overshoot_idx]} ({overshoots[best_overshoot_idx]:.1f}%)"
    )
    print(
        f"🎯 Best Steady State: {names[best_steady_idx]} ({steady_errors[best_steady_idx]:.3f}m error)"
    )
    print(
        f"⚡ Fastest Response: {names[fastest_rise_idx]} ({rise_times[fastest_rise_idx]:.3f}s)"
    )

    print(f"\n📊 Controller Type Performance:")
    p_mask = ki == 0.0
    pi_mask = (ki > 0.0) & (kd == 0.0)
    pid_mask = kd > 0.0

    print(f"   P-only:  avg overshoot = {np.mean(overshoots[p_mask]):.1f}%")
    print(f"   PI:      avg overshoot = {np.mean(overshoots[pi_mask]):.1f}%")
    print(f"   PID:     avg overshoot = {np.mean(overshoots[pid_mask]):.1f}%")

    print(f"\n🎬 Animated GIFs created in: {outdir}/")
    print("   Each GIF shows altitude vs setpoint and thrust vs time")
    print("   Compare the animations to see the control behavior differences!")


if __name__ == "__main__":
    csv_path = "out/grid_search/batch_metrics.csv"
    outdir = "out/grid_search"
    create_summary_plots(csv_path, outdir)
