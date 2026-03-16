"""
Plot ablation study results for thesis.

Reads results.jsonl and training_log_stage*.jsonl from each variant's run directory.
Generates publication-quality figures suitable for a thesis.

Usage:
    cd backend
    uv run python scripts/plot_ablation.py           # all available results
    uv run python scripts/plot_ablation.py --final    # only final (Stage 3) results
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ──────────────────────────────────────────────────────────────
# Style: thesis-quality, archival, technical
# ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

# Muted color palette (archival, not neon)
COLORS = {
    "v3_features": "#2c3e50",    # Dark blue-grey
    "v3_bignet": "#7f8c8d",      # Grey
    "A": "#c0392b",              # Muted red
    "B": "#f39c12",              # Amber
    "C": "#27ae60",              # Green (winner)
    "D": "#2980b9",              # Blue
    "DR": "#8e44ad",             # Purple
}

LABELS = {
    "v3_features": "v3_features (F1-F4, 64×64)",
    "v3_bignet": "v3_bignet (no feat, 128×128×64)",
    "A": "A: brake_focus (F1,F5,F6,F4)",
    "B": "B: kitchen_sink (all 9)",
    "C": "C: minimal (F5,F6 only)",
    "D": "D: energy (F1,F5,F9,F4)",
    "DR": "DR: brake_focus + 128×128 + DR",
}

SCENARIOS = ["easy_60m", "medium_130m", "hard_200m"]
SCENARIO_LABELS = {"easy_60m": "Easy (60m)", "medium_130m": "Medium (130m)", "hard_200m": "Hard (200m)"}


def load_results(runs_dir: Path) -> dict:
    """Load results.jsonl from all variant directories."""
    results = {}

    # Baselines
    for name, pattern in [
        ("v3_features", "guidance_v3_features"),
        ("v3_bignet", "guidance_v3_bignet"),
        ("DR", "guidance_dr"),
    ]:
        d = runs_dir / pattern
        if (d / "results.jsonl").exists():
            with open(d / "results.jsonl") as f:
                results[name] = [json.loads(line) for line in f]

    # Ablation variants
    for v in "ABCD":
        dirs = list(runs_dir.glob(f"ablation_{v}_*"))
        if dirs and (dirs[0] / "results.jsonl").exists():
            with open(dirs[0] / "results.jsonl") as f:
                results[v] = [json.loads(line) for line in f]

    return results


def load_training_logs(runs_dir: Path) -> dict:
    """Load training_log_stage*.jsonl from all variant directories."""
    logs = {}

    for name, pattern in [
        ("v3_features", "guidance_v3_features"),
        ("v3_bignet", "guidance_v3_bignet"),
        ("DR", "guidance_dr"),
    ]:
        d = runs_dir / pattern
        variant_logs = []
        for logfile in sorted(d.glob("training_log_stage*.jsonl")):
            with open(logfile) as f:
                variant_logs.extend([json.loads(line) for line in f])
        if variant_logs:
            logs[name] = variant_logs

    for v in "ABCD":
        dirs = list(runs_dir.glob(f"ablation_{v}_*"))
        if dirs:
            variant_logs = []
            for logfile in sorted(dirs[0].glob("training_log_stage*.jsonl")):
                with open(logfile) as f:
                    variant_logs.extend([json.loads(line) for line in f])
            if variant_logs:
                logs[v] = variant_logs

    return logs


def get_final_results(results: dict) -> dict:
    """Extract the last eval result for each variant."""
    final = {}
    for name, entries in results.items():
        if entries:
            final[name] = entries[-1]
    return final


def plot_bar_comparison(results: dict, output_dir: Path, stage_label="Latest"):
    """Bar chart comparing landing rates across variants and scenarios."""
    final = get_final_results(results)
    if not final:
        print("No results to plot.")
        return

    variants = [k for k in ["v3_features", "v3_bignet", "A", "B", "C", "D", "DR"] if k in final]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for ax, scenario in zip(axes, SCENARIOS):
        rates = []
        colors = []
        labels = []
        for v in variants:
            sc = final[v].get("scenarios", {}).get(scenario, {})
            rate = sc.get("landing_rate", 0) * 100
            rates.append(rate)
            colors.append(COLORS.get(v, "#888"))
            labels.append(v)

        bars = ax.bar(range(len(variants)), rates, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(PercentFormatter())

        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{rate:.0f}%", ha="center", va="bottom", fontsize=7)

    axes[0].set_ylabel("Landing Rate")
    fig.suptitle(f"Feature Ablation — Landing Rate by Scenario ({stage_label})", fontsize=12)
    plt.tight_layout()

    out = output_dir / "ablation_bar_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_stage_progression(results: dict, output_dir: Path):
    """Line chart showing how each variant's Hard 200m rate evolves across stages."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in ["v3_features", "v3_bignet", "A", "B", "C", "D", "DR"]:
        if name not in results:
            continue
        entries = results[name]
        stages = list(range(1, len(entries) + 1))
        hard_rates = []
        for entry in entries:
            rate = entry.get("scenarios", {}).get("hard_200m", {}).get("landing_rate", 0) * 100
            hard_rates.append(rate)

        ax.plot(stages, hard_rates, "o-", color=COLORS.get(name, "#888"),
                label=LABELS.get(name, name), markersize=6)

    ax.set_xlabel("Curriculum Stage")
    ax.set_ylabel("Hard (200m) Landing Rate")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Stage 1\n(80m, 2M)", "Stage 2\n(130m, +2.5M)", "Stage 3\n(200m, +3.5M)"])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Hard (200m) Landing Rate Across Curriculum Stages")
    plt.tight_layout()

    out = output_dir / "ablation_stage_progression.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_training_dynamics(logs: dict, output_dir: Path):
    """Plot landing rate over training steps for each variant."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for name in ["v3_features", "v3_bignet", "A", "B", "C", "D", "DR"]:
        if name not in logs:
            continue
        entries = logs[name]
        steps = [e["steps"] for e in entries]
        rates = [e["landing_rate"] * 100 for e in entries]

        ax.plot(steps, rates, color=COLORS.get(name, "#888"),
                label=LABELS.get(name, name), alpha=0.8)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Landing Rate (rolling 200 episodes)")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax.set_title("Training Dynamics — Landing Rate Over Time")

    # Mark stage boundaries
    for boundary, label in [(2_000_000, "Stage 2"), (4_500_000, "Stage 3")]:
        ax.axvline(x=boundary, color="#bdc3c7", linestyle="--", linewidth=0.8)
        ax.text(boundary + 50000, 95, label, fontsize=7, color="#7f8c8d")

    plt.tight_layout()
    out = output_dir / "ablation_training_dynamics.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_precision_vs_rate(results: dict, output_dir: Path):
    """Scatter: landing rate vs average distance for Hard 200m."""
    final = get_final_results(results)

    fig, ax = plt.subplots(figsize=(7, 5))

    for name in ["v3_features", "v3_bignet", "A", "B", "C", "D", "DR"]:
        if name not in final:
            continue
        sc = final[name].get("scenarios", {}).get("hard_200m", {})
        rate = sc.get("landing_rate", 0) * 100
        dist = sc.get("avg_dist_m", 0)
        if rate > 0:
            ax.scatter(rate, dist, color=COLORS.get(name, "#888"), s=80, zorder=5,
                       edgecolors="white", linewidths=0.5)
            ax.annotate(name, (rate, dist), textcoords="offset points",
                        xytext=(8, 4), fontsize=8, color=COLORS.get(name, "#888"))

    ax.set_xlabel("Landing Rate (%)")
    ax.set_ylabel("Average Landing Distance (m)")
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Hard (200m) — Landing Rate vs. Precision")
    ax.invert_yaxis()  # Lower distance = better, so invert
    plt.tight_layout()

    out = output_dir / "ablation_rate_vs_precision.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def print_summary_table(results: dict):
    """Print a markdown table of latest results."""
    final = get_final_results(results)

    print("\n## Latest Results Summary\n")
    print("| Variant | Easy (60m) | Medium (130m) | Hard (200m) | Stage |")
    print("|---------|------------|---------------|-------------|-------|")

    for name in ["v3_features", "v3_bignet", "A", "B", "C", "D", "DR"]:
        if name not in final:
            continue
        entry = final[name]
        stage = entry.get("stage", "?")
        cols = []
        for sc in SCENARIOS:
            data = entry.get("scenarios", {}).get(sc, {})
            rate = data.get("landing_rate", 0)
            dist = data.get("avg_dist_m", 0)
            cols.append(f"{rate*100:.0f}% / {dist:.1f}m")
        print(f"| {name} | {' | '.join(cols)} | {stage} |")


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--final", action="store_true", help="Only show final results")
    args = parser.parse_args()

    runs_dir = Path("runs")
    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(runs_dir)
    logs = load_training_logs(runs_dir)

    print(f"Found results for: {list(results.keys())}")
    print(f"Found training logs for: {list(logs.keys())}")

    if args.final:
        # Filter to variants with Stage 3 results only
        results = {k: v for k, v in results.items()
                   if v and "Stage 3" in v[-1].get("stage", "")}
        print(f"Filtered to Stage 3 final: {list(results.keys())}")

    print_summary_table(results)

    stage_label = "Final (Stage 3)" if args.final else "Latest"
    print("\nGenerating plots...")
    plot_bar_comparison(results, output_dir, stage_label=stage_label)
    plot_stage_progression(results, output_dir)
    plot_training_dynamics(logs, output_dir)
    plot_precision_vs_rate(results, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
