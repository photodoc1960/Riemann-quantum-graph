#!/usr/bin/env python3
"""Generate Figure 4: GUE surrogate MAD distribution vs Riemann baseline.

Reads results/gue_surrogate_control.jsonl (the K=50 per-surrogate optimization
records) and produces a histogram of best-of-restart MADs across all 50 GUE
surrogates, with a vertical line at the Riemann baseline MAD value.

Visual evidence supporting the manuscript's Section III.G "Riemann-specificity
vs generic GUE alignment" claim: the surrogate distribution lies entirely
above the Riemann value, with Cohen's d = 3.21 and quantile 0/50.

Usage
-----
    python experiments/figure4_surrogate_distribution.py

Outputs
-------
results/figure4_surrogate_distribution.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


JSONL_PATH = Path("results/gue_surrogate_control.jsonl")
SCALING_PATH = Path("results/u2_scaling_results.json")
FIGURE_PATH = Path("results/figure4_surrogate_distribution.png")

RIEMANN_N_VERTICES = 7  # the C_7 baseline used in the K=50 surrogate run


def _load_surrogate_mads() -> list[float]:
    if not JSONL_PATH.exists():
        print(f"ERROR: {JSONL_PATH} does not exist.")
        sys.exit(1)
    mads = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                m = r.get("best_mad_mean_spacings")
                if m is not None and m < 90:
                    mads.append(float(m))
            except json.JSONDecodeError:
                continue
    return mads


def _load_riemann_baseline() -> tuple[float, float]:
    if not SCALING_PATH.exists():
        print(f"ERROR: {SCALING_PATH} does not exist.")
        sys.exit(1)
    with open(SCALING_PATH) as f:
        data = json.load(f)
    entry = next(
        (r for r in data.get("results", []) if r["n"] == RIEMANN_N_VERTICES),
        None,
    )
    if entry is None:
        print(f"ERROR: no C_{RIEMANN_N_VERTICES} baseline in u2_scaling_results.json.")
        sys.exit(1)
    return float(entry["best_score"]), float(entry["best_mad"])


def main() -> None:
    surrogate_mads = _load_surrogate_mads()
    if not surrogate_mads:
        print(f"ERROR: no valid surrogate records in {JSONL_PATH}.")
        sys.exit(1)
    riemann_score, riemann_mad = _load_riemann_baseline()
    n_surrogates = len(surrogate_mads)

    surr_arr = np.array(surrogate_mads)
    mean_surr = float(np.mean(surr_arr))
    std_surr = float(np.std(surr_arr))
    cohens_d = (mean_surr - riemann_mad) / std_surr if std_surr > 1e-6 else 0.0
    n_better = int(np.sum(surr_arr < riemann_mad))
    quantile_pct = 100.0 * n_better / n_surrogates

    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- Histogram of surrogate MADs ----
    bins = np.linspace(
        min(0.14, riemann_mad - 0.01),
        max(0.40, float(np.max(surr_arr)) + 0.01),
        24,
    )
    ax.hist(
        surr_arr,
        bins=bins,
        color="steelblue",
        edgecolor="#1a3a5a",
        alpha=0.85,
        label=f"GUE surrogates (N = {n_surrogates})",
        zorder=2,
    )

    # ---- Vertical line at Riemann baseline ----
    ax.axvline(
        x=riemann_mad,
        color="crimson",
        linestyle="-",
        linewidth=2.5,
        label=f"Riemann MAD = {riemann_mad:.3f}",
        zorder=4,
    )

    # ---- Mean ± std band of surrogate distribution ----
    ax.axvline(
        x=mean_surr,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=f"Surrogate mean = {mean_surr:.3f}",
        zorder=3,
    )
    ax.axvspan(
        mean_surr - std_surr, mean_surr + std_surr,
        color="black", alpha=0.08, zorder=1,
        label=f"Surrogate mean $\\pm$ std",
    )

    # ---- Axes and labels ----
    ax.set_xlabel("Best-of-restart MAD (mean spacings)", fontsize=13)
    ax.set_ylabel("Number of surrogates", fontsize=13)
    ax.set_title(
        "GUE surrogate MAD distribution vs Riemann baseline\n"
        "$C_7$ topology, $U(2)$ vertex scattering, identical optimization pipeline",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=11)

    # ---- Annotation: the verdict ----
    annotation = (
        f"$K = {n_surrogates}$ surrogates\n"
        f"surrogate MAD: ${mean_surr:.3f} \\pm {std_surr:.3f}$\n"
        f"Cohen's $d$ = {cohens_d:.2f}\n"
        f"empirical quantile: {n_better}/{n_surrogates} "
        f"({quantile_pct:.0f}%)\n"
        f"base-rate floor: $1/(K+1) \\approx "
        f"{100.0/(n_surrogates+1):.1f}$%"
    )
    ax.text(
        0.97, 0.55, annotation,
        transform=ax.transAxes,
        fontsize=10.5, ha="right", va="top",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.95,
        ),
    )

    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=200)
    plt.close()

    print(f"Figure saved to {FIGURE_PATH}")
    print()
    print(f"Riemann MAD:        {riemann_mad:.4f}")
    print(f"Surrogate MAD:      {mean_surr:.4f} +/- {std_surr:.4f} (N = {n_surrogates})")
    print(f"Surrogate min:      {float(np.min(surr_arr)):.4f}")
    print(f"Surrogate max:      {float(np.max(surr_arr)):.4f}")
    print(f"Cohen's d:          {cohens_d:.2f}")
    print(f"Quantile:           {n_better}/{n_surrogates} ({quantile_pct:.1f}%)")
    print(f"Base-rate floor:    {100.0/(n_surrogates+1):.2f}%")


if __name__ == "__main__":
    main()
