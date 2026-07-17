#!/usr/bin/env python3
"""Generate Figure 4: Riemann-seed vs GUE-surrogate MAD distributions.

Reads two JSONL files:
    results/riemann_seed_distribution.jsonl  (K_R Riemann seeds)
    results/gue_surrogate_control.jsonl      (K surrogate targets)

Produces a side-by-side histogram of best-of-restart MADs from both
distributions on a shared axis, with the Mann-Whitney U statistic
annotated.

The two distributions are disjoint: every Riemann seed's MAD is smaller
than every surrogate MAD. This is the visual evidence supporting
Section III.G's distribution-vs-distribution specificity claim.

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
from scipy import stats as sp_stats


RIEMANN_JSONL = Path("results/riemann_seed_distribution.jsonl")
SURROGATE_JSONL = Path("results/gue_surrogate_control.jsonl")
FIGURE_PATH = Path("results/figure4_surrogate_distribution.png")


def _load_mads(path: Path, label: str) -> list[float]:
    if not path.exists():
        print(f"ERROR: {path} does not exist (for {label} distribution).")
        sys.exit(1)
    mads: list[float] = []
    with open(path) as f:
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
    if not mads:
        print(f"ERROR: no valid MAD records in {path}.")
        sys.exit(1)
    return mads


def main() -> None:
    riemann_mads = np.asarray(_load_mads(RIEMANN_JSONL, "Riemann"))
    surrogate_mads = np.asarray(_load_mads(SURROGATE_JSONL, "surrogate"))

    r_mean = float(np.mean(riemann_mads))
    r_std = float(np.std(riemann_mads, ddof=1))
    r_med = float(np.median(riemann_mads))
    r_min = float(np.min(riemann_mads))
    r_max = float(np.max(riemann_mads))

    s_mean = float(np.mean(surrogate_mads))
    s_std = float(np.std(surrogate_mads, ddof=1))
    s_med = float(np.median(surrogate_mads))
    s_min = float(np.min(surrogate_mads))

    mw = sp_stats.mannwhitneyu(
        riemann_mads, surrogate_mads,
        alternative="two-sided", method="asymptotic",
    )
    u_stat = float(mw.statistic)
    p_val = float(mw.pvalue)

    fig, ax = plt.subplots(figsize=(10, 6))

    x_low = min(r_min, s_min) - 0.01
    x_high = float(np.max(surrogate_mads)) + 0.02
    bins = np.linspace(x_low, x_high, 30)

    ax.hist(
        riemann_mads, bins=bins,
        color="#c0392b", edgecolor="#7b1c14", alpha=0.85,
        label=f"Riemann seeds ($K_R = {len(riemann_mads)}$)",
        zorder=3,
    )
    ax.hist(
        surrogate_mads, bins=bins,
        color="steelblue", edgecolor="#1a3a5a", alpha=0.85,
        label=f"GUE surrogates ($K = {len(surrogate_mads)}$)",
        zorder=2,
    )

    ax.axvline(
        x=r_med, color="#7b1c14", linestyle="--",
        linewidth=1.3, alpha=0.75,
        label=f"Riemann median = {r_med:.3f}",
        zorder=4,
    )
    ax.axvline(
        x=s_med, color="#1a3a5a", linestyle="--",
        linewidth=1.3, alpha=0.75,
        label=f"Surrogate median = {s_med:.3f}",
        zorder=4,
    )

    ax.set_xlabel("Best-of-restart MAD (mean spacings)", fontsize=13)
    ax.set_ylabel("Number of optimizations", fontsize=13)
    ax.set_title(
        "Riemann-seed vs GUE-surrogate MAD distributions\n"
        "$C_7$, U(2) vertex scattering, identical CMA-ES + Nelder--Mead pipeline",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=11)

    if p_val < 1e-10:
        p_str = f"{p_val:.1e}"
    else:
        p_str = f"{p_val:.2e}"

    annotation = (
        f"Riemann: $K_R = {len(riemann_mads)}$\n"
        f"  MAD $= {r_mean:.3f} \\pm {r_std:.3f}$\n"
        f"  min $= {r_min:.3f}$, max $= {r_max:.3f}$\n"
        f"\n"
        f"Surrogate: $K = {len(surrogate_mads)}$\n"
        f"  MAD $= {s_mean:.3f} \\pm {s_std:.3f}$\n"
        f"  min $= {s_min:.3f}$\n"
        f"\n"
        f"Mann--Whitney $U = {int(u_stat)}$\n"
        f"$p = {p_str}$ (two-sided)"
    )
    ax.text(
        0.98, 0.62, annotation,
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
    print(f"Riemann:   {r_mean:.4f} +/- {r_std:.4f} (K_R = {len(riemann_mads)}, "
          f"min {r_min:.4f}, max {r_max:.4f})")
    print(f"Surrogate: {s_mean:.4f} +/- {s_std:.4f} (K   = {len(surrogate_mads)}, "
          f"min {s_min:.4f})")
    print(f"MW U = {u_stat}, p = {p_val}")


if __name__ == "__main__":
    main()
