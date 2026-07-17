#!/usr/bin/env python3
"""Generate Figure 5: Reflection-magnitude ablation curve.

Reads results/reflection_ablation.jsonl (per-theta best-of-restart records)
and produces a plot of best-of-restart score S versus |cos theta_v| at
each vertex, with the manuscript's key qualitative regions annotated.

The curve is an inverted U: both extremes collapse to S ~ 0.73, with a
peak around |cos theta| ~ 0.55 where S = 0.886. This visual is the
mechanistic evidence for Discussion III.A: reflection is necessary
to break the reflectionless-family ceiling, but too much reflection
decouples the operator.

Usage
-----
    python experiments/figure5_reflection_curve.py

Outputs
-------
results/figure5_reflection_curve.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


JSONL_PATH = Path("results/reflection_ablation.jsonl")
FIGURE_PATH = Path("results/figure5_reflection_curve.png")


def _load_curve() -> list[dict]:
    if not JSONL_PATH.exists():
        print(f"ERROR: {JSONL_PATH} does not exist.")
        sys.exit(1)
    records: list[dict] = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("best_mad_mean_spacings", 99) < 90:
                    records.append(r)
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda r: r["cos_theta_abs"])
    return records


def main() -> None:
    records = _load_curve()
    if not records:
        print(f"ERROR: no valid records in {JSONL_PATH}.")
        sys.exit(1)

    cos_theta = np.array([r["cos_theta_abs"] for r in records])
    scores = np.array([r["best_score"] for r in records])
    mads = np.array([r["best_mad_mean_spacings"] for r in records])

    peak_idx = int(np.argmax(scores))
    peak_cos = float(cos_theta[peak_idx])
    peak_score = float(scores[peak_idx])
    peak_mad = float(mads[peak_idx])

    fig, (ax_s, ax_m) = plt.subplots(
        1, 2, figsize=(13, 5.5), sharex=True,
    )

    # ---- Left panel: score ----
    ax_s.plot(
        cos_theta, scores,
        marker="o", markersize=8,
        color="#1a3a5a", linewidth=2,
        label="Best-of-restart $\\mathcal{S}$",
    )
    ax_s.scatter(
        [peak_cos], [peak_score],
        marker="*", s=280, color="crimson",
        zorder=5, label=f"Peak: $\\mathcal{{S}} = {peak_score:.3f}$",
    )
    ax_s.axhline(
        y=0.72, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
        label="TRS-broken Neumann ceiling ($\\mathcal{S} \\approx 0.72$)",
    )
    ax_s.set_xlabel("$|\\cos\\theta_v|$ (reflection amplitude)", fontsize=13)
    ax_s.set_ylabel("Best-of-restart score $\\mathcal{S}$", fontsize=13)
    ax_s.set_title(
        "Score vs reflection amplitude\n"
        "$C_7$, $\\theta_v$ fixed at every vertex, $(\\alpha, \\beta, \\gamma)$ optimized",
        fontsize=12,
    )
    ax_s.legend(loc="lower center", fontsize=10, framealpha=0.95)
    ax_s.grid(True, alpha=0.2)
    ax_s.tick_params(labelsize=11)
    ax_s.set_ylim(0.70, 0.92)

    # ---- Right panel: MAD ----
    ax_m.plot(
        cos_theta, mads,
        marker="s", markersize=7,
        color="#7b1c14", linewidth=2,
        label="Best-of-restart MAD",
    )
    ax_m.scatter(
        [peak_cos], [peak_mad],
        marker="*", s=280, color="crimson",
        zorder=5, label=f"MAD at peak: {peak_mad:.3f}",
    )
    ax_m.set_xlabel("$|\\cos\\theta_v|$ (reflection amplitude)", fontsize=13)
    ax_m.set_ylabel("Best-of-restart MAD (mean spacings)", fontsize=13)
    ax_m.set_title(
        "MAD vs reflection amplitude\n"
        "Same pipeline, MAD is the position-error signal",
        fontsize=12,
    )
    ax_m.legend(loc="upper center", fontsize=10, framealpha=0.95)
    ax_m.grid(True, alpha=0.2)
    ax_m.tick_params(labelsize=11)

    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=200)
    plt.close()

    print(f"Figure saved to {FIGURE_PATH}")
    print()
    print("Curve (sorted by |cos theta|):")
    for r in records:
        print(
            f"  |cos theta| = {r['cos_theta_abs']:.3f}  "
            f"S = {r['best_score']:.4f}  "
            f"MAD = {r['best_mad_mean_spacings']:.4f}"
        )
    print()
    print(f"Peak at |cos theta| = {peak_cos:.3f}: S = {peak_score:.4f}, MAD = {peak_mad:.4f}")


if __name__ == "__main__":
    main()
