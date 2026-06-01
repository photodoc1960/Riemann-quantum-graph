#!/usr/bin/env python3
"""Reanalyze the GUE surrogate control results on demand.

Reads results/gue_surrogate_control.jsonl (the per-surrogate records
written by the running gue_surrogate_control.py experiment) and runs
the aggregator + verdict logic without disturbing the running
optimization. Useful for peeking at the verdict mid-run, or for
re-running the aggregation against improved verdict criteria after
the optimization is complete.

Usage
-----
    python experiments/gue_surrogate_reanalyze.py
    python experiments/gue_surrogate_reanalyze.py --n 20

The --n flag controls which cycle size's Riemann baseline is used for
comparison (the baseline is pulled from results/u2_scaling_results.json).

This script does not run any optimization. It only reads existing
JSONL records and computes summary statistics. It is safe to run while
the main experiment is in progress.

Output
------
results/gue_surrogate_control_summary.json  (overwritten each call)
Console:                                    full verdict + statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gue_surrogate_control import (
    N_VERTICES_DEFAULT,
    N_ZEROS,
    RESULTS_JSONL,
    SUMMARY_JSON,
    _aggregate_and_verdict,
    _load_all_records,
    _load_riemann_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="On-demand reanalysis of GUE surrogate control results",
    )
    parser.add_argument(
        "--n", type=int, default=N_VERTICES_DEFAULT,
        help=(f"Cycle size for the Riemann baseline comparison "
              f"(default: {N_VERTICES_DEFAULT})"),
    )
    parser.add_argument(
        "--n-restarts", type=int, default=5,
        help="Restarts-per-surrogate that the run was configured with "
             "(used only to populate the config dict; default: 5)",
    )
    parser.add_argument(
        "--n-evals", type=int, default=75_000,
        help="Max evaluations per restart (config field only; default: 75000)",
    )
    args = parser.parse_args()

    if not RESULTS_JSONL.exists():
        print(f"ERROR: {RESULTS_JSONL} does not exist.")
        print("Run gue_surrogate_control.py first to produce surrogate records.")
        sys.exit(1)

    records = _load_all_records()
    if not records:
        print(f"ERROR: {RESULTS_JSONL} contains no valid records.")
        sys.exit(1)

    baseline = _load_riemann_baseline(args.n)
    if baseline is None:
        if args.n == 7:
            baseline = {
                "best_score": 0.897,
                "mad_mean_spacings": 0.145,
                "source": "manuscript_approx",
            }
        elif args.n == 20:
            baseline = {
                "best_score": 0.9258,
                "mad_mean_spacings": 0.097,
                "source": "manuscript_approx",
            }
        else:
            print(f"ERROR: no Riemann baseline available for C_{args.n}.")
            print(f"Either ensure results/u2_scaling_results.json contains "
                  f"an entry for n={args.n}, or use --n with a value that "
                  f"is present in the scaling results.")
            sys.exit(1)

    config = {
        "n_vertices": args.n,
        "n_targets": N_ZEROS,
        "n_surrogates": len(records),
        "n_restarts": args.n_restarts,
        "n_evals": args.n_evals,
        "source": "reanalysis",
    }

    summary = _aggregate_and_verdict(records, baseline, config)

    print("=" * 70)
    print("GUE SURROGATE CONTROL — REANALYSIS")
    print("=" * 70)
    print(f"Records analyzed: {len(records)}")
    print(f"Riemann baseline (C_{args.n}, "
          f"source={baseline.get('source', '?')}):")
    print(f"  best_score:     {baseline['best_score']}")
    print(f"  MAD (mean sp):  {baseline['mad_mean_spacings']}")
    print("")

    if "surrogate_distribution" in summary:
        d = summary["surrogate_distribution"]
        print(f"Surrogate MAD distribution (N = {d['n']}):")
        print(f"  mean:                   {d['best_mad_mean']:.4f}")
        print(f"  std:                    {d['best_mad_std']:.4f}")
        print(f"  median:                 {d['best_mad_median']:.4f}")
        print(f"  min:                    {d['best_mad_min']:.4f}")
        print(f"  max:                    {d['best_mad_max']:.4f}")
        print(f"  surrogates better:      "
              f"{d['n_surrogates_better']}/{d['n_surrogates_total']} "
              f"({d['quantile_pct']:.1f}%)")
        print(f"  z-score (Riemann):      {d['riemann_vs_surrogate_z']:+.2f}")
        print(f"  Cohen's d:              {d['cohens_d']:+.2f}")
        print("")

    print(f"VERDICT: {summary['verdict']}")
    print("")
    msg = summary["verdict_message"]
    print(msg)
    print("")

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
