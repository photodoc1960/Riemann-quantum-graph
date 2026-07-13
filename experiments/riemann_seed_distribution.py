#!/usr/bin/env python3
"""Riemann-seed distribution experiment (companion to gue_surrogate_control.py).

Runs the same C_7 / U(2) CMA-ES + Nelder-Mead pipeline against the actual
Riemann zeros K_R times with independent seeds. The resulting distribution
of best-of-restart MADs is the Riemann-side companion to the K = 50 GUE
surrogate distribution reported in Section 3.7 of the manuscript.

Purpose (reviewer feedback P8, medium-lift item c1):

- Replace point-vs-distribution comparison (single Riemann MAD vs surrogate
  distribution) with distribution-vs-distribution.
- Deliver an honest error bar on the "MAD = 0.147" headline number by
  reporting median / IQR / min / max across K_R independent seeds.
- Enable a Mann-Whitney U test between Riemann and surrogate MAD samples.

Pipeline (bit-identical to gue_surrogate_control.py):
- Cycle C_7, U(2) vertex scattering.
- CMA-ES: sigma_0 = 1.5, population lambda = 4 + floor(3 ln n_param),
  budget 75,000 evals per restart, bounds [0.05, 15] on edges and
  [-10, 10] on scattering params, 5 restarts per seed.
- Nelder-Mead polish (xatol / fatol = 1e-10, maxiter 5000).
- Scoring: 0.70 pos + 0.20 GUE + 0.10 corr.
- Target: first 100 nontrivial Riemann zeros.

Output
------
results/riemann_seed_distribution.jsonl       per-seed best-of-restart record
results/riemann_seed_distribution_summary.json aggregate distribution + comparison
results/riemann_seed_distribution.log         progress log

The script is resume-safe: seeds already recorded in the JSONL are skipped.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "sequential"

import multiprocessing
multiprocessing.set_start_method("forkserver", force=True)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riemann_qg.core.scoring import SpectralScorer

from experiments.gue_surrogate_control import (
    N_VERTICES_DEFAULT,
    N_ZEROS,
    N_RESTARTS_DEFAULT,
    N_EVALS_DEFAULT,
    _optimize_against_target,
)


RESULTS_JSONL = Path("results/riemann_seed_distribution.jsonl")
SUMMARY_JSON = Path("results/riemann_seed_distribution_summary.json")
LOG_PATH = Path("results/riemann_seed_distribution.log")


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def _load_completed_seeds() -> set[int]:
    completed: set[int] = set()
    if not RESULTS_JSONL.exists():
        return completed
    with open(RESULTS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                completed.add(r["seed_index"])
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def _load_all_records() -> list[dict]:
    records: list[dict] = []
    if not RESULTS_JSONL.exists():
        return records
    with open(RESULTS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _load_surrogate_mads() -> list[float] | None:
    """Load per-surrogate best MADs from the sibling surrogate control run,
    if present. Returned as a plain Python list. Used for the two-sample
    Mann-Whitney U test and quantile comparison in the summary.
    """
    surrogate_jsonl = Path("results/gue_surrogate_control.jsonl")
    if not surrogate_jsonl.exists():
        return None
    mads: list[float] = []
    with open(surrogate_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = r.get("best_mad_mean_spacings")
            if v is None:
                continue
            v = float(v)
            if v < 90:
                mads.append(v)
    return mads if mads else None


def _mann_whitney_u(x: list[float], y: list[float]) -> dict:
    """Two-sided Mann-Whitney U test via SciPy. Returned as a dict.

    Reports both the raw U statistic and its normal-approximation z-score.
    """
    from scipy import stats as sp_stats
    if len(x) < 3 or len(y) < 3:
        return {
            "n_x": len(x),
            "n_y": len(y),
            "u_statistic": None,
            "p_value": None,
            "note": "Insufficient data for Mann-Whitney U (need n>=3 each).",
        }
    result = sp_stats.mannwhitneyu(
        x, y, alternative="two-sided", method="asymptotic",
    )
    return {
        "n_x": len(x),
        "n_y": len(y),
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "alternative": "two-sided",
    }


def _aggregate(records: list[dict], config: dict) -> dict:
    mads = [
        r["best_mad_mean_spacings"] for r in records
        if r.get("best_mad_mean_spacings") is not None
        and r["best_mad_mean_spacings"] < 90
    ]
    scores = [
        r["best_score"] for r in records
        if r.get("best_mad_mean_spacings") is not None
        and r["best_mad_mean_spacings"] < 90
    ]
    n_matched = [
        r.get("best_n_matched", 0) for r in records
        if r.get("best_mad_mean_spacings") is not None
        and r["best_mad_mean_spacings"] < 90
    ]

    if len(mads) < 3:
        return {
            "config": config,
            "n_seeds_completed": len(records),
            "n_seeds_valid": len(mads),
            "verdict": "INSUFFICIENT_DATA",
            "message": (
                f"Only {len(mads)} valid Riemann seeds; need at least 3 "
                "for a meaningful distribution."
            ),
        }

    mads_arr = np.asarray(mads, dtype=np.float64)
    quantiles = np.quantile(mads_arr, [0.05, 0.25, 0.50, 0.75, 0.95])

    distribution = {
        "n": int(len(mads)),
        "mad_mean": float(np.mean(mads_arr)),
        "mad_std": float(np.std(mads_arr, ddof=1)),
        "mad_min": float(np.min(mads_arr)),
        "mad_max": float(np.max(mads_arr)),
        "mad_median": float(np.median(mads_arr)),
        "mad_q05": float(quantiles[0]),
        "mad_q25": float(quantiles[1]),
        "mad_q75": float(quantiles[3]),
        "mad_q95": float(quantiles[4]),
        "mad_iqr": float(quantiles[3] - quantiles[1]),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores, ddof=1)),
        "score_max": float(np.max(scores)),
        "n_matched_mean": float(np.mean(n_matched)),
        "n_matched_std": float(np.std(n_matched, ddof=1)),
        "n_matched_min": int(np.min(n_matched)),
        "n_matched_max": int(np.max(n_matched)),
    }

    # ---- Distribution-vs-distribution comparison to surrogates ----
    surrogate_mads = _load_surrogate_mads()
    comparison: dict = {}
    if surrogate_mads is not None:
        surrogate_arr = np.asarray(surrogate_mads, dtype=np.float64)
        # Fraction of surrogates strictly worse (larger MAD) than the
        # median Riemann seed: a distribution-level analogue of the
        # original "0/50 quantile" statistic.
        n_worse = int(np.sum(surrogate_arr > distribution["mad_median"]))
        n_total = int(surrogate_arr.size)
        # Overlap: fraction of pairs (r in Riemann, s in surrogate) with
        # r >= s. Under the null hypothesis of identical distributions
        # this is approximately 0.5; values near 0 indicate the Riemann
        # distribution stochastically dominates below the surrogate one.
        overlap = float(
            np.mean(mads_arr[:, None] >= surrogate_arr[None, :])
        )
        mw = _mann_whitney_u(list(mads_arr), list(surrogate_arr))
        comparison = {
            "surrogate_source": "results/gue_surrogate_control.jsonl",
            "surrogate_n": n_total,
            "surrogate_mad_mean": float(np.mean(surrogate_arr)),
            "surrogate_mad_std": float(np.std(surrogate_arr, ddof=1)),
            "surrogate_mad_median": float(np.median(surrogate_arr)),
            "surrogate_mad_min": float(np.min(surrogate_arr)),
            "riemann_median_vs_surrogate_rank": (
                f"{n_total - n_worse}/{n_total}"
            ),
            "prob_riemann_ge_surrogate": overlap,
            "mann_whitney_u_test": mw,
        }

    verdict = "DESCRIPTIVE"
    if comparison:
        p_val = comparison["mann_whitney_u_test"].get("p_value")
        if p_val is not None:
            if p_val < 0.01 and comparison["prob_riemann_ge_surrogate"] < 0.05:
                verdict = "RIEMANN_DISTRIBUTION_DOMINATES_SURROGATE"
            elif p_val < 0.05:
                verdict = "RIEMANN_DISTRIBUTION_SEPARATED"
            else:
                verdict = "NO_DISTRIBUTIONAL_SEPARATION"

    return {
        "config": config,
        "n_seeds_completed": len(records),
        "n_seeds_valid": len(mads),
        "distribution": distribution,
        "comparison_to_surrogate": comparison,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-seeds", type=int, default=30,
        help="Number of independent Riemann seeds (default: 30).",
    )
    parser.add_argument(
        "--n-vertices", type=int, default=N_VERTICES_DEFAULT,
        help="Cycle graph size (default: 7).",
    )
    parser.add_argument(
        "--n-zeros", type=int, default=N_ZEROS,
        help="Number of Riemann zeros to fit (default: 100).",
    )
    parser.add_argument(
        "--n-restarts", type=int, default=N_RESTARTS_DEFAULT,
        help="CMA-ES restarts per seed (default: 5).",
    )
    parser.add_argument(
        "--n-evals", type=int, default=N_EVALS_DEFAULT,
        help="CMA-ES evaluations per restart (default: 75,000).",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=100_000,
        help=(
            "Master seed offset. Riemann seed i uses "
            "surrogate_seed=(seed_offset + i) inside "
            "_optimize_against_target, which drives the per-restart RNG. "
            "Kept distinct from the surrogate seed space (0..K-1)."
        ),
    )
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    config = {
        "n_vertices": args.n_vertices,
        "n_zeros": args.n_zeros,
        "n_seeds": args.n_seeds,
        "n_restarts": args.n_restarts,
        "n_evals": args.n_evals,
        "seed_offset": args.seed_offset,
    }

    log(f"=== Riemann-seed distribution ===")
    log(f"Config: {json.dumps(config)}")
    log(f"Timestamp start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    scorer = SpectralScorer()
    riemann_zeros = scorer._rz.get_zeros(args.n_zeros)

    completed = _load_completed_seeds()
    log(
        f"Loaded {len(completed)} completed seeds from "
        f"{RESULTS_JSONL}."
    )

    for i in range(args.n_seeds):
        if i in completed:
            log(f"[seed {i}] already completed; skipping.")
            continue

        log(f"[seed {i}] starting optimization "
            f"({args.n_restarts} restarts x {args.n_evals} evals).")
        t0 = time.time()

        seed_key = args.seed_offset + i
        result = _optimize_against_target(
            target=riemann_zeros,
            n=args.n_vertices,
            n_targets=args.n_zeros,
            n_restarts=args.n_restarts,
            n_evals=args.n_evals,
            surrogate_seed=seed_key,
        )

        elapsed = time.time() - t0

        record = {
            "seed_index": i,
            "seed_key": seed_key,
            "best_score": result["best_score"],
            "best_mad_mean_spacings": result["best_mad_mean_spacings"],
            "best_n_matched": result["best_n_matched"],
            "mean_score": result["mean_score"],
            "std_score": result["std_score"],
            "mean_mad": result["mean_mad"],
            "std_mad": result["std_mad"],
            "restart_records": result["restart_records"],
            "elapsed_s": elapsed,
        }

        with open(RESULTS_JSONL, "a") as f:
            f.write(json.dumps(record) + "\n")

        log(
            f"[seed {i}] done. score={result['best_score']:.4f} "
            f"MAD={result['best_mad_mean_spacings']:.4f} "
            f"matched={result['best_n_matched']}/{args.n_zeros} "
            f"elapsed={elapsed:.1f}s"
        )

        # After each seed: refresh the summary aggregate on disk.
        all_records = _load_all_records()
        summary = _aggregate(all_records, config)
        with open(SUMMARY_JSON, "w") as f:
            json.dump(summary, f, indent=2)

    log("=== All requested seeds completed ===")
    log(f"Timestamp end: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_records = _load_all_records()
    summary = _aggregate(all_records, config)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    log(
        f"Aggregate written to {SUMMARY_JSON}. "
        f"Verdict: {summary.get('verdict')}."
    )
    dist = summary.get("distribution", {})
    if dist:
        log(
            f"MAD distribution over {dist['n']} seeds: "
            f"median {dist['mad_median']:.4f}, "
            f"IQR [{dist['mad_q25']:.4f}, {dist['mad_q75']:.4f}], "
            f"min {dist['mad_min']:.4f}, max {dist['mad_max']:.4f}."
        )
    cmp = summary.get("comparison_to_surrogate", {})
    if cmp:
        mw = cmp.get("mann_whitney_u_test", {})
        log(
            f"Mann-Whitney U (Riemann vs surrogate): "
            f"U={mw.get('u_statistic')}, p={mw.get('p_value')}."
        )
        log(
            f"P(Riemann-MAD >= surrogate-MAD) = "
            f"{cmp.get('prob_riemann_ge_surrogate'):.4f}."
        )


if __name__ == "__main__":
    main()
