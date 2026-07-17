#!/usr/bin/env python3
"""Reflection-magnitude ablation (companion to gue_surrogate_control.py
and riemann_seed_distribution.py).

Tests the mechanistic claim in Discussion III.A and Remark 2 of the
manuscript: that the ceiling-breaking mechanism in the transition from
TRS-broken Neumann (S ~ 0.72) to full U(2) (S ~ 0.90) is specifically
the reflection magnitude |cos theta_v| becoming nonzero, not the extra
three phase parameters (alpha, beta, gamma) of U(2).

Procedure
---------
For each fixed theta_fixed on a grid theta_grid:

  1. Fix theta_v = theta_fixed at every vertex of C_n.
  2. Optimize freely over edge lengths and per-vertex (alpha, beta, gamma).
     Parameter count is 4n = 28 for C_7 (n edges + 3n phases), one less
     than a full U(2) restart's 5n = 35.
  3. Run the same CMA-ES + Nelder-Mead pipeline used by
     gue_surrogate_control.py and riemann_seed_distribution.py:
     5 restarts, 75,000 CMA-ES evals per restart, Nelder-Mead polish.
  4. Record best score, best MAD, per-restart records.

Result: a curve of best-of-restart score S vs |cos theta_fixed|.

Interpretation
--------------
- theta_fixed = pi/2: |cos theta| = 0. Reflectionless. Should reproduce
  the TRS-broken Neumann ceiling (~0.72 at N=100 Riemann zeros).
- theta_fixed = 0:    |cos theta| = 1. Fully reflecting; off-diagonals
  of U_v vanish and edges decouple. Degenerate.
- Intermediate theta: partial reflection.

If best-S rises sharply as |cos theta| moves off 0 and plateaus early:
the mechanism is reflection, not phase freedom. Manuscript claim supported.

If best-S rises monotonically all the way as |cos theta| -> 1: reflection
helps at every magnitude with no threshold. Weaker version of the claim.

If best-S is roughly flat in |cos theta| and the improvement comes from
alpha, beta, gamma freedom at any fixed theta: the manuscript's
mechanistic claim is wrong and needs walking back.

Target: first 100 nontrivial Riemann zeros.

Output
------
results/reflection_ablation.jsonl        per-theta best-of-restart record
results/reflection_ablation_summary.json aggregate S(theta) curve
results/reflection_ablation.log          progress log

Resume-safe: theta values already recorded in the JSONL are skipped.
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

import cma
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riemann_qg.core.quantum_graph import QuantumGraph
from riemann_qg.core.scoring import SpectralScorer

from experiments.gue_surrogate_control import (
    N_VERTICES_DEFAULT,
    N_ZEROS,
    N_RESTARTS_DEFAULT,
    N_EVALS_DEFAULT,
    _unitary_2x2,
    _compute_residuals,
)


RESULTS_JSONL = Path("results/reflection_ablation.jsonl")
SUMMARY_JSON = Path("results/reflection_ablation_summary.json")
LOG_PATH = Path("results/reflection_ablation.log")


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


# ---- Fixed-theta pipeline ----

def _build_cycle_unitary_fixed_theta(
    n: int,
    edge_lengths: np.ndarray,
    scat_params_3n: np.ndarray,
    theta_fixed: float,
) -> QuantumGraph:
    """Build C_n under U(2) scattering with theta_v = theta_fixed at every
    vertex. Free scattering parameters are (alpha_v, beta_v, gamma_v) per
    vertex, so scat_params_3n has length 3n.
    """
    adj = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1
    scattering = []
    for v in range(n):
        alpha, beta, gamma = scat_params_3n[v * 3:(v + 1) * 3]
        full_params = np.array([alpha, beta, gamma, theta_fixed])
        scattering.append(_unitary_2x2(full_params))
    return QuantumGraph(adj, list(edge_lengths), scattering)


def _objective_fixed_theta(
    params: np.ndarray,
    n: int,
    scorer: SpectralScorer,
    target: np.ndarray,
    n_targets: int,
    theta_fixed: float,
) -> float:
    """Objective for the fixed-theta ablation. `params` layout:
    [edge_lengths (n), alpha/beta/gamma per vertex (3n)].
    """
    edge_lengths = params[:n]
    scat_params_3n = params[n:]
    if np.any(edge_lengths < 0.01):
        return 1.0
    try:
        qg = _build_cycle_unitary_fixed_theta(
            n, edge_lengths, scat_params_3n, theta_fixed,
        )
        total_length = float(np.sum(edge_lengths))
        k_max = max((n_targets + 5) * np.pi / total_length, 10.0)
        spectrum = qg.compute_spectrum(k_max, n_points=3000)
        result = scorer.score_spectrum(
            spectrum, target, n_targets,
            edge_lengths=qg.edge_lengths,
        )
        return -result.total_score
    except Exception:
        return 1.0


def _optimize_at_fixed_theta(
    theta_fixed: float,
    target: np.ndarray,
    n: int,
    n_targets: int,
    n_restarts: int,
    n_evals: int,
    seed_base: int,
) -> dict:
    """Run CMA-ES + Nelder-Mead pipeline at a single fixed theta_fixed."""
    scorer = SpectralScorer()
    n_edge_params = n
    n_scat_params = 3 * n

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    restart_records = []
    best_score = -1.0
    best_params = None

    for restart in range(n_restarts):
        rng = np.random.default_rng(seed_base * 10000 + restart)

        if restart % 4 == 0:
            edge_init = np.array([
                np.log(primes[i % len(primes)]) for i in range(n)
            ])
        elif restart % 4 == 1:
            edge_init = rng.uniform(0.5, 5.0, n)
        elif restart % 4 == 2:
            edge_init = rng.uniform(1.0, 8.0, n)
        else:
            L = rng.uniform(10.0, 30.0)
            edge_init = np.full(n, L / n)

        scat_init = rng.uniform(0, 2 * np.pi, n_scat_params)
        x0 = np.concatenate([edge_init, scat_init])

        lower = [0.05] * n + [-10.0] * n_scat_params
        upper = [15.0] * n + [10.0] * n_scat_params

        opts = cma.CMAOptions()
        opts.set("maxfevals", n_evals)
        opts.set("tolfun", 1e-10)
        opts.set("tolx", 1e-10)
        opts.set("verbose", -9)
        opts.set("bounds", [lower, upper])
        opts.set("CMA_stds", [2.0] * n + [1.0] * n_scat_params)

        t_start = time.time()
        es = cma.CMAEvolutionStrategy(x0, 1.5, opts)
        while not es.stop():
            sols = es.ask()
            fits = [
                _objective_fixed_theta(
                    x, n, scorer, target, n_targets, theta_fixed,
                )
                for x in sols
            ]
            es.tell(sols, fits)

        cma_score = -es.result.fbest
        cma_params = es.result.xbest

        try:
            polish = minimize(
                _objective_fixed_theta, cma_params,
                args=(n, scorer, target, n_targets, theta_fixed),
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12},
            )
            if -polish.fun > cma_score:
                final_score = -polish.fun
                final_params = polish.x
            else:
                final_score = cma_score
                final_params = cma_params
        except Exception:
            final_score = cma_score
            final_params = cma_params

        elapsed = time.time() - t_start

        edge_lengths = final_params[:n]
        scat_params_3n = final_params[n:]
        qg = _build_cycle_unitary_fixed_theta(
            n, edge_lengths, scat_params_3n, theta_fixed,
        )
        res = _compute_residuals(qg, target, n_targets)

        restart_records.append({
            "restart": restart,
            "cma_score": cma_score,
            "score": final_score,
            "mad": res.get("mad", 99),
            "mad_mean_spacings": res.get("mad_mean_spacings", 99),
            "max_residual": res.get("max_residual", 99),
            "n_matched": res.get("n_matched", 0),
            "total_length": float(np.sum(edge_lengths)),
            "n_cma_evals": int(es.result.evaluations),
            "elapsed_s": elapsed,
            "edge_lengths": [float(x) for x in edge_lengths],
            "scat_params_3n": [float(x) for x in scat_params_3n],
        })

        if final_score > best_score:
            best_score = final_score
            best_params = final_params

        time.sleep(2)  # thermal cooldown

    best_edge_lengths = best_params[:n]
    best_scat_3n = best_params[n:]
    best_qg = _build_cycle_unitary_fixed_theta(
        n, best_edge_lengths, best_scat_3n, theta_fixed,
    )
    best_res = _compute_residuals(best_qg, target, n_targets)

    scores = [r["score"] for r in restart_records]
    mads = [r["mad_mean_spacings"] for r in restart_records]

    return {
        "theta_fixed": float(theta_fixed),
        "cos_theta_abs": float(abs(np.cos(theta_fixed))),
        "best_score": best_score,
        "best_mad_mean_spacings": best_res.get("mad_mean_spacings", 99),
        "best_n_matched": best_res.get("n_matched", 0),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "mean_mad": float(np.mean(mads)),
        "std_mad": float(np.std(mads, ddof=1)) if len(mads) > 1 else 0.0,
        "restart_records": restart_records,
    }


# ---- Persistence and aggregation ----

def _load_completed_thetas() -> set[int]:
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
                completed.add(int(r["theta_index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
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


def _aggregate(records: list[dict], config: dict) -> dict:
    valid = [
        r for r in records
        if r.get("best_mad_mean_spacings") is not None
        and r["best_mad_mean_spacings"] < 90
    ]
    if not valid:
        return {
            "config": config,
            "n_thetas_completed": len(records),
            "n_thetas_valid": 0,
            "verdict": "INSUFFICIENT_DATA",
        }

    valid.sort(key=lambda r: r["theta_fixed"])
    curve = [
        {
            "theta_index": r.get("theta_index"),
            "theta_fixed": r["theta_fixed"],
            "cos_theta_abs": r["cos_theta_abs"],
            "best_score": r["best_score"],
            "best_mad_mean_spacings": r["best_mad_mean_spacings"],
            "best_n_matched": r["best_n_matched"],
            "mean_score": r.get("mean_score"),
            "std_score": r.get("std_score"),
            "mean_mad": r.get("mean_mad"),
            "std_mad": r.get("std_mad"),
        }
        for r in valid
    ]

    scores_arr = np.asarray([c["best_score"] for c in curve])
    cos_arr = np.asarray([c["cos_theta_abs"] for c in curve])

    idx_reflectionless = int(np.argmin(cos_arr))
    idx_max_score = int(np.argmax(scores_arr))

    verdict = "DESCRIPTIVE"
    verdict_msg = ""
    if len(curve) >= 5:
        s_at_zero = scores_arr[idx_reflectionless]
        s_max = scores_arr[idx_max_score]
        gain = float(s_max - s_at_zero)
        # Threshold gain interpreted qualitatively; not a formal test.
        if gain > 0.15:
            verdict = "REFLECTION_LIFTS_CEILING"
            verdict_msg = (
                f"Best score rises from {s_at_zero:.4f} at "
                f"|cos theta| = {cos_arr[idx_reflectionless]:.3f} "
                f"(reflectionless-most sampled point) to "
                f"{s_max:.4f} at |cos theta| = "
                f"{cos_arr[idx_max_score]:.3f}. "
                f"Delta S = {gain:.4f}, consistent with the manuscript's "
                "reflection-magnitude mechanism claim."
            )
        elif gain > 0.05:
            verdict = "MODEST_REFLECTION_EFFECT"
            verdict_msg = (
                f"Delta S = {gain:.4f}. Reflection helps but the effect "
                "is modest; the mechanism claim should be softened to "
                "'reflection contributes' rather than 'reflection is the "
                "mechanism.'"
            )
        else:
            verdict = "REFLECTION_IRRELEVANT"
            verdict_msg = (
                f"Delta S = {gain:.4f}. The manuscript's "
                "reflection-magnitude claim is not supported by this "
                "ablation; the improvement in going from TRS-broken "
                "Neumann to full U(2) must come from the (alpha, beta, "
                "gamma) phase freedom at fixed theta, not from theta itself."
            )

    return {
        "config": config,
        "n_thetas_completed": len(records),
        "n_thetas_valid": len(curve),
        "curve": curve,
        "verdict": verdict,
        "verdict_message": verdict_msg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-thetas", type=int, default=12,
        help="Number of theta grid points (default: 12).",
    )
    parser.add_argument(
        "--theta-min", type=float, default=0.05,
        help=(
            "Minimum theta (radians). Not exactly 0 to avoid the fully"
            "reflecting degenerate limit (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--theta-max", type=float, default=np.pi / 2 - 0.05,
        help=(
            "Maximum theta (radians). Not exactly pi/2 to avoid a hard"
            "coincidence with TRS-broken Neumann (default: pi/2 - 0.05)."
        ),
    )
    parser.add_argument(
        "--n-vertices", type=int, default=N_VERTICES_DEFAULT,
        help="Cycle size (default: 7).",
    )
    parser.add_argument(
        "--n-zeros", type=int, default=N_ZEROS,
        help="Number of Riemann zeros to fit (default: 100).",
    )
    parser.add_argument(
        "--n-restarts", type=int, default=N_RESTARTS_DEFAULT,
        help="CMA-ES restarts per theta (default: 5).",
    )
    parser.add_argument(
        "--n-evals", type=int, default=N_EVALS_DEFAULT,
        help="CMA-ES evaluations per restart (default: 75,000).",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=200_000,
        help=(
            "Master seed offset. Kept distinct from surrogate (0-49) and "
            "Riemann-seed (100000-100029) seed spaces."
        ),
    )
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    theta_grid = np.linspace(args.theta_min, args.theta_max, args.n_thetas)

    config = {
        "n_vertices": args.n_vertices,
        "n_zeros": args.n_zeros,
        "n_thetas": args.n_thetas,
        "theta_min": args.theta_min,
        "theta_max": args.theta_max,
        "theta_grid": theta_grid.tolist(),
        "n_restarts": args.n_restarts,
        "n_evals": args.n_evals,
        "seed_offset": args.seed_offset,
    }

    log("=== Reflection-magnitude ablation ===")
    log(f"Config: {json.dumps(config)}")
    log(f"Timestamp start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    scorer = SpectralScorer()
    riemann_zeros = scorer._rz.get_zeros(args.n_zeros)

    completed = _load_completed_thetas()
    log(
        f"Loaded {len(completed)} completed theta values from "
        f"{RESULTS_JSONL}."
    )

    for i, theta_fixed in enumerate(theta_grid):
        if i in completed:
            log(f"[theta {i} = {theta_fixed:.4f}] already done; skipping.")
            continue

        cos_abs = float(abs(np.cos(theta_fixed)))
        log(
            f"[theta {i} = {theta_fixed:.4f}, |cos theta| = "
            f"{cos_abs:.3f}] starting "
            f"({args.n_restarts} restarts x {args.n_evals} evals)."
        )
        t0 = time.time()

        seed_base = args.seed_offset + i
        result = _optimize_at_fixed_theta(
            theta_fixed=float(theta_fixed),
            target=riemann_zeros,
            n=args.n_vertices,
            n_targets=args.n_zeros,
            n_restarts=args.n_restarts,
            n_evals=args.n_evals,
            seed_base=seed_base,
        )

        elapsed = time.time() - t0

        record = {
            "theta_index": i,
            "seed_base": seed_base,
            **result,
            "elapsed_s": elapsed,
        }

        with open(RESULTS_JSONL, "a") as f:
            f.write(json.dumps(record) + "\n")

        log(
            f"[theta {i}] done. score={result['best_score']:.4f} "
            f"MAD={result['best_mad_mean_spacings']:.4f} "
            f"matched={result['best_n_matched']}/{args.n_zeros} "
            f"elapsed={elapsed:.1f}s"
        )

        all_records = _load_all_records()
        summary = _aggregate(all_records, config)
        with open(SUMMARY_JSON, "w") as f:
            json.dump(summary, f, indent=2)

    log("=== All requested theta values completed ===")
    log(f"Timestamp end: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_records = _load_all_records()
    summary = _aggregate(all_records, config)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Verdict: {summary.get('verdict')}.")
    if summary.get("verdict_message"):
        log(summary["verdict_message"])


if __name__ == "__main__":
    main()
