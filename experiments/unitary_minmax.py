#!/usr/bin/env python3
"""Targeted U(2) re-optimization minimizing MAX residual instead of mean.

The best graph (0.897) has MAD=0.327 but specific outlier zeros with
residuals >1.0 pulling down the score. This script uses a modified
objective that penalizes the worst-case residual, pushing the optimizer
to distribute error uniformly across all zeros rather than sacrificing
a few to improve the average.

Results logged to results/unitary_minmax.jsonl
Best graph saved to results/best_unitary_graph.json (overwrites if better)
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "sequential"

import multiprocessing
multiprocessing.set_start_method("forkserver", force=True)

import json
import sys
import time
from pathlib import Path

import cma
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riemann_qg.core.quantum_graph import QuantumGraph
from riemann_qg.core.scoring import SpectralScorer, _weyl_rescale


RESULTS_FILE = Path("results/unitary_minmax.jsonl")
BEST_GRAPH_FILE = Path("results/best_unitary_graph.json")


def _unitary_2x2(params: np.ndarray) -> np.ndarray:
    """Build a general U(2) matrix from 4 real parameters."""
    alpha, beta, gamma, theta = params
    c, s = np.cos(theta), np.sin(theta)
    phase = np.exp(1j * alpha)
    return phase * np.array([
        [np.exp(1j * beta) * c, -np.exp(-1j * gamma) * s],
        [np.exp(1j * gamma) * s, np.exp(-1j * beta) * c],
    ], dtype=np.complex128)


def _build_cycle_unitary(
    n: int,
    edge_lengths: np.ndarray,
    scat_params: np.ndarray,
) -> QuantumGraph:
    """Build C_n with full U(2) scattering at each vertex."""
    adj = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1

    scattering = []
    for v in range(n):
        u = _unitary_2x2(scat_params[v * 4:(v + 1) * 4])
        scattering.append(u)

    return QuantumGraph(adj, list(edge_lengths), scattering)


def _compute_residuals_raw(
    qg: QuantumGraph,
    zeros: np.ndarray,
    n_zeros: int,
) -> np.ndarray | None:
    """Compute raw per-zero residuals (not wrapped in dict)."""
    total_length = sum(qg.edge_lengths)
    k_max = max((n_zeros + 5) * np.pi / total_length, 10.0)
    spectrum = qg.compute_spectrum(k_max, n_points=3000)

    if len(spectrum) < 3:
        return None

    scaled, _ = _weyl_rescale(np.sort(spectrum), total_length, zeros)

    zero_spacings = np.diff(zeros)
    margin = float(np.mean(zero_spacings)) if len(zero_spacings) > 0 else 1.0
    k_low = zeros[0] - margin
    k_high = zeros[-1] + margin
    windowed = scaled[(scaled >= k_low) & (scaled <= k_high)]

    n_match = min(len(windowed), len(zeros))
    if n_match < 3:
        return None

    return np.abs(windowed[:n_match] - zeros[:n_match])


def _objective_minmax(
    params: np.ndarray,
    n: int,
    scorer: SpectralScorer,
    zeros: np.ndarray,
    n_zeros: int,
    alpha: float,
) -> float:
    """Hybrid objective: (1-alpha)*mean_residual + alpha*max_residual.

    alpha=0: pure MAD minimization (original behavior)
    alpha=1: pure minimax
    alpha=0.5: balanced (recommended)

    Returns positive value for minimization. Lower = better.
    """
    edge_lengths = params[:n]
    scat_params = params[n:]

    if np.any(edge_lengths < 0.01):
        return 100.0

    try:
        qg = _build_cycle_unitary(n, edge_lengths, scat_params)
        residuals = _compute_residuals_raw(qg, zeros, n_zeros)

        if residuals is None or len(residuals) < 3:
            return 100.0

        total_length = sum(qg.edge_lengths)
        zero_spacings = np.diff(zeros[:n_zeros])
        mean_spacing = float(np.mean(zero_spacings))

        # Normalize residuals by mean zero spacing
        norm_residuals = residuals / mean_spacing

        mad = float(np.mean(norm_residuals))
        max_res = float(np.max(norm_residuals))

        # Coverage penalty
        coverage = min(1.0, len(residuals) / n_zeros)

        # Hybrid: blend MAD and max-residual
        objective = (1 - alpha) * mad + alpha * max_res

        # We want to MINIMIZE this, so lower = better.
        # Also factor in coverage (want coverage=1.0)
        return objective / max(coverage, 0.01)

    except Exception:
        return 100.0


def _also_compute_score(
    params: np.ndarray,
    n: int,
    scorer: SpectralScorer,
    zeros: np.ndarray,
    n_zeros: int,
) -> tuple[float, dict]:
    """Compute the standard score and detailed residuals for reporting."""
    edge_lengths = params[:n]
    scat_params = params[n:]

    try:
        qg = _build_cycle_unitary(n, edge_lengths, scat_params)
        total_length = sum(qg.edge_lengths)
        k_max = max((n_zeros + 5) * np.pi / total_length, 10.0)
        spectrum = qg.compute_spectrum(k_max, n_points=3000)
        result = scorer.score_spectrum(
            spectrum, zeros, n_zeros,
            edge_lengths=qg.edge_lengths,
        )

        residuals = _compute_residuals_raw(qg, zeros, n_zeros)
        res_list = list(residuals) if residuals is not None else []

        return result.total_score, {
            "residuals": [float(r) for r in res_list],
            "mad": float(np.mean(res_list)) if res_list else 99.0,
            "max_residual": float(np.max(res_list)) if res_list else 99.0,
            "std_residual": float(np.std(res_list)) if res_list else 99.0,
            "n_matched": len(res_list),
            "n_under_0_1": int(sum(1 for r in res_list if r < 0.1)),
            "n_under_0_5": int(sum(1 for r in res_list if r < 0.5)),
            "n_over_1_0": int(sum(1 for r in res_list if r > 1.0)),
        }
    except Exception:
        return 0.0, {"residuals": [], "mad": 99.0, "max_residual": 99.0}


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def run_experiment(
    n: int = 7,
    n_zeros: int = 100,
    n_restarts: int = 30,
    max_evals: int = 100000,
    alpha: float = 0.5,
) -> None:
    """Run the minimax optimization."""
    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(n_zeros)

    n_edge_params = n
    n_scat_params = 4 * n
    n_params = n_edge_params + n_scat_params

    log("=" * 70)
    log(f"U(2) MINIMAX OPTIMIZATION: C_{n}")
    log("=" * 70)
    log(f"  Params: {n_params} ({n} edges + {n_scat_params} scattering)")
    log(f"  Zeros: {n_zeros}, Restarts: {n_restarts}")
    log(f"  Alpha: {alpha} (0=pure MAD, 1=pure minimax)")
    log(f"  Results: {RESULTS_FILE}")
    log("")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    best_score = 0.0
    best_max_res = 999.0
    best_params = None

    for restart in range(n_restarts):
        rng = np.random.default_rng(restart * 1000 + n * 100 + 99)

        if restart % 4 == 0:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
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
        opts.set("maxfevals", max_evals)
        opts.set("tolfun", 1e-10)
        opts.set("tolx", 1e-10)
        opts.set("verbose", -9)
        opts.set("bounds", [lower, upper])
        opts.set("CMA_stds", [2.0] * n + [1.0] * n_scat_params)

        t0 = time.time()
        es = cma.CMAEvolutionStrategy(x0, 1.5, opts)
        while not es.stop():
            sols = es.ask()
            fits = [
                _objective_minmax(x, n, scorer, zeros, n_zeros, alpha)
                for x in sols
            ]
            es.tell(sols, fits)

        cma_params = es.result.xbest

        # Nelder-Mead polish on the minimax objective
        try:
            polish = minimize(
                _objective_minmax,
                cma_params,
                args=(n, scorer, zeros, n_zeros, alpha),
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12},
            )
            if polish.fun < es.result.fbest:
                final_params = polish.x
            else:
                final_params = cma_params
        except Exception:
            final_params = cma_params

        elapsed = time.time() - t0

        # Compute standard score for comparison
        score, details = _also_compute_score(
            final_params, n, scorer, zeros, n_zeros,
        )

        marker = ""
        if score > best_score:
            best_score = score
            best_max_res = details["max_residual"]
            best_params = final_params
            marker = " <-- BEST SCORE"
        elif details["max_residual"] < best_max_res and score > best_score - 0.01:
            best_max_res = details["max_residual"]
            best_params = final_params
            marker = " <-- BEST MAX_RES"

        log(f"  restart {restart:2d}: score={score:.4f}  "
            f"MAD={details['mad']:.3f}  "
            f"maxRes={details['max_residual']:.3f}  "
            f"<0.5={details['n_under_0_5']}/{details['n_matched']}  "
            f">1.0={details['n_over_1_0']}  "
            f"{elapsed:.0f}s{marker}")

        record = {
            "n_vertices": n,
            "n_zeros": n_zeros,
            "alpha": alpha,
            "restart": restart,
            "score": score,
            "edge_lengths": [float(x) for x in final_params[:n]],
            "scat_params": [float(x) for x in final_params[n:]],
            "total_length": float(np.sum(final_params[:n])),
            "n_cma_evals": int(es.result.evaluations),
            "elapsed_s": elapsed,
            **details,
        }
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        time.sleep(2)

    # Save best graph
    if best_params is not None:
        score, details = _also_compute_score(
            best_params, n, scorer, zeros, n_zeros,
        )
        _save_best_graph(best_params, n, n_zeros, score, details)

    log(f"\nBest score: {best_score:.4f}")
    log(f"Best max residual: {best_max_res:.3f}")


def _save_best_graph(
    params: np.ndarray,
    n: int,
    n_zeros: int,
    score: float,
    details: dict,
) -> None:
    """Save the best graph specification."""
    edge_lengths = params[:n]
    scat_params = params[n:]

    scattering_matrices = []
    for v in range(n):
        p = scat_params[v * 4:(v + 1) * 4]
        u = _unitary_2x2(p)
        # Eigenvalues of the scattering matrix
        eigvals = np.linalg.eigvals(u)
        eigphases = np.angle(eigvals)

        scattering_matrices.append({
            "vertex": v,
            "params_alpha_beta_gamma_theta": [float(x) for x in p],
            "matrix_real": u.real.tolist(),
            "matrix_imag": u.imag.tolist(),
            "determinant_phase": float(np.angle(np.linalg.det(u))),
            "eigenvalue_phases": [float(x) for x in eigphases],
            "trace_real": float(np.trace(u).real),
            "trace_imag": float(np.trace(u).imag),
        })

    spec = {
        "graph_type": f"C_{n} with U(2) scattering (minimax optimized)",
        "n_vertices": n,
        "n_edges": n,
        "score": score,
        "n_zeros": n_zeros,
        "edge_lengths": [float(x) for x in edge_lengths],
        "total_length": float(np.sum(edge_lengths)),
        "scattering_matrices": scattering_matrices,
        **details,
    }

    with open(BEST_GRAPH_FILE, "w") as f:
        json.dump(spec, f, indent=2)

    log(f"Best graph saved to {BEST_GRAPH_FILE}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="U(2) minimax optimization — minimize worst-case residual",
    )
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--n-zeros", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--max-evals", type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="0=pure MAD, 1=pure minimax, 0.5=balanced (default)")
    args = parser.parse_args()

    run_experiment(
        n=args.n, n_zeros=args.n_zeros,
        n_restarts=args.restarts, max_evals=args.max_evals,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
