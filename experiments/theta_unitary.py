#!/usr/bin/env python3
"""U(d) scattering on theta graphs.

Tests whether full unitary vertex conditions at degree-3 hubs (and degree-2
internal vertices) break the U(2)-on-cycle ceiling of ~0.905.

Theta graph: two hub vertices of degree 3 connected by three paths of k
edges each. Total vertices: 2 + 3(k-1). Total edges: 3k.

Parameters per restart:
  - 3k edge lengths
  - 2 × 9 = 18 U(3) hub scattering params
  - 3(k-1) × 4 = 12(k-1) U(2) internal scattering params
  - Total: 15k + 6

Results: results/theta_unitary_results.json, results/theta_unitary_plot.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riemann_qg.core.quantum_graph import QuantumGraph
from riemann_qg.core.scoring import SpectralScorer, _weyl_rescale


# ---- Configuration ----
K_VALUES = [2, 3, 4, 5, 6]      # edges per path
N_ZEROS = 100
N_RESTARTS = 10
N_EVALS = 75_000
RESULTS_FILE = Path("results/theta_unitary_results.json")
PLOT_FILE = Path("results/theta_unitary_plot.png")


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def _unitary_2x2(params: np.ndarray) -> np.ndarray:
    """Standard U(2) parameterization from 4 real params."""
    alpha, beta, gamma, theta = params
    c, s = np.cos(theta), np.sin(theta)
    phase = np.exp(1j * alpha)
    return phase * np.array([
        [np.exp(1j * beta) * c, -np.exp(-1j * gamma) * s],
        [np.exp(1j * gamma) * s, np.exp(-1j * beta) * c],
    ], dtype=np.complex128)


def _unitary_d(d: int, params: np.ndarray) -> np.ndarray:
    """General U(d) from d² real params via exp(iH) with H Hermitian."""
    if d == 1:
        return np.array([[np.exp(1j * params[0])]], dtype=np.complex128)
    if d == 2:
        return _unitary_2x2(params[:4])

    h = np.zeros((d, d), dtype=np.complex128)
    idx = 0
    for i in range(d):
        h[i, i] = params[idx]
        idx += 1
    for i in range(d):
        for j in range(i + 1, d):
            h[i, j] = params[idx] + 1j * params[idx + 1]
            h[j, i] = params[idx] - 1j * params[idx + 1]
            idx += 2

    return expm(1j * h)


def _n_unitary_params(d: int) -> int:
    return d * d


def _build_theta_unitary(
    k_per_path: int,
    edge_lengths: np.ndarray,
    scat_params: np.ndarray,
) -> QuantumGraph:
    """Build theta graph with full U(3) at hubs and U(2) at internal vertices."""
    if k_per_path < 2:
        raise ValueError("Need at least 2 edges per path")

    n_internal_per_path = k_per_path - 1
    n_vertices = 2 + 3 * n_internal_per_path

    adj = np.zeros((n_vertices, n_vertices), dtype=np.int64)
    for path in range(3):
        internal_start = 2 + path * n_internal_per_path
        prev = 0
        for step in range(n_internal_per_path):
            curr = internal_start + step
            adj[prev, curr] = adj[curr, prev] = 1
            prev = curr
        adj[prev, 1] = adj[1, prev] = 1

    n_hub_params = _n_unitary_params(3)  # 9
    n_internal_params = _n_unitary_params(2)  # 4

    scattering = []
    for v in range(2):
        params = scat_params[v * n_hub_params:(v + 1) * n_hub_params]
        scattering.append(_unitary_d(3, params))

    offset = 2 * n_hub_params
    n_internal = n_vertices - 2
    for v in range(n_internal):
        params = scat_params[offset + v * n_internal_params:
                              offset + (v + 1) * n_internal_params]
        scattering.append(_unitary_d(2, params))

    return QuantumGraph(adj, list(edge_lengths), scattering)


def _count_params(k_per_path: int) -> tuple[int, int, int]:
    n_edges = 3 * k_per_path
    n_internal = 3 * (k_per_path - 1)
    n_scat = 2 * _n_unitary_params(3) + n_internal * _n_unitary_params(2)
    return n_edges, n_scat, n_edges + n_scat


def _objective(
    params: np.ndarray,
    k_per_path: int,
    n_edges: int,
    scorer: SpectralScorer,
    zeros: np.ndarray,
    n_zeros: int,
) -> float:
    edge_lengths = params[:n_edges]
    scat_params = params[n_edges:]
    if np.any(edge_lengths < 0.01):
        return 1.0
    try:
        qg = _build_theta_unitary(k_per_path, edge_lengths, scat_params)
        total_length = float(np.sum(edge_lengths))
        k_max = max((n_zeros + 5) * np.pi / total_length, 10.0)
        spectrum = qg.compute_spectrum(k_max, n_points=3000)
        result = scorer.score_spectrum(spectrum, zeros, n_zeros,
                                       edge_lengths=qg.edge_lengths)
        return -result.total_score
    except Exception:
        return 1.0


def _compute_residuals(qg: QuantumGraph, scorer: SpectralScorer,
                       n_zeros: int) -> dict:
    zeros = scorer._rz.get_zeros(n_zeros)
    total_length = sum(qg.edge_lengths)
    k_max = max((n_zeros + 5) * np.pi / total_length, 10.0)
    spectrum = qg.compute_spectrum(k_max, n_points=3000)
    if len(spectrum) < 3:
        return {"residuals": [], "mad": 99.0, "max_residual": 99.0}
    scaled, alpha = _weyl_rescale(np.sort(spectrum), total_length, zeros)
    zero_spacings = np.diff(zeros)
    margin = float(np.mean(zero_spacings)) if len(zero_spacings) > 0 else 1.0
    k_low = zeros[0] - margin
    k_high = zeros[-1] + margin
    windowed = scaled[(scaled >= k_low) & (scaled <= k_high)]
    n_match = min(len(windowed), len(zeros))
    if n_match < 3:
        return {"residuals": [], "mad": 99.0, "max_residual": 99.0}
    residuals = [abs(float(windowed[i] - zeros[i])) for i in range(n_match)]
    mean_spacing = float(np.mean(zero_spacings))
    return {
        "residuals": residuals,
        "mad": float(np.mean(residuals)),
        "mad_mean_spacings": float(np.mean(residuals)) / mean_spacing,
        "max_residual": float(np.max(residuals)),
        "n_matched": n_match,
    }


def _load_existing() -> dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"config": {}, "results": []}


def _save_results(data: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def estimate_runtime() -> None:
    time_per_eval = 0.008
    total = 0
    log("Runtime estimates:")
    for k in K_VALUES:
        n_edges, n_scat, total_params = _count_params(k)
        est = N_EVALS * N_RESTARTS * time_per_eval
        total += est
        log(f"  k={k} ({total_params}D, V={2 + 3*(k-1)}, E={n_edges}): "
            f"~{est/3600:.1f} hours")
    log(f"  Total: ~{total/3600:.1f} hours")
    log("")


def run_experiment() -> None:
    log("=" * 60)
    log("U(d) SCATTERING ON THETA GRAPHS")
    log("=" * 60)
    log(f"k values (edges per path): {K_VALUES}")
    log(f"Hub scattering: U(3) (9 params each)")
    log(f"Internal scattering: U(2) (4 params each)")
    log(f"N zeros: {N_ZEROS}")
    log(f"N restarts: {N_RESTARTS}, max evals: {N_EVALS}")
    log(f"Results: {RESULTS_FILE}")
    log("")

    estimate_runtime()

    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(N_ZEROS)

    existing = _load_existing()
    completed_ks = {r["k_per_path"] for r in existing.get("results", [])}

    existing["config"] = {
        "k_values": K_VALUES,
        "n_zeros": N_ZEROS,
        "n_restarts": N_RESTARTS,
        "n_evals": N_EVALS,
        "hub_dim": 3,
        "internal_dim": 2,
    }

    for k in K_VALUES:
        if k in completed_ks:
            prev = next(r for r in existing["results"]
                        if r["k_per_path"] == k)
            log(f"k={k}: already done (score={prev['best_score']:.4f}), skipping")
            continue

        n_edges, n_scat, total_params = _count_params(k)
        n_vertices = 2 + 3 * (k - 1)

        log(f"\nOptimizing theta k={k} ({total_params}D, V={n_vertices}, "
            f"E={n_edges})...")
        t_start = time.time()

        best_score = -1.0
        best_params = None
        all_scores = []

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        for restart in range(N_RESTARTS):
            rng = np.random.default_rng(k * 10000 + restart)

            if restart % 4 == 0:
                edge_init = np.array([np.log(primes[i % len(primes)])
                                       for i in range(n_edges)])
            elif restart % 4 == 1:
                edge_init = rng.uniform(0.5, 5.0, n_edges)
            elif restart % 4 == 2:
                edge_init = rng.uniform(1.0, 8.0, n_edges)
            else:
                L = rng.uniform(10.0, 30.0)
                edge_init = np.full(n_edges, L / n_edges)

            scat_init = rng.uniform(-3, 3, n_scat)
            x0 = np.concatenate([edge_init, scat_init])

            lower = [0.05] * n_edges + [-10.0] * n_scat
            upper = [15.0] * n_edges + [10.0] * n_scat

            opts = cma.CMAOptions()
            opts.set("maxfevals", N_EVALS)
            opts.set("tolfun", 1e-10)
            opts.set("tolx", 1e-10)
            opts.set("verbose", -9)
            opts.set("bounds", [lower, upper])
            opts.set("CMA_stds", [2.0] * n_edges + [1.0] * n_scat)

            es = cma.CMAEvolutionStrategy(x0, 1.5, opts)
            while not es.stop():
                sols = es.ask()
                fits = [_objective(x, k, n_edges, scorer, zeros, N_ZEROS)
                        for x in sols]
                es.tell(sols, fits)

            cma_score = -es.result.fbest
            cma_params = es.result.xbest

            try:
                polish = minimize(
                    _objective, cma_params,
                    args=(k, n_edges, scorer, zeros, N_ZEROS),
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

            all_scores.append(final_score)

            marker = ""
            if final_score > best_score:
                best_score = final_score
                best_params = final_params
                marker = " <-- BEST"

            log(f"  restart {restart:2d}: {final_score:.4f}{marker}")
            time.sleep(2)

        elapsed = time.time() - t_start

        edge_lengths_best = best_params[:n_edges]
        scat_params_best = best_params[n_edges:]
        qg = _build_theta_unitary(k, edge_lengths_best, scat_params_best)
        res = _compute_residuals(qg, scorer, N_ZEROS)

        result_entry = {
            "k_per_path": k,
            "n_vertices": n_vertices,
            "n_edges": n_edges,
            "total_params": total_params,
            "best_score": best_score,
            "best_mad": res.get("mad_mean_spacings", 99),
            "mean_score": float(np.mean(all_scores)),
            "std_score": float(np.std(all_scores)),
            "best_params": {
                "edge_lengths": [float(x) for x in edge_lengths_best],
                "scat_params": [float(x) for x in scat_params_best],
            },
            "per_zero_residuals": res.get("residuals", []),
            "runtime_hours": elapsed / 3600,
        }

        existing["results"].append(result_entry)
        _save_results(existing)

        log(f"  k={k} ({total_params}D) | Best: {best_score:.4f} | "
            f"MAD: {res.get('mad_mean_spacings', 99):.4f} | "
            f"Mean: {np.mean(all_scores):.4f}+/-{np.std(all_scores):.4f} | "
            f"Time: {elapsed/3600:.1f}h")

    _generate_plot(existing)
    log(f"\nAll results saved to {RESULTS_FILE}")
    log(f"Plot saved to {PLOT_FILE}")


def _generate_plot(data: dict) -> None:
    results = sorted(data["results"], key=lambda r: r["k_per_path"])
    if not results:
        return

    ks = [r["k_per_path"] for r in results]
    scores = [r["best_score"] for r in results]
    mads = [r["best_mad"] for r in results]
    stds = [r["std_score"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.errorbar(ks, scores, yerr=stds, fmt='go-', markersize=8, linewidth=2,
                 capsize=4, label='Theta U(3)+U(2) best score')

    ax1.axhline(y=0.9057, color='blue', linestyle='--', alpha=0.5,
                label='U(2) cycle ceiling ($C_{11}$, 0.9057)')
    ax1.axhline(y=0.955, color='gray', linestyle='--', alpha=0.3,
                label='Self-comparison (0.955)')

    ax1.set_xlabel('Edges per Path ($k$)', fontsize=13)
    ax1.set_ylabel('Best Score $\\mathcal{S}$', fontsize=13, color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    ax2 = ax1.twinx()
    ax2.plot(ks, mads, 'rs--', markersize=6, linewidth=1.5, alpha=0.7,
             label='Best MAD (mean spacings)')
    ax2.set_ylabel('Best MAD (mean spacings)', fontsize=13, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)

    ax1.set_title('U(3)+U(2) Scattering on Theta Graphs vs. Edges per Path',
                  fontsize=14)
    ax1.grid(True, alpha=0.2)

    plt.tight_layout()
    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_FILE, dpi=200)
    plt.close()


if __name__ == "__main__":
    run_experiment()
