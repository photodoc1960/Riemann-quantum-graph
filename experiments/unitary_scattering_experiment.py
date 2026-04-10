#!/usr/bin/env python3
"""Experiment 1: Full U(n) scattering on C_7.

Tests whether the ~0.72 ceiling is imposed by Neumann scattering constraints
or whether fully general unitary vertex conditions can break through.

Results logged to results/unitary_scattering.jsonl
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "sequential"

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


RESULTS_FILE = Path("results/unitary_scattering.jsonl")
N_ZEROS = 100


def _unitary_2x2(params: np.ndarray) -> np.ndarray:
    """Build a general U(2) matrix from 4 real parameters.

    U = exp(iα) [[exp(iβ)cos(θ), -exp(-iγ)sin(θ)],
                  [exp(iγ)sin(θ),  exp(-iβ)cos(θ)]]
    """
    alpha, beta, gamma, theta = params
    c, s = np.cos(theta), np.sin(theta)
    phase = np.exp(1j * alpha)
    return phase * np.array([
        [np.exp(1j * beta) * c, -np.exp(-1j * gamma) * s],
        [np.exp(1j * gamma) * s,  np.exp(-1j * beta) * c],
    ], dtype=np.complex128)


def _unitary_from_params(degree: int, params: np.ndarray) -> np.ndarray:
    """Build a general U(d) matrix from d² real parameters.

    For d=2: uses the standard parameterization (4 params).
    For d>2: uses matrix exponential of skew-Hermitian matrix.
    """
    if degree == 1:
        # U(1) = exp(iθ), 1 parameter
        return np.array([[np.exp(1j * params[0])]], dtype=np.complex128)
    if degree == 2:
        return _unitary_2x2(params[:4])

    # General U(d): parameterize via exp(iH) where H is Hermitian
    # H has d² real parameters: d diagonal + d(d-1)/2 real off-diag + d(d-1)/2 imag off-diag
    n_params = degree * degree
    h = np.zeros((degree, degree), dtype=np.complex128)

    idx = 0
    # Diagonal (real)
    for i in range(degree):
        h[i, i] = params[idx]
        idx += 1
    # Upper triangle (complex)
    for i in range(degree):
        for j in range(i + 1, degree):
            h[i, j] = params[idx] + 1j * params[idx + 1]
            h[j, i] = params[idx] - 1j * params[idx + 1]
            idx += 2

    # exp(iH) is unitary when H is Hermitian
    from scipy.linalg import expm
    return expm(1j * h)


def _n_unitary_params(degree: int) -> int:
    """Number of real parameters for U(d)."""
    return degree * degree


def _build_cycle_with_unitary(
    n: int,
    edge_lengths: list[float],
    scattering_params: np.ndarray,
) -> QuantumGraph:
    """Build C_n with arbitrary unitary scattering at each vertex."""
    adj = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1

    # Each vertex has degree 2 -> U(2) with 4 params
    params_per_vertex = _n_unitary_params(2)
    scattering = []
    for v in range(n):
        start = v * params_per_vertex
        end = start + params_per_vertex
        u = _unitary_from_params(2, scattering_params[start:end])
        scattering.append(u)

    return QuantumGraph(adj, edge_lengths, scattering)


def _objective(
    params: np.ndarray,
    n: int,
    edge_lengths: list[float],
    scorer: SpectralScorer,
    zeros: np.ndarray,
    n_zeros: int,
) -> float:
    """Negative score. Optimize scattering params only, edge lengths fixed."""
    try:
        qg = _build_cycle_with_unitary(n, edge_lengths, params)
        total_length = sum(edge_lengths)
        k_max = max((n_zeros + 5) * np.pi / total_length, 10.0)
        spectrum = qg.compute_spectrum(k_max, n_points=3000)
        result = scorer.score_spectrum(
            spectrum, zeros, n_zeros,
            edge_lengths=qg.edge_lengths,
        )
        return -result.total_score
    except Exception:
        return 1.0


def _objective_joint(
    params: np.ndarray,
    n: int,
    n_scat_params: int,
    scorer: SpectralScorer,
    zeros: np.ndarray,
    n_zeros: int,
) -> float:
    """Negative score. Optimize edge lengths AND scattering jointly."""
    edge_lengths = list(params[:n])
    scat_params = params[n:]
    if any(l <= 0.01 for l in edge_lengths):
        return 1.0
    try:
        qg = _build_cycle_with_unitary(n, edge_lengths, scat_params)
        total_length = sum(edge_lengths)
        k_max = max((n_zeros + 5) * np.pi / total_length, 10.0)
        spectrum = qg.compute_spectrum(k_max, n_points=3000)
        result = scorer.score_spectrum(
            spectrum, zeros, n_zeros,
            edge_lengths=qg.edge_lengths,
        )
        return -result.total_score
    except Exception:
        return 1.0


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def run_experiment(n_trials: int = 50) -> None:
    """Run the full U(n) scattering experiment on C_7."""
    n = 7
    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(N_ZEROS)
    params_per_vertex = _n_unitary_params(2)  # 4 for U(2)
    n_scat_params = n * params_per_vertex  # 28

    log("=" * 70)
    log("EXPERIMENT 1: Full U(2) Scattering on C_7")
    log("=" * 70)
    log(f"  Vertices: {n}, Zeros: {N_ZEROS}")
    log(f"  Scattering params per vertex: {params_per_vertex} (full U(2))")
    log(f"  Total scattering params: {n_scat_params}")
    log(f"  Results: {RESULTS_FILE}")
    log("")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ---- Phase A: fixed edge lengths, optimize scattering only ----
    log("Phase A: Fixed edge lengths (L/n uniform), optimize U(2) scattering")
    log("-" * 60)

    best_score_a = -1
    best_params_a = None

    for trial in range(n_trials):
        rng = np.random.default_rng(trial * 1000 + 7)
        L_total = rng.uniform(15.0, 40.0)
        edge_lengths = [L_total / n] * n
        x0 = rng.uniform(0, 2 * np.pi, size=n_scat_params)

        opts = cma.CMAOptions()
        opts.set("maxfevals", 15000)
        opts.set("tolfun", 1e-9)
        opts.set("verbose", -9)

        t0 = time.time()
        es = cma.CMAEvolutionStrategy(x0, 1.0, opts)
        while not es.stop():
            sols = es.ask()
            fits = [
                _objective(x, n, edge_lengths, scorer, zeros, N_ZEROS)
                for x in sols
            ]
            es.tell(sols, fits)

        score = -es.result.fbest
        elapsed = time.time() - t0

        marker = ""
        if score > best_score_a:
            best_score_a = score
            best_params_a = es.result.xbest
            marker = " <-- BEST"

        log(f"  trial {trial:2d}: score={score:.4f}  L={L_total:.1f}  "
            f"evals={es.result.evaluations}  {elapsed:.0f}s{marker}")

        record = {
            "phase": "A_fixed_lengths",
            "trial": trial,
            "score": score,
            "L_total": L_total,
            "n_evals": int(es.result.evaluations),
            "elapsed_s": elapsed,
        }
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    log(f"\n  Phase A best (scattering only): {best_score_a:.4f}")

    # ---- Phase B: joint optimization of lengths + scattering ----
    log("\nPhase B: Joint optimization (edge lengths + U(2) scattering)")
    log("-" * 60)

    n_joint_params = n + n_scat_params  # 7 + 28 = 35
    best_score_b = -1

    for trial in range(n_trials):
        rng = np.random.default_rng(trial * 1000 + 77)

        edge_init = rng.uniform(1.0, 6.0, size=n)
        scat_init = rng.uniform(0, 2 * np.pi, size=n_scat_params)
        x0 = np.concatenate([edge_init, scat_init])

        lower = [0.05] * n + [-10.0] * n_scat_params
        upper = [15.0] * n + [10.0] * n_scat_params

        opts = cma.CMAOptions()
        opts.set("maxfevals", 20000)
        opts.set("tolfun", 1e-9)
        opts.set("verbose", -9)
        opts.set("bounds", [lower, upper])
        opts.set("CMA_stds", [2.0] * n + [1.0] * n_scat_params)

        t0 = time.time()
        es = cma.CMAEvolutionStrategy(x0, 1.5, opts)
        while not es.stop():
            sols = es.ask()
            fits = [
                _objective_joint(x, n, n_scat_params, scorer, zeros, N_ZEROS)
                for x in sols
            ]
            es.tell(sols, fits)

        score = -es.result.fbest
        elapsed = time.time() - t0

        marker = ""
        if score > best_score_b:
            best_score_b = score
            marker = " <-- BEST"

        log(f"  trial {trial:2d}: score={score:.4f}  "
            f"L={sum(es.result.xbest[:n]):.1f}  "
            f"evals={es.result.evaluations}  {elapsed:.0f}s{marker}")

        record = {
            "phase": "B_joint",
            "trial": trial,
            "score": score,
            "L_total": float(sum(es.result.xbest[:n])),
            "n_evals": int(es.result.evaluations),
            "elapsed_s": elapsed,
        }
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    log(f"\n  Phase B best (joint): {best_score_b:.4f}")

    # ---- Summary ----
    best_un = max(best_score_a, best_score_b)
    neumann_baseline = 0.720

    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)
    log(f"{'Scattering Model':<30s} {'Score':>8s}  {'Notes':<30s}")
    log("-" * 70)
    log(f"{'Neumann (baseline)':<30s} {neumann_baseline:>8.3f}  {'Known result':<30s}")
    log(f"{'U(2) scattering only':<30s} {best_score_a:>8.3f}  {'Fixed edge lengths':<30s}")
    log(f"{'U(2) joint optimization':<30s} {best_score_b:>8.3f}  {'Lengths + scattering':<30s}")
    log("")

    if best_un > 0.76:
        log("INTERPRETATION: Neumann constraint IS the ceiling. Vertex conditions")
        log("are the key variable. Recommend full U(n) evolutionary search.")
    elif best_un < 0.73:
        log("INTERPRETATION: Vertex conditions are NOT the limiting factor.")
        log("Ceiling is structural — finite graph size or fundamental mismatch.")
    else:
        log(f"INTERPRETATION: U(n) provides modest improvement ({best_un - neumann_baseline:+.3f}).")
        log("Vertex conditions matter but are not the sole bottleneck.")

    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiment(n_trials=50)
