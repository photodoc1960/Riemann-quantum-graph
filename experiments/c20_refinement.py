#!/usr/bin/env python3
"""Local refinement of the C_20 U(2) best solution.

Takes the best C_20 result from u2_scaling_results.json and runs 25
restarts initialized in a small Gaussian neighborhood. Tests whether the
0.9258 result is the floor of a deeper basin or a robust local optimum.

Each restart logs to results/c20_refinement.jsonl immediately.
Final best graph saved to results/best_c20_refined.json if it exceeds
the prior best.
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


N = 20
N_ZEROS = 100
N_RESTARTS = 25
N_EVALS = 75_000
SCALING_FILE = Path("results/u2_scaling_results.json")
RESULTS_FILE = Path("results/c20_refinement.jsonl")
BEST_FILE = Path("results/best_c20_refined.json")

EDGE_SIGMA = 0.3
SCAT_SIGMA = 0.5


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def _unitary_2x2(params: np.ndarray) -> np.ndarray:
    alpha, beta, gamma, theta = params
    c, s = np.cos(theta), np.sin(theta)
    phase = np.exp(1j * alpha)
    return phase * np.array([
        [np.exp(1j * beta) * c, -np.exp(-1j * gamma) * s],
        [np.exp(1j * gamma) * s, np.exp(-1j * beta) * c],
    ], dtype=np.complex128)


def _build_cycle_unitary(n: int, edge_lengths: np.ndarray,
                         scat_params: np.ndarray) -> QuantumGraph:
    adj = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1
    scattering = [_unitary_2x2(scat_params[v*4:(v+1)*4]) for v in range(n)]
    return QuantumGraph(adj, list(edge_lengths), scattering)


def _objective(params: np.ndarray, n: int, scorer: SpectralScorer,
               zeros: np.ndarray, n_zeros: int) -> float:
    edge_lengths = params[:n]
    scat_params = params[n:]
    if np.any(edge_lengths < 0.01):
        return 1.0
    try:
        qg = _build_cycle_unitary(n, edge_lengths, scat_params)
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
        return {"residuals": [], "mad": 99.0, "max_residual": 99.0,
                "mad_mean_spacings": 99.0, "n_matched": 0}
    scaled, alpha = _weyl_rescale(np.sort(spectrum), total_length, zeros)
    zero_spacings = np.diff(zeros)
    margin = float(np.mean(zero_spacings)) if len(zero_spacings) > 0 else 1.0
    k_low = zeros[0] - margin
    k_high = zeros[-1] + margin
    windowed = scaled[(scaled >= k_low) & (scaled <= k_high)]
    n_match = min(len(windowed), len(zeros))
    if n_match < 3:
        return {"residuals": [], "mad": 99.0, "max_residual": 99.0,
                "mad_mean_spacings": 99.0, "n_matched": 0}
    residuals = [abs(float(windowed[i] - zeros[i])) for i in range(n_match)]
    mean_spacing = float(np.mean(zero_spacings))
    return {
        "residuals": residuals,
        "mad": float(np.mean(residuals)),
        "mad_mean_spacings": float(np.mean(residuals)) / mean_spacing,
        "max_residual": float(np.max(residuals)),
        "n_matched": n_match,
    }


def _load_c20_best() -> tuple[float, np.ndarray, np.ndarray]:
    if not SCALING_FILE.exists():
        log(f"ERROR: {SCALING_FILE} not found")
        sys.exit(1)
    with open(SCALING_FILE) as f:
        data = json.load(f)
    c20 = next((r for r in data["results"] if r["n"] == N), None)
    if c20 is None:
        log(f"ERROR: C_{N} not found in scaling results")
        sys.exit(1)
    log(f"Prior C_{N} best: {c20['best_score']:.4f}")
    return (
        c20["best_score"],
        np.array(c20["best_params"]["edge_lengths"]),
        np.array(c20["best_params"]["scat_params"]),
    )


def _completed_restarts() -> set[int]:
    completed = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        completed.add(r["restart"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return completed


def run_refinement() -> None:
    log("=" * 60)
    log(f"LOCAL REFINEMENT OF C_{N} BEST SOLUTION")
    log("=" * 60)

    prior_best_score, edge_seed, scat_seed = _load_c20_best()
    n_edges = N
    n_scat = 4 * N
    n_params = n_edges + n_scat

    log(f"Seed total length: {np.sum(edge_seed):.2f}")
    log(f"Parameters: {n_params} ({n_edges} edges + {n_scat} scattering)")
    log(f"Restarts: {N_RESTARTS}, max evals each: {N_EVALS}")
    log(f"Perturbation: edge_sigma={EDGE_SIGMA}, scat_sigma={SCAT_SIGMA}")
    log(f"Results: {RESULTS_FILE}")
    log("")

    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(N_ZEROS)

    completed = _completed_restarts()
    if completed:
        log(f"Resuming: {len(completed)} restarts already done")
        log("")

    best_score = prior_best_score
    best_params = np.concatenate([edge_seed, scat_seed])
    best_record = None

    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if r["score"] > best_score:
                            best_score = r["score"]
                            best_record = r
                    except (json.JSONDecodeError, KeyError):
                        pass

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    for restart in range(N_RESTARTS):
        if restart in completed:
            log(f"  restart {restart:2d}: already done, skipping")
            continue

        rng = np.random.default_rng(20 * 100000 + restart + 1)

        edge_perturbed = edge_seed + rng.normal(0, EDGE_SIGMA, n_edges)
        edge_perturbed = np.clip(edge_perturbed, 0.05, 15.0)
        scat_perturbed = scat_seed + rng.normal(0, SCAT_SIGMA, n_scat)
        # Clip scattering params strictly inside bounds (CMA-ES rejects boundary equality)
        scat_perturbed = np.clip(scat_perturbed, -9.99, 9.99)

        x0 = np.concatenate([edge_perturbed, scat_perturbed])

        lower = [0.05] * n_edges + [-10.0] * n_scat
        upper = [15.0] * n_edges + [10.0] * n_scat

        opts = cma.CMAOptions()
        opts.set("maxfevals", N_EVALS)
        opts.set("tolfun", 1e-10)
        opts.set("tolx", 1e-10)
        opts.set("verbose", -9)
        opts.set("bounds", [lower, upper])
        opts.set("CMA_stds", [1.0] * n_edges + [0.5] * n_scat)

        t_start = time.time()
        es = cma.CMAEvolutionStrategy(x0, 0.5, opts)
        while not es.stop():
            sols = es.ask()
            fits = [_objective(x, N, scorer, zeros, N_ZEROS) for x in sols]
            es.tell(sols, fits)

        cma_score = -es.result.fbest
        cma_params = es.result.xbest

        try:
            polish = minimize(
                _objective, cma_params,
                args=(N, scorer, zeros, N_ZEROS),
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

        edge_lengths = final_params[:n_edges]
        scat_params_final = final_params[n_edges:]
        qg = _build_cycle_unitary(N, edge_lengths, scat_params_final)
        res = _compute_residuals(qg, scorer, N_ZEROS)

        marker = ""
        if final_score > best_score:
            best_score = final_score
            best_params = final_params
            best_record = {
                "restart": restart,
                "score": final_score,
                "mad": res.get("mad", 99),
                "mad_mean_spacings": res.get("mad_mean_spacings", 99),
            }
            marker = "  <-- NEW BEST"

        record = {
            "restart": restart,
            "cma_score": cma_score,
            "score": final_score,
            "mad": res.get("mad", 99),
            "mad_mean_spacings": res.get("mad_mean_spacings", 99),
            "max_residual": res.get("max_residual", 99),
            "edge_lengths": [float(x) for x in edge_lengths],
            "scat_params": [float(x) for x in scat_params_final],
            "total_length": float(np.sum(edge_lengths)),
            "n_cma_evals": int(es.result.evaluations),
            "elapsed_s": elapsed,
            "residuals": res.get("residuals", []),
        }
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        log(f"  restart {restart:2d}: score={final_score:.4f}  "
            f"MAD={res.get('mad_mean_spacings', 99):.4f}  "
            f"L={np.sum(edge_lengths):.1f}  "
            f"{elapsed:.0f}s{marker}")
        time.sleep(2)

    log("")
    log("=" * 60)
    log("REFINEMENT COMPLETE")
    log("=" * 60)
    log(f"Prior best: {prior_best_score:.4f}")
    log(f"Refined best: {best_score:.4f}")
    log(f"Improvement: {best_score - prior_best_score:+.4f}")

    if best_score > prior_best_score:
        edge_lengths = best_params[:n_edges]
        scat_params_final = best_params[n_edges:]
        qg = _build_cycle_unitary(N, edge_lengths, scat_params_final)
        res = _compute_residuals(qg, scorer, N_ZEROS)

        scattering_matrices = []
        for v in range(N):
            p = scat_params_final[v * 4:(v + 1) * 4]
            u = _unitary_2x2(p)
            scattering_matrices.append({
                "vertex": v,
                "params": [float(x) for x in p],
                "matrix_real": u.real.tolist(),
                "matrix_imag": u.imag.tolist(),
            })

        spec = {
            "graph_type": f"C_{N} with U(2) scattering (locally refined)",
            "n_vertices": N,
            "n_edges": N,
            "score": best_score,
            "prior_best_score": prior_best_score,
            "n_zeros": N_ZEROS,
            "edge_lengths": [float(x) for x in edge_lengths],
            "total_length": float(np.sum(edge_lengths)),
            "scattering_matrices": scattering_matrices,
            **res,
        }
        with open(BEST_FILE, "w") as f:
            json.dump(spec, f, indent=2)
        log(f"New best graph saved to {BEST_FILE}")


if __name__ == "__main__":
    run_refinement()
