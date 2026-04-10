#!/usr/bin/env python3
"""Deep U(2) optimization on cycle graphs.

The unitary scattering experiment showed U(2) on C_7 hits 0.896 — far above
the 0.72 Neumann ceiling. This script pushes that result with:
  - Higher CMA-ES budget (100k evals per restart)
  - More zeros (100)
  - More restarts (30)
  - Nelder-Mead polish
  - Full graph specification saved for the best result
  - Sweep over cycle sizes C_5 to C_15

Results logged to results/unitary_deep.jsonl after every restart.
Best graph saved to results/best_unitary_graph.json.
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


RESULTS_FILE = Path("results/unitary_deep.jsonl")
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


def _objective(
    params: np.ndarray,
    n: int,
    scorer: SpectralScorer,
    zeros: np.ndarray,
    n_zeros: int,
) -> float:
    """Negative score. Params = [edge_lengths (n), scat_params (4n)]."""
    edge_lengths = params[:n]
    scat_params = params[n:]

    if np.any(edge_lengths < 0.01):
        return 1.0

    try:
        qg = _build_cycle_unitary(n, edge_lengths, scat_params)
        total_length = float(np.sum(edge_lengths))
        k_max = max((n_zeros + 5) * np.pi / total_length, 10.0)
        spectrum = qg.compute_spectrum(k_max, n_points=3000)
        result = scorer.score_spectrum(
            spectrum, zeros, n_zeros,
            edge_lengths=qg.edge_lengths,
        )
        return -result.total_score
    except Exception:
        return 1.0


def _compute_residuals(
    qg: QuantumGraph,
    scorer: SpectralScorer,
    n_zeros: int,
) -> dict:
    """Compute per-zero residuals."""
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
    return {
        "residuals": residuals,
        "mad": float(np.mean(residuals)),
        "max_residual": float(np.max(residuals)),
        "std_residual": float(np.std(residuals)),
        "n_matched": n_match,
        "weyl_alpha": float(alpha),
    }


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def optimize_single(
    n: int,
    n_zeros: int = 100,
    n_restarts: int = 30,
    max_evals: int = 100000,
) -> dict | None:
    """Deep optimization of C_n with U(2) scattering."""
    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(n_zeros)

    n_edge_params = n
    n_scat_params = 4 * n
    n_params = n_edge_params + n_scat_params

    log(f"\n{'=' * 60}")
    log(f"Deep U(2) optimization: C_{n}")
    log(f"  {n_params} params ({n} edges + {n_scat_params} scattering)")
    log(f"  {n_zeros} zeros, {n_restarts} restarts, {max_evals} max evals")
    log(f"{'=' * 60}")

    best_score = -1.0
    best_params = None
    best_record = None

    # Check existing results for this n
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if r.get("n_vertices") == n and r.get("score", 0) > best_score:
                            best_score = r["score"]
                    except (json.JSONDecodeError, KeyError):
                        pass
        if best_score > 0:
            log(f"  Existing best for C_{n}: {best_score:.4f}")

    overall_best_score = best_score

    for restart in range(n_restarts):
        rng = np.random.default_rng(restart * 1000 + n * 100 + 42)

        # Edge length initialization
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
            # Uniform at a good total length
            L = rng.uniform(12.0, 35.0)
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
            fits = [_objective(x, n, scorer, zeros, n_zeros) for x in sols]
            es.tell(sols, fits)

        cma_score = -es.result.fbest
        cma_params = es.result.xbest

        # Nelder-Mead polish
        try:
            polish = minimize(
                _objective,
                cma_params,
                args=(n, scorer, zeros, n_zeros),
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

        elapsed = time.time() - t0

        # Compute residuals
        edge_lengths = final_params[:n]
        scat_params = final_params[n:]
        qg = _build_cycle_unitary(n, edge_lengths, scat_params)
        res = _compute_residuals(qg, scorer, n_zeros)

        marker = ""
        if final_score > overall_best_score:
            overall_best_score = final_score
            best_params = final_params
            marker = "  <-- BEST"

            best_record = {
                "n_vertices": n,
                "n_zeros": n_zeros,
                "restart": restart,
                "score": final_score,
                "cma_score": cma_score,
                "edge_lengths": [float(x) for x in edge_lengths],
                "scat_params": [float(x) for x in scat_params],
                "total_length": float(np.sum(edge_lengths)),
                "n_cma_evals": int(es.result.evaluations),
                "elapsed_s": elapsed,
                **res,
            }

        log(f"  restart {restart:2d}: CMA={cma_score:.4f} -> "
            f"final={final_score:.4f}  "
            f"MAD={res.get('mad', 99):.3f}  "
            f"L={np.sum(edge_lengths):.1f}  "
            f"evals={es.result.evaluations}  "
            f"{elapsed:.0f}s{marker}")

        # Save every restart
        record = {
            "n_vertices": n,
            "n_zeros": n_zeros,
            "restart": restart,
            "score": final_score,
            "cma_score": cma_score,
            "edge_lengths": [float(x) for x in edge_lengths],
            "scat_params": [float(x) for x in scat_params],
            "total_length": float(np.sum(edge_lengths)),
            "n_cma_evals": int(es.result.evaluations),
            "elapsed_s": elapsed,
            **res,
        }
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        time.sleep(2)  # Thermal cooldown

    log(f"\n  C_{n} best: {overall_best_score:.4f}")
    return best_record


def save_best_graph(record: dict) -> None:
    """Save the complete specification of the best graph."""
    if record is None:
        return

    n = record["n_vertices"]
    edge_lengths = record["edge_lengths"]
    scat_params = record["scat_params"]

    # Reconstruct scattering matrices for display
    scattering_matrices = []
    for v in range(n):
        u = _unitary_2x2(np.array(scat_params[v * 4:(v + 1) * 4]))
        scattering_matrices.append({
            "vertex": v,
            "params": scat_params[v * 4:(v + 1) * 4],
            "matrix_real": u.real.tolist(),
            "matrix_imag": u.imag.tolist(),
        })

    spec = {
        "graph_type": f"C_{n} with U(2) scattering",
        "n_vertices": n,
        "n_edges": n,
        "score": record["score"],
        "n_zeros": record["n_zeros"],
        "edge_lengths": edge_lengths,
        "total_length": record["total_length"],
        "scattering_matrices": scattering_matrices,
        "mad": record.get("mad"),
        "max_residual": record.get("max_residual"),
        "residuals": record.get("residuals"),
    }

    with open(BEST_GRAPH_FILE, "w") as f:
        json.dump(spec, f, indent=2)

    log(f"\nBest graph specification saved to {BEST_GRAPH_FILE}")


def run_sweep(
    n_min: int = 5,
    n_max: int = 15,
    n_zeros: int = 100,
    n_restarts: int = 30,
    max_evals: int = 100000,
) -> None:
    """Sweep C_n with U(2) scattering from n_min to n_max."""
    log("DEEP U(2) SCATTERING SWEEP")
    log(f"C_{n_min} to C_{n_max}, {n_zeros} zeros, "
        f"{n_restarts} restarts, {max_evals} max evals")
    log(f"Results: {RESULTS_FILE}")
    log(f"Best graph: {BEST_GRAPH_FILE}")

    overall_best_score = 0.0
    overall_best_record = None

    for n in range(n_min, n_max + 1):
        record = optimize_single(
            n, n_zeros=n_zeros, n_restarts=n_restarts,
            max_evals=max_evals,
        )

        if record and record.get("score", 0) > overall_best_score:
            overall_best_score = record["score"]
            overall_best_record = record
            save_best_graph(record)
            log(f"  NEW OVERALL BEST: C_{n} = {overall_best_score:.4f}")

    # Final summary
    log("\n" + "=" * 60)
    log("SWEEP COMPLETE")
    log("=" * 60)

    # Read all results
    records: dict[int, list[dict]] = {}
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    n = r["n_vertices"]
                    records.setdefault(n, []).append(r)
                except (json.JSONDecodeError, KeyError):
                    pass

    log(f"\n{'C_n':>5s}  {'Params':>7s}  {'Best':>8s}  {'Mean':>8s}  "
        f"{'MAD':>8s}  {'Runs':>5s}")
    log("-" * 50)
    for n in sorted(records):
        recs = records[n]
        best = max(recs, key=lambda r: r.get("score", 0))
        mean = sum(r["score"] for r in recs) / len(recs)
        log(f"  C_{n:<2d}  {5*n:>7d}  {best['score']:>8.4f}  {mean:>8.4f}  "
            f"{best.get('mad', 99):>8.3f}  {len(recs):>5d}")

    if overall_best_record:
        log(f"\nOverall best: C_{overall_best_record['n_vertices']} = "
            f"{overall_best_score:.4f}")
        log(f"  MAD: {overall_best_record.get('mad', 99):.3f}")
        log(f"  L_total: {overall_best_record.get('total_length', 0):.2f}")

        # Print residuals
        residuals = overall_best_record.get("residuals", [])
        if residuals:
            scorer = SpectralScorer()
            zeros = scorer._rz.get_zeros(overall_best_record["n_zeros"])
            log(f"\n  Per-zero residuals (first 30):")
            for i, r_val in enumerate(residuals[:30]):
                z = zeros[i] if i < len(zeros) else 0
                bar_len = min(int(r_val * 4), 20)
                log(f"    zero {i:2d} ({z:7.2f}): {r_val:.4f}  "
                    f"{'|' * max(bar_len, 1)}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Deep U(2) scattering optimization on cycle graphs",
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Single cycle size to optimize")
    parser.add_argument("--n-min", type=int, default=5)
    parser.add_argument("--n-max", type=int, default=15)
    parser.add_argument("--n-zeros", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--max-evals", type=int, default=100000)
    args = parser.parse_args()

    if args.n is not None:
        record = optimize_single(
            args.n, n_zeros=args.n_zeros,
            n_restarts=args.restarts, max_evals=args.max_evals,
        )
        if record:
            save_best_graph(record)
    else:
        run_sweep(
            n_min=args.n_min, n_max=args.n_max,
            n_zeros=args.n_zeros, n_restarts=args.restarts,
            max_evals=args.max_evals,
        )


if __name__ == "__main__":
    main()
