#!/usr/bin/env python3
"""Deep optimization of theta graphs (two hubs, three paths).

Theta graphs are the minimal topology beyond cycles that have genuine
spectral richness: degree-3 hub vertices create reflection, making every
edge length and phase spectrally relevant.

Parameterization: 3*k edges (k edges per path) + phases at all vertices.
All parameters are spectrally effective (unlike cycles where only L_total
and phase distribution matter).

Usage:
    python theta_deep.py --edges-per-path 3 [--n-zeros 50] [--restarts 20]
    python theta_deep.py --sweep   # sweep edges-per-path from 1 to 10
    python theta_deep.py --analyze
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
import time
from pathlib import Path

import cma
import numpy as np
from scipy.optimize import minimize
from rich.console import Console
from rich.table import Table

from riemann_qg.core.quantum_graph import QuantumGraph, directed_scattering
from riemann_qg.core.scoring import SpectralScorer, _weyl_rescale


RESULTS_PATH = Path("results/theta_deep.jsonl")
console = Console()


def _build_theta(
    k_per_path: int,
    edge_lengths: np.ndarray,
    phases: np.ndarray,
) -> QuantumGraph:
    """Build a theta graph: 2 hubs connected by 3 paths of k edges each.

    Vertices:
      0, 1 = hub vertices (degree 3)
      2..2+k-2 = internal vertices of path 1
      etc.

    For k=1: 2 vertices, 3 edges (multi-graph — need k >= 2)
    For k=2: 5 vertices, 6 edges
    For k=3: 8 vertices, 9 edges
    General: 2 + 3*(k-1) vertices, 3*k edges
    """
    if k_per_path < 2:
        raise ValueError("Need at least 2 edges per path for simple graph")

    n_internal_per_path = k_per_path - 1
    n_vertices = 2 + 3 * n_internal_per_path
    n_edges = 3 * k_per_path

    adj = np.zeros((n_vertices, n_vertices), dtype=np.int64)

    # Build 3 paths: hub0 -- internals -- hub1
    edge_idx = 0
    for path in range(3):
        internal_start = 2 + path * n_internal_per_path
        # hub0 -> first internal
        prev = 0
        for step in range(n_internal_per_path):
            curr = internal_start + step
            adj[prev, curr] = adj[curr, prev] = 1
            edge_idx += 1
            prev = curr
        # last internal -> hub1
        adj[prev, 1] = adj[1, prev] = 1
        edge_idx += 1

    # Build scattering matrices
    degrees = np.sum(adj, axis=1)
    scattering = []
    phase_idx = 0
    for v in range(n_vertices):
        d = int(degrees[v])
        if phase_idx < len(phases):
            scattering.append(directed_scattering(d, float(phases[phase_idx])))
            phase_idx += 1
        else:
            scattering.append(directed_scattering(d, 0.0))

    return QuantumGraph(adj, list(edge_lengths[:n_edges]), scattering)


def _objective(
    params: np.ndarray,
    k_per_path: int,
    n_edges: int,
    n_vertices: int,
    scorer: SpectralScorer,
    zeros: np.ndarray,
    n_zeros: int,
) -> float:
    """Negative score. Params = [edge_lengths..., phases...]."""
    edge_lengths = params[:n_edges]
    phases = params[n_edges:]

    # Reject degenerate graphs
    if np.any(edge_lengths < 0.01) or np.sum(edge_lengths) < 1.0:
        return 1.0

    try:
        qg = _build_theta(k_per_path, edge_lengths, phases)
        total_length = sum(qg.edge_lengths)
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


def optimize_theta(
    k_per_path: int = 3,
    n_zeros: int = 50,
    n_restarts: int = 20,
    max_evals: int = 50000,
) -> dict | None:
    """Deep optimization of theta graph using CMA-ES."""
    n_internal = k_per_path - 1
    n_vertices = 2 + 3 * n_internal
    n_edges = 3 * k_per_path
    n_params = n_edges + n_vertices  # edge lengths + phases

    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(n_zeros)

    console.print(f"\n[bold cyan]Theta graph: {k_per_path} edges/path[/bold cyan]")
    console.print(f"  {n_vertices} vertices, {n_edges} edges, "
                  f"{n_params} parameters")
    console.print(f"  Matching {n_zeros} zeros, {n_restarts} restarts")

    best_score = -1.0
    best_record = None

    overall_best_score = 0.0
    # Check existing results
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if r.get("score", 0) > overall_best_score:
                            overall_best_score = r["score"]
                    except (json.JSONDecodeError, KeyError):
                        pass

    for restart in range(n_restarts):
        rng = np.random.default_rng(restart * 1000 + k_per_path * 100)

        # Initial edge lengths: mix of ln(p)-scale and random
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        if restart % 3 == 0:
            # ln(prime) initialization
            edge_init = np.array([
                np.log(primes[i % len(primes)])
                for i in range(n_edges)
            ])
        elif restart % 3 == 1:
            # Random uniform
            edge_init = rng.uniform(0.5, 5.0, size=n_edges)
        else:
            # Random wide
            edge_init = rng.uniform(0.1, 8.0, size=n_edges)

        phase_init = rng.uniform(0, 2 * np.pi, size=n_vertices)
        x0 = np.concatenate([edge_init, phase_init])

        # CMA-ES bounds
        lower = [0.01] * n_edges + [0.0] * n_vertices
        upper = [15.0] * n_edges + [2 * np.pi] * n_vertices

        sigma0 = 1.5
        opts = cma.CMAOptions()
        opts.set("maxfevals", max_evals)
        opts.set("tolfun", 1e-8)
        opts.set("tolx", 1e-8)
        opts.set("verbose", -9)
        opts.set("bounds", [lower, upper])
        opts.set("CMA_stds",
                 [2.0] * n_edges + [1.0] * n_vertices)

        t0 = time.time()

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop():
            solutions = es.ask()
            fitnesses = [
                _objective(x, k_per_path, n_edges, n_vertices,
                          scorer, zeros, n_zeros)
                for x in solutions
            ]
            es.tell(solutions, fitnesses)

        cma_result = es.result
        cma_score = -cma_result.fbest
        cma_params = cma_result.xbest

        # Nelder-Mead polish (accept only if improves)
        try:
            polish = minimize(
                _objective,
                cma_params,
                args=(k_per_path, n_edges, n_vertices,
                      scorer, zeros, n_zeros),
                method="Nelder-Mead",
                options={"maxiter": 3000, "xatol": 1e-8, "fatol": 1e-10},
            )
            polished_score = -polish.fun
            if polished_score > cma_score:
                final_params = polish.x
                final_score = polished_score
            else:
                final_params = cma_params
                final_score = cma_score
        except Exception:
            final_params = cma_params
            final_score = cma_score

        elapsed = time.time() - t0

        # Build graph and compute residuals
        edge_lengths = final_params[:n_edges]
        phases = final_params[n_edges:]
        qg = _build_theta(k_per_path, edge_lengths, phases)
        residual_info = _compute_residuals(qg, scorer, n_zeros)

        marker = ""
        if final_score > best_score:
            best_score = final_score
            if final_score > overall_best_score:
                overall_best_score = final_score
                marker = "  [bold green]<-- OVERALL BEST![/bold green]"
            else:
                marker = f"  [green]<-- best k={k_per_path}[/green]"

            best_record = {
                "k_per_path": k_per_path,
                "n_vertices": n_vertices,
                "n_edges": n_edges,
                "n_params": n_params,
                "n_zeros": n_zeros,
                "restart": restart,
                "score": final_score,
                "cma_score": cma_score,
                "edge_lengths": list(edge_lengths),
                "phases": list(phases),
                "total_length": float(np.sum(edge_lengths)),
                "path_lengths": [
                    float(np.sum(edge_lengths[k_per_path*p:k_per_path*(p+1)]))
                    for p in range(3)
                ],
                "n_cma_evals": int(cma_result.evaluations),
                "elapsed_s": elapsed,
                **residual_info,
            }

        console.print(
            f"  restart {restart:2d}: CMA={cma_score:.4f} -> "
            f"final={final_score:.4f}  "
            f"MAD={residual_info.get('mad', 99):.3f}  "
            f"L={np.sum(edge_lengths):.1f}  "
            f"[dim]{elapsed:.0f}s[/dim]{marker}"
        )

        # Save incrementally
        record = {
            "k_per_path": k_per_path,
            "n_vertices": n_vertices,
            "n_edges": n_edges,
            "n_params": n_params,
            "n_zeros": n_zeros,
            "restart": restart,
            "score": final_score,
            "cma_score": cma_score,
            "edge_lengths": list(edge_lengths),
            "phases": list(phases),
            "total_length": float(np.sum(edge_lengths)),
            "path_lengths": [
                float(np.sum(edge_lengths[k_per_path*p:k_per_path*(p+1)]))
                for p in range(3)
            ],
            "n_cma_evals": int(cma_result.evaluations),
            "elapsed_s": elapsed,
            **residual_info,
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

        time.sleep(3)

    if best_record:
        console.print(
            f"\n  [bold]Theta k={k_per_path} best: {best_score:.4f}  "
            f"MAD={best_record.get('mad', 99):.3f}[/bold]"
        )

    return best_record


def run_sweep(
    k_min: int = 2,
    k_max: int = 10,
    n_zeros: int = 50,
    n_restarts: int = 15,
    max_evals: int = 40000,
) -> None:
    """Sweep edges-per-path from k_min to k_max."""
    console.print(f"[bold]Theta Graph Sweep: k={k_min} to {k_max} edges/path[/bold]")
    console.print(f"  Zeros: {n_zeros}, Restarts: {n_restarts}")
    console.print()

    for k in range(k_min, k_max + 1):
        optimize_theta(
            k_per_path=k, n_zeros=n_zeros,
            n_restarts=n_restarts, max_evals=max_evals,
        )

    console.print(f"\n[bold green]Sweep complete![/bold green]")


def analyze_results() -> None:
    """Print analysis of theta optimization results."""
    if not RESULTS_PATH.exists():
        console.print("[red]No results found.[/red]")
        return

    records: dict[int, list[dict]] = {}
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    k = r["k_per_path"]
                    records.setdefault(k, []).append(r)
                except (json.JSONDecodeError, KeyError):
                    continue

    console.print(f"\n[bold]Theta Graph Optimization Results[/bold]")

    table = Table(title="Best Score by Path Length")
    table.add_column("k", width=3)
    table.add_column("V", width=3)
    table.add_column("E", width=3)
    table.add_column("Params", width=6)
    table.add_column("Score", style="green", width=8)
    table.add_column("MAD", width=8)
    table.add_column("Max Res", width=8)
    table.add_column("L_total", width=8)
    table.add_column("Paths", width=20)
    table.add_column("Runs", width=4)

    best_overall = 0
    best_k = 0

    for k in sorted(records):
        recs = records[k]
        best = max(recs, key=lambda r: r.get("score", 0))
        score = best.get("score", 0)
        if score > best_overall:
            best_overall = score
            best_k = k

        path_lens = best.get("path_lengths", [])
        path_str = ", ".join(f"{p:.1f}" for p in path_lens)

        table.add_row(
            str(k),
            str(best.get("n_vertices", "?")),
            str(best.get("n_edges", "?")),
            str(best.get("n_params", "?")),
            f"{score:.4f}",
            f"{best.get('mad', 99):.3f}",
            f"{best.get('max_residual', 99):.3f}",
            f"{best.get('total_length', 0):.1f}",
            path_str,
            str(len(recs)),
        )
    console.print(table)

    console.print(f"\n  [bold]Overall best: k={best_k}, "
                  f"score={best_overall:.4f}[/bold]")
    console.print(f"  (Cycle ceiling for reference: ~0.805)")

    # Score chart
    console.print(f"\n[bold]Score vs Edges per Path[/bold]")
    for k in sorted(records):
        best = max(records[k], key=lambda r: r.get("score", 0))
        score = best.get("score", 0)
        bar_len = int(score * 60)
        marker = " <-- BEST" if k == best_k else ""
        console.print(f"  k={k:2d}: [{'#' * bar_len:60s}] {score:.4f}{marker}")

    # Details of best
    if best_k in records:
        best = max(records[best_k], key=lambda r: r.get("score", 0))
        console.print(f"\n[bold]Best Graph Details: theta k={best_k}[/bold]")
        console.print(f"  Score: {best['score']:.6f}")
        console.print(f"  L_total: {best.get('total_length', 0):.4f}")
        console.print(f"  Path lengths: {best.get('path_lengths', [])}")

        residuals = best.get("residuals", [])
        if residuals:
            scorer = SpectralScorer()
            zeros = scorer._rz.get_zeros(best.get("n_zeros", 50))
            console.print(f"\n  Per-zero residuals:")
            for i, r_val in enumerate(residuals):
                z = zeros[i] if i < len(zeros) else 0
                color = "green" if r_val < 0.5 else (
                    "yellow" if r_val < 1.5 else "red")
                bar_len = min(int(r_val * 4), 20)
                console.print(
                    f"    zero {i:2d} ({z:7.2f}): residual={r_val:.4f}  "
                    f"[{color}]{'|' * max(bar_len, 1)}[/{color}]"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep theta graph optimization with CMA-ES",
    )
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--edges-per-path", type=int, default=3)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--n-zeros", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=15)
    parser.add_argument("--max-evals", type=int, default=40000)
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
    elif args.sweep:
        run_sweep(
            k_min=args.k_min, k_max=args.k_max,
            n_zeros=args.n_zeros, n_restarts=args.restarts,
            max_evals=args.max_evals,
        )
    else:
        optimize_theta(
            k_per_path=args.edges_per_path,
            n_zeros=args.n_zeros, n_restarts=args.restarts,
            max_evals=args.max_evals,
        )


if __name__ == "__main__":
    main()
