#!/usr/bin/env python3
"""Deep optimization of cycle graph C_n using analytical insights.

Key insight: the spectrum of C_n depends on only n+1 parameters:
  - L_total = sum(edge_lengths)   [1 scalar]
  - theta_1, ..., theta_n         [n individual phases]
Individual edge lengths are irrelevant — only their sum matters.

This script optimizes over the reduced (n+1)-dimensional space using
CMA-ES (state-of-the-art for continuous optimization) followed by
L-BFGS-B local polishing.  Matches against 50 zeros instead of 20.

Usage:
    python cycle_deep.py --n 7 [--n-zeros 50] [--restarts 20] [--workers 10]
    python cycle_deep.py --sweep --n-min 7 --n-max 50
    python cycle_deep.py --analyze
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


RESULTS_PATH = Path("results/cycle_deep.jsonl")
console = Console()


def _build_cycle(n: int, L_total: float, phases: np.ndarray) -> QuantumGraph:
    """Build C_n from the reduced parameter set: L_total + n phases."""
    adj = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1

    # Edge lengths are uniform (individual lengths don't matter)
    lengths = [L_total / n] * n
    scattering = [directed_scattering(2, float(p)) for p in phases]
    return QuantumGraph(adj, lengths, scattering)


def _objective(
    params: np.ndarray,
    n: int,
    scorer: SpectralScorer,
    zeros: np.ndarray,
    n_zeros: int,
) -> float:
    """Negative score for minimization. Params = [L_total, theta_1, ..., theta_n]."""
    L_total = params[0]
    phases = params[1:]

    if L_total < 1.0:
        return 1.0  # Penalize degenerate graphs

    try:
        qg = _build_cycle(n, L_total, phases)
        k_max = max((n_zeros + 5) * np.pi / L_total, 10.0)
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


def optimize_cycle(
    n: int,
    n_zeros: int = 50,
    n_restarts: int = 20,
    max_evals: int = 50000,
    workers: int = 10,
) -> dict:
    """Deep optimization of C_n using CMA-ES + L-BFGS-B polish."""
    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(n_zeros)
    n_params = n + 1  # L_total + n phases

    console.print(f"\n[bold cyan]Deep optimization: C_{n}[/bold cyan]")
    console.print(f"  Effective parameters: {n_params} "
                  f"(1 total length + {n} phases)")
    console.print(f"  Matching against {n_zeros} zeros")
    console.print(f"  Restarts: {n_restarts}, max evals/restart: {max_evals}")

    best_score = -1.0
    best_params = None
    best_record = None

    for restart in range(n_restarts):
        rng = np.random.default_rng(restart * 1000 + n)

        # Initial guess: L_total ~ 20-40, phases random in [0, 2pi)
        L_init = rng.uniform(15.0, 45.0)
        phases_init = rng.uniform(0, 2 * np.pi, size=n)
        x0 = np.concatenate([[L_init], phases_init])

        # Initial step sizes: ~5 for L_total, ~1 for phases
        sigma0 = 2.0

        t0 = time.time()

        # Phase 1: CMA-ES global search
        opts = cma.CMAOptions()
        opts.set("maxfevals", max_evals)
        opts.set("tolfun", 1e-8)
        opts.set("tolx", 1e-8)
        opts.set("verbose", -9)  # silent
        opts.set("bounds", [
            [1.0] + [0.0] * n,      # lower bounds
            [80.0] + [2*np.pi] * n,  # upper bounds
        ])
        # Scale: L has wider range than phases
        opts.set("CMA_stds", [5.0] + [1.0] * n)

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        while not es.stop():
            solutions = es.ask()
            fitnesses = [
                _objective(x, n, scorer, zeros, n_zeros)
                for x in solutions
            ]
            es.tell(solutions, fitnesses)

        cma_result = es.result
        cma_score = -cma_result.fbest
        cma_params = cma_result.xbest

        # Phase 2: Nelder-Mead local polish (gradient-free, won't overshoot)
        try:
            polish = minimize(
                _objective,
                cma_params,
                args=(n, scorer, zeros, n_zeros),
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10},
            )
            polished_score = -polish.fun
            # Only accept if polish actually improved
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

        # Build optimized graph and compute residuals
        qg = _build_cycle(n, final_params[0], final_params[1:])
        residual_info = _compute_residuals(qg, scorer, n_zeros)

        marker = ""
        if final_score > best_score:
            best_score = final_score
            best_params = final_params
            marker = "  [bold green]<-- BEST[/bold green]"

            best_record = {
                "n_vertices": n,
                "n_zeros": n_zeros,
                "restart": restart,
                "score": final_score,
                "cma_score": cma_score,
                "polished_score": final_score,
                "L_total": float(final_params[0]),
                "phases": list(final_params[1:]),
                "total_phase": float(np.sum(final_params[1:])),
                "n_cma_evals": int(cma_result.evaluations),
                "elapsed_s": elapsed,
                **residual_info,
            }

        console.print(
            f"  restart {restart:2d}: CMA={cma_score:.4f} -> "
            f"polish={final_score:.4f}  "
            f"MAD={residual_info.get('mad', 99):.3f}  "
            f"L={final_params[0]:.1f}  "
            f"[dim]{elapsed:.0f}s[/dim]{marker}"
        )

        # Save every result incrementally
        record = {
            "n_vertices": n,
            "n_zeros": n_zeros,
            "restart": restart,
            "score": final_score,
            "cma_score": cma_score,
            "L_total": float(final_params[0]),
            "phases": list(final_params[1:]),
            "total_phase": float(np.sum(final_params[1:])),
            "n_cma_evals": int(cma_result.evaluations),
            "elapsed_s": elapsed,
            **residual_info,
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Thermal cooldown
        time.sleep(3)

    console.print(f"\n  [bold]C_{n} best: {best_score:.4f}  "
                  f"MAD={best_record.get('mad', 99):.3f}  "
                  f"L={best_params[0]:.2f}[/bold]")

    return best_record


def run_sweep(
    n_min: int = 7,
    n_max: int = 50,
    n_zeros: int = 50,
    n_restarts: int = 10,
    max_evals: int = 30000,
    workers: int = 10,
) -> None:
    """Sweep C_n from n_min to n_max with deep optimization."""
    console.print(f"[bold]Deep Cycle Sweep: C_{n_min} to C_{n_max}[/bold]")
    console.print(f"  Zeros: {n_zeros}, Restarts: {n_restarts}")
    console.print()

    overall_best_score = 0.0
    overall_best_n = 0

    for n in range(n_min, n_max + 1):
        result = optimize_cycle(
            n, n_zeros=n_zeros, n_restarts=n_restarts,
            max_evals=max_evals, workers=workers,
        )

        if result and result.get("score", 0) > overall_best_score:
            overall_best_score = result["score"]
            overall_best_n = n
            console.print(
                f"  [bold green]NEW OVERALL BEST: C_{n} = "
                f"{overall_best_score:.4f}[/bold green]"
            )

    console.print(f"\n[bold green]Sweep complete![/bold green]")
    console.print(f"  Overall best: C_{overall_best_n} = {overall_best_score:.4f}")


def analyze_results() -> None:
    """Print analysis of deep optimization results."""
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
                    n = r["n_vertices"]
                    records.setdefault(n, []).append(r)
                except (json.JSONDecodeError, KeyError):
                    continue

    console.print(f"\n[bold]Deep Cycle Optimization Results[/bold]")

    # Summary table
    table = Table(title="Best Score by Cycle Size")
    table.add_column("C_n", style="cyan", width=5)
    table.add_column("DOF", width=5)
    table.add_column("Score", style="green", width=8)
    table.add_column("MAD", width=8)
    table.add_column("Max Res", width=8)
    table.add_column("L_total", width=8)
    table.add_column("Sum(θ)", width=8)
    table.add_column("Matched", width=7)
    table.add_column("Zeros", width=5)
    table.add_column("Runs", width=4)

    best_overall_score = 0
    best_overall_n = 0

    for n in sorted(records):
        recs = records[n]
        best = max(recs, key=lambda r: r.get("score", 0))
        score = best.get("score", 0)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_n = n

        table.add_row(
            str(n), str(n + 1),
            f"{score:.4f}",
            f"{best.get('mad', 99):.3f}",
            f"{best.get('max_residual', 99):.3f}",
            f"{best.get('L_total', 0):.1f}",
            f"{best.get('total_phase', 0):.2f}",
            f"{best.get('n_matched', 0)}/{best.get('n_zeros', 20)}",
            str(best.get("n_zeros", 20)),
            str(len(recs)),
        )
    console.print(table)

    console.print(f"\n  [bold]Overall best: C_{best_overall_n} = "
                  f"{best_overall_score:.4f}[/bold]")

    # Score vs n chart
    console.print(f"\n[bold]Score vs Cycle Size[/bold]")
    for n in sorted(records):
        best = max(records[n], key=lambda r: r.get("score", 0))
        score = best.get("score", 0)
        bar_len = int(score * 60)
        marker = " <-- BEST" if n == best_overall_n else ""
        console.print(f"  C_{n:2d}: [{'#' * bar_len:60s}] {score:.4f}{marker}")

    # Detailed best graph
    if best_overall_n in records:
        best = max(records[best_overall_n], key=lambda r: r.get("score", 0))
        console.print(f"\n[bold]Best Graph Details: C_{best_overall_n}[/bold]")
        console.print(f"  Score: {best['score']:.6f}")
        console.print(f"  L_total: {best.get('L_total', 0):.6f}")
        console.print(f"  Sum(phases): {best.get('total_phase', 0):.6f}")
        console.print(f"  N zeros matched: {best.get('n_matched', 0)}")

        phases = best.get("phases", [])
        if phases:
            console.print(f"  Phases ({len(phases)}):")
            for i, p in enumerate(phases):
                console.print(f"    θ_{i:2d} = {p:.6f}")

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
        description="Deep cycle graph optimization with CMA-ES",
    )
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep C_n from n-min to n-max")
    parser.add_argument("--n", type=int, default=7,
                        help="Cycle size for single optimization (default: 7)")
    parser.add_argument("--n-min", type=int, default=7)
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument("--n-zeros", type=int, default=50,
                        help="Number of Riemann zeros to match (default: 50)")
    parser.add_argument("--restarts", type=int, default=20,
                        help="Random restarts (default: 20)")
    parser.add_argument("--max-evals", type=int, default=50000,
                        help="Max CMA-ES evaluations per restart (default: 50000)")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
    elif args.sweep:
        run_sweep(
            n_min=args.n_min, n_max=args.n_max,
            n_zeros=args.n_zeros, n_restarts=args.restarts,
            max_evals=args.max_evals, workers=args.workers,
        )
    else:
        optimize_cycle(
            args.n, n_zeros=args.n_zeros, n_restarts=args.restarts,
            max_evals=args.max_evals, workers=args.workers,
        )


if __name__ == "__main__":
    main()
