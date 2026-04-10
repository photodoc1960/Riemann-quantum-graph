#!/usr/bin/env python3
"""Sweep cycle graphs C_n for n=7..50 to find the optimal cycle size.

For each C_n, jointly optimizes n edge lengths and n scattering phases
to maximize spectral match against the first 20 Riemann zeta zeros.
Runs multiple random restarts per cycle size to avoid local optima.

Results are saved incrementally to results/cycle_sweep.jsonl so the
sweep can be interrupted and resumed.

Usage:
    python cycle_sweep.py [--n-min 7] [--n-max 50] [--restarts 5]
    python cycle_sweep.py --analyze
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

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from riemann_qg.core.quantum_graph import QuantumGraph
from riemann_qg.core.scoring import SpectralScorer, _weyl_rescale
from riemann_qg.search.optimizer import joint_optimize


RESULTS_PATH = Path("results/cycle_sweep.jsonl")
N_ZEROS = 20
console = Console()


def _make_cycle_graph(
    n: int,
    length_mode: str = "ln_prime",
    seed: int = 0,
) -> QuantumGraph:
    """Build a cycle graph C_n with initial edge lengths."""
    adj = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
              31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
              73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
              127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
              179, 181, 191, 193, 197, 199, 211, 223, 227, 229]

    if length_mode == "ln_prime":
        lengths = [float(np.log(primes[i % len(primes)])) for i in range(n)]
    elif length_mode == "random":
        rng = np.random.default_rng(seed)
        lengths = list(rng.uniform(0.5, 5.0, size=n))
    elif length_mode == "random_wide":
        rng = np.random.default_rng(seed)
        lengths = list(rng.uniform(0.1, 10.0, size=n))
    elif length_mode == "uniform":
        # Total length chosen so Weyl density gives ~20 eigenvalues in zero range
        # N(k) = L*k/pi, want ~25 eigenvalues up to k ~ 3.0 -> L ~ 25*pi/3 ~ 26
        total = 26.0
        lengths = [total / n] * n
    else:
        lengths = [1.0] * n

    return QuantumGraph(adj, lengths)


def _compute_residuals(
    qg: QuantumGraph,
    scorer: SpectralScorer,
) -> dict:
    """Compute per-zero residuals for a quantum graph."""
    zeros = scorer._rz.get_zeros(N_ZEROS)
    total_length = sum(qg.edge_lengths)
    k_max = max((N_ZEROS + 5) * np.pi / total_length, 10.0)
    spectrum = qg.compute_spectrum(k_max, n_points=2000)

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


def _already_completed() -> dict[int, list[dict]]:
    """Read completed results grouped by n."""
    completed: dict[int, list[dict]] = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        n = record["n_vertices"]
                        completed.setdefault(n, []).append(record)
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


def run_sweep(
    n_min: int = 7,
    n_max: int = 50,
    n_restarts: int = 5,
    max_iter: int = 200,
    workers: int = 10,
) -> None:
    """Run the C_n sweep from n_min to n_max."""
    completed = _already_completed()

    console.print(f"[bold]Cycle Graph Sweep: C_{n_min} to C_{n_max}[/bold]")
    console.print(f"  Restarts per size: {n_restarts}")
    console.print(f"  Max DE iterations: {max_iter}")
    console.print(f"  Workers: {workers}")
    console.print()

    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(N_ZEROS)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    init_modes = ["ln_prime", "random", "random_wide", "uniform"]

    # Track overall best across all cycle sizes
    all_existing = [r for recs in completed.values() for r in recs]
    overall_best_score = max(
        (r.get("score", 0) for r in all_existing), default=0,
    )
    overall_best_n = 0
    if all_existing:
        overall_best = max(all_existing, key=lambda r: r.get("score", 0))
        overall_best_n = overall_best.get("n_vertices", 0)

    for n in range(n_min, n_max + 1):
        existing = completed.get(n, [])
        n_done = len(existing)

        if n_done >= n_restarts:
            best = max(existing, key=lambda r: r.get("score", 0))
            console.print(
                f"  C_{n:2d}: already done ({n_done} runs), "
                f"best={best['score']:.4f}"
            )
            continue

        best_score = max((r.get("score", 0) for r in existing), default=0)
        best_record = None

        console.print(f"\n[bold cyan]Optimizing C_{n}[/bold cyan] "
                      f"({2*n} dims, {n_restarts - n_done} restarts remaining)")

        for restart in range(n_done, n_restarts):
            mode = init_modes[restart % len(init_modes)]
            seed = n * 10000 + restart

            qg = _make_cycle_graph(n, length_mode=mode, seed=seed)

            t0 = time.time()
            try:
                result = joint_optimize(
                    qg, scorer,
                    riemann_zeros=zeros,
                    n_compare=N_ZEROS,
                    max_iter=max_iter,
                    seed=seed,
                    workers=workers,
                )
            except Exception as exc:
                console.print(f"    restart {restart} ({mode}): [red]FAILED: {exc}[/red]")
                continue
            elapsed = time.time() - t0

            # Build optimized graph for residual computation
            opt_qg = qg.with_edge_lengths(list(result.edge_lengths))
            if result.phases:
                opt_qg = opt_qg.with_scattering_phases(list(result.phases))

            residual_info = _compute_residuals(opt_qg, scorer)

            record = {
                "n_vertices": n,
                "n_edges": n,
                "restart": restart,
                "init_mode": mode,
                "score": result.score,
                "n_evaluations": result.n_evaluations,
                "converged": result.converged,
                "edge_lengths": list(result.edge_lengths),
                "phases": list(result.phases),
                "total_length": sum(result.edge_lengths),
                "elapsed_s": elapsed,
                **residual_info,
            }

            # Save incrementally
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")

            marker = ""
            if result.score > best_score:
                best_score = result.score
                best_record = record
                if result.score > overall_best_score:
                    overall_best_score = result.score
                    overall_best_n = n
                    marker = "  [bold green]<-- OVERALL BEST![/bold green]"
                else:
                    marker = "  [green]<-- best C_{0}[/green]".format(n)

            console.print(
                f"    restart {restart} ({mode:12s}): "
                f"score={result.score:.4f}  MAD={residual_info.get('mad', 99):.3f}  "
                f"L={sum(result.edge_lengths):.1f}  "
                f"[dim]{elapsed:.0f}s[/dim]{marker}"
            )

            # Thermal cooldown
            time.sleep(3)

        if best_record:
            console.print(
                f"  [bold]C_{n} best: {best_score:.4f}  "
                f"MAD={best_record.get('mad', 99):.3f}[/bold]"
            )

    console.print(f"\n[bold green]Sweep complete![/bold green]")


def analyze_results() -> None:
    """Print analysis of cycle sweep results."""
    completed = _already_completed()

    if not completed:
        console.print("[red]No results found. Run the sweep first.[/red]")
        return

    console.print(f"\n[bold]Cycle Graph Sweep Results[/bold]")
    console.print(f"  Cycle sizes evaluated: {sorted(completed.keys())}")
    console.print()

    # Summary table: best score per cycle size
    table = Table(title="Best Score by Cycle Size")
    table.add_column("C_n", style="cyan", width=5)
    table.add_column("Dims", width=5)
    table.add_column("Score", style="green", width=8)
    table.add_column("MAD", width=8)
    table.add_column("Max Res", width=8)
    table.add_column("Std Res", width=8)
    table.add_column("L_total", width=8)
    table.add_column("Matched", width=7)
    table.add_column("Runs", width=4)
    table.add_column("Best Init", width=12)

    best_overall_score = 0
    best_overall_n = 0

    for n in sorted(completed):
        records = completed[n]
        best = max(records, key=lambda r: r.get("score", 0))
        score = best.get("score", 0)

        if score > best_overall_score:
            best_overall_score = score
            best_overall_n = n

        table.add_row(
            str(n),
            str(2 * n),
            f"{score:.4f}",
            f"{best.get('mad', 99):.3f}",
            f"{best.get('max_residual', 99):.3f}",
            f"{best.get('std_residual', 99):.3f}",
            f"{best.get('total_length', 0):.1f}",
            f"{best.get('n_matched', 0)}/{N_ZEROS}",
            str(len(records)),
            best.get("init_mode", "?"),
        )
    console.print(table)

    console.print(f"\n  [bold]Overall best: C_{best_overall_n} "
                  f"with score {best_overall_score:.4f}[/bold]")

    # Score trend
    console.print(f"\n[bold]Score vs Cycle Size[/bold]")
    ns = sorted(completed)
    for n in ns:
        best = max(completed[n], key=lambda r: r.get("score", 0))
        score = best.get("score", 0)
        bar_len = int(score * 60)
        bar = "#" * bar_len
        marker = " <-- BEST" if n == best_overall_n else ""
        console.print(f"  C_{n:2d}: [{bar:60s}] {score:.4f}{marker}")

    # Detailed breakdown of the overall best
    best_records = completed[best_overall_n]
    best = max(best_records, key=lambda r: r.get("score", 0))

    console.print(f"\n[bold]Best Graph: C_{best_overall_n}[/bold]")
    console.print(f"  Score: {best['score']:.4f}")
    console.print(f"  Total length: {best.get('total_length', 0):.4f}")
    console.print(f"  Init mode: {best.get('init_mode', '?')}")
    console.print(f"  Converged: {best.get('converged', '?')}")
    console.print(f"  DE evaluations: {best.get('n_evaluations', '?')}")

    lengths = best.get("edge_lengths", [])
    if lengths:
        console.print(f"  Edge lengths ({len(lengths)}):")
        for i, l in enumerate(lengths):
            console.print(f"    edge {i:2d}: {l:.6f}")

        console.print(f"\n  Length statistics:")
        console.print(f"    Mean: {np.mean(lengths):.4f}")
        console.print(f"    Std:  {np.std(lengths):.4f}")
        console.print(f"    CV:   {np.std(lengths)/np.mean(lengths):.4f}")
        console.print(f"    Min:  {min(lengths):.4f}")
        console.print(f"    Max:  {max(lengths):.4f}")

        # ln(p) proximity
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                  53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        ln_primes = [np.log(p) for p in primes]
        n_match = 0
        console.print(f"\n  ln(p) proximity:")
        for i, l in enumerate(lengths):
            dists = [(abs(l - lp), p) for lp, p in zip(ln_primes, primes)]
            best_dist, best_p = min(dists, key=lambda x: x[0])
            match = " <-- MATCH" if best_dist < 0.05 else ""
            if best_dist < 0.05:
                n_match += 1
            console.print(
                f"    edge {i:2d}: L={l:.4f}, nearest ln({best_p})="
                f"{np.log(best_p):.4f}, dist={best_dist:.4f}{match}"
            )
        console.print(f"  ln(p) matches: {n_match}/{len(lengths)}")

    # Residuals
    residuals = best.get("residuals", [])
    if residuals:
        scorer = SpectralScorer()
        zeros = scorer._rz.get_zeros(N_ZEROS)
        console.print(f"\n  Per-zero residuals:")
        for i, r_val in enumerate(residuals):
            z = zeros[i] if i < len(zeros) else 0
            bar_len = min(int(r_val * 4), 20)
            color = "green" if r_val < 0.5 else ("yellow" if r_val < 1.5 else "red")
            console.print(
                f"    zero {i:2d} ({z:6.2f}): "
                f"residual={r_val:.3f}  [{color}]{'|' * max(bar_len, 1)}[/{color}]"
            )

    # All restarts for the best n
    console.print(f"\n  All restarts for C_{best_overall_n}:")
    for r in sorted(best_records, key=lambda x: x.get("score", 0), reverse=True):
        console.print(
            f"    {r.get('init_mode', '?'):12s}: score={r['score']:.4f}  "
            f"MAD={r.get('mad', 99):.3f}  L={r.get('total_length', 0):.1f}  "
            f"evals={r.get('n_evaluations', '?')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep cycle graphs C_n for optimal Riemann zero matching",
    )
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results")
    parser.add_argument("--n-min", type=int, default=7,
                        help="Minimum cycle size (default: 7)")
    parser.add_argument("--n-max", type=int, default=50,
                        help="Maximum cycle size (default: 50)")
    parser.add_argument("--restarts", type=int, default=5,
                        help="Random restarts per cycle size (default: 5)")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Max DE iterations per restart (default: 200)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers for DE (default: 10)")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
    else:
        run_sweep(
            n_min=args.n_min,
            n_max=args.n_max,
            n_restarts=args.restarts,
            max_iter=args.max_iter,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
