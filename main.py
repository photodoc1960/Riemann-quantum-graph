#!/usr/bin/env python3
"""Entry point for the Riemann quantum graph search system.

Usage:
    python main.py run [--population-size N] [--n-generations N] ...
    python main.py resume --checkpoint PATH
    python main.py analyze --graph PATH
    python main.py test
"""

from __future__ import annotations

# Set BLAS to single-threaded BEFORE numpy import.
# For 36×36 matrices, single-thread is faster (no thread-pool overhead).
# Frees all 16 cores for multiprocessing in differential_evolution.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import pickle
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Riemann quantum graph evolutionary search",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Fresh run with default or custom config")
    run_parser.add_argument("--population-size", type=int, default=200)
    run_parser.add_argument("--n-generations", type=int, default=100)
    run_parser.add_argument("--k-max", type=float, default=80.0)
    run_parser.add_argument("--n-zeros-compare", type=int, default=20)
    run_parser.add_argument("--top-fraction", type=float, default=0.10)
    run_parser.add_argument("--elite-fraction", type=float, default=0.05)
    run_parser.add_argument("--n-workers", type=int, default=-1)
    run_parser.add_argument("--min-vertices", type=int, default=3)
    run_parser.add_argument("--max-vertices", type=int, default=12)
    run_parser.add_argument("--min-edges", type=int, default=3)
    run_parser.add_argument("--max-edges", type=int, default=30)
    run_parser.add_argument("--prime-length-bias", type=float, default=0.7)
    run_parser.add_argument("--phase-breaking", action="store_true", default=True)
    run_parser.add_argument("--no-phase-breaking", dest="phase_breaking", action="store_false")
    run_parser.add_argument("--target-score", type=float, default=0.85)
    run_parser.add_argument("--checkpoint-every", type=int, default=10)
    run_parser.add_argument("--results-dir", type=str, default="results")
    run_parser.add_argument("--verbose", action="store_true", default=True)
    run_parser.add_argument("--quiet", dest="verbose", action="store_false")
    run_parser.add_argument("--random-seed", type=int, default=42)

    # --- resume ---
    resume_parser = subparsers.add_parser("resume", help="Resume from checkpoint")
    resume_parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint .pkl file",
    )

    # --- analyze ---
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a saved graph")
    analyze_parser.add_argument(
        "--graph", type=str, required=True,
        help="Path to pickled QuantumGraph or CheckpointState",
    )

    # --- trajectory ---
    traj_parser = subparsers.add_parser(
        "trajectory", help="Longitudinal analysis of search trajectory",
    )
    traj_parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory containing trajectory.jsonl",
    )

    # --- test ---
    subparsers.add_parser("test", help="Run pytest suite")

    return parser


def _config_from_args(args: argparse.Namespace) -> "SearchConfig":
    from riemann_qg.agents.orchestrator import SearchConfig

    return SearchConfig(
        population_size=args.population_size,
        n_generations=args.n_generations,
        k_max=args.k_max,
        n_zeros_compare=args.n_zeros_compare,
        top_fraction=args.top_fraction,
        elite_fraction=args.elite_fraction,
        n_workers=args.n_workers,
        min_vertices=args.min_vertices,
        max_vertices=args.max_vertices,
        min_edges=args.min_edges,
        max_edges=args.max_edges,
        prime_length_bias=args.prime_length_bias,
        phase_breaking=args.phase_breaking,
        target_score=args.target_score,
        checkpoint_every=args.checkpoint_every,
        results_dir=args.results_dir,
        verbose=args.verbose,
        random_seed=args.random_seed,
    )


def _cmd_run(args: argparse.Namespace) -> None:
    from riemann_qg.agents.orchestrator import Orchestrator

    config = _config_from_args(args)
    orchestrator = Orchestrator()

    try:
        final_state = orchestrator.run(config)
    except KeyboardInterrupt:
        print("\nInterrupted -- saving emergency checkpoint ...")
        # The orchestrator saves periodic checkpoints; print what we know.
        print(
            f"Check {config.results_dir}/ for the latest checkpoint. "
            "Re-run with 'resume --checkpoint <path>' to continue."
        )
        sys.exit(130)

    print(f"\nSearch complete. Best score: {final_state.best_score:.4f}")
    if final_state.best_graph is not None:
        out_path = Path(config.results_dir) / "best_graph.pkl"
        with open(out_path, "wb") as fh:
            pickle.dump(final_state.best_graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Best graph saved to {out_path}")


def _cmd_resume(args: argparse.Namespace) -> None:
    from riemann_qg.agents.orchestrator import Orchestrator

    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    orchestrator = Orchestrator()
    try:
        final_state = orchestrator.resume(checkpoint_path)
    except KeyboardInterrupt:
        print("\nInterrupted during resume. Check results/ for checkpoints.")
        sys.exit(130)

    print(f"\nResumed search complete. Best score: {final_state.best_score:.4f}")


def _cmd_analyze(args: argparse.Namespace) -> None:
    from riemann_qg.agents.orchestrator import CheckpointState, Orchestrator
    from riemann_qg.agents.pattern_agent import PatternAgent, PopulationInsights
    from riemann_qg.agents.spectral_agent import SpectralAgent
    from riemann_qg.agents.symbolic_agent import SymbolicAgent
    from riemann_qg.core.quantum_graph import QuantumGraph

    graph_path = args.graph
    if not Path(graph_path).exists():
        print(f"File not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    with open(graph_path, "rb") as fh:
        obj = pickle.load(fh)

    if isinstance(obj, CheckpointState):
        graph = obj.best_graph
        if graph is None:
            print("Checkpoint has no best_graph.", file=sys.stderr)
            sys.exit(1)
    elif isinstance(obj, QuantumGraph):
        graph = obj
    else:
        print(f"Unexpected object type in file: {type(obj)}", file=sys.stderr)
        sys.exit(1)

    print(f"Graph: {graph.n_vertices} vertices, {graph.n_edges} edges")
    print(f"Edge lengths: {graph.edge_lengths}")

    spectral = SpectralAgent()
    result = spectral.evaluate_graph(graph, k_max=80.0, n_zeros_compare=20)
    print(f"Score: {result.scoring_result.total_score:.4f}")
    print(f"Spectrum ({len(result.spectrum)} eigenvalues): {result.spectrum[:10]}...")

    symbolic = SymbolicAgent()
    fits = symbolic.fit_prime_combination(list(graph.edge_lengths), n_primes=15)
    exact_count = sum(1 for f in fits if f.is_exact)
    print(f"Symbolic: {exact_count}/{len(fits)} edges are exact log-prime combinations")

    for i, fit in enumerate(fits):
        status = "EXACT" if fit.is_exact else f"residual={fit.residual:.6f}"
        print(f"  Edge {i}: L={fit.length:.6f}  [{status}]  coeffs={fit.coefficients}")


def _cmd_trajectory(args: argparse.Namespace) -> None:
    """Longitudinal analysis: what the population learned, generation by generation."""
    import json
    from pathlib import Path

    traj_path = Path(args.results_dir) / "trajectory.jsonl"
    if not traj_path.exists():
        print(f"No trajectory log found at {traj_path}", file=sys.stderr)
        print("Run a search first, or check --results-dir.", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(traj_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("Trajectory log is empty.")
        sys.exit(0)

    print(f"{'='*72}")
    print(f"LONGITUDINAL TRAJECTORY ANALYSIS ({len(records)} generations)")
    print(f"{'='*72}\n")

    # --- Phase detection: find inflection points in score ---
    scores = [r["best_score"] for r in records]
    phases: list[tuple[int, int, str]] = []
    phase_start = 0
    prev_score = scores[0]

    for i in range(1, len(scores)):
        # Detect plateaus (no improvement for 5+ gens) or jumps (>0.02 improvement)
        if i - phase_start >= 5 and scores[i] - scores[phase_start] < 0.005:
            phases.append((phase_start, i, "plateau"))
            phase_start = i
        elif scores[i] - prev_score > 0.02:
            phases.append((phase_start, i, "climbing"))
            phase_start = i
        prev_score = scores[i]
    phases.append((phase_start, len(scores) - 1, "final"))

    # --- Score trajectory ---
    print("SCORE TRAJECTORY")
    print("-" * 40)
    for r in records:
        gen = r["generation"]
        bar_len = int(r["best_score"] * 50)
        bar = "#" * bar_len + "." * (50 - bar_len)
        marker = ""
        if gen == 0:
            marker = " <-- start"
        elif r["best_score"] == max(scores):
            marker = " <-- best"
        if gen % 5 == 0 or marker:
            print(f"  Gen {gen:3d}: [{bar}] {r['best_score']:.4f}{marker}")
    print()

    # --- Topology evolution ---
    print("TOPOLOGY EVOLUTION")
    print("-" * 40)
    prev_topo = None
    for r in records:
        topo = f"{r['best_n_vertices']}v/{r['best_n_edges']}e"
        if topo != prev_topo:
            print(
                f"  Gen {r['generation']:3d}: topology shifted to {topo} "
                f"(score {r['best_score']:.4f}, "
                f"{r['n_unique_topologies']} unique topologies in pop)"
            )
            prev_topo = topo
    print()

    # --- Edge length story ---
    print("EDGE LENGTH EVOLUTION")
    print("-" * 40)
    for r in records:
        gen = r["generation"]
        if gen % 10 == 0 or gen == len(records) - 1:
            print(
                f"  Gen {gen:3d}: alpha={r['best_alpha']:.3f}, "
                f"corr(ln p)={r['correlation_with_log_primes']:.3f}, "
                f"mean dist to ln(p)={r['mean_nearest_prime_log_dist']:.4f}, "
                f"prime overlap={r['mean_prime_overlap']:.3f}"
            )
    print()

    # --- Scattering matrix evolution ---
    print("SCATTERING MATRIX EVOLUTION")
    print("-" * 40)
    for r in records:
        gen = r["generation"]
        if gen % 10 == 0 or gen == len(records) - 1:
            print(
                f"  Gen {gen:3d}: {r['fraction_near_neumann']:.0%} near-Neumann, "
                f"mean phase={r['mean_phase']:.4f}"
            )
    print()

    # --- Symbolic rules discovered ---
    print("SYMBOLIC RULES DISCOVERED")
    print("-" * 40)
    prev_formula = ""
    for r in records:
        formula = r.get("symbolic_formula", "")
        residual = r.get("symbolic_residual", float("inf"))
        if formula and formula != prev_formula and residual < 1.0:
            print(
                f"  Gen {r['generation']:3d}: {formula} "
                f"(residual {residual:.4f})"
            )
            prev_formula = formula
    print()

    # --- Diversity trajectory ---
    print("DIVERSITY")
    print("-" * 40)
    for r in records:
        gen = r["generation"]
        if gen % 10 == 0 or gen == len(records) - 1:
            print(
                f"  Gen {gen:3d}: {r['n_unique_topologies']} unique topologies, "
                f"score std={r['std_score']:.4f}"
            )
    print()

    # --- Summary narrative ---
    first = records[0]
    last = records[-1]
    best_gen = max(records, key=lambda r: r["best_score"])

    print("NARRATIVE SUMMARY")
    print("=" * 40)
    print(
        f"The search ran for {len(records)} generations.\n"
        f"\n"
        f"Starting point (gen 0): score {first['best_score']:.4f}, "
        f"{first['best_n_vertices']}v/{first['best_n_edges']}e, "
        f"alpha={first['best_alpha']:.3f}, "
        f"log-prime correlation={first['correlation_with_log_primes']:.3f}\n"
        f"\n"
        f"Peak (gen {best_gen['generation']}): score {best_gen['best_score']:.4f}, "
        f"{best_gen['best_n_vertices']}v/{best_gen['best_n_edges']}e, "
        f"alpha={best_gen['best_alpha']:.3f}, "
        f"log-prime correlation={best_gen['correlation_with_log_primes']:.3f}\n"
        f"\n"
        f"Final (gen {last['generation']}): score {last['best_score']:.4f}, "
        f"{last['best_n_vertices']}v/{last['best_n_edges']}e, "
        f"alpha={last['best_alpha']:.3f}, "
        f"log-prime correlation={last['correlation_with_log_primes']:.3f}"
    )

    score_delta = last["best_score"] - first["best_score"]
    corr_delta = last["correlation_with_log_primes"] - first["correlation_with_log_primes"]
    print(f"\nScore improved by {score_delta:+.4f}")
    print(f"Log-prime correlation changed by {corr_delta:+.3f}")

    if last["n_unique_topologies"] <= 3:
        print("\nWARNING: Population diversity collapsed. Consider increasing fresh injection rate.")
    if last["correlation_with_log_primes"] > 0.8:
        print("\nNOTE: Strong log-prime correlation in edge lengths — prime encoding is emerging.")
    if last["fraction_near_neumann"] < 0.5:
        print("\nNOTE: Population has moved away from Neumann scattering — TRS breaking is active.")

    print(f"\nBest graph edge lengths: {best_gen['best_edge_lengths']}")
    print()


def _cmd_test(_args: argparse.Namespace) -> None:
    import pytest

    sys.exit(pytest.main(["-v", "riemann_qg/tests/"]))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    commands = {
        "run": _cmd_run,
        "resume": _cmd_resume,
        "analyze": _cmd_analyze,
        "trajectory": _cmd_trajectory,
        "test": _cmd_test,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
