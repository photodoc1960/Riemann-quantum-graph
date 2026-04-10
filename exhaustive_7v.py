#!/usr/bin/env python3
"""Exhaustive search over all 853 non-isomorphic connected 7-vertex graphs.

For each topology, jointly optimizes edge lengths and scattering phases
to maximize spectral match against the first 20 Riemann zeta zeros.

Results are saved incrementally to results/exhaustive_7v.jsonl so the
search can be interrupted and resumed.

Usage:
    python exhaustive_7v.py [--workers 10] [--max-iter 50]
    python exhaustive_7v.py --analyze   # print summary of results
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

import networkx as nx
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from riemann_qg.core.quantum_graph import QuantumGraph
from riemann_qg.core.scoring import SpectralScorer, _weyl_rescale
from riemann_qg.search.optimizer import joint_optimize


RESULTS_PATH = Path("results/exhaustive_7v.jsonl")
N_VERTICES = 7
N_ZEROS = 20
console = Console()


def _all_connected_7v_graphs() -> list[nx.Graph]:
    """Return all 853 non-isomorphic connected graphs on 7 vertices."""
    all_graphs = nx.graph_atlas_g()
    return [
        g for g in all_graphs
        if g.number_of_nodes() == N_VERTICES and nx.is_connected(g)
    ]


def _graph_invariants(g: nx.Graph) -> dict:
    """Compute topological invariants for analysis."""
    degrees = sorted(dict(g.degree()).values(), reverse=True)
    n_edges = g.number_of_edges()
    return {
        "n_edges": n_edges,
        "degree_sequence": degrees,
        "max_degree": degrees[0],
        "min_degree": degrees[-1],
        "betti_number": n_edges - N_VERTICES + 1,
        "is_regular": len(set(degrees)) == 1,
        "regularity": degrees[0] if len(set(degrees)) == 1 else -1,
        "diameter": nx.diameter(g),
        "girth": _girth(g),
        "algebraic_connectivity": float(
            sorted(nx.laplacian_spectrum(g))[1]
        ),
        "n_triangles": sum(nx.triangles(g).values()) // 3,
        "is_bipartite": nx.is_bipartite(g),
        "is_planar": nx.is_planar(g),
        "vertex_connectivity": nx.node_connectivity(g),
        "edge_connectivity": nx.edge_connectivity(g),
    }


def _girth(g: nx.Graph) -> int:
    """Shortest cycle length in the graph."""
    min_cycle = float("inf")
    for node in g.nodes():
        # BFS from node, looking for back-edges
        dist: dict[int, int] = {node: 0}
        queue = [node]
        parent: dict[int, int | None] = {node: None}
        while queue:
            v = queue.pop(0)
            for w in g.neighbors(v):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    queue.append(w)
                elif parent[v] != w:
                    cycle_len = dist[v] + dist[w] + 1
                    min_cycle = min(min_cycle, cycle_len)
    return int(min_cycle) if min_cycle < float("inf") else 0


def _nx_to_quantum_graph(
    g: nx.Graph,
    length_mode: str = "ln_prime",
    seed: int = 0,
) -> QuantumGraph:
    """Convert networkx graph to QuantumGraph with initial edge lengths."""
    adj = nx.to_numpy_array(g, dtype=np.int64)
    n_edges = g.number_of_edges()

    if length_mode == "ln_prime":
        # Use ln(p) for primes 2, 3, 5, 7, 11, ...
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                  31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        lengths = [float(np.log(primes[i % len(primes)])) for i in range(n_edges)]
    elif length_mode == "random":
        rng = np.random.default_rng(seed)
        lengths = list(rng.uniform(0.5, 5.0, size=n_edges))
    else:
        lengths = [1.0] * n_edges

    return QuantumGraph.from_adjacency(adj, edge_lengths=lengths)


def _already_completed() -> set[int]:
    """Read completed graph indices from results file."""
    completed = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        completed.add(record["graph_index"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


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

    scaled, _ = _weyl_rescale(np.sort(spectrum), total_length, zeros)

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
        "n_matched": n_match,
    }


def _optimize_single_graph(
    idx: int,
    g: nx.Graph,
    scorer: SpectralScorer,
    zeros: np.ndarray,
    max_iter: int,
    workers: int,
) -> dict:
    """Optimize one graph and return full result record."""
    invariants = _graph_invariants(g)

    # Try two initial length strategies, keep the best
    best_score = -1.0
    best_result = None
    best_qg = None

    for mode_idx, mode in enumerate(["ln_prime", "random"]):
        qg = _nx_to_quantum_graph(g, length_mode=mode, seed=idx * 100 + mode_idx)

        try:
            result = joint_optimize(
                qg, scorer,
                riemann_zeros=zeros,
                n_compare=N_ZEROS,
                max_iter=max_iter,
                seed=idx * 1000 + mode_idx,
                workers=workers,
            )
            if result.score > best_score:
                best_score = result.score
                best_result = result
                best_qg = qg.with_edge_lengths(list(result.edge_lengths))
                if result.phases:
                    best_qg = best_qg.with_scattering_phases(list(result.phases))
        except Exception as exc:
            console.print(f"  [red]Graph {idx} mode={mode} failed: {exc}[/red]")
            continue

    if best_result is None or best_qg is None:
        return {
            "graph_index": idx,
            "score": 0.0,
            "error": "optimization_failed",
            **invariants,
        }

    # Compute per-zero residuals for the optimized graph
    residual_info = _compute_residuals(best_qg, scorer)

    return {
        "graph_index": idx,
        "score": best_result.score,
        "n_evaluations": best_result.n_evaluations,
        "converged": best_result.converged,
        "edge_lengths": list(best_result.edge_lengths),
        "phases": list(best_result.phases),
        **residual_info,
        **invariants,
    }


def run_exhaustive(max_iter: int = 50, workers: int = 10) -> None:
    """Run the exhaustive search over all 7-vertex connected graphs."""
    graphs = _all_connected_7v_graphs()
    completed = _already_completed()

    console.print(f"[bold]Exhaustive 7-vertex search[/bold]")
    console.print(f"  Total topologies: {len(graphs)}")
    console.print(f"  Already completed: {len(completed)}")
    console.print(f"  Remaining: {len(graphs) - len(completed)}")
    console.print(f"  Workers: {workers}, max_iter: {max_iter}")
    console.print()

    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(N_ZEROS)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_overall_score = 0.0
    best_overall_idx = -1
    t_start = time.time()

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Optimizing...", total=len(graphs) - len(completed))

        for idx, g in enumerate(graphs):
            if idx in completed:
                continue

            t0 = time.time()
            record = _optimize_single_graph(
                idx, g, scorer, zeros, max_iter, workers,
            )
            elapsed = time.time() - t0

            # Save incrementally
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")

            score = record.get("score", 0.0)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_idx = idx

            n_edges = record.get("n_edges", "?")
            mad = record.get("mad", 99.0)
            progress.update(task, advance=1)
            console.print(
                f"  Graph {idx:3d} ({n_edges:2}e): "
                f"score={score:.4f}  MAD={mad:.3f}  "
                f"[dim]{elapsed:.0f}s[/dim]"
                + (f"  [green]<-- new best![/green]" if idx == best_overall_idx and score > 0 else "")
            )

            # Thermal cooldown between graphs
            time.sleep(3)

    total_time = time.time() - t_start
    console.print(f"\n[bold green]Complete![/bold green]")
    console.print(f"  Total time: {total_time/3600:.1f} hours")
    console.print(f"  Best score: {best_overall_score:.4f} (graph {best_overall_idx})")


def analyze_results() -> None:
    """Print analysis of exhaustive search results."""
    if not RESULTS_PATH.exists():
        console.print("[red]No results found. Run the search first.[/red]")
        return

    records = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    records.sort(key=lambda r: r.get("score", 0), reverse=True)

    console.print(f"\n[bold]Exhaustive 7-Vertex Search Results[/bold]")
    console.print(f"  Total graphs evaluated: {len(records)}")
    console.print()

    # Top 20 table
    table = Table(title="Top 20 Topologies")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Idx", width=4)
    table.add_column("Score", style="green", width=8)
    table.add_column("Edges", width=5)
    table.add_column("Betti", width=5)
    table.add_column("Degrees", width=20)
    table.add_column("MAD", width=8)
    table.add_column("Max Res", width=8)
    table.add_column("Reg?", width=4)
    table.add_column("Girth", width=5)
    table.add_column("Diam", width=4)

    for rank, r in enumerate(records[:20]):
        table.add_row(
            str(rank + 1),
            str(r.get("graph_index", "?")),
            f"{r.get('score', 0):.4f}",
            str(r.get("n_edges", "?")),
            str(r.get("betti_number", "?")),
            str(r.get("degree_sequence", "?")),
            f"{r.get('mad', 99):.3f}",
            f"{r.get('max_residual', 99):.3f}",
            "Y" if r.get("is_regular") else "N",
            str(r.get("girth", "?")),
            str(r.get("diameter", "?")),
        )
    console.print(table)

    # Statistical summary by topological feature
    console.print("\n[bold]Score Distribution by Topological Feature[/bold]\n")

    # By edge count
    edge_groups: dict[int, list[float]] = {}
    for r in records:
        ne = r.get("n_edges", 0)
        s = r.get("score", 0)
        edge_groups.setdefault(ne, []).append(s)

    table2 = Table(title="Score by Edge Count")
    table2.add_column("Edges", width=6)
    table2.add_column("Count", width=6)
    table2.add_column("Mean Score", width=10)
    table2.add_column("Max Score", width=10)
    table2.add_column("Best in Top 20?", width=14)

    top20_edges = {r.get("n_edges") for r in records[:20]}
    for ne in sorted(edge_groups):
        scores_list = edge_groups[ne]
        in_top = "***" if ne in top20_edges else ""
        table2.add_row(
            str(ne),
            str(len(scores_list)),
            f"{np.mean(scores_list):.4f}",
            f"{np.max(scores_list):.4f}",
            in_top,
        )
    console.print(table2)

    # Regular vs irregular
    reg_scores = [r["score"] for r in records if r.get("is_regular")]
    irreg_scores = [r["score"] for r in records if not r.get("is_regular")]
    console.print(f"\n  Regular graphs: n={len(reg_scores)}, "
                  f"mean={np.mean(reg_scores):.4f}, max={np.max(reg_scores):.4f}" if reg_scores else "")
    console.print(f"  Irregular graphs: n={len(irreg_scores)}, "
                  f"mean={np.mean(irreg_scores):.4f}, max={np.max(irreg_scores):.4f}" if irreg_scores else "")

    # Bipartite vs non-bipartite
    bip = [r["score"] for r in records if r.get("is_bipartite")]
    nonbip = [r["score"] for r in records if not r.get("is_bipartite")]
    console.print(f"  Bipartite: n={len(bip)}, "
                  f"mean={np.mean(bip):.4f}, max={np.max(bip):.4f}" if bip else "")
    console.print(f"  Non-bipartite: n={len(nonbip)}, "
                  f"mean={np.mean(nonbip):.4f}, max={np.max(nonbip):.4f}" if nonbip else "")

    # By Betti number
    betti_groups: dict[int, list[float]] = {}
    for r in records:
        b = r.get("betti_number", 0)
        betti_groups.setdefault(b, []).append(r.get("score", 0))

    console.print(f"\n  [bold]By Betti number (independent cycles):[/bold]")
    for b in sorted(betti_groups):
        scores_list = betti_groups[b]
        console.print(f"    beta_1={b}: n={len(scores_list)}, "
                      f"mean={np.mean(scores_list):.4f}, "
                      f"max={np.max(scores_list):.4f}")

    # Edge lengths of top graph
    if records:
        top = records[0]
        console.print(f"\n[bold]Best Graph Details[/bold]")
        console.print(f"  Index: {top.get('graph_index')}")
        console.print(f"  Score: {top.get('score', 0):.4f}")
        console.print(f"  Topology: {top.get('n_edges')} edges, "
                      f"degrees={top.get('degree_sequence')}")
        console.print(f"  Betti number: {top.get('betti_number')}")
        console.print(f"  Girth: {top.get('girth')}, Diameter: {top.get('diameter')}")

        lengths = top.get("edge_lengths", [])
        if lengths:
            console.print(f"  Edge lengths: {[f'{l:.4f}' for l in lengths]}")
            # Check proximity to ln(p)
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            ln_primes = [np.log(p) for p in primes]
            console.print(f"  ln(p) comparison:")
            for i, l in enumerate(lengths):
                dists = [(abs(l - lp), p) for lp, p in zip(ln_primes, primes)]
                best_dist, best_p = min(dists, key=lambda x: x[0])
                marker = " <-- MATCH" if best_dist < 0.05 else ""
                console.print(
                    f"    edge {i}: L={l:.4f}, nearest ln({best_p})="
                    f"{np.log(best_p):.4f}, dist={best_dist:.4f}{marker}"
                )

        residuals = top.get("residuals", [])
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exhaustive search over all 7-vertex quantum graphs",
    )
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results instead of running search")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers for DE (default: 10)")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Max DE iterations per graph (default: 50)")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
    else:
        run_exhaustive(max_iter=args.max_iter, workers=args.workers)


if __name__ == "__main__":
    main()
