#!/usr/bin/env python3
"""Experiment 2: Scaling law measurement.

Measures score ceiling as a function of graph size. Tests cycle graphs C_n
for n=3 through 20, plus theta/complete/random graphs at n=7.
Neumann scattering fixed throughout — size is the only variable.

Results: results/scaling_results.csv, results/scaling_plot.png,
         results/mad_by_zero.png
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "sequential"

import csv
import json
import sys
import time
from pathlib import Path

import cma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riemann_qg.core.quantum_graph import QuantumGraph, neumann_scattering
from riemann_qg.core.scoring import SpectralScorer, _weyl_rescale


N_ZEROS = 20
RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "scaling_results.csv"
LOG_PATH = RESULTS_DIR / "scaling_experiment.log"


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def _build_cycle(n: int, edge_lengths: list[float]) -> QuantumGraph:
    """Build C_n with Neumann scattering (no TRS breaking)."""
    adj = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1
    return QuantumGraph(adj, edge_lengths)


def _build_theta(path_lengths: list[float]) -> QuantumGraph:
    """Build theta graph: 2 hubs + 3 paths, each path = 1 edge."""
    # 5 vertices: 0,1 = hubs; 2,3,4 = midpoints
    adj = np.zeros((5, 5), dtype=np.int64)
    adj[0, 2] = adj[2, 0] = 1; adj[2, 1] = adj[1, 2] = 1
    adj[0, 3] = adj[3, 0] = 1; adj[3, 1] = adj[1, 3] = 1
    adj[0, 4] = adj[4, 0] = 1; adj[4, 1] = adj[1, 4] = 1
    # 6 edges: split each path length into 2 edges
    lengths = []
    for pl in path_lengths:
        lengths.extend([pl / 2, pl / 2])
    return QuantumGraph(adj, lengths)


def _build_complete(n: int, edge_lengths: list[float]) -> QuantumGraph:
    """Build complete graph K_n with Neumann scattering."""
    adj = np.ones((n, n), dtype=np.int64) - np.eye(n, dtype=np.int64)
    return QuantumGraph(adj, edge_lengths)


def _score_objective(
    lengths: np.ndarray,
    build_fn,
    build_args: dict,
    scorer: SpectralScorer,
    zeros: np.ndarray,
) -> float:
    """Generic negative score objective for edge length optimization."""
    if np.any(lengths < 0.01):
        return 1.0
    try:
        qg = build_fn(edge_lengths=list(lengths), **build_args)
        total_length = sum(qg.edge_lengths)
        k_max = max((N_ZEROS + 5) * np.pi / total_length, 10.0)
        spectrum = qg.compute_spectrum(k_max, n_points=2000)
        result = scorer.score_spectrum(
            spectrum, zeros, N_ZEROS,
            edge_lengths=qg.edge_lengths,
        )
        return -result.total_score
    except Exception:
        return 1.0


def _cycle_objective(lengths: np.ndarray, n: int, scorer, zeros) -> float:
    if np.any(lengths < 0.01):
        return 1.0
    try:
        qg = _build_cycle(n, list(lengths))
        total_length = sum(qg.edge_lengths)
        k_max = max((N_ZEROS + 5) * np.pi / total_length, 10.0)
        spectrum = qg.compute_spectrum(k_max, n_points=2000)
        result = scorer.score_spectrum(
            spectrum, zeros, N_ZEROS,
            edge_lengths=qg.edge_lengths,
        )
        return -result.total_score
    except Exception:
        return 1.0


def _generic_objective(lengths: np.ndarray, adj, scorer, zeros) -> float:
    if np.any(lengths < 0.01):
        return 1.0
    try:
        qg = QuantumGraph(adj, list(lengths))
        total_length = sum(qg.edge_lengths)
        k_max = max((N_ZEROS + 5) * np.pi / total_length, 10.0)
        spectrum = qg.compute_spectrum(k_max, n_points=2000)
        result = scorer.score_spectrum(
            spectrum, zeros, N_ZEROS,
            edge_lengths=qg.edge_lengths,
        )
        return -result.total_score
    except Exception:
        return 1.0


def optimize_graph(
    objective_fn,
    obj_args: tuple,
    n_dims: int,
    n_restarts: int,
    max_evals: int,
    length_bounds: tuple[float, float] = (0.05, 10.0),
) -> tuple[float, np.ndarray, int]:
    """CMA-ES optimization with restarts. Returns (best_score, best_params, total_evals)."""
    best_score = -1.0
    best_params = None
    total_evals = 0

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    for restart in range(n_restarts):
        rng = np.random.default_rng(restart * 1000 + n_dims)

        if restart == 0:
            x0 = np.array([np.log(primes[i % len(primes)]) for i in range(n_dims)])
        elif restart == 1:
            x0 = np.full(n_dims, 3.7)  # ~ln(40)
        else:
            x0 = rng.uniform(length_bounds[0], length_bounds[1], size=n_dims)

        opts = cma.CMAOptions()
        opts.set("maxfevals", max_evals)
        opts.set("tolfun", 1e-8)
        opts.set("verbose", -9)
        opts.set("bounds", [
            [length_bounds[0]] * n_dims,
            [length_bounds[1]] * n_dims,
        ])

        es = cma.CMAEvolutionStrategy(x0, 1.5, opts)
        while not es.stop():
            sols = es.ask()
            fits = [objective_fn(x, *obj_args) for x in sols]
            es.tell(sols, fits)

        score = -es.result.fbest
        total_evals += es.result.evaluations
        if score > best_score:
            best_score = score
            best_params = es.result.xbest

    return best_score, best_params, total_evals


def compute_detailed_residuals(
    qg: QuantumGraph,
    scorer: SpectralScorer,
) -> dict:
    """Compute per-zero residuals for detailed analysis."""
    zeros = scorer._rz.get_zeros(N_ZEROS)
    total_length = sum(qg.edge_lengths)
    k_max = max((N_ZEROS + 5) * np.pi / total_length, 10.0)
    spectrum = qg.compute_spectrum(k_max, n_points=2000)

    if len(spectrum) < 3:
        return {"residuals": [], "mad": 99.0, "n_matched_01": 0, "n_matched_05": 0}

    scaled, alpha = _weyl_rescale(np.sort(spectrum), total_length, zeros)

    zero_spacings = np.diff(zeros)
    margin = float(np.mean(zero_spacings)) if len(zero_spacings) > 0 else 1.0
    k_low = zeros[0] - margin
    k_high = zeros[-1] + margin
    windowed = scaled[(scaled >= k_low) & (scaled <= k_high)]

    n_match = min(len(windowed), len(zeros))
    if n_match < 3:
        return {"residuals": [], "mad": 99.0, "n_matched_01": 0, "n_matched_05": 0}

    residuals = np.array([abs(float(windowed[i] - zeros[i])) for i in range(n_match)])
    return {
        "residuals": list(residuals),
        "mad": float(np.mean(residuals)),
        "n_matched_01": int(np.sum(residuals < 0.1)),
        "n_matched_05": int(np.sum(residuals < 0.5)),
        "n_windowed": n_match,
    }


def run_scaling_experiment() -> None:
    """Run the full scaling experiment."""
    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(N_ZEROS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Clear log
    with open(LOG_PATH, "w") as f:
        f.write("")

    log("=" * 70)
    log("SCALING EXPERIMENT: Score Ceiling vs. Graph Size")
    log("=" * 70)
    log(f"N_ZEROS: {N_ZEROS}, Scattering: Neumann (fixed)")
    log(f"Results: {CSV_PATH}")
    log("")

    results = []

    # ---- Family 1: Cycle Graphs C_3 to C_20 ----
    log("FAMILY 1: Cycle Graphs C_3 to C_20")
    log("-" * 50)

    for n in range(3, 21):
        n_edges = n
        if n <= 10:
            max_evals, n_restarts = 15000, 5
        else:
            max_evals, n_restarts = 8000, 3

        t0 = time.time()
        best_score, best_params, total_evals = optimize_graph(
            _cycle_objective, (n, scorer, zeros),
            n_dims=n_edges,
            n_restarts=n_restarts,
            max_evals=max_evals,
        )
        elapsed = time.time() - t0

        # Compute detailed residuals
        qg = _build_cycle(n, list(best_params))
        details = compute_detailed_residuals(qg, scorer)

        row = {
            "graph_family": "cycle",
            "n_vertices": n,
            "n_edges": n_edges,
            "graph_type": f"C_{n}",
            "best_score": best_score,
            "best_MAD": details["mad"],
            "n_zeros_matched_01": details["n_matched_01"],
            "n_zeros_matched_05": details["n_matched_05"],
            "total_length": float(sum(best_params)),
            "edge_lengths": [float(x) for x in best_params],
            "init_strategy": "mixed",
            "n_evals": total_evals,
            "runtime_seconds": elapsed,
            "residuals": details.get("residuals", []),
        }
        results.append(row)

        log(f"  C_{n:2d}: score={best_score:.4f}  MAD={details['mad']:.3f}  "
            f"matched(±0.5)={details['n_matched_05']}/{N_ZEROS}  "
            f"L={sum(best_params):.1f}  {elapsed:.0f}s")

    # ---- Family 2: Selected comparisons at n=7 ----
    log("\nFAMILY 2: Topology Comparison at n=7")
    log("-" * 50)

    # C_7 (already done, include for completeness)
    c7_result = next(r for r in results if r["graph_type"] == "C_7")
    log(f"  C_7 (cycle):    score={c7_result['best_score']:.4f}  "
        f"(from Family 1)")

    # K_7 complete graph (21 edges)
    n_k7_edges = 7 * 6 // 2
    adj_k7 = np.ones((7, 7), dtype=np.int64) - np.eye(7, dtype=np.int64)

    t0 = time.time()
    best_k7, params_k7, evals_k7 = optimize_graph(
        _generic_objective, (adj_k7, scorer, zeros),
        n_dims=n_k7_edges, n_restarts=3, max_evals=8000,
    )
    elapsed_k7 = time.time() - t0
    qg_k7 = QuantumGraph(adj_k7, list(params_k7))
    det_k7 = compute_detailed_residuals(qg_k7, scorer)

    results.append({
        "graph_family": "complete", "n_vertices": 7, "n_edges": n_k7_edges,
        "graph_type": "K_7", "best_score": best_k7,
        "best_MAD": det_k7["mad"],
        "n_zeros_matched_01": det_k7["n_matched_01"],
        "n_zeros_matched_05": det_k7["n_matched_05"],
        "total_length": float(sum(params_k7)),
        "edge_lengths": [float(x) for x in params_k7],
        "init_strategy": "mixed", "n_evals": evals_k7,
        "runtime_seconds": elapsed_k7,
        "residuals": det_k7.get("residuals", []),
    })
    log(f"  K_7 (complete): score={best_k7:.4f}  MAD={det_k7['mad']:.3f}  "
        f"{elapsed_k7:.0f}s")

    # Theta graph (5v, 6e)
    def _theta_obj(lengths, scorer, zeros):
        if np.any(lengths < 0.01):
            return 1.0
        try:
            qg = _build_theta(list(lengths[:3]))
            total_length = sum(qg.edge_lengths)
            k_max = max((N_ZEROS + 5) * np.pi / total_length, 10.0)
            spectrum = qg.compute_spectrum(k_max, n_points=2000)
            result = scorer.score_spectrum(
                spectrum, zeros, N_ZEROS, edge_lengths=qg.edge_lengths)
            return -result.total_score
        except Exception:
            return 1.0

    t0 = time.time()
    best_theta, params_theta, evals_theta = optimize_graph(
        _theta_obj, (scorer, zeros),
        n_dims=3, n_restarts=5, max_evals=15000,
        length_bounds=(0.5, 20.0),
    )
    elapsed_theta = time.time() - t0
    qg_theta = _build_theta(list(params_theta[:3]))
    det_theta = compute_detailed_residuals(qg_theta, scorer)

    results.append({
        "graph_family": "theta", "n_vertices": 5, "n_edges": 6,
        "graph_type": "Theta_3path", "best_score": best_theta,
        "best_MAD": det_theta["mad"],
        "n_zeros_matched_01": det_theta["n_matched_01"],
        "n_zeros_matched_05": det_theta["n_matched_05"],
        "total_length": float(sum(qg_theta.edge_lengths)),
        "edge_lengths": [float(x) for x in qg_theta.edge_lengths],
        "init_strategy": "mixed", "n_evals": evals_theta,
        "runtime_seconds": elapsed_theta,
        "residuals": det_theta.get("residuals", []),
    })
    log(f"  Theta (3-path): score={best_theta:.4f}  MAD={det_theta['mad']:.3f}  "
        f"{elapsed_theta:.0f}s")

    # Petersen-like 3-regular on 7 vertices (not possible — Petersen is 10v;
    # use a random 3-regular on 8 vertices as proxy, or the best 7v 3-reg we can build)
    # 3-regular on 7v doesn't exist (sum of degrees must be even: 7*3=21 is odd)
    # Use 3-regular on 8 vertices instead
    try:
        g_reg = nx.random_regular_graph(3, 8, seed=42)
        adj_reg = nx.to_numpy_array(g_reg, dtype=np.int64)
        n_reg_edges = g_reg.number_of_edges()

        t0 = time.time()
        best_reg, params_reg, evals_reg = optimize_graph(
            _generic_objective, (adj_reg, scorer, zeros),
            n_dims=n_reg_edges, n_restarts=3, max_evals=8000,
        )
        elapsed_reg = time.time() - t0
        qg_reg = QuantumGraph(adj_reg, list(params_reg))
        det_reg = compute_detailed_residuals(qg_reg, scorer)

        results.append({
            "graph_family": "regular", "n_vertices": 8, "n_edges": n_reg_edges,
            "graph_type": "3-reg_8v", "best_score": best_reg,
            "best_MAD": det_reg["mad"],
            "n_zeros_matched_01": det_reg["n_matched_01"],
            "n_zeros_matched_05": det_reg["n_matched_05"],
            "total_length": float(sum(params_reg)),
            "edge_lengths": [float(x) for x in params_reg],
            "init_strategy": "mixed", "n_evals": evals_reg,
            "runtime_seconds": elapsed_reg,
            "residuals": det_reg.get("residuals", []),
        })
        log(f"  3-regular 8v:   score={best_reg:.4f}  MAD={det_reg['mad']:.3f}  "
            f"{elapsed_reg:.0f}s")
    except Exception as e:
        log(f"  3-regular 8v: FAILED ({e})")

    # 5 random connected graphs on 7 vertices
    for trial in range(5):
        g_rand = None
        for attempt in range(100):
            g_rand = nx.erdos_renyi_graph(7, 0.5, seed=trial * 100 + attempt)
            if nx.is_connected(g_rand):
                break
        if g_rand is None or not nx.is_connected(g_rand):
            continue

        adj_rand = nx.to_numpy_array(g_rand, dtype=np.int64)
        n_rand_edges = g_rand.number_of_edges()

        t0 = time.time()
        best_rand, params_rand, evals_rand = optimize_graph(
            _generic_objective, (adj_rand, scorer, zeros),
            n_dims=n_rand_edges, n_restarts=3, max_evals=8000,
        )
        elapsed_rand = time.time() - t0
        qg_rand = QuantumGraph(adj_rand, list(params_rand))
        det_rand = compute_detailed_residuals(qg_rand, scorer)

        results.append({
            "graph_family": "random", "n_vertices": 7, "n_edges": n_rand_edges,
            "graph_type": f"Random_7v_{trial}", "best_score": best_rand,
            "best_MAD": det_rand["mad"],
            "n_zeros_matched_01": det_rand["n_matched_01"],
            "n_zeros_matched_05": det_rand["n_matched_05"],
            "total_length": float(sum(params_rand)),
            "edge_lengths": [float(x) for x in params_rand],
            "init_strategy": "mixed", "n_evals": evals_rand,
            "runtime_seconds": elapsed_rand,
            "residuals": det_rand.get("residuals", []),
        })
        log(f"  Random 7v #{trial}: score={best_rand:.4f}  "
            f"edges={n_rand_edges}  MAD={det_rand['mad']:.3f}  {elapsed_rand:.0f}s")

    # ---- Family 3: Extended Cycles (multiples of 7) ----
    log("\nFAMILY 3: Extended Cycles (multiples of 7)")
    log("-" * 50)

    for n in [14, 21, 28]:
        n_edges = n
        t0 = time.time()
        best_ext, params_ext, evals_ext = optimize_graph(
            _cycle_objective, (n, scorer, zeros),
            n_dims=n_edges, n_restarts=3, max_evals=8000,
        )
        elapsed_ext = time.time() - t0
        qg_ext = _build_cycle(n, list(params_ext))
        det_ext = compute_detailed_residuals(qg_ext, scorer)

        results.append({
            "graph_family": "cycle_extended", "n_vertices": n, "n_edges": n_edges,
            "graph_type": f"C_{n}", "best_score": best_ext,
            "best_MAD": det_ext["mad"],
            "n_zeros_matched_01": det_ext["n_matched_01"],
            "n_zeros_matched_05": det_ext["n_matched_05"],
            "total_length": float(sum(params_ext)),
            "edge_lengths": [float(x) for x in params_ext],
            "init_strategy": "mixed", "n_evals": evals_ext,
            "runtime_seconds": elapsed_ext,
            "residuals": det_ext.get("residuals", []),
        })
        log(f"  C_{n:2d}: score={best_ext:.4f}  MAD={det_ext['mad']:.3f}  "
            f"{elapsed_ext:.0f}s")

    # ---- Save CSV ----
    csv_fields = [
        "graph_family", "n_vertices", "n_edges", "graph_type", "best_score",
        "best_MAD", "n_zeros_matched_01", "n_zeros_matched_05",
        "total_length", "init_strategy", "n_evals", "runtime_seconds",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    log(f"\nCSV saved to {CSV_PATH}")

    # ---- Save detailed JSONL (includes residuals) ----
    jsonl_path = RESULTS_DIR / "scaling_experiment.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log(f"Detailed JSONL saved to {jsonl_path}")

    # ---- Generate plots ----
    _generate_scaling_plot(results)
    _generate_mad_plot(results)

    # ---- Console summary ----
    _print_summary(results)


def _generate_scaling_plot(results: list[dict]) -> None:
    """Generate scaling_plot.png."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Cycle graphs (connected line)
    cycle_results = sorted(
        [r for r in results if r["graph_family"] == "cycle"],
        key=lambda r: r["n_edges"],
    )
    cycle_edges = [r["n_edges"] for r in cycle_results]
    cycle_scores = [r["best_score"] for r in cycle_results]
    ax.plot(cycle_edges, cycle_scores, "bo-", label="Cycle C_n", markersize=6, linewidth=2)

    # Extended cycles
    ext_results = [r for r in results if r["graph_family"] == "cycle_extended"]
    if ext_results:
        ext_edges = [r["n_edges"] for r in ext_results]
        ext_scores = [r["best_score"] for r in ext_results]
        ax.plot(ext_edges, ext_scores, "bs", label="Extended cycle", markersize=8)

    # Other topologies as scatter
    markers = {"complete": ("r^", "Complete K_n"), "theta": ("gD", "Theta"),
               "regular": ("mp", "3-regular"), "random": ("kx", "Random 7v")}
    for family, (marker_str, label) in markers.items():
        fam_results = [r for r in results if r["graph_family"] == family]
        if fam_results:
            edges = [r["n_edges"] for r in fam_results]
            scores = [r["best_score"] for r in fam_results]
            ax.plot(edges, scores, marker_str, label=label, markersize=8)

    # Reference lines
    ax.axhline(y=0.72, color="gray", linestyle="--", alpha=0.5, label="Current ceiling (0.72)")
    ax.axhline(y=0.80, color="orange", linestyle="--", alpha=0.5, label="Level 3 (0.80)")
    ax.axhline(y=0.955, color="green", linestyle="--", alpha=0.3, label="Self-comparison (0.955)")

    ax.set_xlabel("Number of Edges", fontsize=12)
    ax.set_ylabel("Best Score Achieved", fontsize=12)
    ax.set_title("Quantum Graph Spectral Ceiling vs. Graph Size", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.0)

    plt.tight_layout()
    path = RESULTS_DIR / "scaling_plot.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"Plot saved to {path}")


def _generate_mad_plot(results: list[dict]) -> None:
    """Generate mad_by_zero.png for selected graph sizes."""
    fig, ax = plt.subplots(figsize=(12, 7))

    target_sizes = [7, 10, 14, 18]
    colors = ["b", "r", "g", "m"]

    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(N_ZEROS)

    for n, color in zip(target_sizes, colors):
        # Find the cycle result for this n
        cycle_r = next(
            (r for r in results
             if r["graph_family"] in ("cycle", "cycle_extended")
             and r["n_vertices"] == n),
            None,
        )
        if cycle_r is None or not cycle_r.get("residuals"):
            continue

        residuals = cycle_r["residuals"]
        n_match = len(residuals)
        indices = list(range(1, n_match + 1))
        ax.plot(indices, residuals, f"{color}o-",
                label=f"C_{n} (score={cycle_r['best_score']:.3f})",
                markersize=4, linewidth=1.5)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="|residual| = 0.5")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="|residual| = 1.0")

    ax.set_xlabel("Zero Index", fontsize=12)
    ax.set_ylabel("|Mapped Eigenvalue - Zero|", fontsize=12)
    ax.set_title("Per-Zero Residuals by Graph Size (Neumann Scattering)", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = RESULTS_DIR / "mad_by_zero.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"Plot saved to {path}")


def _print_summary(results: list[dict]) -> None:
    """Print final summary with interpretive conclusion."""
    log("\n" + "=" * 70)
    log("              SCALING EXPERIMENT RESULTS")
    log("=" * 70)
    log(f"{'Graph':<16s} {'Edges':>6s} {'Score':>8s} {'MAD':>8s} {'Zeros matched (±0.5)':>22s}")
    log("-" * 62)

    for r in sorted(results, key=lambda x: (x["graph_family"], x["n_edges"])):
        log(f"{r['graph_type']:<16s} {r['n_edges']:>6d} {r['best_score']:>8.4f} "
            f"{r['best_MAD']:>8.3f}  {r['n_zeros_matched_05']:>4d} / {N_ZEROS}")

    # Trend analysis on cycle graphs
    cycle_results = sorted(
        [r for r in results if r["graph_family"] == "cycle"],
        key=lambda r: r["n_edges"],
    )
    if len(cycle_results) >= 5:
        edges = np.array([r["n_edges"] for r in cycle_results])
        scores = np.array([r["best_score"] for r in cycle_results])

        # Check for monotonic rise, plateau, or rise-then-plateau
        first_half = scores[:len(scores) // 2]
        second_half = scores[len(scores) // 2:]
        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)

        # Fit log trend
        log_edges = np.log(edges)
        coeffs = np.polyfit(log_edges, scores, 1)
        slope = coeffs[0]

        # Check plateau (std of last 5 scores)
        tail_std = np.std(scores[-5:])
        tail_mean = np.mean(scores[-5:])

        log("\n" + "-" * 62)

        if slope > 0.02 and tail_std > 0.01:
            # Still rising
            a, b = coeffs
            target_score = 0.85
            if a > 0:
                target_log_n = (target_score - b) / a
                target_n = int(np.exp(target_log_n))
            else:
                target_n = -1
            log(f"RESULT: Ceiling is size-dependent. Scaling law: "
                f"score ≈ {a:.4f}·log(n) + {b:.4f}.")
            if target_n > 0:
                log(f"Extrapolated graph size to reach 0.85: {target_n} edges.")
            log("Recommend extended search with larger graphs.")
        elif tail_std < 0.005 and abs(mean_second - mean_first) < 0.01:
            # Plateau
            log(f"RESULT: Ceiling is size-independent. Finite quantum graphs appear "
                f"to have a fundamental correspondence limit of approximately "
                f"{tail_mean:.3f} against the Riemann zeros under the current "
                f"framework. This is consistent with the quasi-periodicity hypothesis. "
                f"The limitation is structural, not topological.")
        else:
            # Rise then plateau
            # Find plateau onset
            diffs = np.abs(np.diff(scores))
            plateau_idx = len(scores) - 1
            for i in range(len(diffs) - 1, max(0, len(diffs) - 6), -1):
                if diffs[i] > 0.01:
                    plateau_idx = i + 1
                    break
            plateau_n = int(edges[plateau_idx])
            plateau_score = float(scores[plateau_idx:].mean())
            log(f"RESULT: Ceiling rises until n={plateau_n} then stabilizes at "
                f"{plateau_score:.3f}. The effective spectral complexity saturates "
                f"at this graph size. Diminishing returns above {plateau_n} edges.")

    total_time = sum(r["runtime_seconds"] for r in results)
    log(f"\nTotal runtime: {total_time / 3600:.1f} hours")
    log(f"Results saved to {CSV_PATH}")


if __name__ == "__main__":
    run_scaling_experiment()
