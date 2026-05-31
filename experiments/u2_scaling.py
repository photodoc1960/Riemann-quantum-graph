#!/usr/bin/env python3
"""U(2) scaling experiment: score vs cycle size under optimized U(2) scattering.

Measures how spectral correspondence varies with cycle size n for C_n graphs
under full U(2) vertex scattering optimization.

Results: results/u2_scaling_results.json, results/u2_scaling_plot.png
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
import math
import sys
import time
from pathlib import Path

import cma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riemann_qg.core.quantum_graph import QuantumGraph
from riemann_qg.core.scoring import SpectralScorer, _weyl_rescale


# ---- Configuration ----
CYCLE_SIZES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
N_ZEROS = 100
N_RESTARTS = 15
N_EVALS = 100_000
RESULTS_FILE = Path("results/u2_scaling_results.json")
PLOT_FILE = Path("results/u2_scaling_plot.png")


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def _unitary_2x2(params: np.ndarray) -> np.ndarray:
    """Build a general U(2) matrix from 4 real parameters."""
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


def _compute_monodromy(scat_params: np.ndarray, n: int) -> list[float]:
    M = np.eye(2, dtype=np.complex128)
    for v in range(n):
        u = _unitary_2x2(scat_params[v*4:(v+1)*4])
        M = M @ u
    eigvals = np.linalg.eigvals(M)
    return sorted([float(np.angle(ev)) for ev in eigvals])


def _load_existing() -> dict:
    """Load existing results for checkpoint resumption."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"config": {}, "results": []}


def _save_results(data: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def estimate_runtime() -> tuple[int, int]:
    """Estimate runtime and adjust parameters if needed."""
    time_per_eval = 0.007
    total = 0
    log("Runtime estimates:")
    for n in CYCLE_SIZES:
        dims = 5 * n
        est = N_EVALS * N_RESTARTS * time_per_eval
        total += est
        log(f"  C_{n:2d} ({dims}D): ~{est/3600:.1f} hours")
    log(f"  Total: ~{total/3600:.1f} hours")

    n_restarts = N_RESTARTS
    n_evals = N_EVALS

    if total / 3600 > 20:
        n_restarts = 10
        n_evals = 75_000
        adjusted = n_evals * n_restarts * time_per_eval * len(CYCLE_SIZES)
        log(f"\n  ADJUSTED (>20h): restarts={n_restarts}, evals={n_evals}")
        log(f"  Adjusted total: ~{adjusted/3600:.1f} hours")

    log("")
    return n_restarts, n_evals


def run_experiment() -> None:
    """Run the full U(2) scaling experiment."""
    log("=" * 60)
    log("U(2) SCALING EXPERIMENT: Score vs. Cycle Size")
    log("=" * 60)
    log(f"Cycle sizes: {CYCLE_SIZES}")
    log(f"N zeros: {N_ZEROS}")
    log(f"Results: {RESULTS_FILE}")
    log("")

    n_restarts, n_evals = estimate_runtime()

    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(N_ZEROS)

    existing = _load_existing()
    completed_ns = {r["n"] for r in existing.get("results", [])}

    existing["config"] = {
        "cycle_sizes": CYCLE_SIZES,
        "n_zeros": N_ZEROS,
        "n_restarts": n_restarts,
        "n_evals": n_evals,
    }

    for n in CYCLE_SIZES:
        if n in completed_ns:
            prev = next(r for r in existing["results"] if r["n"] == n)
            log(f"C_{n:2d}: already done (score={prev['best_score']:.4f}), skipping")
            continue

        dims = 5 * n
        n_edge_params = n
        n_scat_params = 4 * n

        log(f"\nOptimizing C_{n} ({dims}D, {n_restarts} restarts, {n_evals} evals)...")
        t_start = time.time()

        best_score = -1.0
        best_params = None
        all_scores = []

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        for restart in range(n_restarts):
            rng = np.random.default_rng(n * 1000 + restart)

            if restart % 4 == 0:
                edge_init = np.array([np.log(primes[i % len(primes)]) for i in range(n)])
            elif restart % 4 == 1:
                edge_init = rng.uniform(0.5, 5.0, n)
            elif restart % 4 == 2:
                edge_init = rng.uniform(1.0, 8.0, n)
            else:
                L = rng.uniform(10.0, 30.0)
                edge_init = np.full(n, L / n)

            scat_init = rng.uniform(0, 2 * np.pi, n_scat_params)
            x0 = np.concatenate([edge_init, scat_init])

            lower = [0.05] * n + [-10.0] * n_scat_params
            upper = [15.0] * n + [10.0] * n_scat_params

            opts = cma.CMAOptions()
            opts.set("maxfevals", n_evals)
            opts.set("tolfun", 1e-10)
            opts.set("tolx", 1e-10)
            opts.set("verbose", -9)
            opts.set("bounds", [lower, upper])
            opts.set("CMA_stds", [2.0] * n + [1.0] * n_scat_params)

            es = cma.CMAEvolutionStrategy(x0, 1.5, opts)
            while not es.stop():
                sols = es.ask()
                fits = [_objective(x, n, scorer, zeros, N_ZEROS) for x in sols]
                es.tell(sols, fits)

            cma_score = -es.result.fbest
            cma_params = es.result.xbest

            # Nelder-Mead polish
            try:
                polish = minimize(
                    _objective, cma_params,
                    args=(n, scorer, zeros, N_ZEROS),
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

            time.sleep(2)  # thermal cooldown

        elapsed = time.time() - t_start

        # Compute detailed results for best graph
        edge_lengths = best_params[:n]
        scat_params_best = best_params[n:]
        qg = _build_cycle_unitary(n, edge_lengths, scat_params_best)
        res = _compute_residuals(qg, scorer, N_ZEROS)
        mono_phases = _compute_monodromy(scat_params_best, n)

        result_entry = {
            "n": n,
            "dims": dims,
            "best_score": best_score,
            "best_mad": res.get("mad_mean_spacings", 99),
            "mean_score": float(np.mean(all_scores)),
            "std_score": float(np.std(all_scores)),
            "best_params": {
                "edge_lengths": [float(x) for x in edge_lengths],
                "scat_params": [float(x) for x in scat_params_best],
            },
            "per_zero_residuals": res.get("residuals", []),
            "monodromy_eigenvalue_phases": mono_phases,
            "near_minus_one_monodromy": any(
                abs(abs(p) - np.pi) < 0.15 for p in mono_phases
            ),
            "runtime_hours": elapsed / 3600,
            "n_restarts": n_restarts,
            "n_evals": n_evals,
        }

        existing["results"].append(result_entry)
        _save_results(existing)

        log(f"  C_{n:2d} ({dims}D) | Best: {best_score:.4f} | "
            f"MAD: {res.get('mad_mean_spacings', 99):.4f} | "
            f"Mean: {np.mean(all_scores):.4f}+/-{np.std(all_scores):.4f} | "
            f"Mono -1: {result_entry['near_minus_one_monodromy']} | "
            f"Time: {elapsed/3600:.1f}h")

    # ---- Trend analysis ----
    _analyze_trend(existing)

    # ---- Generate plot ----
    _generate_plot(existing)

    # ---- Generate summary ----
    _generate_summary(existing)

    log(f"\nAll results saved to {RESULTS_FILE}")
    log(f"Plot saved to {PLOT_FILE}")


def _analyze_trend(data: dict) -> None:
    """Fit scaling trends to the data."""
    results = sorted(data["results"], key=lambda r: r["n"])
    ns = np.array([r["n"] for r in results], dtype=float)
    scores = np.array([r["best_score"] for r in results])

    if len(ns) < 4:
        log("\nToo few data points for trend analysis")
        return

    log("\n" + "=" * 60)
    log("TREND ANALYSIS")
    log("=" * 60)

    # Linear fit
    try:
        coeffs_lin = np.polyfit(ns, scores, 1)
        pred_lin = np.polyval(coeffs_lin, ns)
        ss_res = np.sum((scores - pred_lin) ** 2)
        ss_tot = np.sum((scores - np.mean(scores)) ** 2)
        r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        log(f"  Linear: S(n) = {coeffs_lin[0]:.6f}*n + {coeffs_lin[1]:.4f}, R² = {r2_lin:.4f}")
    except Exception:
        r2_lin = 0

    # Log fit
    try:
        coeffs_log = np.polyfit(np.log(ns), scores, 1)
        pred_log = np.polyval(coeffs_log, np.log(ns))
        ss_res = np.sum((scores - pred_log) ** 2)
        ss_tot = np.sum((scores - np.mean(scores)) ** 2)
        r2_log = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        log(f"  Log: S(n) = {coeffs_log[0]:.6f}*ln(n) + {coeffs_log[1]:.4f}, R² = {r2_log:.4f}")
    except Exception:
        r2_log = 0

    # Determine trend
    score_range = np.max(scores) - np.min(scores)
    peak_n = int(ns[np.argmax(scores)])
    tail_std = np.std(scores[-3:]) if len(scores) >= 3 else 99

    if score_range < 0.01:
        trend = "PLATEAU"
        trend_msg = (f"Score saturates at S ~ {np.mean(scores):.3f} across all cycle sizes. "
                     f"Additional degrees of freedom do not improve spectral correspondence.")
    elif peak_n == ns[-1] and coeffs_lin[0] > 0.002:
        trend = "IMPROVING"
        trend_msg = (f"Score increases with cycle size. "
                     f"Best fit: S(n) = {coeffs_lin[0]:.4f}*n + {coeffs_lin[1]:.4f}. "
                     f"Recommend extending to C_20 and beyond.")
    elif peak_n < ns[-1] - 1 and scores[-1] < np.max(scores) - 0.01:
        trend = f"PEAKED_AT_{peak_n}"
        trend_msg = (f"Score peaks at C_{peak_n} and degrades for larger cycles. "
                     f"C_{peak_n} appears to be the optimal cycle topology.")
    else:
        trend = "PLATEAU"
        trend_msg = (f"Score approximately constant (range {score_range:.4f}). "
                     f"Cycle size has minimal effect under U(2) scattering.")

    log(f"\n  TREND: {trend}")
    log(f"  {trend_msg}")

    data["summary"] = {
        "optimal_n": peak_n,
        "optimal_score": float(np.max(scores)),
        "optimal_mad": float(results[np.argmax(scores)]["best_mad"]),
        "trend": trend,
        "scaling_law": f"S(n) = {coeffs_lin[0]:.6f}*n + {coeffs_lin[1]:.4f}" if r2_lin > 0.5 else "none",
        "r2_linear": r2_lin,
        "r2_log": r2_log,
    }


def _generate_plot(data: dict) -> None:
    """Generate the scaling plot."""
    results = sorted(data["results"], key=lambda r: r["n"])
    ns = [r["n"] for r in results]
    scores = [r["best_score"] for r in results]
    mads = [r["best_mad"] for r in results]
    stds = [r["std_score"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: score
    ax1.errorbar(ns, scores, yerr=stds, fmt='bo-', markersize=6, linewidth=2,
                 capsize=4, label='Best score $\\mathcal{S}$')

    # Mark C_7
    if 7 in ns:
        idx7 = ns.index(7)
        ax1.plot(7, scores[idx7], 'b*', markersize=15, zorder=5)

    # Reference lines
    ax1.axhline(y=0.720, color='blue', linestyle='--', alpha=0.4, linewidth=1,
                label='$C_7$ Neumann baseline (0.720)')
    ax1.axhline(y=0.897, color='green', linestyle='--', alpha=0.4, linewidth=1,
                label='Previous best U(2) (0.897)')
    ax1.axhline(y=0.955, color='gray', linestyle='--', alpha=0.3, linewidth=1,
                label='Self-comparison ceiling (0.955)')

    ax1.set_xlabel('Cycle Size $n$', fontsize=13)
    ax1.set_ylabel('Best Score $\\mathcal{S}$', fontsize=13, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)

    # Right axis: MAD
    ax2 = ax1.twinx()
    ax2.plot(ns, mads, 'rs--', markersize=6, linewidth=1.5, alpha=0.7,
             label='Best MAD (mean spacings)')
    ax2.set_ylabel('Best MAD (mean spacings)', fontsize=13, color='red')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=11)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right',
               fontsize=9, framealpha=0.9)

    ax1.set_title('U(2) Spectral Correspondence vs. Cycle Size', fontsize=14)
    ax1.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200)
    plt.close()
    log(f"Plot saved to {PLOT_FILE}")


def _generate_summary(data: dict) -> None:
    """Generate the strengthening summary markdown."""
    summary_path = Path("results/strengthening_summary.md")

    # Load PSLQ results if available
    pslq_path = Path("results/pslq_analysis.json")
    pslq_verdict = "NOT YET RUN"
    pslq_findings = "N/A"
    if pslq_path.exists():
        with open(pslq_path) as f:
            pslq = json.load(f)
        pslq_verdict = pslq.get("verdict", "UNKNOWN")
        n_findings = len(pslq.get("significant_relations", []))
        n_pi = len(pslq.get("pi_multiples", []))
        pslq_findings = f"{n_findings} PSLQ relations, {n_pi} pi-multiple phases"

    s = data.get("summary", {})
    results = sorted(data["results"], key=lambda r: r["n"])

    with open(summary_path, "w") as f:
        f.write("# Strengthening Analysis Summary\n\n")

        f.write("## PSLQ Algebraic Structure Analysis\n")
        f.write(f"**Verdict**: {pslq_verdict}\n")
        f.write(f"**Key findings**: {pslq_findings}\n\n")

        f.write("## U(2) Scaling Experiment\n")
        f.write(f"**Optimal cycle size**: C_{s.get('optimal_n', '?')} "
                f"(n={s.get('optimal_n', '?')})\n")
        f.write(f"**Best score**: {s.get('optimal_score', 0):.4f} | "
                f"**Best MAD**: {s.get('optimal_mad', 0):.4f} mean spacings\n")
        f.write(f"**Trend**: {s.get('trend', '?')}\n")
        f.write(f"**Scaling law**: {s.get('scaling_law', 'none')}\n\n")

        f.write("### Results Table\n")
        f.write("| C_n | Dims | Best Score | MAD | Mean±Std | Mono -1? |\n")
        f.write("|-----|------|-----------|-----|----------|----------|\n")
        for r in results:
            f.write(f"| C_{r['n']} | {r['dims']} | {r['best_score']:.4f} | "
                    f"{r['best_mad']:.4f} | {r['mean_score']:.4f}+/-{r['std_score']:.4f} | "
                    f"{'Yes' if r.get('near_minus_one_monodromy') else 'No'} |\n")

        f.write("\n## Recommended Next Steps\n")
        if s.get("trend") == "IMPROVING":
            f.write("- Extend U(2) search to C_20 and beyond\n")
            f.write("- The spectral correspondence continues improving with cycle size\n")
        elif "PEAKED" in s.get("trend", ""):
            f.write(f"- Focus optimization effort on C_{s.get('optimal_n')}\n")
            f.write("- Investigate why this cycle size is optimal\n")
        else:
            f.write("- Cycle size has minimal effect; focus on scattering optimization\n")
            f.write("- Consider alternative topologies with U(2) scattering\n")

    log(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    run_experiment()
