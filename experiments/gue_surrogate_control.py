#!/usr/bin/env python3
"""GUE surrogate control experiment.

Tests whether the U(2) cycle-graph spectral correspondence at MAD ~0.13
mean spacings is specific to the Riemann zeros, or whether the same
construction would fit any GUE-distributed target sequence with matching
smooth counting density equally well.

Procedure
---------
1. For each of K surrogate trials:
   a. Sample eigenvalues from a large GUE matrix.
   b. Take the central window of length N (matching the Riemann test).
   c. Rescale so the smooth counting density matches the Riemann-von
      Mangoldt density at the location of the first N Riemann zeros.
   d. Run the identical CMA-ES + Nelder-Mead U(2) optimization pipeline
      against this surrogate target.
   e. Record best score, best MAD, mean across restarts.

2. Aggregate: report the distribution of best-MAD across surrogates
   alongside the Riemann baseline.

Verdict
-------
- If surrogate MAD distribution centers at ~Riemann MAD (0.145 at C_7):
  the construction is a generic spectral fitter. The "discovery" reading
  of the manuscript's U(2) result is weakened.
- If surrogate MAD distribution centers materially worse than Riemann
  (e.g., 0.25+): the Riemann zeros are genuinely more graph-realizable
  than a random GUE sequence. The discovery reading is strengthened.

Output
------
results/gue_surrogate_control.jsonl        per-restart records
results/gue_surrogate_control_summary.json aggregate summary + verdict
results/gue_surrogate_control.log          progress log
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
import sys
import time
from pathlib import Path

import cma
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riemann_qg.core.quantum_graph import QuantumGraph
from riemann_qg.core.scoring import SpectralScorer, _weyl_rescale, _n_smooth


# ---- Configuration defaults ----
N_VERTICES_DEFAULT = 7
N_ZEROS = 100
N_SURROGATES_DEFAULT = 20
N_RESTARTS_DEFAULT = 5
N_EVALS_DEFAULT = 75_000
GUE_MATRIX_SIZE = 400

RESULTS_JSONL = Path("results/gue_surrogate_control.jsonl")
SUMMARY_JSON = Path("results/gue_surrogate_control_summary.json")
LOG_PATH = Path("results/gue_surrogate_control.log")


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


# ---- U(2) machinery (matches u2_scaling.py exactly) ----

def _unitary_2x2(params: np.ndarray) -> np.ndarray:
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
    adj = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1
    scattering = [_unitary_2x2(scat_params[v * 4:(v + 1) * 4]) for v in range(n)]
    return QuantumGraph(adj, list(edge_lengths), scattering)


def _objective(
    params: np.ndarray,
    n: int,
    scorer: SpectralScorer,
    target: np.ndarray,
    n_targets: int,
) -> float:
    """Negative score. Identical to u2_scaling._objective but accepts an
    arbitrary `target` sequence rather than fetching the Riemann zeros."""
    edge_lengths = params[:n]
    scat_params = params[n:]
    if np.any(edge_lengths < 0.01):
        return 1.0
    try:
        qg = _build_cycle_unitary(n, edge_lengths, scat_params)
        total_length = float(np.sum(edge_lengths))
        k_max = max((n_targets + 5) * np.pi / total_length, 10.0)
        spectrum = qg.compute_spectrum(k_max, n_points=3000)
        result = scorer.score_spectrum(
            spectrum, target, n_targets,
            edge_lengths=qg.edge_lengths,
        )
        return -result.total_score
    except Exception:
        return 1.0


def _compute_residuals(
    qg: QuantumGraph,
    target: np.ndarray,
    n_targets: int,
) -> dict:
    """Compute MAD and per-target residuals against an arbitrary target."""
    total_length = sum(qg.edge_lengths)
    k_max = max((n_targets + 5) * np.pi / total_length, 10.0)
    spectrum = qg.compute_spectrum(k_max, n_points=3000)
    if len(spectrum) < 3:
        return {"residuals": [], "mad": 99.0, "max_residual": 99.0,
                "mad_mean_spacings": 99.0, "n_matched": 0}
    target_sorted = np.sort(target)
    scaled, alpha = _weyl_rescale(np.sort(spectrum), total_length, target_sorted)
    spacings = np.diff(target_sorted)
    margin = float(np.mean(spacings)) if len(spacings) > 0 else 1.0
    t_low = target_sorted[0] - margin
    t_high = target_sorted[-1] + margin
    windowed = scaled[(scaled >= t_low) & (scaled <= t_high)]
    n_match = min(len(windowed), len(target_sorted))
    if n_match < 3:
        return {"residuals": [], "mad": 99.0, "max_residual": 99.0,
                "mad_mean_spacings": 99.0, "n_matched": 0}
    residuals = [abs(float(windowed[i] - target_sorted[i])) for i in range(n_match)]
    mean_spacing = float(np.mean(spacings))
    return {
        "residuals": residuals,
        "mad": float(np.mean(residuals)),
        "mad_mean_spacings": float(np.mean(residuals)) / mean_spacing,
        "max_residual": float(np.max(residuals)),
        "n_matched": n_match,
    }


# ---- GUE surrogate generation ----

def _gue_eigenvalues(size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample eigenvalues from an n x n GUE matrix.

    GUE: Hermitian matrices H with H_{ij} = (X_{ij} + i Y_{ij})/sqrt(2)
    for i != j, with X_{ij}, Y_{ij} ~ N(0, 1) independent, and H_{ii}
    ~ N(0, 1). Returns sorted real eigenvalues.
    """
    h = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    h = (h + h.conj().T) / 2.0
    eigvals = np.linalg.eigvalsh(h).real
    return np.sort(eigvals)


def _make_surrogate_target(
    n_targets: int,
    riemann_zeros: np.ndarray,
    rng: np.random.Generator,
    gue_size: int = GUE_MATRIX_SIZE,
) -> np.ndarray:
    """Generate a GUE surrogate target sequence with smooth counting density
    matching the first n_targets Riemann zeros.

    Procedure:
    1. Sample GUE eigenvalues at large size.
    2. Take central n_targets eigenvalues (cleanest bulk statistics).
    3. Unfold via polynomial fit to the staircase function.
    4. Map back through the inverse Riemann-von Mangoldt N_smooth.

    Result: GUE local statistics + Riemann smooth counting density,
    with no actual Riemann arithmetic information.
    """
    # 1-2. Sample GUE eigenvalues, take central n_targets.
    gue = _gue_eigenvalues(gue_size, rng)
    start = (gue_size - n_targets) // 2
    central = gue[start:start + n_targets]

    # 3. Unfold via polynomial fit to staircase i = f(eigenvalue).
    indices = np.arange(1, n_targets + 1, dtype=np.float64)
    poly_deg = 5
    coeffs = np.polyfit(central, indices, poly_deg)
    unfolded = np.polyval(coeffs, central)

    # 4. Map back to Riemann-density coordinates via N_smooth^{-1}.
    from scipy.optimize import brentq

    def _invert_n_smooth(n_target: float) -> float:
        if n_target < -0.125:
            return 2.0 * np.pi
        def f(t: float) -> float:
            return _n_smooth(t) - n_target
        t_low = 1.0
        t_high = max(200.0, 4.0 * np.pi * (n_target + 2))
        while f(t_high) < 0:
            t_high *= 2
        try:
            return brentq(f, t_low, t_high, xtol=1e-6)
        except ValueError:
            return t_high

    surrogate = np.array([_invert_n_smooth(u) for u in unfolded])
    surrogate = np.sort(surrogate)

    # Shift correction: ensure midpoint aligns with the Riemann midpoint
    # (handles drift from unfolding offsets).
    target_midpoint = riemann_zeros[n_targets // 2]
    surrogate_midpoint = surrogate[n_targets // 2]
    shift = target_midpoint - surrogate_midpoint
    if abs(shift) > 0.5:
        surrogate = surrogate + shift

    surrogate = np.maximum(surrogate, 0.1)
    return np.sort(surrogate)


# ---- Optimization driver for one surrogate ----

def _optimize_against_target(
    target: np.ndarray,
    n: int,
    n_targets: int,
    n_restarts: int,
    n_evals: int,
    surrogate_seed: int,
) -> dict:
    """Run the full CMA-ES + Nelder-Mead pipeline against one target sequence.

    Identical to u2_scaling.run_experiment but with the target passed in.
    """
    scorer = SpectralScorer()
    n_edge_params = n
    n_scat_params = 4 * n

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    restart_records = []
    best_score = -1.0
    best_params = None

    for restart in range(n_restarts):
        rng = np.random.default_rng(surrogate_seed * 10000 + restart)

        if restart % 4 == 0:
            edge_init = np.array([
                np.log(primes[i % len(primes)]) for i in range(n)
            ])
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

        t_start = time.time()
        es = cma.CMAEvolutionStrategy(x0, 1.5, opts)
        while not es.stop():
            sols = es.ask()
            fits = [_objective(x, n, scorer, target, n_targets) for x in sols]
            es.tell(sols, fits)

        cma_score = -es.result.fbest
        cma_params = es.result.xbest

        try:
            polish = minimize(
                _objective, cma_params,
                args=(n, scorer, target, n_targets),
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

        edge_lengths = final_params[:n]
        scat_params = final_params[n:]
        qg = _build_cycle_unitary(n, edge_lengths, scat_params)
        res = _compute_residuals(qg, target, n_targets)

        restart_records.append({
            "restart": restart,
            "cma_score": cma_score,
            "score": final_score,
            "mad": res.get("mad", 99),
            "mad_mean_spacings": res.get("mad_mean_spacings", 99),
            "max_residual": res.get("max_residual", 99),
            "n_matched": res.get("n_matched", 0),
            "total_length": float(np.sum(edge_lengths)),
            "n_cma_evals": int(es.result.evaluations),
            "elapsed_s": elapsed,
            "edge_lengths": [float(x) for x in edge_lengths],
            "scat_params": [float(x) for x in scat_params],
        })

        if final_score > best_score:
            best_score = final_score
            best_params = final_params

        time.sleep(2)  # thermal cooldown

    # Best-restart residuals
    best_edge_lengths = best_params[:n]
    best_scat = best_params[n:]
    best_qg = _build_cycle_unitary(n, best_edge_lengths, best_scat)
    best_res = _compute_residuals(best_qg, target, n_targets)

    scores = [r["score"] for r in restart_records]
    mads = [r["mad_mean_spacings"] for r in restart_records]

    return {
        "best_score": best_score,
        "best_mad_mean_spacings": best_res.get("mad_mean_spacings", 99),
        "best_n_matched": best_res.get("n_matched", 0),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "mean_mad": float(np.mean(mads)),
        "std_mad": float(np.std(mads)),
        "restart_records": restart_records,
    }


# ---- Existing-results checkpoint ----

def _load_completed_surrogates() -> set[int]:
    completed = set()
    if RESULTS_JSONL.exists():
        with open(RESULTS_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    completed.add(r["surrogate_index"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def _load_all_records() -> list[dict]:
    records = []
    if not RESULTS_JSONL.exists():
        return records
    with open(RESULTS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


# ---- Aggregation and verdict ----

def _aggregate_and_verdict(
    results: list[dict],
    riemann_baseline: dict,
    config: dict,
) -> dict:
    best_mads = [r["best_mad_mean_spacings"] for r in results
                 if r["best_mad_mean_spacings"] < 90]
    best_scores = [r["best_score"] for r in results
                   if r["best_mad_mean_spacings"] < 90]
    best_n_matched = [r.get("best_n_matched", 0) for r in results
                      if r["best_mad_mean_spacings"] < 90]

    if len(best_mads) < 3:
        verdict = "INSUFFICIENT_DATA"
        verdict_msg = (
            f"Only {len(best_mads)} valid surrogates; need at least 3 "
            "for a meaningful distribution."
        )
        return {
            "config": config,
            "riemann_baseline": riemann_baseline,
            "n_surrogates_completed": len(results),
            "n_surrogates_valid": len(best_mads),
            "verdict": verdict,
            "verdict_message": verdict_msg,
        }

    distribution = {
        "n": len(best_mads),
        "best_mad_mean": float(np.mean(best_mads)),
        "best_mad_std": float(np.std(best_mads)),
        "best_mad_min": float(np.min(best_mads)),
        "best_mad_max": float(np.max(best_mads)),
        "best_mad_median": float(np.median(best_mads)),
        "best_score_mean": float(np.mean(best_scores)),
        "best_score_std": float(np.std(best_scores)),
        # Matched-eigenvalue counts: how many of the N_ZEROS targets the
        # optimizer was able to place a graph eigenvalue against (within
        # the +/-1 mean-spacing window). Asymmetry with the Riemann
        # n_matched (typically ~74/100 in the headline result) would mean
        # surrogate MADs are computed over different subset sizes than
        # the Riemann MAD, which would need to be acknowledged in any
        # comparison.
        "n_matched_mean": float(np.mean(best_n_matched)),
        "n_matched_std": float(np.std(best_n_matched)),
        "n_matched_min": int(np.min(best_n_matched)),
        "n_matched_max": int(np.max(best_n_matched)),
        "n_matched_median": float(np.median(best_n_matched)),
    }

    riemann_mad = riemann_baseline["mad_mean_spacings"]
    surrogate_mean_mad = distribution["best_mad_mean"]
    surrogate_std_mad = distribution["best_mad_std"]

    # ---- Multiple complementary effect-size measures ----
    # 1. Empirical quantile: fraction of surrogates that beat Riemann.
    #    (A "better" fit has smaller MAD.)
    n_better = int(sum(1 for m in best_mads if m < riemann_mad))
    n_total = len(best_mads)
    quantile_pct = 100.0 * n_better / n_total

    # 2. Z-score (kept for continuity, no longer drives verdict)
    if surrogate_std_mad > 1e-6:
        z = (surrogate_mean_mad - riemann_mad) / surrogate_std_mad
    else:
        z = 0.0

    # 3. Cohen's d: standardized effect size, distribution-agnostic
    if surrogate_std_mad > 1e-6:
        cohens_d = (surrogate_mean_mad - riemann_mad) / surrogate_std_mad
    else:
        cohens_d = 0.0

    distribution["riemann_vs_surrogate_z"] = float(z)
    distribution["quantile_pct"] = float(quantile_pct)
    distribution["cohens_d"] = float(cohens_d)
    distribution["n_surrogates_better"] = n_better
    distribution["n_surrogates_total"] = n_total

    # ---- Verdict based on the empirical quantile ----
    # GENERIC:           Riemann within the bulk of surrogate distribution.
    # SUGGESTIVE:        Riemann better than 75-95% of surrogates.
    # RIEMANN_SPECIFIC:  Riemann better than 95% of surrogates.
    #
    # Base-rate caveat: at K=20 surrogates, even a quantile of 0% is
    # consistent with the GENERIC hypothesis at a one-in-21 base rate
    # (~4.8%). Strong specificity claims require K >= 50.
    if quantile_pct <= 5.0:
        verdict = "RIEMANN_SPECIFIC"
        verdict_msg = (
            f"Riemann MAD {riemann_mad:.4f} is better than "
            f"{n_total - n_better} of {n_total} surrogates "
            f"({100.0 - quantile_pct:.1f}th percentile of surrogate MADs). "
            f"Surrogate distribution: mean {surrogate_mean_mad:.4f} +/- "
            f"{surrogate_std_mad:.4f} (Cohen's d = {cohens_d:.2f}). "
            "The U(2) construction fits the Riemann zeros materially "
            "better than matched-density GUE surrogates. The 'discovery' "
            "reading of the U(2) result is supported."
        )
        if n_total < 50:
            verdict_msg += (
                f" Caveat: at K={n_total} surrogates, this percentile "
                "has a base-rate floor of roughly "
                f"{100.0 / (n_total + 1):.1f}%; strong specificity claims "
                "should be confirmed at K >= 50."
            )
    elif quantile_pct <= 25.0:
        verdict = "SUGGESTIVE"
        verdict_msg = (
            f"Riemann MAD {riemann_mad:.4f} is better than "
            f"{n_total - n_better} of {n_total} surrogates "
            f"({100.0 - quantile_pct:.1f}th percentile of surrogate MADs). "
            f"Surrogate distribution: mean {surrogate_mean_mad:.4f} +/- "
            f"{surrogate_std_mad:.4f} (Cohen's d = {cohens_d:.2f}). "
            "The Riemann result sits in the upper tail of the surrogate "
            "distribution but not decisively outside it. Suggestive but "
            "inconclusive; K >= 50 surrogates would tighten the verdict."
        )
    else:
        verdict = "GENERIC"
        verdict_msg = (
            f"Riemann MAD {riemann_mad:.4f} sits at the "
            f"{100.0 - quantile_pct:.1f}th percentile of the surrogate "
            f"distribution: {n_better} of {n_total} surrogates achieved a "
            "smaller MAD than the Riemann result. Surrogate distribution: "
            f"mean {surrogate_mean_mad:.4f} +/- {surrogate_std_mad:.4f} "
            f"(Cohen's d = {cohens_d:.2f}). The U(2) construction fits "
            "matched-density GUE surrogates approximately as well as the "
            "Riemann zeros, indicating it is a generic GUE-density fitter "
            "rather than a Riemann-specific construction. The 'discovery' "
            "reading of the U(2) result is weakened."
        )

    return {
        "config": config,
        "riemann_baseline": riemann_baseline,
        "n_surrogates_completed": len(results),
        "n_surrogates_valid": len(best_mads),
        "surrogate_distribution": distribution,
        "verdict": verdict,
        "verdict_message": verdict_msg,
    }


def _load_riemann_baseline(n_vertices: int) -> dict | None:
    scaling_path = Path("results/u2_scaling_results.json")
    if not scaling_path.exists():
        return None
    with open(scaling_path) as f:
        data = json.load(f)
    entry = next((r for r in data.get("results", [])
                  if r["n"] == n_vertices), None)
    if entry is None:
        return None
    return {
        "best_score": entry["best_score"],
        "mad_mean_spacings": entry["best_mad"],
        "mean_score": entry["mean_score"],
        "std_score": entry["std_score"],
        "source": "u2_scaling_results.json",
    }


# ---- Main experiment driver ----

def run_experiment(
    n_vertices: int = N_VERTICES_DEFAULT,
    n_surrogates: int = N_SURROGATES_DEFAULT,
    n_restarts: int = N_RESTARTS_DEFAULT,
    n_evals: int = N_EVALS_DEFAULT,
) -> None:
    log("=" * 70)
    log("GUE SURROGATE CONTROL EXPERIMENT")
    log("=" * 70)
    log(f"Cycle: C_{n_vertices} (parameter dimension {5 * n_vertices})")
    log(f"Targets per surrogate: {N_ZEROS}")
    log(f"Surrogates: {n_surrogates}")
    log(f"Restarts per surrogate: {n_restarts}")
    log(f"Max CMA-ES evaluations per restart: {n_evals}")
    log("")
    log("Goal: test whether U(2) C_n fits matched-density GUE surrogates")
    log("as well as it fits the Riemann zeros. Identical MAD => generic")
    log("fitter; substantially worse on surrogates => Riemann-specific.")
    log("")

    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    scorer = SpectralScorer()
    riemann_zeros = scorer._rz.get_zeros(N_ZEROS)

    baseline = _load_riemann_baseline(n_vertices)
    if baseline is None:
        log(f"WARNING: no Riemann baseline found for C_{n_vertices} in "
            f"results/u2_scaling_results.json.")
        if n_vertices == 7:
            baseline = {"best_score": 0.897, "mad_mean_spacings": 0.145,
                        "source": "manuscript_approx"}
        elif n_vertices == 20:
            baseline = {"best_score": 0.9258, "mad_mean_spacings": 0.097,
                        "source": "manuscript_approx"}
        else:
            baseline = {"best_score": None, "mad_mean_spacings": None,
                        "source": "unknown"}
    log(f"Riemann baseline (C_{n_vertices}): score = "
        f"{baseline['best_score']}, MAD = {baseline['mad_mean_spacings']}")
    log("")

    time_per_eval = 0.007
    est_per_restart = n_evals * time_per_eval
    est_per_surrogate = est_per_restart * n_restarts
    est_total = est_per_surrogate * n_surrogates
    log(f"Runtime estimate: ~{est_per_surrogate/3600:.1f} h per surrogate, "
        f"~{est_total/3600:.1f} h total ({n_surrogates} surrogates).")
    log("")

    completed = _load_completed_surrogates()
    if completed:
        log(f"Resuming: {len(completed)} surrogates already done.")

    config = {
        "n_vertices": n_vertices,
        "n_targets": N_ZEROS,
        "n_surrogates": n_surrogates,
        "n_restarts": n_restarts,
        "n_evals": n_evals,
        "gue_matrix_size": GUE_MATRIX_SIZE,
    }

    all_results = list(_load_all_records())

    for surrogate_idx in range(n_surrogates):
        if surrogate_idx in completed:
            log(f"Surrogate {surrogate_idx}: already done, skipping.")
            continue

        log(f"\n--- Surrogate {surrogate_idx + 1}/{n_surrogates} ---")
        seed = 17 * surrogate_idx + 1
        rng = np.random.default_rng(seed)

        target = _make_surrogate_target(N_ZEROS, riemann_zeros, rng)
        log(f"Surrogate target range: [{target[0]:.2f}, {target[-1]:.2f}]")
        log(f"Mean spacing: {float(np.mean(np.diff(target))):.4f}  "
            f"(Riemann: {float(np.mean(np.diff(riemann_zeros))):.4f})")

        t_surrogate = time.time()
        opt_result = _optimize_against_target(
            target, n_vertices, N_ZEROS,
            n_restarts, n_evals, seed,
        )
        elapsed = time.time() - t_surrogate

        log(f"  Best score: {opt_result['best_score']:.4f}, "
            f"Best MAD: {opt_result['best_mad_mean_spacings']:.4f}, "
            f"Mean: {opt_result['mean_score']:.4f}+/-"
            f"{opt_result['std_score']:.4f}, "
            f"Matched: {opt_result['best_n_matched']}/{N_ZEROS}, "
            f"Time: {elapsed/3600:.1f}h")

        record = {
            "surrogate_index": surrogate_idx,
            "seed": seed,
            "target_first": float(target[0]),
            "target_last": float(target[-1]),
            "target_mean_spacing": float(np.mean(np.diff(target))),
            "elapsed_h": elapsed / 3600,
            **opt_result,
        }
        with open(RESULTS_JSONL, "a") as f:
            f.write(json.dumps(record) + "\n")
        all_results.append(record)

    # ---- Aggregate ----
    log("")
    log("=" * 70)
    log("AGGREGATE RESULTS")
    log("=" * 70)

    summary = _aggregate_and_verdict(all_results, baseline, config)

    if "surrogate_distribution" in summary:
        d = summary["surrogate_distribution"]
        log(f"Riemann MAD (baseline):       {baseline['mad_mean_spacings']}")
        log(f"Surrogate MAD distribution:")
        log(f"  N:                          {d['n']}")
        log(f"  Mean:                       {d['best_mad_mean']:.4f}")
        log(f"  Std:                        {d['best_mad_std']:.4f}")
        log(f"  Median:                     {d['best_mad_median']:.4f}")
        log(f"  Min:                        {d['best_mad_min']:.4f}")
        log(f"  Max:                        {d['best_mad_max']:.4f}")
        log(f"  Riemann z-score:            {d['riemann_vs_surrogate_z']:.2f}")
        log(f"Surrogate matched-eigenvalue counts (out of {N_ZEROS}):")
        log(f"  Mean:                       {d['n_matched_mean']:.1f}")
        log(f"  Std:                        {d['n_matched_std']:.1f}")
        log(f"  Median:                     {d['n_matched_median']:.1f}")
        log(f"  Min:                        {d['n_matched_min']}")
        log(f"  Max:                        {d['n_matched_max']}")

    log("")
    log(f"VERDICT: {summary['verdict']}")
    log("")
    log(summary["verdict_message"])

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    log("")
    log(f"Summary written to {SUMMARY_JSON}")
    log(f"Per-surrogate records written to {RESULTS_JSONL}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GUE surrogate control experiment for U(2) cycle scattering",
    )
    parser.add_argument("--n", type=int, default=N_VERTICES_DEFAULT,
                        help=f"Cycle size (default: {N_VERTICES_DEFAULT})")
    parser.add_argument("--n-surrogates", type=int,
                        default=N_SURROGATES_DEFAULT,
                        help=f"Number of GUE surrogates "
                             f"(default: {N_SURROGATES_DEFAULT})")
    parser.add_argument("--restarts", type=int, default=N_RESTARTS_DEFAULT,
                        help=f"CMA-ES restarts per surrogate "
                             f"(default: {N_RESTARTS_DEFAULT})")
    parser.add_argument("--max-evals", type=int, default=N_EVALS_DEFAULT,
                        help=f"Max CMA-ES evaluations per restart "
                             f"(default: {N_EVALS_DEFAULT})")
    args = parser.parse_args()
    run_experiment(
        n_vertices=args.n,
        n_surrogates=args.n_surrogates,
        n_restarts=args.restarts,
        n_evals=args.max_evals,
    )


if __name__ == "__main__":
    main()
