#!/usr/bin/env python3
"""Verify the C_20 best score of 0.9258 by re-scoring the saved parameters.

Loads the exact edge lengths and scattering parameters from
results/u2_scaling_results.json (the C_20 entry) and computes the score
from scratch, with no optimization. Tests sensitivity to spectrum
sampling resolution to identify any numerical fragility.

If the reproduced score matches 0.9258, the result is mathematically
valid (just not reproducible by independent optimization).

If the reproduced score is substantially different, the original was
likely an optimizer artifact and should not be reported as the headline.

Output: results/c20_verification.json + console report.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "sequential"

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riemann_qg.core.quantum_graph import QuantumGraph
from riemann_qg.core.scoring import SpectralScorer, _weyl_rescale


N = 20
N_ZEROS = 100
N_REPLICATES = 5
SCALING_FILE = Path("results/u2_scaling_results.json")
OUTPUT_FILE = Path("results/c20_verification.json")


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


def _score_with_n_points(qg: QuantumGraph, scorer: SpectralScorer,
                         zeros: np.ndarray, n_zeros: int,
                         n_points: int) -> dict:
    total_length = sum(qg.edge_lengths)
    k_max = max((n_zeros + 5) * np.pi / total_length, 10.0)
    spectrum = qg.compute_spectrum(k_max, n_points=n_points)
    result = scorer.score_spectrum(spectrum, zeros, n_zeros,
                                   edge_lengths=qg.edge_lengths)
    return {
        "score": result.total_score,
        "position_score": result.absolute_match,
        "spacing_score": result.spacing_distribution,
        "pair_score": result.pair_correlation,
        "n_windowed": result.n_windowed,
        "weyl_alpha": result.weyl_alpha,
        "n_raw_eigenvalues": len(spectrum),
        "k_max_used": float(k_max),
        "n_points_used": n_points,
    }


def _compute_residuals(qg: QuantumGraph, scorer: SpectralScorer,
                       n_zeros: int, n_points: int) -> dict:
    zeros = scorer._rz.get_zeros(n_zeros)
    total_length = sum(qg.edge_lengths)
    k_max = max((n_zeros + 5) * np.pi / total_length, 10.0)
    spectrum = qg.compute_spectrum(k_max, n_points=n_points)
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


def run_verification() -> None:
    log("=" * 60)
    log("VERIFICATION: C_20 BEST SCORE REPRODUCIBILITY")
    log("=" * 60)

    if not SCALING_FILE.exists():
        log(f"ERROR: {SCALING_FILE} not found")
        sys.exit(1)

    with open(SCALING_FILE) as f:
        data = json.load(f)

    c20 = next((r for r in data["results"] if r["n"] == N), None)
    if c20 is None:
        log(f"ERROR: C_{N} not found")
        sys.exit(1)

    prior_score = c20["best_score"]
    edge_lengths = np.array(c20["best_params"]["edge_lengths"])
    scat_params = np.array(c20["best_params"]["scat_params"])

    log(f"Prior reported best: {prior_score:.6f}")
    log(f"Total edge length:   {np.sum(edge_lengths):.4f}")
    log(f"Edges:               {len(edge_lengths)}")
    log(f"Scattering params:   {len(scat_params)}")
    log("")

    scorer = SpectralScorer()
    zeros = scorer._rz.get_zeros(N_ZEROS)
    qg = _build_cycle_unitary(N, edge_lengths, scat_params)

    log("Re-scoring with original optimizer settings (n_points=3000):")
    primary = _score_with_n_points(qg, scorer, zeros, N_ZEROS, 3000)
    log(f"  Score:               {primary['score']:.6f}")
    log(f"  Position score:      {primary['position_score']:.6f}")
    log(f"  Spacing score:       {primary['spacing_score']:.6f}")
    log(f"  Pair correlation:    {primary['pair_score']:.6f}")
    log(f"  Window:              {primary['n_windowed']}/{N_ZEROS}")
    log(f"  Raw eigenvalues:     {primary['n_raw_eigenvalues']}")
    log(f"  Weyl alpha:          {primary['weyl_alpha']:.4f}")
    log("")

    diff_primary = primary["score"] - prior_score
    log(f"Difference from reported: {diff_primary:+.6f}")
    log("")

    log(f"Running {N_REPLICATES} replicate scorings:")
    replicate_scores = []
    for i in range(N_REPLICATES):
        rep = _score_with_n_points(qg, scorer, zeros, N_ZEROS, 3000)
        replicate_scores.append(rep["score"])
        log(f"  replicate {i}: {rep['score']:.8f}")

    score_std = float(np.std(replicate_scores))
    log(f"  std: {score_std:.2e}")
    log("")

    log("Sensitivity to spectrum sampling resolution:")
    sensitivity = {}
    for n_pts in [1500, 2000, 3000, 5000, 8000]:
        s = _score_with_n_points(qg, scorer, zeros, N_ZEROS, n_pts)
        sensitivity[n_pts] = s["score"]
        log(f"  n_points={n_pts:5d}: score={s['score']:.6f}  "
            f"n_eig={s['n_raw_eigenvalues']:3d}  "
            f"window={s['n_windowed']}/{N_ZEROS}")
    log("")

    res = _compute_residuals(qg, scorer, N_ZEROS, n_points=8000)
    log(f"At n_points=8000:")
    log(f"  Score: {sensitivity[8000]:.6f}")
    log(f"  MAD:   {res['mad']:.6f} T-units")
    log(f"  MAD:   {res['mad_mean_spacings']:.6f} mean spacings")
    log(f"  Max:   {res['max_residual']:.6f}")
    log(f"  n_matched: {res['n_matched']}/{N_ZEROS}")
    log("")

    abs_diff = abs(diff_primary)
    if abs_diff < 0.001:
        verdict = "REPRODUCIBLE"
        message = ("The score reproduces to within 0.001. "
                   "The 0.9258 result is mathematically valid.")
    elif abs_diff < 0.01:
        verdict = "PARTIAL"
        message = (f"The score reproduces within {abs_diff:.4f} but not exactly. "
                   "Likely caused by spectrum sampling resolution. "
                   "Report the reproducible value as headline.")
    else:
        verdict = "FAILED_TO_REPRODUCE"
        message = (f"The score differs by {abs_diff:.4f} from the reported value. "
                   "The original 0.9258 appears to be an optimizer artifact. "
                   "Do not report it as the headline result.")

    log("=" * 60)
    log(f"VERDICT: {verdict}")
    log(message)
    log("=" * 60)

    output = {
        "prior_reported_score": prior_score,
        "reproduced_score": primary["score"],
        "score_difference": diff_primary,
        "reproduced_mad_T_units": res["mad"],
        "reproduced_mad_mean_spacings": res["mad_mean_spacings"],
        "reproduced_max_residual": res["max_residual"],
        "n_matched": res["n_matched"],
        "n_replicates": N_REPLICATES,
        "replicate_scores": replicate_scores,
        "replicate_score_std": score_std,
        "sensitivity_n_points": {str(k): float(v) for k, v in sensitivity.items()},
        "primary_score_details": {
            "position_score": primary["position_score"],
            "spacing_score": primary["spacing_score"],
            "pair_score": primary["pair_score"],
            "n_windowed": primary["n_windowed"],
            "n_raw_eigenvalues": primary["n_raw_eigenvalues"],
            "weyl_alpha": primary["weyl_alpha"],
            "k_max_used": primary["k_max_used"],
        },
        "per_zero_residuals": res["residuals"],
        "verdict": verdict,
        "verdict_message": message,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_verification()
