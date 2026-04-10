#!/usr/bin/env python3
"""Analyze the structure of the optimized U(2) scattering matrices.

Reads the best graph from results/best_unitary_graph.json and tests
whether the scattering matrices have hidden mathematical structure:
- Constant determinant across vertices
- Eigenvalue phases that are rational multiples of π
- Relationships between vertex parameters and edge lengths
- Product of all scattering matrices (monodromy) structure
- Distance from Neumann scattering
- Lie algebra structure

Output: results/scattering_analysis.txt + console
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BEST_GRAPH_FILE = Path("results/best_unitary_graph.json")
OUTPUT_FILE = Path("results/scattering_analysis.txt")


def log(msg: str, fh=None) -> None:
    print(msg)
    sys.stdout.flush()
    if fh:
        fh.write(msg + "\n")


def _is_rational_multiple_of_pi(phase: float, max_denom: int = 20, tol: float = 0.02) -> tuple[bool, str]:
    """Test if phase ≈ (p/q)π for small p, q."""
    ratio = phase / np.pi
    frac = Fraction(ratio).limit_denominator(max_denom)
    approx = float(frac) * np.pi
    error = abs(phase - approx)
    is_rational = error < tol
    return is_rational, f"{frac}π" if is_rational else f"{phase:.4f}"


def run_analysis() -> None:
    """Run the full scattering matrix structure analysis."""
    if not BEST_GRAPH_FILE.exists():
        print(f"No best graph found at {BEST_GRAPH_FILE}")
        sys.exit(1)

    with open(BEST_GRAPH_FILE) as f:
        spec = json.load(f)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as fh:
        log("=" * 70, fh)
        log("SCATTERING MATRIX STRUCTURE ANALYSIS", fh)
        log("=" * 70, fh)
        log(f"Graph: {spec['graph_type']}", fh)
        log(f"Score: {spec['score']:.6f}", fh)
        log(f"N zeros: {spec['n_zeros']}", fh)
        log(f"MAD: {spec.get('mad', 'N/A')}", fh)
        log(f"Max residual: {spec.get('max_residual', 'N/A')}", fh)
        log("", fh)

        n = spec["n_vertices"]
        edge_lengths = np.array(spec["edge_lengths"])
        matrices = []

        # ---- 1. Individual matrix properties ----
        log("1. INDIVIDUAL SCATTERING MATRICES", fh)
        log("-" * 50, fh)

        det_phases = []
        eig_phases_all = []
        traces = []
        distances_from_neumann = []

        # Neumann scattering for degree 2
        s_neumann = np.array([[0, 1], [1, 0]], dtype=np.complex128)

        for sm in spec["scattering_matrices"]:
            v = sm["vertex"]
            u = np.array(sm["matrix_real"], dtype=np.float64) + \
                1j * np.array(sm["matrix_imag"], dtype=np.float64)
            matrices.append(u)

            det_val = np.linalg.det(u)
            det_phase = np.angle(det_val)
            det_phases.append(det_phase)

            eigvals = np.linalg.eigvals(u)
            eig_phases = np.sort(np.angle(eigvals))
            eig_phases_all.append(eig_phases)

            tr = np.trace(u)
            traces.append(tr)

            # Frobenius distance from Neumann
            dist = np.linalg.norm(u - s_neumann, "fro")
            distances_from_neumann.append(dist)

            # Check if det phase is rational multiple of π
            is_rat, rat_str = _is_rational_multiple_of_pi(det_phase)

            log(f"  Vertex {v}:", fh)
            log(f"    det = {abs(det_val):.6f} * exp(i * {det_phase:.6f})"
                f"  {'≈ ' + rat_str if is_rat else ''}", fh)
            log(f"    eigenvalue phases: [{eig_phases[0]:.6f}, {eig_phases[1]:.6f}]", fh)

            # Check eigenvalue phases
            for ep in eig_phases:
                is_r, r_str = _is_rational_multiple_of_pi(ep)
                if is_r:
                    log(f"      {ep:.6f} ≈ {r_str}", fh)

            log(f"    trace = {tr.real:.6f} + {tr.imag:.6f}i", fh)
            log(f"    |trace| = {abs(tr):.6f}", fh)
            log(f"    distance from Neumann = {dist:.6f}", fh)
            log(f"    edge length = {edge_lengths[v]:.6f}", fh)
            log("", fh)

        # ---- 2. Cross-vertex patterns ----
        log("\n2. CROSS-VERTEX PATTERNS", fh)
        log("-" * 50, fh)

        det_phases_arr = np.array(det_phases)
        log(f"  Determinant phases: {[f'{p:.4f}' for p in det_phases]}", fh)
        log(f"    Std dev: {np.std(det_phases_arr):.6f}", fh)
        log(f"    Are they constant? {'YES' if np.std(det_phases_arr) < 0.05 else 'NO'}"
            f" (std < 0.05)", fh)
        log(f"    Sum of det phases: {np.sum(det_phases_arr):.6f}", fh)
        is_sum_rat, sum_rat_str = _is_rational_multiple_of_pi(np.sum(det_phases_arr))
        log(f"    Sum ≈ rational×π? {'YES: ' + sum_rat_str if is_sum_rat else 'NO'}", fh)
        log("", fh)

        # Eigenvalue phase patterns
        log("  Eigenvalue phase sums:", fh)
        eig_phase_sums = [np.sum(ep) for ep in eig_phases_all]
        log(f"    Per-vertex sums: {[f'{s:.4f}' for s in eig_phase_sums]}", fh)
        log(f"    Total sum: {sum(eig_phase_sums):.6f}", fh)
        is_total_rat, total_rat_str = _is_rational_multiple_of_pi(sum(eig_phase_sums))
        log(f"    Total ≈ rational×π? {'YES: ' + total_rat_str if is_total_rat else 'NO'}", fh)
        log("", fh)

        # Trace patterns
        traces_arr = np.array(traces)
        log(f"  Traces: {[f'{t.real:.4f}+{t.imag:.4f}i' for t in traces]}", fh)
        log(f"    |traces|: {[f'{abs(t):.4f}' for t in traces]}", fh)
        mean_trace_mag = np.mean([abs(t) for t in traces])
        log(f"    Mean |trace|: {mean_trace_mag:.6f}", fh)
        log(f"    Neumann |trace| = 0 (for reference)", fh)
        log("", fh)

        # Distance from Neumann
        dists = np.array(distances_from_neumann)
        log(f"  Frobenius distances from Neumann:", fh)
        log(f"    Per vertex: {[f'{d:.4f}' for d in dists]}", fh)
        log(f"    Mean: {np.mean(dists):.6f}", fh)
        log(f"    Std: {np.std(dists):.6f}", fh)
        log(f"    Max: {np.max(dists):.6f}", fh)
        log(f"    The matrices are {'CLOSE to' if np.mean(dists) < 0.5 else 'FAR from'} Neumann", fh)
        log("", fh)

        # ---- 3. Monodromy matrix (product around the cycle) ----
        log("\n3. MONODROMY MATRIX (product around cycle)", fh)
        log("-" * 50, fh)

        monodromy = np.eye(2, dtype=np.complex128)
        for u in matrices:
            monodromy = monodromy @ u

        mono_det = np.linalg.det(monodromy)
        mono_eigvals = np.linalg.eigvals(monodromy)
        mono_eig_phases = np.sort(np.angle(mono_eigvals))
        mono_trace = np.trace(monodromy)

        log(f"  M = S_1 * S_2 * ... * S_n", fh)
        log(f"  det(M) = {abs(mono_det):.6f} * exp(i * {np.angle(mono_det):.6f})", fh)
        is_det_rat, det_rat_str = _is_rational_multiple_of_pi(np.angle(mono_det))
        log(f"    det phase ≈ rational×π? {'YES: ' + det_rat_str if is_det_rat else 'NO'}", fh)
        log(f"  eigenvalue phases: [{mono_eig_phases[0]:.6f}, {mono_eig_phases[1]:.6f}]", fh)
        for ep in mono_eig_phases:
            is_r, r_str = _is_rational_multiple_of_pi(ep)
            log(f"    {ep:.6f} {'≈ ' + r_str if is_r else ''}", fh)
        log(f"  trace = {mono_trace.real:.6f} + {mono_trace.imag:.6f}i", fh)
        log(f"  |trace| = {abs(mono_trace):.6f}", fh)
        log(f"  Is M = ±I? {np.allclose(monodromy, np.eye(2), atol=0.1) or np.allclose(monodromy, -np.eye(2), atol=0.1)}", fh)
        log("", fh)

        # ---- 4. Edge length / scattering correlations ----
        log("\n4. EDGE LENGTH / SCATTERING CORRELATIONS", fh)
        log("-" * 50, fh)

        # Correlation between edge lengths and det phases
        if len(edge_lengths) == len(det_phases):
            corr_det = np.corrcoef(edge_lengths, det_phases)[0, 1]
            log(f"  Corr(edge_length, det_phase): {corr_det:.4f}", fh)

        # Correlation between edge lengths and |trace|
        trace_mags = [abs(t) for t in traces]
        if len(edge_lengths) == len(trace_mags):
            corr_trace = np.corrcoef(edge_lengths, trace_mags)[0, 1]
            log(f"  Corr(edge_length, |trace|): {corr_trace:.4f}", fh)

        # Correlation between edge lengths and distance from Neumann
        if len(edge_lengths) == len(distances_from_neumann):
            corr_dist = np.corrcoef(edge_lengths, distances_from_neumann)[0, 1]
            log(f"  Corr(edge_length, dist_from_Neumann): {corr_dist:.4f}", fh)

        # Product: edge_length * det_phase
        products = edge_lengths * np.array(det_phases)
        log(f"\n  L_v * det_phase_v products: {[f'{p:.4f}' for p in products]}", fh)
        log(f"    Std dev: {np.std(products):.6f}", fh)
        log(f"    Are they constant? {'POSSIBLY' if np.std(products) / (np.mean(np.abs(products)) + 1e-10) < 0.3 else 'NO'}", fh)
        log("", fh)

        # ---- 5. SU(2) structure check ----
        log("\n5. SU(2) vs U(2) STRUCTURE", fh)
        log("-" * 50, fh)
        log("  SU(2) ⊂ U(2) has det(S) = 1 (zero det phase).", fh)
        log("  If the optimizer prefers SU(2), det phases would be ~0.", fh)
        log("", fh)

        n_su2 = sum(1 for dp in det_phases if abs(dp) < 0.1)
        log(f"  Vertices with |det_phase| < 0.1: {n_su2}/{n}", fh)
        log(f"  Vertices with |det_phase| < 0.3: "
            f"{sum(1 for dp in det_phases if abs(dp) < 0.3)}/{n}", fh)
        log(f"  The matrices are {'close to SU(2)' if n_su2 > n // 2 else 'genuinely U(2)'}", fh)
        log("", fh)

        # ---- 6. Reflection / transmission coefficients ----
        log("\n6. REFLECTION vs TRANSMISSION", fh)
        log("-" * 50, fh)
        log("  For degree-2 vertices, S = [[r, t'], [t, r']]", fh)
        log("  Neumann: r=0, t=1 (pure transmission)", fh)
        log("  How much reflection did the optimizer introduce?", fh)
        log("", fh)

        for v, u in enumerate(matrices):
            r_coeff = abs(u[0, 0])  # reflection
            t_coeff = abs(u[0, 1])  # transmission
            log(f"  Vertex {v}: |r|={r_coeff:.4f}, |t|={t_coeff:.4f}, "
                f"|r|²={r_coeff**2:.4f}, |t|²={t_coeff**2:.4f}  "
                f"({'mostly transmit' if t_coeff > r_coeff else 'mostly reflect'})", fh)

        mean_r = np.mean([abs(u[0, 0]) for u in matrices])
        mean_t = np.mean([abs(u[0, 1]) for u in matrices])
        log(f"\n  Mean |r| = {mean_r:.4f}, Mean |t| = {mean_t:.4f}", fh)
        log(f"  Mean |r|² = {mean_r**2:.4f}, Mean |t|² = {mean_t**2:.4f}", fh)
        log(f"  The optimizer {'introduced significant reflection' if mean_r > 0.3 else 'kept mostly transmission-like'}", fh)
        log("", fh)

        # ---- 7. Summary ----
        log("\n" + "=" * 70, fh)
        log("SUMMARY", fh)
        log("=" * 70, fh)

        findings = []
        if np.std(det_phases_arr) < 0.05:
            findings.append("Constant determinant phase across vertices")
        if is_sum_rat:
            findings.append(f"Sum of det phases = {sum_rat_str}")
        if is_det_rat:
            findings.append(f"Monodromy determinant phase = {det_rat_str}")
        if np.mean(dists) < 0.5:
            findings.append("Matrices close to Neumann")
        elif np.mean(dists) > 1.5:
            findings.append("Matrices far from Neumann")
        if mean_r > 0.3:
            findings.append(f"Significant reflection: mean |r|={mean_r:.3f}")
        if n_su2 > n // 2:
            findings.append("Matrices approximately SU(2)")
        if abs(corr_det) > 0.7:
            findings.append(f"Strong edge-length/det-phase correlation: {corr_det:.3f}")

        if findings:
            log("  Structural patterns found:", fh)
            for f_str in findings:
                log(f"    - {f_str}", fh)
        else:
            log("  No strong structural patterns detected.", fh)
            log("  The scattering matrices appear numerically generic.", fh)

        log("", fh)
        log(f"Results saved to {OUTPUT_FILE}", fh)


if __name__ == "__main__":
    run_analysis()
