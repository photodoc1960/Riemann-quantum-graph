#!/usr/bin/env python3
"""PSLQ algebraic structure analysis of the optimal U(2) scattering matrices.

Determines whether the optimized vertex conditions contain hidden algebraic
structure — integer relations involving known mathematical constants.

Reads: results/best_unitary_graph.json
Writes: results/pslq_analysis.json, console report
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

mpmath.mp.dps = 50  # 50 decimal places throughout

GRAPH_FILE = Path("results/best_unitary_graph.json")
OUTPUT_FILE = Path("results/pslq_analysis.json")

# Constants for PSLQ basis
CONST_NAMES = [
    "pi", "ln2", "ln3", "ln5", "ln7", "ln11", "ln13",
    "euler", "sqrt2", "sqrt3", "sqrt5", "phi",
]
CONST_VALUES = [
    mpmath.pi,
    mpmath.log(2),
    mpmath.log(3),
    mpmath.log(5),
    mpmath.log(7),
    mpmath.log(11),
    mpmath.log(13),
    mpmath.euler,
    mpmath.sqrt(2),
    mpmath.sqrt(3),
    mpmath.sqrt(5),
    mpmath.phi,
]


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def _try_identify(x: float, label: str) -> list[dict]:
    """Try mpmath.identify on a value."""
    findings = []
    try:
        result = mpmath.identify(mpmath.mpf(x), tol=1e-20)
        if result is not None:
            findings.append({
                "label": label,
                "value": float(x),
                "identified_as": str(result),
                "method": "mpmath.identify",
            })
    except Exception:
        pass
    return findings


def _try_pslq_linear(x: float, label: str) -> list[dict]:
    """PSLQ: test x = a₀ + a₁π + a₂ln2 + a₃ln3 + a₄ln5 + a₅ln7 + a₆ln11 + a₇ln13."""
    findings = []
    basis = [mpmath.mpf(x), mpmath.mpf(1)] + CONST_VALUES[:7]  # pi, ln2..ln13
    basis_names = [label, "1"] + CONST_NAMES[:7]

    try:
        rel = mpmath.pslq(basis, maxcoeff=10, tol=1e-20)
        if rel is not None:
            # rel[0]*x + rel[1]*1 + rel[2]*pi + ... = 0
            # Check that not all zero and rel[0] != 0
            if rel[0] != 0 and any(r != 0 for r in rel[1:]):
                residual = float(abs(sum(c * v for c, v in zip(rel, basis))))
                if residual < 1e-20:
                    # Express as x = -(rel[1] + rel[2]*pi + ...)/rel[0]
                    terms = []
                    for i, (c, name) in enumerate(zip(rel[1:], basis_names[1:])):
                        if c != 0:
                            terms.append(f"{c}*{name}")
                    formula = f"{label} = -({' + '.join(terms)})/{rel[0]}"
                    findings.append({
                        "label": label,
                        "value": float(x),
                        "relation": [int(c) for c in rel],
                        "formula": formula,
                        "residual": residual,
                        "method": "pslq_linear",
                    })
    except Exception:
        pass
    return findings


def _try_pslq_quadratic(x: float, label: str) -> list[dict]:
    """PSLQ: test x² = a₀ + a₁π + a₂ln2 + a₃ln3 + a₄ln5."""
    findings = []
    x2 = mpmath.mpf(x) ** 2
    basis = [x2, mpmath.mpf(1)] + CONST_VALUES[:5]

    try:
        rel = mpmath.pslq(basis, maxcoeff=10, tol=1e-20)
        if rel is not None and rel[0] != 0 and any(r != 0 for r in rel[1:]):
            residual = float(abs(sum(c * v for c, v in zip(rel, basis))))
            if residual < 1e-20:
                findings.append({
                    "label": f"{label}²",
                    "value": float(x2),
                    "relation": [int(c) for c in rel],
                    "residual": residual,
                    "method": "pslq_quadratic",
                })
    except Exception:
        pass
    return findings


def _try_pslq_pair(x1: float, x2: float, l1: str, l2: str) -> list[dict]:
    """PSLQ: test relation between two values and constants."""
    findings = []
    basis = [mpmath.mpf(x1), mpmath.mpf(x2), mpmath.mpf(1),
             mpmath.pi, mpmath.log(2), mpmath.log(3)]

    try:
        rel = mpmath.pslq(basis, maxcoeff=10, tol=1e-20)
        if rel is not None:
            n_nonzero = sum(1 for r in rel if r != 0)
            if n_nonzero >= 2 and (rel[0] != 0 or rel[1] != 0):
                residual = float(abs(sum(c * v for c, v in zip(rel, basis))))
                if residual < 1e-20:
                    findings.append({
                        "labels": [l1, l2],
                        "values": [float(x1), float(x2)],
                        "relation": [int(c) for c in rel],
                        "residual": residual,
                        "method": "pslq_pair",
                    })
    except Exception:
        pass
    return findings


def _test_phase_rationals(phases: list[float], labels: list[str],
                          max_denom: int = 20, tol: float = 1e-8) -> list[dict]:
    """Test pairwise phase ratios against rationals."""
    findings = []
    for i in range(len(phases)):
        for j in range(i + 1, len(phases)):
            if abs(phases[j]) < 1e-15:
                continue
            ratio = phases[i] / phases[j]
            frac = Fraction(ratio).limit_denominator(max_denom)
            approx = float(frac)
            error = abs(ratio - approx)
            if error < tol:
                findings.append({
                    "labels": [labels[i], labels[j]],
                    "ratio": float(ratio),
                    "fraction": str(frac),
                    "error": error,
                })
    return findings


def _test_phase_pi_multiples(phases: list[float], labels: list[str],
                             tol: float = 1e-8) -> list[dict]:
    """Test if phases are rational multiples of π."""
    findings = []
    for phase, label in zip(phases, labels):
        ratio = phase / float(mpmath.pi)
        frac = Fraction(ratio).limit_denominator(20)
        approx = float(frac) * float(mpmath.pi)
        error = abs(phase - approx)
        if error < tol:
            findings.append({
                "label": label,
                "phase": float(phase),
                "pi_multiple": str(frac),
                "error": error,
            })
    return findings


def run_analysis() -> None:
    """Run the full PSLQ analysis."""
    if not GRAPH_FILE.exists():
        log(f"No graph found at {GRAPH_FILE}")
        sys.exit(1)

    with open(GRAPH_FILE) as f:
        spec = json.load(f)

    log("=" * 60)
    log("PSLQ ALGEBRAIC STRUCTURE ANALYSIS")
    log("=" * 60)
    log(f"Graph: {spec['graph_type']}")
    log(f"Score: {spec['score']:.6f}")
    log("")

    n = spec["n_vertices"]
    edge_lengths = spec["edge_lengths"]

    # Reconstruct matrices at high precision
    matrices = []
    all_params = []
    for sm in spec["scattering_matrices"]:
        mr = np.array(sm["matrix_real"])
        mi = np.array(sm["matrix_imag"])
        u = mr + 1j * mi
        matrices.append(u)
        all_params.extend(sm["params"])

    all_findings = []

    # ---- Step 1: Extract all numerical parameters ----
    log("STEP 1: Extracting numerical parameters...")

    # Matrix entries (real and imag parts)
    matrix_entries_real = []
    matrix_entries_imag = []
    entry_labels_real = []
    entry_labels_imag = []
    for v, u in enumerate(matrices):
        for i in range(2):
            for j in range(2):
                matrix_entries_real.append(u[i, j].real)
                matrix_entries_imag.append(u[i, j].imag)
                entry_labels_real.append(f"Re(S{v}[{i},{j}])")
                entry_labels_imag.append(f"Im(S{v}[{i},{j}])")

    # Eigenvalue phases
    eig_phases = []
    eig_labels = []
    for v, u in enumerate(matrices):
        eigvals = np.linalg.eigvals(u)
        for k, ev in enumerate(sorted(eigvals, key=lambda x: np.angle(x))):
            eig_phases.append(float(np.angle(ev)))
            eig_labels.append(f"arg(eig{k}_v{v})")

    # Determinant phases
    det_phases = []
    det_labels = []
    for v, u in enumerate(matrices):
        det_phases.append(float(np.angle(np.linalg.det(u))))
        det_labels.append(f"arg(det_v{v})")

    # Monodromy
    mono = np.eye(2, dtype=np.complex128)
    for u in matrices:
        mono = mono @ u
    mono_eigvals = np.linalg.eigvals(mono)
    mono_phases = [float(np.angle(ev)) for ev in sorted(mono_eigvals, key=lambda x: np.angle(x))]
    mono_labels = ["arg(mono_eig0)", "arg(mono_eig1)"]

    # CMA-ES raw params
    param_labels = []
    for v in range(n):
        for pname in ["alpha", "beta", "gamma", "theta"]:
            param_labels.append(f"{pname}_v{v}")

    log(f"  Matrix entries: {len(matrix_entries_real)} real + {len(matrix_entries_imag)} imag")
    log(f"  Eigenvalue phases: {len(eig_phases)}")
    log(f"  Determinant phases: {len(det_phases)}")
    log(f"  Monodromy phases: {len(mono_phases)}")
    log(f"  Edge lengths: {len(edge_lengths)}")
    log(f"  CMA-ES params: {len(all_params)}")
    log("")

    # ---- Step 2: PSLQ integer relation search ----
    log("STEP 2: PSLQ integer relation search...")

    # Test eigenvalue phases (most likely to have structure)
    log("  Testing eigenvalue phases...")
    for phase, label in zip(eig_phases, eig_labels):
        all_findings.extend(_try_identify(phase, label))
        all_findings.extend(_try_pslq_linear(phase, label))

    # Test determinant phases
    log("  Testing determinant phases...")
    for phase, label in zip(det_phases, det_labels):
        all_findings.extend(_try_identify(phase, label))
        all_findings.extend(_try_pslq_linear(phase, label))

    # Test monodromy phases (highest priority)
    log("  Testing monodromy phases...")
    for phase, label in zip(mono_phases, mono_labels):
        all_findings.extend(_try_identify(phase, label))
        all_findings.extend(_try_pslq_linear(phase, label))
        all_findings.extend(_try_pslq_quadratic(phase, label))

    # Test edge lengths
    log("  Testing edge lengths...")
    for length, v in zip(edge_lengths, range(n)):
        label = f"L_{v}"
        all_findings.extend(_try_identify(length, label))
        all_findings.extend(_try_pslq_linear(length, label))

    # Test CMA-ES params
    log("  Testing CMA-ES raw parameters...")
    for param, label in zip(all_params, param_labels):
        all_findings.extend(_try_identify(param, label))
        all_findings.extend(_try_pslq_linear(param, label))

    # Test selected pairs
    log("  Testing parameter pairs...")
    # Pairs of eigenvalue phases within each vertex
    for v in range(n):
        p1, p2 = eig_phases[2*v], eig_phases[2*v+1]
        l1, l2 = eig_labels[2*v], eig_labels[2*v+1]
        all_findings.extend(_try_pslq_pair(p1, p2, l1, l2))

    # Pairs of det phases
    for i in range(n):
        for j in range(i+1, min(n, i+3)):
            all_findings.extend(_try_pslq_pair(
                det_phases[i], det_phases[j],
                det_labels[i], det_labels[j],
            ))

    # Edge length vs det phase at same vertex
    for v in range(n):
        all_findings.extend(_try_pslq_pair(
            edge_lengths[v], det_phases[v],
            f"L_{v}", det_labels[v],
        ))

    log(f"  Found {len(all_findings)} PSLQ relations")
    log("")

    # ---- Step 3: Phase ratio analysis ----
    log("STEP 3: Phase ratio analysis...")

    phase_rationals = _test_phase_rationals(eig_phases, eig_labels)
    phase_rationals.extend(_test_phase_rationals(det_phases, det_labels))
    log(f"  Found {len(phase_rationals)} rational phase ratios")

    pi_multiples = _test_phase_pi_multiples(eig_phases, eig_labels)
    pi_multiples.extend(_test_phase_pi_multiples(det_phases, det_labels))
    pi_multiples.extend(_test_phase_pi_multiples(mono_phases, mono_labels))
    log(f"  Found {len(pi_multiples)} phases that are rational multiples of pi")
    log("")

    # ---- Step 4: Monodromy deep analysis ----
    log("STEP 4: Monodromy deep analysis...")

    # High-precision monodromy
    mono_hp = mpmath.eye(2)
    for sm in spec["scattering_matrices"]:
        mr = sm["matrix_real"]
        mi = sm["matrix_imag"]
        u_hp = mpmath.matrix([
            [mpmath.mpf(mr[0][0]) + mpmath.mpf(mi[0][0]) * 1j,
             mpmath.mpf(mr[0][1]) + mpmath.mpf(mi[0][1]) * 1j],
            [mpmath.mpf(mr[1][0]) + mpmath.mpf(mi[1][0]) * 1j,
             mpmath.mpf(mr[1][1]) + mpmath.mpf(mi[1][1]) * 1j],
        ])
        mono_hp = mono_hp * u_hp

    mono_det = mpmath.det(mono_hp)
    mono_trace = mono_hp[0, 0] + mono_hp[1, 1]

    log(f"  det(M) = {float(abs(mono_det)):.15f} * exp(i*{float(mpmath.arg(mono_det)):.15f})")
    log(f"  trace(M) = {float(mono_trace.real):.15f} + {float(mono_trace.imag):.15f}i")

    # Test M^k for small k
    monodromy_powers = {}
    mk = mpmath.eye(2)
    for k in range(1, 15):
        mk = mk * mono_hp
        mk_trace = mk[0, 0] + mk[1, 1]
        is_I = (abs(mk[0, 0] - 1) < 1e-10 and abs(mk[1, 1] - 1) < 1e-10 and
                abs(mk[0, 1]) < 1e-10 and abs(mk[1, 0]) < 1e-10)
        is_neg_I = (abs(mk[0, 0] + 1) < 1e-10 and abs(mk[1, 1] + 1) < 1e-10 and
                    abs(mk[0, 1]) < 1e-10 and abs(mk[1, 0]) < 1e-10)
        if is_I or is_neg_I:
            monodromy_powers[k] = "I" if is_I else "-I"
            log(f"  M^{k} = {'I' if is_I else '-I'}  *** EXACT ***")
        elif abs(float(mk_trace.real) - 2) < 0.01 and abs(float(mk_trace.imag)) < 0.01:
            log(f"  M^{k}: trace ≈ 2 (near I)")
        elif abs(float(mk_trace.real) + 2) < 0.01 and abs(float(mk_trace.imag)) < 0.01:
            log(f"  M^{k}: trace ≈ -2 (near -I)")

    # Characteristic polynomial: λ² - trace(M)·λ + det(M) = 0
    char_poly_coeffs = [mpmath.mpf(1), -mono_trace, mono_det]
    log(f"\n  Characteristic polynomial: lambda^2 + ({float(-mono_trace.real):.10f}"
        f"{float(-mono_trace.imag):+.10f}i)*lambda + ({float(mono_det.real):.10f}"
        f"{float(mono_det.imag):+.10f}i)")

    # Test trace and det against known constants
    trace_findings = _try_identify(float(mono_trace.real), "Re(trace_M)")
    trace_findings.extend(_try_identify(float(mono_trace.imag), "Im(trace_M)"))
    all_findings.extend(trace_findings)

    monodromy_analysis = {
        "det_magnitude": float(abs(mono_det)),
        "det_phase": float(mpmath.arg(mono_det)),
        "trace_real": float(mono_trace.real),
        "trace_imag": float(mono_trace.imag),
        "eigenvalue_phases": mono_phases,
        "powers_equal_I_or_neg_I": monodromy_powers,
        "near_minus_one": abs(abs(mono_phases[0]) - float(mpmath.pi)) < 0.01,
    }

    log("")

    # ---- Step 5: Edge length analysis ----
    log("STEP 5: Edge length analysis...")

    # PSLQ against extended set of ln(primes)
    primes_extended = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    ln_primes = [mpmath.log(p) for p in primes_extended]

    edge_length_findings = []
    for v, length in enumerate(edge_lengths):
        basis = [mpmath.mpf(length), mpmath.mpf(1)] + ln_primes
        try:
            rel = mpmath.pslq(basis, maxcoeff=10, tol=1e-20)
            if rel is not None and rel[0] != 0 and any(r != 0 for r in rel[1:]):
                residual = float(abs(sum(c * v for c, v in zip(rel, basis))))
                if residual < 1e-20:
                    terms = []
                    for i, (c, p) in enumerate(zip(rel[2:], primes_extended)):
                        if c != 0:
                            terms.append(f"{c}*ln({p})")
                    if rel[1] != 0:
                        terms.insert(0, str(rel[1]))
                    formula = f"L_{v} = -({' + '.join(terms)})/{rel[0]}"
                    edge_length_findings.append({
                        "vertex": v,
                        "length": float(length),
                        "formula": formula,
                        "relation": [int(c) for c in rel],
                        "residual": residual,
                    })
        except Exception:
            pass

    # Total length
    total_L = sum(edge_lengths)
    total_findings = _try_identify(total_L, "L_total")
    total_findings.extend(_try_pslq_linear(total_L, "L_total"))

    # Length ratios
    length_rationals = _test_phase_rationals(edge_lengths,
                                              [f"L_{v}" for v in range(n)])

    edge_analysis = {
        "pslq_ln_prime_relations": edge_length_findings,
        "total_length": float(total_L),
        "total_length_findings": total_findings,
        "length_rationals": length_rationals,
    }

    log(f"  Edge lengths with ln(p) relations: {len(edge_length_findings)}")
    log(f"  Total length findings: {len(total_findings)}")
    log(f"  Rational length ratios: {len(length_rationals)}")
    log("")

    # ---- Determine verdict ----
    n_significant = len(all_findings) + len(pi_multiples) + len(edge_length_findings)
    n_monodromy_structure = len(monodromy_powers) + (1 if monodromy_analysis["near_minus_one"] else 0)

    if n_significant > 10 or len(monodromy_powers) > 0:
        verdict = "STRUCTURED"
    elif n_significant > 3 or n_monodromy_structure > 0:
        verdict = "PARTIALLY_STRUCTURED"
    else:
        verdict = "GENERIC"

    # ---- Save results ----
    output = {
        "summary": f"Analyzed {n} vertex scattering matrices from best U(2) graph",
        "significant_relations": all_findings,
        "phase_rationals": phase_rationals,
        "pi_multiples": pi_multiples,
        "monodromy_analysis": monodromy_analysis,
        "edge_length_analysis": edge_analysis,
        "verdict": verdict,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ---- Console report ----
    log("=" * 60)
    log("PSLQ ALGEBRAIC STRUCTURE ANALYSIS — RESULTS")
    log("=" * 60)
    log(f"Matrix entries: {len([f for f in all_findings if 'Re(S' in f.get('label','') or 'Im(S' in f.get('label','')])} relations found")
    log(f"Eigenvalue phases: {len([f for f in all_findings if 'eig' in f.get('label','')])} relations found")
    log(f"  Pi-multiples: {len([f for f in pi_multiples if 'eig' in f.get('label','')])} phases")
    log(f"Determinant phases: {len([f for f in all_findings if 'det' in f.get('label','')])} relations found")
    log(f"  Pi-multiples: {len([f for f in pi_multiples if 'det' in f.get('label','')])} phases")
    log(f"Monodromy eigenvalues: phases = [{mono_phases[0]:.8f}, {mono_phases[1]:.8f}]")
    log(f"  Near -1: {monodromy_analysis['near_minus_one']}")
    log(f"  Powers equal ±I: {monodromy_powers if monodromy_powers else 'none found (k=1..14)'}")
    log(f"Edge lengths: {len(edge_length_findings)} ln(p) relations found")
    log(f"Phase rationals: {len(phase_rationals)} rational ratios found")

    if all_findings:
        log("\nSIGNIFICANT FINDINGS:")
        for f_item in all_findings[:20]:
            if "formula" in f_item:
                log(f"  {f_item['formula']} (residual: {f_item.get('residual', '?'):.2e})")
            elif "identified_as" in f_item:
                log(f"  {f_item['label']} = {f_item['identified_as']}")
            else:
                log(f"  {f_item.get('label', f_item.get('labels', '?'))}: {f_item.get('relation', '?')}")

    if pi_multiples:
        log("\nPI-MULTIPLE PHASES:")
        for pm in pi_multiples[:15]:
            log(f"  {pm['label']} = {pm['pi_multiple']}*pi (error: {pm['error']:.2e})")

    if edge_length_findings:
        log("\nEDGE LENGTH ln(p) RELATIONS:")
        for ef in edge_length_findings:
            log(f"  {ef['formula']} (residual: {ef['residual']:.2e})")

    log(f"\nVERDICT: {verdict}")
    if verdict == "STRUCTURED":
        log("MATHEMATICAL STRUCTURE DETECTED — recommend analytic investigation")
    elif verdict == "PARTIALLY_STRUCTURED":
        log("Partial structure detected — primarily in monodromy and phase ratios.")
        log("Individual matrices appear generic; structure emerges in composition.")
    else:
        log("No algebraic structure detected. Optimal matrices appear numerically")
        log("generic. Structure, if any, lies in the composition (monodromy)")
        log("rather than individual matrices.")

    log(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_analysis()
