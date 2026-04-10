#!/usr/bin/env python3
"""Periodicity analysis: check if the best cycle graph's edge lengths
have commensurable ratios that would produce quasi-periodic spectra.

Reads results from the scaling experiment and analyzes edge length ratios.
Output: results/periodicity_analysis.txt
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "periodicity_analysis.txt"
SCALING_JSONL = RESULTS_DIR / "scaling_experiment.jsonl"


def log(msg: str, fh=None) -> None:
    print(msg)
    sys.stdout.flush()
    if fh:
        fh.write(msg + "\n")


def analyze_commensurability(lengths: list[float], tol: float = 0.02) -> dict:
    """Check if edge lengths are approximately commensurable.

    Two lengths l_i, l_j are commensurable if l_i/l_j ≈ p/q for small p, q.
    If ALL pairs are commensurable, the spectrum is quasi-periodic.
    """
    n = len(lengths)
    ratios = []
    commensurable_pairs = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            ratio = lengths[i] / lengths[j]
            # Find best rational approximation with small denominator
            frac = Fraction(ratio).limit_denominator(20)
            approx = float(frac)
            error = abs(ratio - approx) / max(abs(ratio), 1e-10)

            is_comm = error < tol
            if is_comm:
                commensurable_pairs += 1
            total_pairs += 1

            ratios.append({
                "i": i, "j": j,
                "ratio": ratio,
                "fraction": str(frac),
                "error": error,
                "commensurable": is_comm,
            })

    return {
        "commensurable_pairs": commensurable_pairs,
        "total_pairs": total_pairs,
        "fraction_commensurable": commensurable_pairs / max(total_pairs, 1),
        "ratios": ratios,
    }


def run_analysis() -> None:
    """Run periodicity analysis on scaling experiment results."""
    if not SCALING_JSONL.exists():
        print(f"No scaling results found at {SCALING_JSONL}.")
        print("Run scaling_experiment.py first.")
        sys.exit(1)

    records = []
    with open(SCALING_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Find the best cycle graphs at key sizes
    cycle_records = [
        r for r in records
        if r["graph_family"] in ("cycle", "cycle_extended")
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as fh:
        log("=" * 70, fh)
        log("PERIODICITY / COMMENSURABILITY ANALYSIS", fh)
        log("=" * 70, fh)
        log("", fh)
        log("Question: Do the optimized edge lengths form commensurable ratios", fh)
        log("(l_i/l_j ≈ p/q for small integers p,q)? If yes, the spectrum is", fh)
        log("quasi-periodic, which limits how well it can match the aperiodic", fh)
        log("Riemann zeros.", fh)
        log("", fh)

        for r in sorted(cycle_records, key=lambda x: x["n_vertices"]):
            n = r["n_vertices"]
            # Edge lengths aren't stored in the scaling CSV directly
            # We need them from the JSONL
            lengths_raw = r.get("edge_lengths") or r.get("best_edge_lengths")
            if not lengths_raw:
                # Reconstruct: for cycles with Neumann, the optimizer found
                # these lengths. If not available, skip.
                continue

            lengths = [float(l) for l in lengths_raw]
            total_L = sum(lengths)

            log(f"--- C_{n} (score={r['best_score']:.4f}, L={total_L:.2f}) ---", fh)
            log(f"  Edge lengths: {[f'{l:.4f}' for l in lengths]}", fh)

            result = analyze_commensurability(lengths)

            log(f"  Commensurable pairs: {result['commensurable_pairs']}"
                f"/{result['total_pairs']} "
                f"({result['fraction_commensurable']:.0%})", fh)

            # Show the ratio table
            for ratio in result["ratios"]:
                status = "COMM" if ratio["commensurable"] else "    "
                log(f"    L_{ratio['i']}/L_{ratio['j']} = {ratio['ratio']:.4f} "
                    f"≈ {ratio['fraction']:>5s} "
                    f"(err={ratio['error']:.4f}) {status}", fh)
            log("", fh)

        # Overall conclusion
        log("=" * 70, fh)
        log("CONCLUSION", fh)
        log("=" * 70, fh)

        # Check if the best graph has high commensurability
        best_cycle = max(cycle_records, key=lambda x: x.get("best_score", 0))
        best_lengths = best_cycle.get("edge_lengths") or best_cycle.get("best_edge_lengths")

        if best_lengths:
            best_result = analyze_commensurability([float(l) for l in best_lengths])
            frac = best_result["fraction_commensurable"]

            if frac > 0.7:
                log(f"The best graph (C_{best_cycle['n_vertices']}) has "
                    f"{frac:.0%} commensurable edge length pairs.", fh)
                log("This indicates a QUASI-PERIODIC spectrum, which fundamentally", fh)
                log("limits how well it can approximate the aperiodic Riemann zeros.", fh)
                log("The scoring ceiling may be a consequence of this quasi-periodicity.", fh)
            elif frac > 0.3:
                log(f"The best graph has {frac:.0%} commensurable pairs — partial", fh)
                log("commensurability. The spectrum is neither fully periodic nor", fh)
                log("fully incommensurable.", fh)
            else:
                log(f"The best graph has only {frac:.0%} commensurable pairs.", fh)
                log("Edge lengths are largely incommensurable, suggesting the", fh)
                log("optimizer is NOT constrained to quasi-periodic spectra.", fh)
                log("The ceiling is not caused by quasi-periodicity.", fh)
        else:
            log("Could not analyze best graph — edge lengths not available in results.", fh)
            log("Re-run the scaling experiment to capture edge lengths.", fh)

    log(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_analysis()
