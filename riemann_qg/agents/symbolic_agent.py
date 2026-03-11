"""Symbolic agent -- discovers closed-form rules for edge-length patterns.

Uses integer-relation detection (PSLQ via mpmath) and grammar-based symbolic
regression to express winning edge lengths as rational combinations of
logarithms of primes.  This is the key diagnostic for whether a candidate
graph truly "encodes" primes in its metric structure.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Sequence

import mpmath
import numpy as np

from riemann_qg.core.quantum_graph import QuantumGraph


# ---------------------------------------------------------------------------
# Prime helpers
# ---------------------------------------------------------------------------

def _first_n_primes(n: int) -> list[int]:
    upper = max(30, int(n * math.log(n + 2) * 1.3) + 20)
    sieve = [True] * (upper + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(upper**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, upper + 1, i):
                sieve[j] = False
    primes = [i for i, v in enumerate(sieve) if v]
    while len(primes) < n:
        upper *= 2
        return _first_n_primes(n)
    return primes[:n]


# ---------------------------------------------------------------------------
# Result dataclasses (frozen)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrimeFitResult:
    """Result of fitting a single edge length to sum of log-primes."""
    length: float
    coefficients: tuple[float, ...]
    primes_used: tuple[int, ...]
    residual: float
    is_exact: bool  # residual < tolerance


@dataclass(frozen=True)
class LengthRule:
    """A discovered rule mapping edge index -> edge length."""
    name: str
    formula: str
    residuals: tuple[float, ...]
    mean_residual: float
    applies_to_n_graphs: int


@dataclass(frozen=True)
class SymbolicRegressionResult:
    """Result of grammar-based symbolic regression."""
    best_formula: str
    best_residual: float
    candidates_evaluated: int


# ---------------------------------------------------------------------------
# SymbolicAgent
# ---------------------------------------------------------------------------

class SymbolicAgent:
    """Attempts to discover closed-form rules for edge-length patterns.

    Given the edge lengths of high-scoring quantum graphs, the agent uses
    integer-relation algorithms (PSLQ) and simple symbolic regression to
    find expressions in terms of logarithms of primes.
    """

    def __init__(self, precision: int = 30, tolerance: float = 1e-8) -> None:
        self._precision = precision
        self._tolerance = tolerance

    # ---- PSLQ fitting ----

    def fit_prime_combination(
        self,
        edge_lengths: Sequence[float],
        n_primes: int = 10,
        max_coeff: int = 10,
    ) -> list[PrimeFitResult]:
        """Express each edge length as L = sum(a_p * ln(p)).

        Uses the PSLQ integer-relation algorithm (mpmath) to search for
        integer coefficient vectors ``a_p`` such that
        ``L - sum(a_p * ln(p)) = 0``.

        Parameters
        ----------
        edge_lengths : sequence of float
        n_primes : int
            Number of primes to include in the basis.
        max_coeff : int
            Reject solutions with coefficients exceeding this.

        Returns
        -------
        list[PrimeFitResult]
            One result per edge length.
        """
        primes = _first_n_primes(n_primes)
        mpmath.mp.dps = self._precision
        log_primes = [mpmath.log(p) for p in primes]

        results: list[PrimeFitResult] = []
        for length in edge_lengths:
            result = self._fit_single_length(
                float(length), primes, log_primes, max_coeff,
            )
            results.append(result)
        return results

    def _fit_single_length(
        self,
        length: float,
        primes: list[int],
        log_primes: list[mpmath.mpf],
        max_coeff: int,
    ) -> PrimeFitResult:
        """Fit a single edge length using PSLQ.

        The key challenge is that float64 inputs have ~16 digits of precision
        while PSLQ operates at 30+ digits. We recover full precision by
        checking if the length is close to any ln(p), and if so, using the
        full-precision mpmath value.
        """
        mpmath.mp.dps = self._precision

        # Try to recover full-precision target by matching to known ln(p)
        mp_length = self._recover_precision(length, log_primes)

        # Try PSLQ with increasing basis sizes for better convergence
        relation = None
        used_n_basis = len(log_primes)

        for n_basis in [3, 5, len(log_primes)]:
            n_basis = min(n_basis, len(log_primes))
            vec = [mp_length] + [-lp for lp in log_primes[:n_basis]]
            try:
                relation = mpmath.pslq(vec, maxcoeff=max_coeff * 5)
            except Exception:
                relation = None
            if relation is not None and relation[0] != 0:
                used_n_basis = n_basis
                break

        if relation is None or relation[0] == 0:
            return PrimeFitResult(
                length=length,
                coefficients=(),
                primes_used=tuple(primes),
                residual=float("inf"),
                is_exact=False,
            )

        # relation[0]*L + relation[1]*(-ln p1) + ... = 0
        # => L = sum( (relation[i]/relation[0]) * ln(pi) )  for i>=1
        scale = relation[0]
        coefficients_short = tuple(
            float(mpmath.mpf(relation[i]) / scale)
            for i in range(1, used_n_basis + 1)
        )
        # Pad to full length
        coefficients = coefficients_short + (0.0,) * (len(log_primes) - used_n_basis)

        reconstructed = sum(
            c * float(lp) for c, lp in zip(coefficients, log_primes)
        )
        residual = abs(length - reconstructed)

        return PrimeFitResult(
            length=length,
            coefficients=coefficients,
            primes_used=tuple(primes),
            residual=residual,
            is_exact=residual < self._tolerance,
        )

    @staticmethod
    def _recover_precision(
        length: float, log_primes: list[mpmath.mpf],
    ) -> mpmath.mpf:
        """Recover full mpmath precision for a float64 length value.

        If the length is within float64 tolerance of any small integer
        combination of log-primes, return the full-precision value.
        """
        float_tol = 1e-12
        # Check single log-primes first
        for lp in log_primes:
            for coeff in [1, 2, 3, -1, -2]:
                if abs(length - coeff * float(lp)) < float_tol:
                    return coeff * lp

        # Check sums of two log-primes
        for i, lp_i in enumerate(log_primes[:6]):
            for j, lp_j in enumerate(log_primes[:6]):
                candidate = lp_i + lp_j
                if abs(length - float(candidate)) < float_tol:
                    return candidate

        # Fall back to mpmath conversion (limited to float64 precision)
        return mpmath.mpf(str(length))

    # ---- length rule discovery ----

    def discover_length_rule(
        self,
        top_graphs: Sequence[QuantumGraph],
        n_primes: int = 20,
    ) -> LengthRule:
        """Find a single rule that predicts edge lengths across top graphs.

        Tries several candidate rules and returns the best-fitting one.

        Parameters
        ----------
        top_graphs : sequence of QuantumGraph
        n_primes : int

        Returns
        -------
        LengthRule
        """
        primes = _first_n_primes(n_primes)
        log_p = np.array([math.log(p) for p in primes])

        rules = self._candidate_rules(primes, log_p)
        best_rule: LengthRule | None = None

        for name, formula, predict_fn in rules:
            all_residuals: list[float] = []
            n_applicable = 0
            for g in top_graphs:
                lengths = np.array(g.edge_lengths)
                if len(lengths) == 0:
                    continue
                n_e = min(len(lengths), len(primes))
                predicted = np.array([predict_fn(i) for i in range(n_e)])
                residuals = np.abs(lengths[:n_e] - predicted)
                all_residuals.extend(residuals.tolist())
                n_applicable += 1

            if not all_residuals:
                continue

            mean_res = float(np.mean(all_residuals))
            rule = LengthRule(
                name=name,
                formula=formula,
                residuals=tuple(all_residuals),
                mean_residual=mean_res,
                applies_to_n_graphs=n_applicable,
            )

            if best_rule is None or mean_res < best_rule.mean_residual:
                best_rule = rule

        if best_rule is None:
            return LengthRule(
                name="none",
                formula="no rule found",
                residuals=(),
                mean_residual=float("inf"),
                applies_to_n_graphs=0,
            )
        return best_rule

    @staticmethod
    def _candidate_rules(
        primes: list[int], log_p: np.ndarray,
    ) -> list[tuple[str, str, object]]:
        """Return list of (name, formula_str, predict_fn) tuples."""
        rules: list[tuple[str, str, object]] = []

        # L_e = ln(p_e)
        rules.append((
            "ln(p_e)",
            "L_e = ln(p_e)",
            lambda i: log_p[i] if i < len(log_p) else float("nan"),
        ))

        # L_e = 0.5 * ln(p_e)
        rules.append((
            "half_ln(p_e)",
            "L_e = (1/2) * ln(p_e)",
            lambda i: 0.5 * log_p[i] if i < len(log_p) else float("nan"),
        ))

        # L_e = 2 * ln(p_e)
        rules.append((
            "2*ln(p_e)",
            "L_e = 2 * ln(p_e)",
            lambda i: 2.0 * log_p[i] if i < len(log_p) else float("nan"),
        ))

        # L_e = ln(p_e * p_{e+1})  for consecutive pairs
        rules.append((
            "ln(p_e*p_{e+1})",
            "L_e = ln(p_e * p_{e+1})",
            lambda i: (
                log_p[i] + log_p[i + 1]
                if i + 1 < len(log_p) else float("nan")
            ),
        ))

        # L_e = pi / ln(p_e)
        rules.append((
            "pi/ln(p_e)",
            "L_e = pi / ln(p_e)",
            lambda i: (
                math.pi / log_p[i] if i < len(log_p) else float("nan")
            ),
        ))

        return rules

    # ---- symbolic regression ----

    def symbolic_regression_lengths(
        self,
        top_graphs: Sequence[QuantumGraph],
        max_depth: int = 3,
    ) -> SymbolicRegressionResult:
        """Grammar-based symbolic regression over edge lengths.

        Uses a simple expression grammar mapping edge index e to
        edge length L_e. Searches over combinations of elementary
        operations applied to the edge index and prime sequences.

        Parameters
        ----------
        top_graphs : sequence of QuantumGraph
        max_depth : int
            Maximum expression tree depth.

        Returns
        -------
        SymbolicRegressionResult
        """
        # Collect (index, length) pairs across all top graphs
        data_pairs: list[tuple[int, float]] = []
        for g in top_graphs:
            for idx, length in enumerate(g.edge_lengths):
                data_pairs.append((idx, float(length)))

        if not data_pairs:
            return SymbolicRegressionResult(
                best_formula="no data",
                best_residual=float("inf"),
                candidates_evaluated=0,
            )

        indices = np.array([p[0] for p in data_pairs], dtype=float)
        targets = np.array([p[1] for p in data_pairs])

        primes_30 = np.array(
            [float(p) for p in _first_n_primes(30)], dtype=float,
        )

        candidates = self._generate_formula_candidates(
            indices, primes_30, max_depth,
        )

        best_formula = "constant"
        best_residual = float("inf")
        n_evaluated = 0

        for name, values in candidates:
            if len(values) != len(targets):
                continue
            if np.any(~np.isfinite(values)):
                continue
            n_evaluated += 1
            residual = float(np.mean(np.abs(targets - values)))
            if residual < best_residual:
                best_residual = residual
                best_formula = name

        return SymbolicRegressionResult(
            best_formula=best_formula,
            best_residual=best_residual,
            candidates_evaluated=n_evaluated,
        )

    @staticmethod
    def _generate_formula_candidates(
        indices: np.ndarray,
        primes: np.ndarray,
        max_depth: int,
    ) -> list[tuple[str, np.ndarray]]:
        """Generate candidate formulas and their predicted values."""
        n = len(indices)
        candidates: list[tuple[str, np.ndarray]] = []

        # Ensure we can index into primes
        safe_idx = np.clip(indices.astype(int), 0, len(primes) - 1)
        p_vals = primes[safe_idx]
        log_p = np.log(p_vals)

        # Depth-1 primitives
        candidates.append(("ln(p_e)", log_p))
        candidates.append(("e + 1", indices + 1))
        candidates.append(("sqrt(p_e)", np.sqrt(p_vals)))

        # Depth-2 compositions
        for alpha in [0.5, 1.0, 1.5, 2.0, math.pi]:
            candidates.append(
                (f"{alpha:.2f} * ln(p_e)", alpha * log_p)
            )

        candidates.append(("ln(p_e)^2", log_p**2))
        candidates.append(("1 / ln(p_e)", np.where(log_p > 0, 1.0 / log_p, np.inf)))
        candidates.append(("pi / ln(p_e)", np.where(log_p > 0, math.pi / log_p, np.inf)))
        candidates.append(("ln(p_e) + ln(2)", log_p + math.log(2)))

        if max_depth >= 3:
            candidates.append(("ln(ln(p_e))", np.where(log_p > 0, np.log(log_p), np.inf)))
            candidates.append(("ln(p_e) * (e+1)", log_p * (indices + 1)))

        return candidates
