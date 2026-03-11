"""Symbolic regression and integer relation finding for edge length patterns.

Discovers closed-form rules for quantum graph edge lengths using PSLQ
integer relation algorithms and simple grammar-based symbolic regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import mpmath
import numpy as np
from sympy import primerange


@dataclass(frozen=True)
class PSLQResult:
    """Immutable result from a PSLQ integer relation search."""

    coefficients: tuple[int, ...]
    basis_labels: tuple[str, ...]
    residual: float
    found: bool


@dataclass(frozen=True)
class DecompositionResult:
    """Immutable result from prime log decomposition of edge lengths."""

    length_index: int
    original_value: float
    coefficients: tuple[int, ...]
    primes_used: tuple[int, ...]
    reconstruction: float
    residual: float


@dataclass(frozen=True)
class RegressionResult:
    """Immutable result from symbolic regression."""

    expression: str
    complexity: int
    mse: float
    r_squared: float


def _empty_pslq_result(basis_len: int) -> PSLQResult:
    """Return an empty PSLQ result for failed searches."""
    return PSLQResult(
        coefficients=(),
        basis_labels=tuple(f"b{i}" for i in range(basis_len)),
        residual=float("inf"),
        found=False,
    )


def _build_pslq_vector(
    values: list[float],
    basis: list[float],
    mp_values: list | None,
    mp_basis: list | None,
) -> list:
    """Build the mpmath vector for PSLQ from float or mpf inputs."""
    if mp_values is not None and mp_basis is not None:
        return list(mp_basis) + [mp_values[0]]
    target = mpmath.mpf(str(values[0]))
    return [mpmath.mpf(str(b)) for b in basis] + [target]


def pslq_fit(
    values: list[float],
    basis: list[float],
    precision: int = 50,
    max_coeff: int = 20,
    mp_values: list | None = None,
    mp_basis: list | None = None,
) -> PSLQResult:
    """Find integer relations among values using PSLQ algorithm.

    For best results, pass mp_values and mp_basis as mpmath.mpf objects
    computed at full precision.

    Args:
        values: Target values (uses first element).
        basis: Basis elements for the relation.
        precision: Decimal places for mpmath.
        max_coeff: Maximum allowed coefficient magnitude.
        mp_values: Optional high-precision mpmath targets.
        mp_basis: Optional high-precision mpmath basis.
    """
    mpmath.mp.dps = precision

    if not values or not basis:
        return _empty_pslq_result(0)

    vector = _build_pslq_vector(values, basis, mp_values, mp_basis)
    relation = mpmath.pslq(vector, maxcoeff=max_coeff * 10)

    if relation is None:
        return _empty_pslq_result(len(basis))

    coeffs = tuple(int(c) for c in relation)
    labels = tuple(f"b{i}" for i in range(len(basis))) + ("target",)

    if all(abs(c) <= max_coeff for c in coeffs):
        residual = float(abs(sum(c * b for c, b in zip(coeffs, vector))))
        return PSLQResult(
            coefficients=coeffs,
            basis_labels=labels,
            residual=residual,
            found=True,
        )

    return PSLQResult(
        coefficients=coeffs, basis_labels=labels,
        residual=float("inf"), found=False,
    )


def prime_log_decomposition(
    lengths: list[float],
    n_primes: int = 15,
    precision: int = 30,
) -> list[DecompositionResult]:
    """Decompose edge lengths into sums of logarithms of primes.

    For each length L, attempts to find integers a_p such that
    L ~ sum(a_p * ln(p)) for primes p.

    Args:
        lengths: Edge lengths to decompose.
        n_primes: Number of primes to use in the basis.
        precision: Decimal precision for PSLQ.

    Returns:
        List of DecompositionResult, one per edge length.
    """
    primes = _first_n_primes(n_primes)
    mpmath.mp.dps = precision
    mp_prime_logs = [mpmath.log(p) for p in primes]
    prime_logs = [float(pl) for pl in mp_prime_logs]

    results = []
    for idx, length in enumerate(lengths):
        result = _decompose_single_length(
            idx, length, primes, prime_logs, mp_prime_logs, precision
        )
        results.append(result)

    return results


def _decompose_single_length(
    idx: int,
    length: float,
    primes: list[int],
    prime_logs: list[float],
    mp_prime_logs: list,
    precision: int,
    max_coeff: int = 5,
    tolerance: float = 1e-6,
) -> DecompositionResult:
    """Attempt decomposition of a single edge length into prime logs.

    First tries PSLQ with mpmath precision. Falls back to brute-force
    search over small integer coefficient combinations.
    """
    mpmath.mp.dps = precision
    mp_length = mpmath.mpf(str(length))

    # Try PSLQ first
    pslq_result = pslq_fit(
        [length], prime_logs, precision=precision,
        mp_values=[mp_length], mp_basis=mp_prime_logs,
    )

    if pslq_result.found and len(pslq_result.coefficients) > 0:
        decomp = _extract_pslq_decomposition(
            idx, length, pslq_result, primes, prime_logs
        )
        if decomp is not None:
            return decomp

    # Fall back to brute-force search over small coefficients
    return _brute_force_decomposition(
        idx, length, primes, prime_logs, max_coeff, tolerance
    )


def _extract_pslq_decomposition(
    idx: int,
    length: float,
    pslq_result: PSLQResult,
    primes: list[int],
    prime_logs: list[float],
) -> DecompositionResult | None:
    """Extract decomposition from a successful PSLQ result."""
    basis_coeffs = pslq_result.coefficients[:-1]
    target_coeff = pslq_result.coefficients[-1]

    if target_coeff == 0:
        return None

    adjusted = tuple(-c / target_coeff for c in basis_coeffs)
    reconstruction = sum(c * pl for c, pl in zip(adjusted, prime_logs))
    residual = abs(length - reconstruction)

    if residual > 0.01:
        return None

    return DecompositionResult(
        length_index=idx,
        original_value=length,
        coefficients=tuple(int(round(c)) for c in adjusted),
        primes_used=tuple(primes[: len(basis_coeffs)]),
        reconstruction=reconstruction,
        residual=residual,
    )


def _brute_force_decomposition(
    idx: int,
    length: float,
    primes: list[int],
    prime_logs: list[float],
    max_coeff: int,
    tolerance: float,
) -> DecompositionResult:
    """Search small integer coefficients for prime-log decomposition."""
    n = min(len(prime_logs), 6)  # Limit search space
    best_residual = abs(length)
    best_coeffs: tuple[int, ...] = ()

    for depth in range(1, n + 1):
        coeffs, residual = _search_coefficients(
            length, prime_logs[:depth], max_coeff
        )
        if residual < best_residual:
            best_residual = residual
            best_coeffs = coeffs + (0,) * (n - len(coeffs))
            if residual < tolerance:
                break

    if best_residual < tolerance and best_coeffs:
        reconstruction = sum(
            c * pl for c, pl in zip(best_coeffs, prime_logs[:n])
        )
        return DecompositionResult(
            length_index=idx,
            original_value=length,
            coefficients=best_coeffs[:n],
            primes_used=tuple(primes[:n]),
            reconstruction=reconstruction,
            residual=best_residual,
        )

    return DecompositionResult(
        length_index=idx,
        original_value=length,
        coefficients=(),
        primes_used=(),
        reconstruction=0.0,
        residual=abs(length),
    )


def _search_coefficients(
    target: float,
    prime_logs: list[float],
    max_coeff: int,
) -> tuple[tuple[int, ...], float]:
    """Search for integer coefficients minimizing residual."""
    if len(prime_logs) == 0:
        return (), abs(target)

    best_coeffs: tuple[int, ...] = ()
    best_residual = abs(target)

    best_coeffs, best_residual = _search_single_prime(
        target, prime_logs, max_coeff, best_coeffs, best_residual
    )

    if len(prime_logs) >= 2 and best_residual > 1e-8:
        best_coeffs, best_residual = _search_two_primes(
            target, prime_logs, max_coeff, best_coeffs, best_residual
        )

    if len(prime_logs) >= 3 and best_residual > 1e-8:
        best_coeffs, best_residual = _search_three_primes(
            target, prime_logs, max_coeff, best_coeffs, best_residual
        )

    return best_coeffs, best_residual


def _search_single_prime(
    target: float,
    prime_logs: list[float],
    max_coeff: int,
    best_coeffs: tuple[int, ...],
    best_residual: float,
) -> tuple[tuple[int, ...], float]:
    """Search single-prime coefficient matches."""
    for i, pl in enumerate(prime_logs):
        for c in range(-max_coeff, max_coeff + 1):
            if c == 0:
                continue
            residual = abs(target - c * pl)
            if residual < best_residual:
                best_residual = residual
                coeffs_list = [0] * len(prime_logs)
                coeffs_list[i] = c
                best_coeffs = tuple(coeffs_list)
    return best_coeffs, best_residual


def _search_two_primes(
    target: float,
    prime_logs: list[float],
    max_coeff: int,
    best_coeffs: tuple[int, ...],
    best_residual: float,
) -> tuple[tuple[int, ...], float]:
    """Search two-prime coefficient combinations."""
    for i in range(len(prime_logs)):
        for j in range(i, len(prime_logs)):
            for ci in range(-max_coeff, max_coeff + 1):
                remainder = target - ci * prime_logs[i]
                if prime_logs[j] > 1e-15:
                    cj = round(remainder / prime_logs[j])
                    if abs(cj) <= max_coeff:
                        residual = abs(target - ci * prime_logs[i] - cj * prime_logs[j])
                        if residual < best_residual:
                            best_residual = residual
                            cl = [0] * len(prime_logs)
                            cl[i] = ci
                            cl[j] += cj
                            best_coeffs = tuple(cl)
    return best_coeffs, best_residual


def _search_three_primes(
    target: float,
    prime_logs: list[float],
    max_coeff: int,
    best_coeffs: tuple[int, ...],
    best_residual: float,
) -> tuple[tuple[int, ...], float]:
    """Search three-prime coefficient combinations."""
    n = min(len(prime_logs), 6)
    for i in range(min(n, 4)):
        for j in range(i, min(n, 5)):
            for k in range(j, n):
                for ci in range(-max_coeff, max_coeff + 1):
                    for cj in range(-max_coeff, max_coeff + 1):
                        remainder = target - ci * prime_logs[i] - cj * prime_logs[j]
                        if prime_logs[k] > 1e-15:
                            ck = round(remainder / prime_logs[k])
                            if abs(ck) <= max_coeff:
                                residual = abs(remainder - ck * prime_logs[k])
                                if residual < best_residual:
                                    best_residual = residual
                                    cl = [0] * len(prime_logs)
                                    cl[i] = ci
                                    cl[j] += cj
                                    cl[k] += ck
                                    best_coeffs = tuple(cl)
    return best_coeffs, best_residual


def grammar_regression(
    x: np.ndarray,
    y: np.ndarray,
    max_complexity: int = 5,
) -> RegressionResult:
    """Simple symbolic regression using polynomial and log-polynomial basis.

    Tries polynomial, log, and mixed bases up to max_complexity and
    returns the best-fitting expression.

    Args:
        x: Independent variable values.
        y: Dependent variable values.
        max_complexity: Maximum polynomial degree / basis complexity.

    Returns:
        RegressionResult with the best expression found.
    """
    if len(x) == 0 or len(y) == 0:
        return RegressionResult(
            expression="0", complexity=0, mse=float("inf"), r_squared=0.0
        )

    best = RegressionResult(
        expression="0", complexity=0, mse=float("inf"), r_squared=0.0
    )

    for degree in range(1, max_complexity + 1):
        result = _try_polynomial_fit(x, y, degree)
        if result.mse < best.mse:
            best = result

    log_result = _try_log_fit(x, y)
    if log_result.mse < best.mse:
        best = log_result

    mixed_result = _try_mixed_fit(x, y, max_complexity)
    if mixed_result.mse < best.mse:
        best = mixed_result

    return best


def _try_polynomial_fit(
    x: np.ndarray, y: np.ndarray, degree: int
) -> RegressionResult:
    """Fit a polynomial of given degree."""
    try:
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)
        mse = float(np.mean((y - y_pred) ** 2))
        r_sq = _r_squared(y, y_pred)

        terms = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if abs(c) < 1e-10:
                continue
            if power == 0:
                terms.append(f"{c:.4f}")
            elif power == 1:
                terms.append(f"{c:.4f}*x")
            else:
                terms.append(f"{c:.4f}*x^{power}")

        expr = " + ".join(terms) if terms else "0"
        return RegressionResult(
            expression=expr, complexity=degree, mse=mse, r_squared=r_sq
        )
    except (np.linalg.LinAlgError, ValueError):
        return RegressionResult(
            expression="0", complexity=degree, mse=float("inf"), r_squared=0.0
        )


def _try_log_fit(x: np.ndarray, y: np.ndarray) -> RegressionResult:
    """Fit y = a * ln(x) + b."""
    try:
        valid = x > 0
        if np.sum(valid) < 2:
            return RegressionResult(
                expression="0", complexity=1, mse=float("inf"), r_squared=0.0
            )
        log_x = np.log(x[valid])
        coeffs = np.polyfit(log_x, y[valid], 1)
        y_pred = np.polyval(coeffs, log_x)
        mse = float(np.mean((y[valid] - y_pred) ** 2))
        r_sq = _r_squared(y[valid], y_pred)

        expr = f"{coeffs[0]:.4f}*ln(x) + {coeffs[1]:.4f}"
        return RegressionResult(
            expression=expr, complexity=2, mse=mse, r_squared=r_sq
        )
    except (np.linalg.LinAlgError, ValueError):
        return RegressionResult(
            expression="0", complexity=2, mse=float("inf"), r_squared=0.0
        )


def _try_mixed_fit(
    x: np.ndarray, y: np.ndarray, max_degree: int
) -> RegressionResult:
    """Fit y = a * ln(x) + polynomial(x)."""
    try:
        valid = x > 0
        if np.sum(valid) < 3:
            return RegressionResult(
                expression="0", complexity=3, mse=float("inf"), r_squared=0.0
            )

        xv = x[valid]
        yv = y[valid]
        log_x = np.log(xv)

        degree = min(max_degree, 2)
        basis = np.column_stack([log_x] + [xv**i for i in range(degree + 1)])

        coeffs, _, _, _ = np.linalg.lstsq(basis, yv, rcond=None)
        y_pred = basis @ coeffs
        mse = float(np.mean((yv - y_pred) ** 2))
        r_sq = _r_squared(yv, y_pred)

        terms = [f"{coeffs[0]:.4f}*ln(x)"]
        for i in range(degree + 1):
            if abs(coeffs[i + 1]) > 1e-10:
                if i == 0:
                    terms.append(f"{coeffs[i + 1]:.4f}")
                else:
                    terms.append(f"{coeffs[i + 1]:.4f}*x^{i}")

        expr = " + ".join(terms)
        return RegressionResult(
            expression=expr, complexity=degree + 2, mse=mse, r_squared=r_sq
        )
    except (np.linalg.LinAlgError, ValueError):
        return RegressionResult(
            expression="0", complexity=3, mse=float("inf"), r_squared=0.0
        )


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared goodness of fit."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _first_n_primes(n: int) -> list[int]:
    """Return the first n prime numbers."""
    upper = _prime_upper_bound(n)
    primes = list(primerange(2, upper))[:n]
    return primes


def _prime_upper_bound(n: int) -> int:
    """Upper bound for the nth prime."""
    if n < 6:
        return 15
    import math
    return int(n * (math.log(n) + math.log(math.log(n)))) + 10
