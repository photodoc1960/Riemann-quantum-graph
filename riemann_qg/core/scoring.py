"""Spectral scoring: compare quantum graph eigenvalues to Riemann zeros.

Scoring philosophy (per advisor review, 2026-03-10):
- Absolute positional match of eigenvalues to zeros DOMINATES (0.70).
- GUE spacing and pair correlation are secondary validators (0.20 / 0.10).
- Before comparison, eigenvalues are Weyl-rescaled so their density
  matches Riemann zero density in the comparison window.  This is a
  deterministic preprocessing step, not part of optimisation.
- Only eigenvalues falling in the comparison window around the target
  zeros contribute to the score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats
from scipy.special import erf


@dataclass(frozen=True)
class ScoringResult:
    """Immutable result of spectral comparison."""

    absolute_match: float
    spacing_distribution: float
    pair_correlation: float
    prime_overlap: float          # diagnostic only — not in total_score
    total_score: float
    n_windowed: int               # how many eigenvalues landed in window
    weyl_alpha: float             # Weyl rescaling factor applied
    summary: str


# ---------------------------------------------------------------------------
# Weights — positional match must dominate
# ---------------------------------------------------------------------------

_WEIGHT_POSITION = 0.70
_WEIGHT_SPACING = 0.20
_WEIGHT_PAIR_CORR = 0.10


class SpectralScorer:
    """Score quantum graph spectra against Riemann zeros."""

    def __init__(self) -> None:
        from riemann_qg.core.riemann_zeros import RiemannZeros
        self._rz = RiemannZeros()

    def score_spectrum(
        self,
        candidate_eigenvalues: NDArray[np.float64],
        riemann_zeros: NDArray[np.float64] | None = None,
        n_compare: int = 20,
        edge_lengths: tuple[float, ...] | None = None,
    ) -> ScoringResult:
        """Compute multi-component score comparing spectrum to zeros.

        Pipeline:
        1. Weyl-rescale eigenvalues so density matches zero density
        2. Select eigenvalues inside comparison window around zeros
        3. Absolute positional match (sequential, no unfolding)
        4. GUE spacing + pair correlation (on unfolded windowed spectrum)
        5. Weighted total with coverage penalty
        """
        if riemann_zeros is None:
            riemann_zeros = self._rz.get_zeros(n_compare)

        zeros = np.sort(riemann_zeros[:n_compare])
        n_zeros = len(zeros)

        if len(candidate_eigenvalues) < 3 or n_zeros < 3:
            return ScoringResult(
                absolute_match=0.0,
                spacing_distribution=0.0,
                pair_correlation=0.0,
                prime_overlap=0.0,
                total_score=0.0,
                n_windowed=0,
                weyl_alpha=1.0,
                summary="Insufficient data (need >= 3 eigenvalues and zeros)",
            )

        # ------ 1. Weyl rescaling ------
        if edge_lengths and len(edge_lengths) > 0:
            total_length = sum(edge_lengths)
            scaled_eigs, alpha = _weyl_rescale(
                np.sort(candidate_eigenvalues), total_length, zeros,
            )
        else:
            # No edge lengths (e.g. self-comparison test): no rescaling
            scaled_eigs = np.sort(candidate_eigenvalues)
            alpha = 1.0

        # ------ 2. Comparison window ------
        zero_spacings = np.diff(zeros)
        mean_zero_spacing = (
            float(np.mean(zero_spacings)) if len(zero_spacings) > 0 else 1.0
        )
        k_low = zeros[0] - mean_zero_spacing
        k_high = zeros[-1] + mean_zero_spacing

        mask = (scaled_eigs >= k_low) & (scaled_eigs <= k_high)
        windowed = scaled_eigs[mask]
        n_match = min(len(windowed), n_zeros)

        # ------ 3. Positional match (dominant) ------
        if n_match >= 3:
            position_score = _position_match_score(
                windowed[:n_match], zeros[:n_match], mean_zero_spacing,
            )
        else:
            position_score = 0.0

        # ------ 4. GUE statistics (secondary) ------
        if n_match >= 5:
            unfolded = normalize_spectrum(windowed[:n_match])
            spacing_score = _spacing_distribution_score(unfolded)
            pair_score = _pair_correlation_score(unfolded)
        else:
            spacing_score = 0.0
            pair_score = 0.0

        # ------ 5. Coverage penalty ------
        coverage = min(1.0, n_match / n_zeros) if n_zeros > 0 else 0.0

        total = (
            _WEIGHT_POSITION * position_score
            + _WEIGHT_SPACING * spacing_score
            + _WEIGHT_PAIR_CORR * pair_score
        ) * coverage

        # Prime overlap as diagnostic only
        prime_score = (
            _prime_overlap_score(edge_lengths) if edge_lengths else 0.0
        )

        summary = (
            f"Total={total:.4f} "
            f"[Pos={position_score:.4f}*{_WEIGHT_POSITION}, "
            f"Spacing={spacing_score:.4f}*{_WEIGHT_SPACING}, "
            f"PairCorr={pair_score:.4f}*{_WEIGHT_PAIR_CORR}, "
            f"Cov={coverage:.2f}, "
            f"Window={n_match}/{n_zeros}, alpha={alpha:.2f}, "
            f"PrimeOlap={prime_score:.4f}(diag)]"
        )

        return ScoringResult(
            absolute_match=position_score,
            spacing_distribution=spacing_score,
            pair_correlation=pair_score,
            prime_overlap=prime_score,
            total_score=total,
            n_windowed=n_match,
            weyl_alpha=alpha,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Weyl rescaling
# ---------------------------------------------------------------------------

def _n_smooth(t: float) -> float:
    """Smooth part of the Riemann zero counting function.

    N_smooth(T) = (T/2pi) ln(T/2pi) - T/2pi + 7/8
    """
    if t <= 1e-10:
        return 0.0
    u = t / (2.0 * np.pi)
    return u * np.log(u) - u + 7.0 / 8.0


def _invert_n_smooth(n_target: float) -> float:
    """Find T such that N_smooth(T) = n_target.

    Uses Brent's method.  N_smooth is monotonically increasing for
    T > 2*pi (u > 1) and has a minimum of -1/8 at T = 2*pi.
    """
    from scipy.optimize import brentq

    if n_target < -0.125:
        return 2.0 * np.pi  # below minimum — return minimum location

    def f(t: float) -> float:
        return _n_smooth(t) - n_target

    # Bracket: lower bound just above 0, upper generous
    t_low = 1.0
    t_high = max(200.0, 4.0 * np.pi * (n_target + 2))
    # Widen bracket if needed
    while f(t_high) < 0:
        t_high *= 2
    try:
        return brentq(f, t_low, t_high, xtol=1e-6)
    except ValueError:
        return t_high


def _weyl_rescale(
    eigenvalues: NDArray[np.float64],
    total_length: float,
    target_zeros: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Nonlinear Weyl rescaling via Riemann-von Mangoldt formula.

    For each eigenvalue k_i:
      1. Weyl ordinal:  n_i = L * k_i / pi
      2. Map to zero scale:  T_i = N_smooth^{-1}(n_i)

    This removes the systematic density mismatch between the constant
    graph density (L/pi) and the logarithmic zero density.  The linear
    rescaling k -> alpha*k only matches average density; this matches
    it pointwise.

    Returns (mapped_eigenvalues, effective_alpha) where effective_alpha
    is the median ratio T_i / k_i for diagnostic display.
    """
    n = len(target_zeros)
    if n < 2 or total_length <= 0:
        return eigenvalues, 1.0

    mapped = np.empty(len(eigenvalues), dtype=np.float64)
    for i, k in enumerate(eigenvalues):
        ordinal = total_length * k / np.pi
        mapped[i] = _invert_n_smooth(ordinal)

    # Effective alpha for display (median ratio, ignoring near-zero k)
    valid = eigenvalues > 1e-10
    if np.any(valid):
        ratios = mapped[valid] / eigenvalues[valid]
        alpha_eff = float(np.median(ratios))
    else:
        alpha_eff = 1.0

    return mapped, alpha_eff


# ---------------------------------------------------------------------------
# Positional match (the primary score component)
# ---------------------------------------------------------------------------

def _position_match_score(
    windowed_eigs: NDArray[np.float64],
    zeros: NDArray[np.float64],
    mean_zero_spacing: float,
) -> float:
    """Score absolute positional match between eigenvalues and zeros.

    Sequential matching: i-th windowed eigenvalue vs i-th zero.
    MAD normalised by mean zero spacing so the score is scale-invariant.

    exp(-x): MAD of 1 spacing -> 0.37, 0.5 spacing -> 0.61, 2 -> 0.14
    """
    n = min(len(windowed_eigs), len(zeros))
    if n < 2 or mean_zero_spacing <= 0:
        return 0.0

    residuals = np.abs(windowed_eigs[:n] - zeros[:n])
    mad = float(np.mean(residuals))
    normalized_mad = mad / mean_zero_spacing

    return float(np.exp(-normalized_mad))


# ---------------------------------------------------------------------------
# GUE statistics (secondary validators)
# ---------------------------------------------------------------------------

def normalize_spectrum(
    eigenvalues: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Local unfolding to unit mean density.

    Uses a smooth polynomial fit to the staircase function.
    """
    sorted_eigs = np.sort(eigenvalues)
    n = len(sorted_eigs)
    staircase = np.arange(1, n + 1, dtype=np.float64)

    degree = min(5, max(1, n // 10))
    coeffs = np.polyfit(sorted_eigs, staircase, degree)
    unfolded = np.polyval(coeffs, sorted_eigs)

    spacings = np.diff(unfolded)
    mean_spacing = np.mean(spacings) if len(spacings) > 0 else 1.0
    if mean_spacing > 0:
        unfolded = unfolded / mean_spacing

    return unfolded


def _spacing_distribution_score(
    unfolded: NDArray[np.float64],
) -> float:
    """KS test of nearest-neighbor spacings vs GUE Wigner surmise."""
    spacings = np.diff(unfolded)
    if len(spacings) < 3:
        return 0.0

    mean_s = np.mean(spacings)
    if mean_s <= 0:
        return 0.0
    spacings_norm = spacings / mean_s

    ks_stat, _ = sp_stats.kstest(spacings_norm, _wigner_surmise_cdf)
    return float(max(0.0, 1.0 - ks_stat))


def _wigner_surmise_cdf(s: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """CDF of GUE Wigner surmise P(s) = (32/pi^2) s^2 exp(-4s^2/pi)."""
    s_arr = np.asarray(s, dtype=np.float64)
    return np.clip(
        erf(2.0 * s_arr / np.sqrt(np.pi))
        - (4.0 * s_arr / np.pi) * np.exp(-4.0 * s_arr**2 / np.pi),
        0.0,
        1.0,
    )


def _pair_correlation_score(
    unfolded: NDArray[np.float64],
    r_max: float = 3.0,
    n_bins: int = 50,
) -> float:
    """Score pair correlation against GUE prediction."""
    n = len(unfolded)
    if n < 5:
        return 0.0

    diffs = _compute_pair_diffs(unfolded, r_max)
    if len(diffs) < 5:
        return 0.0

    bins = np.linspace(0, r_max, n_bins + 1)
    hist, _ = np.histogram(diffs, bins=bins, density=True)

    centers = 0.5 * (bins[:-1] + bins[1:])
    gue_r2 = _gue_pair_correlation(centers)
    dr = bins[1] - bins[0]
    gue_integral = np.sum(gue_r2) * dr
    if gue_integral > 1e-10:
        gue_r2_norm = gue_r2 / gue_integral
    else:
        gue_r2_norm = gue_r2

    deviation = float(np.sum((hist - gue_r2_norm) ** 2) * dr)
    return float(np.exp(-deviation / 1.0))


def _compute_pair_diffs(
    unfolded: NDArray[np.float64], r_max: float,
) -> NDArray[np.float64]:
    """Compute all pair differences within r_max (vectorized)."""
    diffs_2d = np.abs(unfolded[:, None] - unfolded[None, :])
    upper = np.triu(diffs_2d, k=1)
    mask = (upper > 0) & (upper <= r_max)
    return upper[mask]


def _gue_pair_correlation(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """GUE pair correlation R_2(r) = 1 - (sin(pi*r)/(pi*r))^2."""
    pi_r = np.pi * r
    sinc_sq = np.where(
        np.abs(r) < 1e-15,
        1.0,
        (np.sin(pi_r) / pi_r) ** 2,
    )
    return 1.0 - sinc_sq


# ---------------------------------------------------------------------------
# Prime overlap (diagnostic only — not included in total_score)
# ---------------------------------------------------------------------------

_PRIME_LOGS: NDArray[np.float64] | None = None


def _get_prime_logs(n: int = 50) -> NDArray[np.float64]:
    """Cache and return ln(p) for first n primes."""
    global _PRIME_LOGS
    if _PRIME_LOGS is None or len(_PRIME_LOGS) < n:
        primes: list[int] = []
        candidate = 2
        while len(primes) < n:
            if all(candidate % p != 0 for p in primes):
                primes.append(candidate)
            candidate += 1
        _PRIME_LOGS = np.array([math.log(p) for p in primes])
    return _PRIME_LOGS[:n]


def _prime_overlap_score(
    edge_lengths: tuple[float, ...],
    n_primes: int = 30,
) -> float:
    """Score how well edge lengths align with logarithms of primes."""
    if not edge_lengths or len(edge_lengths) < 2:
        return 0.0

    lengths = np.array(edge_lengths)
    log_primes = _get_prime_logs(n_primes)

    n = min(len(lengths), len(log_primes))
    sorted_l = np.sort(lengths)[:n]
    lp = log_primes[:n]
    denom = np.dot(lp, lp)
    if denom < 1e-15:
        return 0.0
    best_alpha = np.dot(sorted_l, lp) / denom

    scaled_primes = best_alpha * log_primes
    match_count = 0
    for length in lengths:
        min_dist = np.min(np.abs(scaled_primes - length))
        if min_dist < 0.05 * max(abs(length), 0.01):
            match_count += 1
    match_fraction = match_count / len(lengths)

    predicted = best_alpha * lp
    if np.std(sorted_l) > 1e-10:
        corr = float(np.corrcoef(sorted_l, predicted)[0, 1])
        corr = max(0.0, corr)
    else:
        corr = 0.0

    return 0.5 * match_fraction + 0.5 * corr
