"""Tests for spectral normalization and scoring against Riemann zeros."""

from __future__ import annotations

import numpy as np
import pytest

from riemann_qg.core.riemann_zeros import RiemannZeros
from riemann_qg.core.scoring import (
    SpectralScorer,
    normalize_spectrum,
    _spacing_distribution_score,
)


# -----------------------------------------------------------------------
# 1. Normalization gives unit mean spacing
# -----------------------------------------------------------------------


class TestNormalization:
    """After unfolding, mean nearest-neighbor spacing should be ~1."""

    def test_unit_mean_spacing_linear(self):
        """Evenly spaced eigenvalues should unfold to unit spacing."""
        eigs = np.linspace(1.0, 50.0, 100)
        unfolded = normalize_spectrum(eigs)
        spacings = np.diff(unfolded)
        mean_spacing = np.mean(spacings)
        assert abs(mean_spacing - 1.0) < 0.15, (
            f"Mean spacing {mean_spacing:.4f} should be near 1.0"
        )

    def test_unit_mean_spacing_riemann_zeros(self):
        """Riemann zeros after unfolding should have mean spacing ~1."""
        rz = RiemannZeros()
        zeros = rz.get_zeros(50)
        unfolded = normalize_spectrum(zeros)
        spacings = np.diff(unfolded)
        mean_spacing = np.mean(spacings)
        assert abs(mean_spacing - 1.0) < 0.2, (
            f"Riemann zeros mean spacing {mean_spacing:.4f} should be near 1.0"
        )


# -----------------------------------------------------------------------
# 2. Scoring: self-comparison → high; random → low; positional dominates
# -----------------------------------------------------------------------


class TestScoringQuality:
    """Self-comparison should score near 1.0; random should score lower."""

    def test_self_comparison_high_score(self):
        """Comparing Riemann zeros to themselves should give score ~1.0.

        Without edge_lengths, no Weyl rescaling occurs and eigenvalues
        ARE the zeros, giving perfect positional match (1.0).
        With weights 0.70*pos + 0.20*spacing + 0.10*pair_corr, expected ~0.95.
        """
        rz = RiemannZeros()
        zeros = rz.get_zeros(30)

        scorer = SpectralScorer()
        result = scorer.score_spectrum(
            candidate_eigenvalues=zeros,
            riemann_zeros=zeros,
            n_compare=20,
        )

        assert result.total_score > 0.90, (
            f"Self-comparison score {result.total_score:.4f} should be > 0.90. "
            f"Summary: {result.summary}"
        )
        assert result.absolute_match > 0.99, (
            f"Position match {result.absolute_match:.4f} should be ~1.0 for self"
        )

    def test_random_spectrum_low_score(self):
        """Random eigenvalues should score much lower than Riemann zeros."""
        rz = RiemannZeros()
        zeros = rz.get_zeros(30)

        rng = np.random.default_rng(42)
        random_eigs = np.sort(rng.uniform(10, 250, size=50))

        scorer = SpectralScorer()
        result_self = scorer.score_spectrum(
            candidate_eigenvalues=zeros,
            riemann_zeros=zeros,
            n_compare=20,
        )
        result_random = scorer.score_spectrum(
            candidate_eigenvalues=random_eigs,
            riemann_zeros=zeros,
            n_compare=20,
        )

        assert result_self.total_score > result_random.total_score, (
            f"Self-score {result_self.total_score:.4f} should exceed "
            f"random score {result_random.total_score:.4f}"
        )

    def test_insufficient_data_returns_zero(self):
        """Fewer than 3 values should return total=0."""
        scorer = SpectralScorer()
        result = scorer.score_spectrum(
            candidate_eigenvalues=np.array([14.0, 21.0]),
            riemann_zeros=np.array([14.13, 21.02]),
            n_compare=2,
        )
        assert result.total_score == 0.0

    def test_weyl_rescaling_applied(self):
        """When edge_lengths provided, Weyl rescaling should be applied."""
        rz = RiemannZeros()
        zeros = rz.get_zeros(20)

        scorer = SpectralScorer()
        # Graph eigenvalues at 0.5-5 range with edge_lengths
        eigs = np.linspace(0.5, 5.0, 30)
        result = scorer.score_spectrum(
            candidate_eigenvalues=eigs,
            riemann_zeros=zeros,
            n_compare=20,
            edge_lengths=tuple([1.0] * 10),
        )
        assert result.weyl_alpha != 1.0, "Weyl scaling should be applied"
        assert result.n_windowed >= 0, "Window count should be non-negative"

    def test_position_dominates_scoring(self):
        """Positional match should be the largest component of the score."""
        scorer = SpectralScorer()
        rz = RiemannZeros()
        zeros = rz.get_zeros(20)

        # Score zeros against themselves
        result = scorer.score_spectrum(zeros, zeros, 20)
        # 0.70 * 1.0 = 0.70 from position alone
        assert result.absolute_match * 0.70 >= 0.50, (
            "Position component should contribute at least 0.50 to score"
        )


# -----------------------------------------------------------------------
# 3. GUE spacing score: synthetic GUE > uniform random
# -----------------------------------------------------------------------


class TestGUESpacingScore:
    """Spectra drawn from GUE-like distributions should score higher
    on the spacing distribution metric than uniform random spacings.
    """

    def test_gue_beats_uniform(self):
        """GUE Wigner-surmise distributed spacings should score higher."""
        rng = np.random.default_rng(123)

        gue_spacings = _sample_wigner_surmise(rng, n=200)
        gue_cumsum = np.cumsum(gue_spacings)

        uniform_spacings = rng.exponential(1.0, size=200)
        uniform_cumsum = np.cumsum(uniform_spacings)

        score_gue = _spacing_distribution_score(gue_cumsum)
        score_uniform = _spacing_distribution_score(uniform_cumsum)

        assert score_gue > score_uniform, (
            f"GUE spacing score {score_gue:.4f} should exceed "
            f"uniform spacing score {score_uniform:.4f}"
        )


def _sample_wigner_surmise(
    rng: np.random.Generator, n: int,
) -> np.ndarray:
    """Sample n spacings from GUE Wigner surmise via rejection sampling."""
    samples: list[float] = []
    max_val = 32.0 / (np.pi**2) * (np.pi / 8.0) * np.exp(-0.5)
    envelope_scale = max_val * 3.0

    while len(samples) < n:
        s_candidate = rng.exponential(1.0)
        p_s = (32.0 / np.pi**2) * s_candidate**2 * np.exp(-4 * s_candidate**2 / np.pi)
        u = rng.uniform(0, envelope_scale * np.exp(-s_candidate))
        if u < p_s:
            samples.append(s_candidate)

    return np.array(samples)
