"""Edge length and scattering matrix optimization for quantum graphs.

Uses scipy.optimize to find edge lengths and scattering phases that
maximize spectral match scores against Riemann zeta zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import differential_evolution, minimize

if TYPE_CHECKING:
    from riemann_qg.core.quantum_graph import QuantumGraph
    from riemann_qg.core.scoring import SpectralScorer


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable result from an optimization run."""

    edge_lengths: tuple[float, ...]
    phases: tuple[float, ...]
    score: float
    n_evaluations: int
    converged: bool


def optimize_edge_lengths(
    graph: "QuantumGraph",
    scorer: "SpectralScorer",
    riemann_zeros: np.ndarray,
    n_compare: int,
    bounds_factor: float = 3.0,
    max_iter: int = 200,
    seed: int | None = None,
    workers: int = 1,
) -> OptimizationResult:
    """Optimize edge lengths for a fixed topology to maximize spectral match.

    Uses differential evolution for global search over edge length space.

    Args:
        graph: Quantum graph with topology to optimize.
        scorer: Spectral scorer instance.
        riemann_zeros: Known Riemann zeros for comparison.
        n_compare: Number of zeros to compare.
        bounds_factor: Multiplier for initial length bounds.
        max_iter: Maximum optimizer iterations.
        workers: Number of parallel workers for DE. -1 uses all cores.

    Returns:
        OptimizationResult with best edge lengths and score.
    """
    n_edges = graph.n_edges
    current_lengths = np.array(graph.edge_lengths)
    mean_length = max(np.mean(current_lengths), 0.1)

    bounds = [(0.01, mean_length * bounds_factor)] * n_edges

    result = differential_evolution(
        func=_edge_length_objective,
        bounds=bounds,
        args=(graph, scorer, riemann_zeros, n_compare),
        maxiter=max_iter,
        seed=seed,
        tol=1e-6,
        polish=True,
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
    )

    return OptimizationResult(
        edge_lengths=tuple(result.x),
        phases=tuple(
            _extract_phases(graph)
        ),
        score=-result.fun,
        n_evaluations=result.nfev,
        converged=result.success,
    )


def _edge_length_objective(
    lengths: np.ndarray,
    graph: "QuantumGraph",
    scorer: "SpectralScorer",
    riemann_zeros: np.ndarray,
    n_compare: int,
) -> float:
    """Negative score for minimization (we maximize score)."""
    candidate = graph.with_edge_lengths(list(lengths))
    try:
        k_max = _estimate_k_max(candidate, n_compare)
        spectrum = candidate.compute_spectrum(k_max, n_points=2000)
        result = scorer.score_spectrum(
            spectrum, riemann_zeros, n_compare,
            edge_lengths=candidate.edge_lengths,
        )
        return -result.total_score
    except Exception:
        return 1.0  # Worse than any valid graph (score in [0,1], negated)


def optimize_scattering(
    graph: "QuantumGraph",
    scorer: "SpectralScorer",
    riemann_zeros: np.ndarray,
    n_compare: int,
    max_iter: int = 200,
    seed: int | None = None,
    workers: int = 1,
) -> OptimizationResult:
    """Optimize scattering matrix phases for fixed topology and edge lengths.

    Parameterizes scattering matrices by phase angles and optimizes
    over those angles.

    Args:
        graph: Quantum graph to optimize.
        scorer: Spectral scorer instance.
        riemann_zeros: Known Riemann zeros.
        n_compare: Number of zeros to compare.
        max_iter: Maximum iterations.
        workers: Number of parallel workers for DE. -1 uses all cores.

    Returns:
        OptimizationResult with best phases and score.
    """
    n_phases = _count_phase_parameters(graph)
    bounds = [(0.0, 2.0 * np.pi)] * n_phases

    result = differential_evolution(
        func=_scattering_objective,
        bounds=bounds,
        args=(graph, scorer, riemann_zeros, n_compare),
        maxiter=max_iter,
        seed=seed,
        tol=1e-6,
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
    )

    return OptimizationResult(
        edge_lengths=tuple(graph.edge_lengths),
        phases=tuple(result.x),
        score=-result.fun,
        n_evaluations=result.nfev,
        converged=result.success,
    )


def _scattering_objective(
    phases: np.ndarray,
    graph: "QuantumGraph",
    scorer: "SpectralScorer",
    riemann_zeros: np.ndarray,
    n_compare: int,
) -> float:
    """Negative score for scattering phase optimization."""
    candidate = graph.with_scattering_phases(list(phases))
    try:
        k_max = _estimate_k_max(candidate, n_compare)
        spectrum = candidate.compute_spectrum(k_max, n_points=2000)
        result = scorer.score_spectrum(
            spectrum, riemann_zeros, n_compare,
            edge_lengths=candidate.edge_lengths,
        )
        return -result.total_score
    except Exception:
        return 1.0


def _build_joint_bounds(
    graph: "QuantumGraph",
) -> tuple[list[tuple[float, float]], int, int]:
    """Build parameter bounds for joint length + phase optimization."""
    n_edges = graph.n_edges
    n_phases = _count_phase_parameters(graph)

    current_lengths = np.array(graph.edge_lengths)
    mean_length = max(np.mean(current_lengths), 0.1)

    length_bounds = [(0.01, mean_length * 3.0)] * n_edges
    phase_bounds = [(0.0, 2.0 * np.pi)] * n_phases
    return length_bounds + phase_bounds, n_edges, n_phases


def joint_optimize(
    graph: "QuantumGraph",
    scorer: "SpectralScorer",
    riemann_zeros: np.ndarray,
    n_compare: int,
    max_iter: int = 300,
    seed: int | None = None,
    workers: int = -1,
) -> OptimizationResult:
    """Optimize both edge lengths and scattering phases jointly.

    Args:
        graph: Quantum graph to optimize.
        scorer: Spectral scorer instance.
        riemann_zeros: Known Riemann zeros.
        n_compare: Number of zeros to compare.
        max_iter: Maximum iterations.
        workers: Number of parallel workers for DE population evaluation.
                 -1 uses all available CPU cores.
    """
    bounds, n_edges, n_phases = _build_joint_bounds(graph)

    result = differential_evolution(
        func=_joint_objective,
        bounds=bounds,
        args=(graph, scorer, riemann_zeros, n_compare, n_edges),
        maxiter=max_iter,
        seed=seed,
        tol=1e-6,
        polish=True,
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
    )

    return OptimizationResult(
        edge_lengths=tuple(result.x[:n_edges]),
        phases=tuple(result.x[n_edges:]),
        score=-result.fun,
        n_evaluations=result.nfev,
        converged=result.success,
    )


def _joint_objective(
    params: np.ndarray,
    graph: "QuantumGraph",
    scorer: "SpectralScorer",
    riemann_zeros: np.ndarray,
    n_compare: int,
    n_edges: int,
) -> float:
    """Negative score for joint length + phase optimization."""
    lengths = params[:n_edges]
    phases = params[n_edges:]

    candidate = graph.with_edge_lengths(list(lengths))
    candidate = candidate.with_scattering_phases(list(phases))

    try:
        k_max = _estimate_k_max(candidate, n_compare)
        spectrum = candidate.compute_spectrum(k_max, n_points=2000)
        result = scorer.score_spectrum(
            spectrum, riemann_zeros, n_compare,
            edge_lengths=candidate.edge_lengths,
        )
        return -result.total_score
    except Exception:
        return 1.0


def _estimate_k_max(graph: "QuantumGraph", n_zeros: int) -> float:
    """Estimate k_max needed to capture n_zeros eigenvalues via Weyl's law.

    N(k) ~ (L_total * k) / pi, so k ~ n_zeros * pi / L_total.
    """
    total_length = sum(graph.edge_lengths)
    if total_length <= 0:
        return 100.0
    k_max = (n_zeros + 5) * np.pi / total_length
    return max(k_max, 10.0)


def _extract_phases(graph: "QuantumGraph") -> list[float]:
    """Extract TRS-breaking phase angles from the graph's scattering matrices.

    For D @ S_neumann @ D† with D = diag(1, e^{iθ}, ...), element (0,1)
    carries the phase as S_neumann[0,1] * e^{-iθ}.  Since S_neumann[0,1]
    is real positive (2/d), the TRS phase θ = -angle(s[0,1]).
    """
    phases = []
    for matrix in graph.vertex_scattering:
        if matrix is not None and matrix.shape[0] >= 2:
            phase = float(-np.angle(matrix[0, 1]))
            phases.append(phase)
        else:
            phases.append(0.0)
    return phases


def _count_phase_parameters(graph: "QuantumGraph") -> int:
    """Count the number of independent phase parameters.

    One phase per vertex for the time-reversal breaking parameter.
    """
    return graph.n_vertices
