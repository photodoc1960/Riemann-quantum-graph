"""Spectral agent -- the compute workhorse.

Evaluates quantum graphs by computing their eigenvalue spectra and scoring
them against the Riemann zeta zeros.  Supports parallel batch evaluation
via ProcessPoolExecutor and adaptive k_max estimation through Weyl's law.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm

from riemann_qg.core.quantum_graph import QuantumGraph
from riemann_qg.core.scoring import ScoringResult, SpectralScorer


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalResult:
    """Immutable result of evaluating a single quantum graph."""
    graph: QuantumGraph
    scoring_result: ScoringResult
    spectrum: np.ndarray

    @property
    def score(self) -> float:
        return self.scoring_result.total_score


# ---------------------------------------------------------------------------
# Module-level helper for pickling across processes
# ---------------------------------------------------------------------------

def _evaluate_single(
    graph: QuantumGraph,
    k_max: float,
    n_points: int,
    n_zeros_compare: int,
) -> tuple[QuantumGraph, ScoringResult, np.ndarray]:
    """Evaluate one graph (top-level function for multiprocessing)."""
    scorer = SpectralScorer()
    effective_k_max = _adaptive_k_max_for_graph(graph, n_zeros_compare, k_max)
    spectrum = graph.compute_spectrum(effective_k_max, n_points=n_points)
    scoring_result = scorer.score_spectrum(
        candidate_eigenvalues=spectrum,
        n_compare=n_zeros_compare,
        edge_lengths=graph.edge_lengths,
    )
    return (graph, scoring_result, spectrum)


def _adaptive_k_max_for_graph(
    graph: QuantumGraph,
    n_zeros_target: int,
    fallback_k_max: float,
) -> float:
    """Compute adaptive k_max for a graph, floored at fallback_k_max.

    Uses Weyl's law: N(k) ~ (L_total * k) / pi.
    """
    total_length = sum(graph.edge_lengths)
    if total_length <= 0:
        return fallback_k_max
    k_estimate = (n_zeros_target * math.pi) / total_length * 1.2
    return max(k_estimate, fallback_k_max)


# ---------------------------------------------------------------------------
# SpectralAgent
# ---------------------------------------------------------------------------

class SpectralAgent:
    """Computes quantum graph spectra and scores them against Riemann zeros.

    The spectral agent wraps the computationally intensive loop of
    spectrum-computation + scoring, providing both single-graph and
    parallel-batch interfaces.  An adaptive k_max estimator uses Weyl's
    law to size the spectral window appropriately.
    """

    def __init__(self, n_points: int = 10_000) -> None:
        """
        Parameters
        ----------
        n_points : int
            Number of sample points for dense root search in
            ``QuantumGraph.compute_spectrum``.
        """
        self._n_points = n_points

    # ----- single evaluation -----

    def evaluate_graph(
        self,
        graph: QuantumGraph,
        scorer: SpectralScorer | None = None,
        k_max: float = 80.0,
        n_zeros_compare: int = 20,
    ) -> EvalResult:
        """Evaluate a single quantum graph.

        Parameters
        ----------
        graph : QuantumGraph
        scorer : SpectralScorer or None
            Reusable scorer; a fresh one is created if None.
        k_max : float
            Upper bound of the wavenumber window.
        n_zeros_compare : int
            How many zeros to use for comparison.

        Returns
        -------
        EvalResult
        """
        scorer = scorer or SpectralScorer()
        effective_k_max = _adaptive_k_max_for_graph(
            graph, n_zeros_compare, k_max,
        )
        spectrum = graph.compute_spectrum(
            effective_k_max, n_points=self._n_points,
        )
        scoring_result = scorer.score_spectrum(
            candidate_eigenvalues=spectrum,
            n_compare=n_zeros_compare,
            edge_lengths=graph.edge_lengths,
        )
        return EvalResult(
            graph=graph,
            scoring_result=scoring_result,
            spectrum=spectrum,
        )

    # ----- batch evaluation -----

    def evaluate_batch(
        self,
        graphs: Sequence[QuantumGraph],
        scorer: SpectralScorer | None = None,
        k_max: float = 80.0,
        n_zeros_compare: int = 20,
        n_workers: int = 1,
    ) -> list[EvalResult]:
        """Evaluate a batch of graphs, optionally in parallel.

        Parameters
        ----------
        graphs : sequence of QuantumGraph
        scorer : SpectralScorer or None
        k_max : float
        n_zeros_compare : int
        n_workers : int
            Number of worker processes.  1 means sequential (no subprocess
            overhead).  Values <= 0 are interpreted as ``cpu_count + n_workers``.

        Returns
        -------
        list[EvalResult]
            Sorted best-first by total score.
        """
        if n_workers <= 0:
            import os
            n_workers = max(1, os.cpu_count() + n_workers)  # type: ignore[operator]

        results: list[EvalResult] = []

        if n_workers == 1:
            results = self._evaluate_sequential(
                graphs, scorer, k_max, n_zeros_compare,
            )
        else:
            results = self._evaluate_parallel(
                graphs, k_max, n_zeros_compare, n_workers,
            )

        return sorted(results, key=lambda r: r.score, reverse=True)

    def _evaluate_sequential(
        self,
        graphs: Sequence[QuantumGraph],
        scorer: SpectralScorer | None,
        k_max: float,
        n_zeros_compare: int,
    ) -> list[EvalResult]:
        scorer = scorer or SpectralScorer()
        results: list[EvalResult] = []
        for graph in tqdm(graphs, desc="Evaluating graphs", unit="graph"):
            try:
                result = self.evaluate_graph(
                    graph, scorer, k_max, n_zeros_compare,
                )
                results.append(result)
            except Exception as exc:
                _warn_eval_failure(graph, exc)
        return results

    def _evaluate_parallel(
        self,
        graphs: Sequence[QuantumGraph],
        k_max: float,
        n_zeros_compare: int,
        n_workers: int,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(
                    _evaluate_single,
                    graph,
                    k_max,
                    self._n_points,
                    n_zeros_compare,
                ): i
                for i, graph in enumerate(graphs)
            }
            with tqdm(total=len(graphs), desc="Evaluating graphs", unit="graph") as pbar:
                for future in as_completed(future_to_idx):
                    pbar.update(1)
                    try:
                        graph, scoring_result, spectrum = future.result()
                        results.append(EvalResult(
                            graph=graph,
                            scoring_result=scoring_result,
                            spectrum=spectrum,
                        ))
                    except Exception as exc:
                        idx = future_to_idx[future]
                        _warn_eval_failure(graphs[idx], exc)
        return results

    # ----- adaptive k_max -----

    @staticmethod
    def adaptive_k_max(
        n_zeros_target: int,
        total_length: float | None = None,
        graph: QuantumGraph | None = None,
    ) -> float:
        """Estimate k_max needed to capture *n_zeros_target* eigenvalues.

        Uses Weyl's law for quantum graphs:
            N(k) ~ (total_length * k) / pi

        Parameters
        ----------
        n_zeros_target : int
            Desired number of eigenvalues.
        total_length : float or None
            Sum of edge lengths.  Inferred from *graph* if not given.
        graph : QuantumGraph or None
            Used to compute total_length when not explicitly supplied.

        Returns
        -------
        float
            Estimated k_max with a 20% safety margin.
        """
        if total_length is None:
            if graph is None:
                raise ValueError(
                    "Provide either total_length or graph."
                )
            total_length = sum(graph.edge_lengths)

        if total_length <= 0:
            raise ValueError("total_length must be positive.")

        k_estimate = (n_zeros_target * math.pi) / total_length
        return k_estimate * 1.2  # 20% safety margin


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _warn_eval_failure(graph: QuantumGraph, exc: Exception) -> None:
    """Log a non-fatal evaluation failure."""
    import warnings
    warnings.warn(
        f"Graph evaluation failed (n_v={graph.n_vertices}, "
        f"n_e={graph.n_edges}): {exc}",
        RuntimeWarning,
        stacklevel=3,
    )
