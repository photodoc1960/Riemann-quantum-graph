"""Identifies shared features of high-scoring quantum graphs.

Extracts topology, edge length, and scattering matrix statistics from
graph populations and compares enrichment in top performers vs. overall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from sympy import primerange

if TYPE_CHECKING:
    from riemann_qg.core.quantum_graph import QuantumGraph


@dataclass(frozen=True)
class EnrichmentResult:
    """Immutable result from population comparison analysis."""

    feature_name: str
    top_mean: float
    all_mean: float
    enrichment_ratio: float
    significant: bool


def extract_topology_features(
    graphs: list["QuantumGraph"],
) -> pd.DataFrame:
    """Extract topology statistics as a DataFrame.

    Computes vertex count, edge count, degree statistics, clustering,
    and diameter for each graph.

    Args:
        graphs: List of quantum graphs to analyze.

    Returns:
        DataFrame with one row per graph and topology feature columns.
    """
    records = [_topology_record(g, idx) for idx, g in enumerate(graphs)]
    return pd.DataFrame(records)


def _topology_record(graph: "QuantumGraph", idx: int) -> dict:
    """Build a single topology feature record for one graph."""
    nx_graph = nx.from_numpy_array(graph.adjacency)
    degrees = [d for _, d in nx_graph.degree()]

    return {
        "index": idx,
        "n_vertices": graph.n_vertices,
        "n_edges": graph.n_edges,
        "mean_degree": float(np.mean(degrees)),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "clustering_coeff": nx.average_clustering(nx_graph),
        "diameter": nx.diameter(nx_graph) if nx.is_connected(nx_graph) else -1,
        "density": nx.density(nx_graph),
        "is_regular": len(set(degrees)) == 1,
    }


def extract_length_features(
    graphs: list["QuantumGraph"],
    n_primes: int = 25,
) -> pd.DataFrame:
    """Analyze edge lengths vs logarithms of primes.

    For each graph, computes statistics on how well edge lengths align
    with the set {ln(p) : p prime}.

    Args:
        graphs: List of quantum graphs.
        n_primes: Number of primes to compare against.

    Returns:
        DataFrame with edge length analysis per graph.
    """
    prime_logs = _prime_logs(n_primes)
    records = [
        _length_record(g, idx, prime_logs) for idx, g in enumerate(graphs)
    ]
    return pd.DataFrame(records)


def _length_record(
    graph: "QuantumGraph", idx: int, prime_logs: np.ndarray
) -> dict:
    """Build edge length feature record for one graph."""
    lengths = np.array(graph.edge_lengths)
    if len(lengths) == 0:
        return {"index": idx, "mean_length": 0.0, "prime_log_corr": 0.0}

    deviations = _min_prime_log_deviations(lengths, prime_logs)

    return {
        "index": idx,
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": float(np.min(lengths)),
        "max_length": float(np.max(lengths)),
        "mean_prime_deviation": float(np.mean(deviations)),
        "max_prime_deviation": float(np.max(deviations)),
        "prime_log_corr": _prime_log_correlation(lengths, prime_logs),
        "n_near_prime_log": int(np.sum(deviations < 0.05)),
    }


def _min_prime_log_deviations(
    lengths: np.ndarray, prime_logs: np.ndarray
) -> np.ndarray:
    """Compute minimum distance from each length to nearest prime log."""
    deviations = np.array([
        float(np.min(np.abs(length - prime_logs)))
        for length in lengths
    ])
    return deviations


def _prime_log_correlation(
    lengths: np.ndarray, prime_logs: np.ndarray
) -> float:
    """Correlation between sorted edge lengths and prime logs."""
    sorted_lengths = np.sort(lengths)
    n = min(len(sorted_lengths), len(prime_logs))
    if n < 2:
        return 0.0
    corr_matrix = np.corrcoef(sorted_lengths[:n], prime_logs[:n])
    return float(corr_matrix[0, 1])


def extract_scattering_features(
    graphs: list["QuantumGraph"],
) -> pd.DataFrame:
    """Analyze scattering matrix phase structure.

    Extracts phase angles, unitarity deviation, and symmetry-breaking
    indicators from vertex scattering matrices.

    Args:
        graphs: List of quantum graphs.

    Returns:
        DataFrame with scattering feature columns.
    """
    records = [_scattering_record(g, idx) for idx, g in enumerate(graphs)]
    return pd.DataFrame(records)


def _scattering_record(graph: "QuantumGraph", idx: int) -> dict:
    """Build scattering feature record for one graph."""
    phases = []
    unitarity_devs = []

    for matrix in graph.vertex_scattering:
        if matrix is None or matrix.size == 0:
            continue
        phases.extend(np.angle(matrix).flatten().tolist())
        identity = np.eye(matrix.shape[0])
        unitarity_dev = np.linalg.norm(
            matrix @ matrix.conj().T - identity
        )
        unitarity_devs.append(float(unitarity_dev))

    return {
        "index": idx,
        "mean_phase": float(np.mean(phases)) if phases else 0.0,
        "std_phase": float(np.std(phases)) if phases else 0.0,
        "mean_unitarity_dev": (
            float(np.mean(unitarity_devs)) if unitarity_devs else 0.0
        ),
        "has_complex_phases": bool(
            any(abs(p) > 1e-10 for p in phases)
        ),
        "n_vertices_with_phase": sum(
            1 for p in phases if abs(p) > 1e-10
        ),
    }


def compare_populations(
    top_graphs: list["QuantumGraph"],
    all_graphs: list["QuantumGraph"],
    significance_threshold: float = 1.5,
) -> list[EnrichmentResult]:
    """Compare features enriched in top-scoring vs. all graphs.

    Computes enrichment ratios for each numeric topology feature.

    Args:
        top_graphs: High-scoring subset of graphs.
        all_graphs: Full population of graphs.
        significance_threshold: Minimum enrichment ratio for significance.

    Returns:
        List of EnrichmentResult, one per feature.
    """
    top_topo = extract_topology_features(top_graphs)
    all_topo = extract_topology_features(all_graphs)

    numeric_cols = top_topo.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "index"]

    results = []
    for col in numeric_cols:
        top_mean = float(top_topo[col].mean())
        all_mean = float(all_topo[col].mean())
        ratio = top_mean / all_mean if abs(all_mean) > 1e-10 else 0.0

        results.append(EnrichmentResult(
            feature_name=col,
            top_mean=top_mean,
            all_mean=all_mean,
            enrichment_ratio=ratio,
            significant=abs(ratio) > significance_threshold,
        ))

    return results


def _prime_logs(n_primes: int) -> np.ndarray:
    """Compute logarithms of the first n primes."""
    primes = list(primerange(2, _nth_prime_upper_bound(n_primes)))[:n_primes]
    return np.log(np.array(primes, dtype=float))


def _nth_prime_upper_bound(n: int) -> int:
    """Upper bound for the nth prime (for sieve range)."""
    if n < 6:
        return 15
    import math
    return int(n * (math.log(n) + math.log(math.log(n)))) + 10
