"""Pattern agent -- extracts structural patterns from high-scoring graphs.

Analyses the top-performing fraction of the population across four axes:
topology, edge lengths, scattering matrices, and periodic-orbit structure.
The resulting ``PopulationInsights`` dataclass is consumed by the orchestrator
and symbolic agent to guide subsequent search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import networkx as nx
import numpy as np
from scipy import stats as sp_stats

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


def _log_primes(n: int) -> np.ndarray:
    return np.array([math.log(p) for p in _first_n_primes(n)])


# ---------------------------------------------------------------------------
# Insight dataclasses (all frozen / immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopologyStats:
    """Summary statistics of graph topologies in the top fraction."""
    mean_n_vertices: float
    std_n_vertices: float
    mean_n_edges: float
    std_n_edges: float
    degree_sequence_mode: tuple[int, ...]
    mean_clustering: float
    mean_diameter: float


@dataclass(frozen=True)
class EdgeLengthStats:
    """How edge lengths relate to logarithms of primes."""
    mean_length: float
    std_length: float
    min_nearest_prime_log_dist: float
    mean_nearest_prime_log_dist: float
    correlation_with_log_primes: float
    best_alpha: float  # best-fit scaling L ≈ alpha * ln(p)


@dataclass(frozen=True)
class ScatteringStats:
    """Phase structure of vertex scattering matrices."""
    mean_phase: float
    std_phase: float
    fraction_near_neumann: float  # fraction of matrices close to Neumann form


@dataclass(frozen=True)
class OrbitStats:
    """Periodic orbit alignment with prime logarithms."""
    mean_prime_overlap: float
    std_prime_overlap: float


@dataclass(frozen=True)
class PopulationInsights:
    """Complete analysis of the top-performing sub-population."""
    topology: TopologyStats
    edge_lengths: EdgeLengthStats
    scattering: ScatteringStats
    orbits: OrbitStats
    n_graphs_analyzed: int
    summary: str


# ---------------------------------------------------------------------------
# PatternAgent
# ---------------------------------------------------------------------------

class PatternAgent:
    """Extracts what high-scoring quantum graphs have in common.

    Given a population with scores, the agent selects the top fraction and
    computes distributional statistics over topology, edge lengths, scattering
    matrices, and periodic-orbit structure.
    """

    def __init__(self, n_primes_ref: int = 50) -> None:
        self._log_primes = _log_primes(n_primes_ref)

    def analyze_population(
        self,
        graphs: Sequence[QuantumGraph],
        scores: np.ndarray,
        top_fraction: float = 0.1,
    ) -> PopulationInsights:
        """Analyse the top-performing graphs.

        Parameters
        ----------
        graphs : sequence of QuantumGraph
        scores : ndarray
        top_fraction : float in (0, 1]

        Returns
        -------
        PopulationInsights
        """
        scores_arr = np.asarray(scores, dtype=float)
        n_top = max(1, int(len(graphs) * top_fraction))
        top_idx = np.argsort(scores_arr)[-n_top:]
        top_graphs = [graphs[i] for i in top_idx]

        topo = self._topology_stats(top_graphs)
        edge = self._edge_length_stats(top_graphs)
        scat = self._scattering_stats(top_graphs)
        orb = self._orbit_stats(top_graphs)

        summary = self._build_summary(topo, edge, scat, orb, n_top)

        return PopulationInsights(
            topology=topo,
            edge_lengths=edge,
            scattering=scat,
            orbits=orb,
            n_graphs_analyzed=n_top,
            summary=summary,
        )

    # ---- topology ----

    def _topology_stats(
        self, graphs: Sequence[QuantumGraph],
    ) -> TopologyStats:
        n_verts = [g.n_vertices for g in graphs]
        n_edges_list = [g.n_edges for g in graphs]

        degree_seqs: list[tuple[int, ...]] = []
        clusterings: list[float] = []
        diameters: list[float] = []

        for g in graphs:
            nx_g = nx.from_numpy_array(g.adjacency)
            deg_seq = tuple(sorted(
                (d for _, d in nx_g.degree()), reverse=True,
            ))
            degree_seqs.append(deg_seq)
            clusterings.append(nx.average_clustering(nx_g))
            if nx.is_connected(nx_g):
                diameters.append(float(nx.diameter(nx_g)))
            else:
                diameters.append(float("inf"))

        # Mode of degree sequences (most common)
        mode_seq = max(set(degree_seqs), key=degree_seqs.count)

        return TopologyStats(
            mean_n_vertices=float(np.mean(n_verts)),
            std_n_vertices=float(np.std(n_verts)),
            mean_n_edges=float(np.mean(n_edges_list)),
            std_n_edges=float(np.std(n_edges_list)),
            degree_sequence_mode=mode_seq,
            mean_clustering=float(np.mean(clusterings)),
            mean_diameter=float(np.mean([d for d in diameters if d < float("inf")])) if any(d < float("inf") for d in diameters) else float("inf"),
        )

    # ---- edge lengths ----

    def _edge_length_stats(
        self, graphs: Sequence[QuantumGraph],
    ) -> EdgeLengthStats:
        all_lengths: list[float] = []
        for g in graphs:
            all_lengths.extend(g.edge_lengths)

        if not all_lengths:
            return EdgeLengthStats(
                mean_length=0.0, std_length=0.0,
                min_nearest_prime_log_dist=float("inf"),
                mean_nearest_prime_log_dist=float("inf"),
                correlation_with_log_primes=0.0,
                best_alpha=1.0,
            )

        arr = np.array(all_lengths)

        # Distance to nearest ln(p)
        dists = np.array([
            float(np.min(np.abs(self._log_primes - l)))
            for l in arr
        ])

        # Correlation with log-primes (use first len(arr) primes)
        n_cmp = min(len(arr), len(self._log_primes))
        sorted_lengths = np.sort(arr)[:n_cmp]
        lp = self._log_primes[:n_cmp]
        if n_cmp >= 2 and np.std(sorted_lengths) > 0:
            corr = float(np.corrcoef(sorted_lengths, lp)[0, 1])
        else:
            corr = 0.0

        # Best-fit alpha: L ≈ alpha * ln(p)
        best_alpha = self._fit_alpha(arr)

        return EdgeLengthStats(
            mean_length=float(np.mean(arr)),
            std_length=float(np.std(arr)),
            min_nearest_prime_log_dist=float(np.min(dists)),
            mean_nearest_prime_log_dist=float(np.mean(dists)),
            correlation_with_log_primes=corr,
            best_alpha=best_alpha,
        )

    def _fit_alpha(self, lengths: np.ndarray) -> float:
        """Find alpha minimising sum |L_i - alpha * ln(p_i)|."""
        n = min(len(lengths), len(self._log_primes))
        sorted_l = np.sort(lengths)[:n]
        lp = self._log_primes[:n]
        denom = np.dot(lp, lp)
        if denom < 1e-15:
            return 1.0
        return float(np.dot(sorted_l, lp) / denom)

    # ---- scattering matrices ----

    def _scattering_stats(
        self, graphs: Sequence[QuantumGraph],
    ) -> ScatteringStats:
        phases: list[float] = []
        near_neumann_count = 0
        total_count = 0

        for g in graphs:
            if not g.vertex_scattering:
                continue
            for mat in g.vertex_scattering:
                eigenvalues = np.linalg.eigvals(mat)
                mat_phases = np.angle(eigenvalues)
                phases.extend(mat_phases.tolist())

                # Check closeness to Neumann form
                deg = mat.shape[0]
                neumann = (2.0 / deg) * np.ones((deg, deg)) - np.eye(deg)
                if np.allclose(mat, neumann, atol=0.1):
                    near_neumann_count += 1
                total_count += 1

        if not phases:
            return ScatteringStats(
                mean_phase=0.0,
                std_phase=0.0,
                fraction_near_neumann=1.0,
            )

        return ScatteringStats(
            mean_phase=float(np.mean(phases)),
            std_phase=float(np.std(phases)),
            fraction_near_neumann=(
                near_neumann_count / total_count if total_count else 1.0
            ),
        )

    # ---- periodic orbits ----

    def _orbit_stats(
        self, graphs: Sequence[QuantumGraph],
    ) -> OrbitStats:
        overlaps: list[float] = []
        for g in graphs:
            overlap = self._prime_orbit_overlap_quick(g)
            overlaps.append(overlap)

        return OrbitStats(
            mean_prime_overlap=float(np.mean(overlaps)) if overlaps else 0.0,
            std_prime_overlap=float(np.std(overlaps)) if overlaps else 0.0,
        )

    def _prime_orbit_overlap_quick(self, graph: QuantumGraph) -> float:
        """Quick estimate of orbit-prime overlap using edge lengths only."""
        if not graph.edge_lengths:
            return 0.0
        lengths = np.array(graph.edge_lengths)
        # Count edges whose length is close to some ln(p)
        matches = 0
        for l_val in lengths:
            dists = np.abs(self._log_primes - l_val)
            if np.min(dists) < 0.05 * l_val:
                matches += 1
        return matches / len(lengths)

    # ---- summary ----

    def _build_summary(
        self,
        topo: TopologyStats,
        edge: EdgeLengthStats,
        scat: ScatteringStats,
        orb: OrbitStats,
        n_analyzed: int,
    ) -> str:
        lines = [
            f"Population analysis ({n_analyzed} top graphs):",
            f"  Topology: {topo.mean_n_vertices:.1f} +/- {topo.std_n_vertices:.1f} vertices, "
            f"{topo.mean_n_edges:.1f} +/- {topo.std_n_edges:.1f} edges, "
            f"clustering {topo.mean_clustering:.3f}",
            f"  Edge lengths: mean {edge.mean_length:.3f}, "
            f"corr with ln(p) = {edge.correlation_with_log_primes:.3f}, "
            f"best alpha = {edge.best_alpha:.3f}",
            f"  Scattering: mean phase {scat.mean_phase:.3f}, "
            f"{scat.fraction_near_neumann:.0%} near Neumann",
            f"  Orbits: prime overlap {orb.mean_prime_overlap:.3f} "
            f"+/- {orb.std_prime_overlap:.3f}",
        ]
        return "\n".join(lines)
