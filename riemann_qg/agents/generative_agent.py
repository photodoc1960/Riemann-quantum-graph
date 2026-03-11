"""Generative agent for proposing candidate quantum graphs.

This agent implements four strategies for producing new quantum graph candidates:
- PrimeLengthStrategy: edge lengths as scaled logarithms of primes
- MutationStrategy: small perturbations of high-scoring graphs
- TemplateStrategy: known topologies with parameter variation
- CrossoverStrategy: combining features of two parent graphs

The generative agent is the main source of diversity in the search population.
Fitness-proportionate selection ensures high-scoring graphs contribute more
offspring, while each strategy provides a different mode of exploration.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import networkx as nx
import numpy as np

from riemann_qg.core.quantum_graph import QuantumGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRIMES_CACHE: list[int] = []


def _primes_up_to(n: int) -> list[int]:
    """Return list of primes up to *n* via sieve of Eratosthenes."""
    global _PRIMES_CACHE
    if _PRIMES_CACHE and _PRIMES_CACHE[-1] >= n:
        return [p for p in _PRIMES_CACHE if p <= n]
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    _PRIMES_CACHE = [i for i, v in enumerate(sieve) if v]
    return list(_PRIMES_CACHE)


def _first_n_primes(n: int) -> list[int]:
    """Return the first *n* prime numbers."""
    upper = max(30, int(n * math.log(n + 2) * 1.3) + 20)
    while True:
        primes = _primes_up_to(upper)
        if len(primes) >= n:
            return primes[:n]
        upper *= 2


def _fitness_proportionate_select(
    population: Sequence[QuantumGraph],
    scores: np.ndarray,
    rng: np.random.Generator,
) -> QuantumGraph:
    """Select one graph with probability proportional to score."""
    shifted = scores - scores.min() + 1e-8
    probs = shifted / shifted.sum()
    idx = rng.choice(len(population), p=probs)
    return population[idx]


def _ensure_connected(adj: np.ndarray) -> np.ndarray:
    """Add edges to make adjacency matrix represent a connected graph."""
    g = nx.from_numpy_array(adj)
    components = list(nx.connected_components(g))
    result = adj.copy()
    for i in range(len(components) - 1):
        u = min(components[i])
        v = min(components[i + 1])
        result[u, v] = max(1, result[u, v])
        result[v, u] = result[u, v]
    return result


# ---------------------------------------------------------------------------
# Strategy enum and base
# ---------------------------------------------------------------------------

class StrategyType(Enum):
    PRIME_LENGTH = auto()
    MUTATION = auto()
    TEMPLATE = auto()
    CROSSOVER = auto()


class _Strategy(ABC):
    """Base class for graph generation strategies."""

    @abstractmethod
    def generate(
        self,
        population: Sequence[QuantumGraph],
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> QuantumGraph:
        ...


# ---------------------------------------------------------------------------
# PrimeLengthStrategy
# ---------------------------------------------------------------------------

class PrimeLengthStrategy(_Strategy):
    """Generate graphs with edge lengths = alpha * ln(p)."""

    def __init__(
        self,
        min_vertices: int = 3,
        max_vertices: int = 10,
        alpha_range: tuple[float, float] = (0.3, 3.0),
    ) -> None:
        self._min_v = min_vertices
        self._max_v = max_vertices
        self._alpha_range = alpha_range

    def generate(
        self,
        population: Sequence[QuantumGraph],
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> QuantumGraph:
        n_v = int(rng.integers(self._min_v, self._max_v + 1))
        edge_prob = rng.uniform(0.3, 0.8)
        adj = (rng.random((n_v, n_v)) < edge_prob).astype(int)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)
        adj = _ensure_connected(adj)

        n_edges = int(np.triu(adj, k=1).sum())
        primes = _first_n_primes(max(n_edges, 1))
        alpha = rng.uniform(*self._alpha_range)
        lengths = tuple(alpha * math.log(p) for p in primes[:n_edges])

        return QuantumGraph.from_adjacency(adj, edge_lengths=lengths)


# ---------------------------------------------------------------------------
# MutationStrategy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationParams:
    """Tunable knobs for mutation."""
    length_perturb_scale: float = 0.1
    phase_perturb_scale: float = 0.3
    add_edge_prob: float = 0.15
    remove_edge_prob: float = 0.15
    swap_lengths_prob: float = 0.20
    perturb_length_prob: float = 0.30
    scattering_perturb_prob: float = 0.20


class MutationStrategy(_Strategy):
    """Mutate a high-scoring parent graph."""

    def __init__(self, params: MutationParams | None = None) -> None:
        self._params = params or MutationParams()

    def generate(
        self,
        population: Sequence[QuantumGraph],
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> QuantumGraph:
        parent = _fitness_proportionate_select(population, scores, rng)
        roll = rng.random()
        p = self._params

        # Cumulative thresholds — must sum to 1.0
        t1 = p.add_edge_prob
        t2 = t1 + p.remove_edge_prob
        t3 = t2 + p.swap_lengths_prob
        t4 = t3 + p.perturb_length_prob

        if roll < t1:
            return self._add_edge(parent, rng)
        if roll < t2:
            return self._remove_edge(parent, rng)
        if roll < t3:
            return self._swap_lengths(parent, rng)
        if roll < t4:
            return self._perturb_length(parent, rng)
        return self._perturb_scattering(parent, rng)

    # -- individual mutation operators --

    def _perturb_length(
        self, graph: QuantumGraph, rng: np.random.Generator,
    ) -> QuantumGraph:
        if not graph.edge_lengths:
            return graph
        lengths = list(graph.edge_lengths)
        idx = int(rng.integers(0, len(lengths)))
        # Scale perturbation relative to edge length (not absolute)
        scale = max(lengths[idx] * self._params.length_perturb_scale, 0.01)
        eps = rng.normal(0, scale)
        lengths[idx] = max(1e-6, lengths[idx] + eps)
        return graph.with_edge_lengths(lengths)

    def _swap_lengths(
        self, graph: QuantumGraph, rng: np.random.Generator,
    ) -> QuantumGraph:
        if len(graph.edge_lengths) < 2:
            return graph
        lengths = list(graph.edge_lengths)
        i, j = rng.choice(len(lengths), size=2, replace=False)
        lengths[i], lengths[j] = lengths[j], lengths[i]
        return graph.with_edge_lengths(lengths)

    def _add_edge(
        self, graph: QuantumGraph, rng: np.random.Generator,
    ) -> QuantumGraph:
        adj = graph.adjacency.copy()
        n = adj.shape[0]
        if n < 2:
            return graph
        zeros = list(zip(*np.where(np.triu(adj, k=1) == 0)))
        if not zeros:
            return graph
        i, j = zeros[int(rng.integers(0, len(zeros)))]
        adj[i, j] = adj[j, i] = 1
        new_len = rng.uniform(0.5, 3.0)

        # Build lengths in canonical upper-triangle order to stay aligned
        old_lengths = list(graph.edge_lengths)
        old_edge_idx = 0
        lengths: list[float] = []
        for r in range(n):
            for c in range(r + 1, n):
                if adj[r, c] > 0:
                    if (r, c) == (i, j):
                        lengths.append(new_len)
                    else:
                        if old_edge_idx < len(old_lengths):
                            lengths.append(old_lengths[old_edge_idx])
                        else:
                            lengths.append(rng.uniform(0.5, 3.0))
                        old_edge_idx += 1
        return QuantumGraph.from_adjacency(adj, edge_lengths=tuple(lengths))

    def _remove_edge(
        self, graph: QuantumGraph, rng: np.random.Generator,
    ) -> QuantumGraph:
        adj = graph.adjacency.copy()
        edges = list(zip(*np.where(np.triu(adj, k=1) > 0)))
        if len(edges) <= graph.n_vertices - 1:
            return graph  # keep connected
        idx = int(rng.integers(0, len(edges)))
        i, j = edges[idx]
        adj[i, j] = adj[j, i] = 0
        if not nx.is_connected(nx.from_numpy_array(adj)):
            return graph  # reject disconnecting removals
        lengths = list(graph.edge_lengths)
        if idx < len(lengths):
            lengths.pop(idx)
        return QuantumGraph.from_adjacency(adj, edge_lengths=tuple(lengths))

    def _perturb_scattering(
        self, graph: QuantumGraph, rng: np.random.Generator,
    ) -> QuantumGraph:
        if not graph.vertex_scattering:
            return graph
        # Extract TRS-breaking phase from off-diagonal element.
        # For D @ S_neumann @ D† with D = diag(1, e^{iθ}, ...),
        # element (0,1) = S_neumann[0,1] * e^{-iθ}.  Since S_neumann[0,1]
        # is real positive (= 2/d), the TRS phase θ = -angle(s[0,1]).
        # NOTE: s[0,0] is always S_neumann[0,0] regardless of θ — useless.
        current_phases = [
            float(-np.angle(s[0, 1])) if s.shape[0] >= 2 else 0.0
            for s in graph.vertex_scattering
        ]
        idx = int(rng.integers(0, len(current_phases)))
        current_phases[idx] += rng.uniform(
            -self._params.phase_perturb_scale,
            self._params.phase_perturb_scale,
        )
        return graph.with_scattering_phases(current_phases)


# ---------------------------------------------------------------------------
# TemplateStrategy
# ---------------------------------------------------------------------------

class TemplateStrategy(_Strategy):
    """Generate graphs from known high-performing topologies."""

    def __init__(
        self,
        alpha_range: tuple[float, float] = (0.3, 3.0),
    ) -> None:
        self._alpha_range = alpha_range

    def generate(
        self,
        population: Sequence[QuantumGraph],
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> QuantumGraph:
        template_type = rng.choice(["complete", "cycle", "expander"])
        n_v = int(rng.integers(3, 10))
        alpha = rng.uniform(*self._alpha_range)

        if template_type == "complete":
            g = nx.complete_graph(n_v)
        elif template_type == "cycle":
            g = nx.cycle_graph(n_v)
        else:
            # Approximate expander via random regular graph
            degree = min(n_v - 1, max(3, int(rng.integers(3, 6))))
            if degree >= n_v or (n_v * degree) % 2 != 0:
                degree = min(n_v - 1, degree + 1) if (n_v * degree) % 2 else degree
                if (n_v * degree) % 2 != 0:
                    degree = max(2, degree - 1)
            try:
                g = nx.random_regular_graph(degree, n_v)
            except nx.NetworkXError:
                g = nx.complete_graph(n_v)

        adj = nx.to_numpy_array(g).astype(int)
        n_edges = int(np.triu(adj, k=1).sum())
        primes = _first_n_primes(max(n_edges, 1))
        lengths = tuple(alpha * math.log(p) for p in primes[:n_edges])

        return QuantumGraph.from_adjacency(adj, edge_lengths=lengths)


# ---------------------------------------------------------------------------
# CrossoverStrategy
# ---------------------------------------------------------------------------

class CrossoverStrategy(_Strategy):
    """Combine two parent graphs via graph crossover."""

    def generate(
        self,
        population: Sequence[QuantumGraph],
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> QuantumGraph:
        parent_a = _fitness_proportionate_select(population, scores, rng)
        # Retry up to 3 times for a distinct second parent
        parent_b = parent_a
        for _ in range(3):
            parent_b = _fitness_proportionate_select(population, scores, rng)
            if parent_b is not parent_a:
                break

        n_v = max(parent_a.n_vertices, parent_b.n_vertices)

        # Combine adjacency: union of edges, randomly thinned
        adj_a = _pad_adjacency(parent_a.adjacency, n_v)
        adj_b = _pad_adjacency(parent_b.adjacency, n_v)
        mask = (rng.random((n_v, n_v)) < 0.5).astype(int)
        mask = np.maximum(mask, mask.T)
        combined = np.where(mask, adj_a, adj_b)
        np.fill_diagonal(combined, 0)
        combined = _ensure_connected(combined)

        n_edges = int(np.triu(combined, k=1).sum())
        all_lengths = list(parent_a.edge_lengths) + list(parent_b.edge_lengths)
        if not all_lengths:
            all_lengths = [1.0]
        all_lengths_arr = np.array(all_lengths)
        lengths = tuple(
            float(rng.choice(all_lengths_arr)) for _ in range(n_edges)
        )

        return QuantumGraph.from_adjacency(combined, edge_lengths=lengths)


def _pad_adjacency(adj: np.ndarray, size: int) -> np.ndarray:
    """Pad adjacency matrix to *size* x *size*."""
    if adj.shape[0] >= size:
        return adj[:size, :size]
    padded = np.zeros((size, size), dtype=int)
    n = adj.shape[0]
    padded[:n, :n] = adj
    return padded


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

_STRATEGIES: dict[StrategyType, type[_Strategy]] = {
    StrategyType.PRIME_LENGTH: PrimeLengthStrategy,
    StrategyType.MUTATION: MutationStrategy,
    StrategyType.TEMPLATE: TemplateStrategy,
    StrategyType.CROSSOVER: CrossoverStrategy,
}


class GenerativeAgent:
    """Proposes candidate quantum graphs using configurable strategies.

    The generative agent maintains a registry of strategies and selects among
    them (or uses a caller-specified strategy) to produce batches of new
    candidate graphs for spectral evaluation.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._strategies: dict[StrategyType, _Strategy] = {
            StrategyType.PRIME_LENGTH: PrimeLengthStrategy(),
            StrategyType.MUTATION: MutationStrategy(),
            StrategyType.TEMPLATE: TemplateStrategy(),
            StrategyType.CROSSOVER: CrossoverStrategy(),
        }

    def propose_batch(
        self,
        population: Sequence[QuantumGraph],
        scores: np.ndarray,
        n_new: int,
        strategy: StrategyType | None = None,
    ) -> list[QuantumGraph]:
        """Generate *n_new* candidate graphs.

        Parameters
        ----------
        population : sequence of QuantumGraph
            Current population of graphs.
        scores : ndarray
            Scores aligned with *population*.
        n_new : int
            Number of new candidates to produce.
        strategy : StrategyType or None
            If None, cycle through strategies uniformly.

        Returns
        -------
        list[QuantumGraph]
            New candidate graphs (never mutates inputs).
        """
        scores_arr = np.asarray(scores, dtype=float)
        candidates: list[QuantumGraph] = []
        strategy_keys = list(self._strategies.keys())

        for i in range(n_new):
            key = strategy if strategy is not None else strategy_keys[i % len(strategy_keys)]
            strat = self._strategies[key]
            try:
                candidate = strat.generate(population, scores_arr, self._rng)
                candidates.append(candidate)
            except Exception:
                # Fallback: prime-length strategy
                fallback = self._strategies[StrategyType.PRIME_LENGTH]
                candidates.append(
                    fallback.generate(population, scores_arr, self._rng)
                )

        return candidates
