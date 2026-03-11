"""Evolutionary algorithm over quantum graph populations.

Uses the DEAP framework for tournament selection, crossover, and mutation
operators adapted to quantum graph topology and parameter spaces.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from deap import base, creator, tools

from riemann_qg.search.topology_search import (
    crossover_topologies,
    mutate_topology,
)

if TYPE_CHECKING:
    from riemann_qg.core.quantum_graph import QuantumGraph


@dataclass(frozen=True)
class EvolutionConfig:
    """Immutable configuration for evolutionary search."""

    population_size: int = 200
    tournament_size: int = 3
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    elite_fraction: float = 0.05
    length_mutation_scale: float = 0.1
    phase_mutation_scale: float = 0.2


@dataclass(frozen=True)
class GenerationResult:
    """Immutable result from one generation of evolution."""

    population: tuple["QuantumGraph", ...]
    scores: tuple[float, ...]
    best_score: float
    mean_score: float
    generation: int


def _ensure_deap_types() -> None:
    """Register DEAP fitness and individual types if not already present."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", dict, fitness=creator.FitnessMax)


class EvolutionarySearch:
    """Evolutionary search engine for quantum graph optimization.

    Manages a population of quantum graphs and evolves them using
    tournament selection, topology crossover, and parameter mutation.
    """

    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self._config = config or EvolutionConfig()
        _ensure_deap_types()
        self._toolbox = self._build_toolbox()

    def _build_toolbox(self) -> base.Toolbox:
        """Configure DEAP toolbox with selection operator."""
        tb = base.Toolbox()
        tb.register(
            "select",
            tools.selTournament,
            tournsize=self._config.tournament_size,
        )
        return tb

    def evolve(
        self,
        population: list["QuantumGraph"],
        scores: list[float],
        config: EvolutionConfig | None = None,
    ) -> list["QuantumGraph"]:
        """Produce the next generation from current population and scores.

        Applies elitism, tournament selection, crossover, and mutation.

        Args:
            population: Current generation of quantum graphs.
            scores: Fitness scores corresponding to each graph.
            config: Optional override config for this generation.

        Returns:
            New population of quantum graphs for the next generation.
        """
        cfg = config or self._config
        n_elite = max(1, int(len(population) * cfg.elite_fraction))

        elites = _select_elites(population, scores, n_elite)
        individuals = _wrap_individuals(population, scores)
        selected = self._toolbox.select(
            individuals, len(population) - n_elite
        )

        offspring = _produce_offspring(
            selected, cfg.crossover_prob, cfg.mutation_prob, cfg
        )

        return list(elites) + offspring

    @property
    def config(self) -> EvolutionConfig:
        """Return the current evolution configuration."""
        return self._config


def _select_elites(
    population: list["QuantumGraph"],
    scores: list[float],
    n_elite: int,
) -> list["QuantumGraph"]:
    """Select the top-n graphs by score (elitism)."""
    paired = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
    return [graph for _, graph in paired[:n_elite]]


def _wrap_individuals(
    population: list["QuantumGraph"],
    scores: list[float],
) -> list:
    """Wrap graphs as DEAP individuals with fitness values."""
    individuals = []
    for graph, score in zip(population, scores):
        ind = creator.Individual({"graph": graph})
        ind.fitness.values = (score,)
        individuals.append(ind)
    return individuals


def _produce_offspring(
    selected: list,
    crossover_prob: float,
    mutation_prob: float,
    config: EvolutionConfig,
) -> list["QuantumGraph"]:
    """Generate offspring via crossover and mutation operators."""
    offspring = []
    parents = [ind["graph"] for ind in selected]

    i = 0
    while i < len(parents):
        if i + 1 < len(parents) and random.random() < crossover_prob:
            child = _crossover_graphs(parents[i], parents[i + 1], config)
            offspring.append(child)
            i += 2
        else:
            child = _maybe_mutate(parents[i], mutation_prob, config)
            offspring.append(child)
            i += 1

    while len(offspring) < len(selected):
        parent = random.choice(parents)
        offspring.append(_maybe_mutate(parent, mutation_prob, config))

    return offspring[: len(selected)]


def _crossover_graphs(
    parent1: "QuantumGraph",
    parent2: "QuantumGraph",
    config: EvolutionConfig,
) -> "QuantumGraph":
    """Crossover two quantum graphs: topology + blended parameters."""
    child_adj = crossover_topologies(parent1.adjacency, parent2.adjacency)

    n_edges_child = int(np.sum(np.triu(child_adj, k=1)))
    blended_lengths = _blend_edge_lengths(
        parent1.edge_lengths, parent2.edge_lengths, n_edges_child
    )

    return parent1.from_adjacency(child_adj, blended_lengths)


def _blend_edge_lengths(
    lengths1: list[float],
    lengths2: list[float],
    target_n: int,
) -> list[float]:
    """Blend edge lengths from two parents for the child graph."""
    all_lengths = list(lengths1) + list(lengths2)
    if not all_lengths:
        return [1.0] * target_n

    result = []
    for i in range(target_n):
        alpha = random.random()
        l1 = all_lengths[i % len(all_lengths)]
        l2 = all_lengths[(i + 1) % len(all_lengths)]
        result.append(alpha * l1 + (1 - alpha) * l2)
    return result


def _maybe_mutate(
    graph: "QuantumGraph",
    mutation_prob: float,
    config: EvolutionConfig,
) -> "QuantumGraph":
    """Conditionally apply mutation to a quantum graph."""
    if random.random() >= mutation_prob:
        return graph

    mutation_type = random.choice(["topology", "lengths", "phases"])

    if mutation_type == "topology":
        return _mutate_graph_topology(graph)
    elif mutation_type == "lengths":
        return _mutate_edge_lengths(graph, config.length_mutation_scale)
    else:
        return _mutate_phases(graph, config.phase_mutation_scale)


def _mutate_graph_topology(graph: "QuantumGraph") -> "QuantumGraph":
    """Mutate the graph topology while preserving connectivity."""
    new_adj = mutate_topology(graph.adjacency)
    n_edges = int(np.sum(np.triu(new_adj, k=1)))

    old_lengths = list(graph.edge_lengths)
    new_lengths = _adapt_lengths_to_new_topology(old_lengths, n_edges)

    return graph.from_adjacency(new_adj, new_lengths)


def _mutate_edge_lengths(
    graph: "QuantumGraph", scale: float
) -> "QuantumGraph":
    """Perturb edge lengths by small random amounts."""
    lengths = np.array(graph.edge_lengths)
    perturbation = np.random.normal(0, scale, size=len(lengths))
    new_lengths = np.maximum(lengths + perturbation * lengths, 0.01)
    return graph.with_edge_lengths(list(new_lengths))


def _mutate_phases(
    graph: "QuantumGraph", scale: float
) -> "QuantumGraph":
    """Perturb scattering matrix phases."""
    phases = [
        np.angle(m[0, 0]) if m is not None and m.size > 0 else 0.0
        for m in graph.vertex_scattering
    ]
    perturbation = np.random.normal(0, scale, size=len(phases))
    new_phases = [float(p + dp) for p, dp in zip(phases, perturbation)]
    return graph.with_scattering_phases(new_phases)


def _adapt_lengths_to_new_topology(
    old_lengths: list[float], n_edges_new: int
) -> list[float]:
    """Adapt edge lengths when topology changes edge count."""
    if not old_lengths:
        return [1.0] * n_edges_new

    result = []
    for i in range(n_edges_new):
        result.append(old_lengths[i % len(old_lengths)])
    return result
