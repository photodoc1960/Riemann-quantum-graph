"""Search and optimization algorithms."""

from riemann_qg.search.evolutionary import (
    EvolutionarySearch,
    EvolutionConfig,
    GenerationResult,
)
from riemann_qg.search.optimizer import (
    OptimizationResult,
    joint_optimize,
    optimize_edge_lengths,
    optimize_scattering,
)
from riemann_qg.search.topology_search import (
    crossover_topologies,
    enumerate_topologies,
    mutate_topology,
)

__all__ = [
    "EvolutionarySearch",
    "EvolutionConfig",
    "GenerationResult",
    "OptimizationResult",
    "crossover_topologies",
    "enumerate_topologies",
    "joint_optimize",
    "mutate_topology",
    "optimize_edge_lengths",
    "optimize_scattering",
]
