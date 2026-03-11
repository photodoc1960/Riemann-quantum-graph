"""Graph connectivity enumeration and mutation for quantum graph topology search.

Generates connected graph topologies and provides mutation/crossover operators
for evolutionary search over the space of graph structures.
"""

from __future__ import annotations

import random
from typing import Iterator

import networkx as nx
import numpy as np


def enumerate_topologies(
    min_v: int,
    max_v: int,
    min_e: int,
    max_e: int,
) -> Iterator[np.ndarray]:
    """Generate connected graph topologies within vertex/edge bounds.

    Yields adjacency matrices for connected graphs with vertex count in
    [min_v, max_v] and edge count in [min_e, max_e].

    Args:
        min_v: Minimum number of vertices.
        max_v: Maximum number of vertices.
        min_e: Minimum number of edges.
        max_e: Maximum number of edges.

    Yields:
        Adjacency matrices as numpy int arrays.
    """
    for n_vertices in range(min_v, max_v + 1):
        for n_edges in range(min_e, max_e + 1):
            if n_edges < n_vertices - 1:
                continue  # Cannot be connected
            max_possible = n_vertices * (n_vertices - 1) // 2
            if n_edges > max_possible:
                continue
            graph = _random_connected_graph(n_vertices, n_edges)
            if graph is not None:
                yield nx.to_numpy_array(graph, dtype=int)


def _random_connected_graph(
    n_vertices: int, n_edges: int, max_attempts: int = 50
) -> nx.Graph | None:
    """Build a random connected graph with exact vertex/edge counts.

    Starts from a random spanning tree, then adds random edges until the
    target edge count is reached.
    """
    if n_edges < n_vertices - 1:
        return None
    max_possible = n_vertices * (n_vertices - 1) // 2
    if n_edges > max_possible:
        return None

    for _ in range(max_attempts):
        tree = nx.random_labeled_tree(n_vertices)
        graph = nx.Graph(tree)
        all_possible = [
            (i, j)
            for i in range(n_vertices)
            for j in range(i + 1, n_vertices)
            if not graph.has_edge(i, j)
        ]
        extra_needed = n_edges - (n_vertices - 1)
        if extra_needed > len(all_possible):
            return None
        if extra_needed > 0:
            chosen = random.sample(all_possible, extra_needed)
            graph.add_edges_from(chosen)
        if graph.number_of_edges() == n_edges:
            return graph
    return None


def mutate_topology(adjacency: np.ndarray) -> np.ndarray:
    """Apply a random mutation to a graph topology, preserving connectivity.

    Mutations: add an edge, remove an edge (if graph stays connected),
    or rewire an edge.

    Args:
        adjacency: Square symmetric integer adjacency matrix.

    Returns:
        A new adjacency matrix (original is not modified).
    """
    adj = adjacency.copy()
    n = adj.shape[0]
    graph = nx.from_numpy_array(adj)

    mutation_type = random.choice(["add", "remove", "rewire"])

    if mutation_type == "add":
        adj = _try_add_edge(graph, n)
    elif mutation_type == "remove":
        adj = _try_remove_edge(graph, n)
    else:
        adj = _try_rewire_edge(graph, n)

    return adj


def _try_add_edge(graph: nx.Graph, n: int) -> np.ndarray:
    """Add a random edge to the graph."""
    non_edges = list(nx.non_edges(graph))
    if non_edges:
        u, v = random.choice(non_edges)
        new_graph = graph.copy()
        new_graph.add_edge(u, v)
        return nx.to_numpy_array(new_graph, dtype=int)
    return nx.to_numpy_array(graph, dtype=int)


def _try_remove_edge(graph: nx.Graph, n: int) -> np.ndarray:
    """Remove a random edge while keeping the graph connected."""
    edges = list(graph.edges())
    random.shuffle(edges)
    for u, v in edges:
        candidate = graph.copy()
        candidate.remove_edge(u, v)
        if nx.is_connected(candidate):
            return nx.to_numpy_array(candidate, dtype=int)
    return nx.to_numpy_array(graph, dtype=int)


def _try_rewire_edge(graph: nx.Graph, n: int) -> np.ndarray:
    """Rewire: remove one edge and add another, keeping connected."""
    edges = list(graph.edges())
    non_edges = list(nx.non_edges(graph))
    if not non_edges:
        return nx.to_numpy_array(graph, dtype=int)

    random.shuffle(edges)
    for u, v in edges:
        candidate = graph.copy()
        candidate.remove_edge(u, v)
        if nx.is_connected(candidate):
            new_u, new_v = random.choice(non_edges)
            candidate.add_edge(new_u, new_v)
            return nx.to_numpy_array(candidate, dtype=int)
    return nx.to_numpy_array(graph, dtype=int)


def crossover_topologies(
    adj1: np.ndarray, adj2: np.ndarray
) -> np.ndarray:
    """Combine two graph topologies via edge-set union-then-prune.

    Takes the union of both edge sets, then randomly removes edges until
    the edge count matches the average of the parents, while preserving
    connectivity.

    Args:
        adj1: First parent adjacency matrix.
        adj2: Second parent adjacency matrix.

    Returns:
        A new adjacency matrix for the child topology.
    """
    n = max(adj1.shape[0], adj2.shape[0])
    padded1 = _pad_adjacency(adj1, n)
    padded2 = _pad_adjacency(adj2, n)

    union = np.maximum(padded1, padded2)
    graph = nx.from_numpy_array(union)

    target_edges = (
        np.count_nonzero(np.triu(padded1, k=1))
        + np.count_nonzero(np.triu(padded2, k=1))
    ) // 2
    target_edges = max(target_edges, n - 1)

    graph = _prune_to_target(graph, target_edges)
    graph = _remove_isolated_vertices(graph)

    return nx.to_numpy_array(graph, dtype=int)


def _pad_adjacency(adj: np.ndarray, target_size: int) -> np.ndarray:
    """Pad an adjacency matrix to a larger size with zeros."""
    if adj.shape[0] >= target_size:
        return adj[:target_size, :target_size].copy()
    padded = np.zeros((target_size, target_size), dtype=int)
    n = adj.shape[0]
    padded[:n, :n] = adj
    return padded


def _prune_to_target(graph: nx.Graph, target_edges: int) -> nx.Graph:
    """Remove random edges until target count, preserving connectivity."""
    result = graph.copy()
    edges = list(result.edges())
    random.shuffle(edges)

    while result.number_of_edges() > target_edges and edges:
        u, v = edges.pop()
        if not result.has_edge(u, v):
            continue
        candidate = result.copy()
        candidate.remove_edge(u, v)
        if nx.is_connected(candidate):
            result = candidate

    return result


def _remove_isolated_vertices(graph: nx.Graph) -> nx.Graph:
    """Return a new graph with isolated vertices removed."""
    isolated = list(nx.isolates(graph))
    if not isolated:
        return graph
    result = graph.copy()
    result.remove_nodes_from(isolated)
    return nx.convert_node_labels_to_integers(result)
