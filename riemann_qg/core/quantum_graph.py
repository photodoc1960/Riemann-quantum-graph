"""Quantum graph construction and spectral computation.

Implements the Kottos & Smilansky secular equation framework for
computing eigenvalues of quantum graphs with arbitrary vertex scattering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar


@dataclass(frozen=True)
class QuantumGraphSpec:
    """Immutable specification of a quantum graph."""

    adjacency: NDArray[np.int64]
    edge_lengths: tuple[float, ...]
    vertex_scattering: tuple[NDArray[np.complex128], ...]
    n_vertices: int
    n_edges: int
    bond_pairs: tuple[tuple[int, int], ...]


class QuantumGraph:
    """Quantum graph with vertex scattering matrices and spectral computation.

    Uses the Kottos & Smilansky convention: the secular equation is
    det[I - S * U(k)] = 0 where U(k) is the diagonal bond propagation
    matrix and S is the block-diagonal vertex scattering matrix.
    """

    def __init__(
        self,
        adjacency: NDArray[np.int64],
        edge_lengths: list[float],
        vertex_scattering: Optional[list[NDArray[np.complex128]]] = None,
    ) -> None:
        self._validate_inputs(adjacency, edge_lengths)
        self._adjacency = adjacency.copy()
        self._n_vertices = adjacency.shape[0]

        # Build directed bond list from adjacency
        self._bond_pairs = _build_bond_pairs(adjacency)
        self._n_edges = len(edge_lengths)

        # Map each bond to its edge length
        self._edge_lengths = tuple(edge_lengths)
        self._bond_lengths = _assign_bond_lengths(
            self._bond_pairs, adjacency, edge_lengths
        )

        if vertex_scattering is None:
            degrees = np.sum(adjacency, axis=1)
            self._vertex_scattering = tuple(
                neumann_scattering(int(d)) for d in degrees
            )
        else:
            self._vertex_scattering = tuple(
                s.copy() for s in vertex_scattering
            )

        self._verify_unitarity()
        self._build_full_scattering_matrix()

    @staticmethod
    def _validate_inputs(
        adjacency: NDArray[np.int64], edge_lengths: list[float]
    ) -> None:
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Adjacency must be a square matrix")
        if not np.allclose(adjacency, adjacency.T):
            raise ValueError("Adjacency matrix must be symmetric")
        n_edges_from_adj = int(np.sum(adjacency)) // 2
        if len(edge_lengths) != n_edges_from_adj:
            raise ValueError(
                f"Expected {n_edges_from_adj} edge lengths, got {len(edge_lengths)}"
            )
        if any(length <= 0 for length in edge_lengths):
            raise ValueError("All edge lengths must be positive")

    def _verify_unitarity(self) -> None:
        """Verify each vertex scattering matrix is unitary."""
        for i, s_mat in enumerate(self._vertex_scattering):
            product = s_mat @ s_mat.conj().T
            identity = np.eye(s_mat.shape[0], dtype=np.complex128)
            if not np.allclose(product, identity, atol=1e-10):
                raise ValueError(
                    f"Vertex {i} scattering matrix is not unitary"
                )

    def _build_full_scattering_matrix(self) -> None:
        """Build the full 2E x 2E scattering matrix (block-diagonal)."""
        n_bonds = len(self._bond_pairs)
        self._full_s = np.zeros(
            (n_bonds, n_bonds), dtype=np.complex128
        )
        # For each vertex, fill in the scattering block
        for v_idx in range(self._n_vertices):
            s_v = self._vertex_scattering[v_idx]
            # Bonds arriving at vertex v_idx
            incoming = [
                i for i, (_, to_v) in enumerate(self._bond_pairs)
                if to_v == v_idx
            ]
            # Bonds leaving vertex v_idx
            outgoing = [
                i for i, (from_v, _) in enumerate(self._bond_pairs)
                if from_v == v_idx
            ]
            for row_local, out_b in enumerate(outgoing):
                for col_local, in_b in enumerate(incoming):
                    self._full_s[out_b, in_b] = s_v[row_local, col_local]

    @property
    def spec(self) -> QuantumGraphSpec:
        """Return immutable specification of this graph."""
        return QuantumGraphSpec(
            adjacency=self._adjacency.copy(),
            edge_lengths=self._edge_lengths,
            vertex_scattering=tuple(
                s.copy() for s in self._vertex_scattering
            ),
            n_vertices=self._n_vertices,
            n_edges=self._n_edges,
            bond_pairs=self._bond_pairs,
        )

    @property
    def n_vertices(self) -> int:
        return self._n_vertices

    @property
    def n_edges(self) -> int:
        return self._n_edges

    @property
    def bond_pairs(self) -> tuple[tuple[int, int], ...]:
        return self._bond_pairs

    @property
    def adjacency(self) -> NDArray[np.int64]:
        return self._adjacency.copy()

    @property
    def edge_lengths(self) -> tuple[float, ...]:
        return self._edge_lengths

    @property
    def vertex_scattering(self) -> tuple[NDArray[np.complex128], ...]:
        return self._vertex_scattering

    def with_edge_lengths(
        self, edge_lengths: tuple[float, ...] | list[float]
    ) -> QuantumGraph:
        """Return a new graph with different edge lengths."""
        return QuantumGraph(
            self._adjacency, list(edge_lengths), list(self._vertex_scattering)
        )

    def with_scattering_phases(
        self, phases: list[float]
    ) -> QuantumGraph:
        """Return a new graph with phase-modified scattering matrices."""
        degrees = np.sum(self._adjacency, axis=1)
        new_scattering = [
            directed_scattering(int(d), p) for d, p in zip(degrees, phases)
        ]
        return QuantumGraph(
            self._adjacency, list(self._edge_lengths), new_scattering
        )

    @classmethod
    def from_adjacency(
        cls,
        adjacency: NDArray[np.int64],
        edge_lengths: tuple[float, ...] | list[float] | None = None,
        vertex_scattering: list[NDArray[np.complex128]] | None = None,
    ) -> QuantumGraph:
        """Create a QuantumGraph from an adjacency matrix.

        If edge_lengths is None, defaults to unit lengths.
        """
        adj = np.asarray(adjacency, dtype=np.int64)
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0)
        n_edges = int(np.sum(np.triu(adj, k=1)))
        if n_edges == 0:
            adj[0, 1] = adj[1, 0] = 1
            n_edges = 1
        if edge_lengths is None:
            lengths = [1.0] * n_edges
        else:
            lengths = list(edge_lengths)
            if len(lengths) != n_edges:
                # Adjust length list to match topology.  Truncate excess or
                # pad with the mean of existing lengths (not silent 1.0).
                if len(lengths) > n_edges:
                    lengths = lengths[:n_edges]
                else:
                    fill = sum(lengths) / len(lengths) if lengths else 1.0
                    lengths.extend([fill] * (n_edges - len(lengths)))
        return cls(adj, lengths, vertex_scattering)

    def secular_det(self, k: float) -> complex:
        """Compute det[I - S * U(k)] as a complex value."""
        n_bonds = len(self._bond_pairs)
        lengths_arr = np.array(self._bond_lengths, dtype=np.float64)
        phases = np.exp(1j * k * lengths_arr)
        # S @ diag(phases) == S * phases[None, :] via broadcasting (avoids diag alloc)
        matrix = np.eye(n_bonds, dtype=np.complex128) - self._full_s * phases[None, :]
        return np.linalg.det(matrix)

    def secular_equation(self, k: float) -> float:
        """Compute |det[I - S * U(k)]| for eigenvalue detection."""
        return abs(self.secular_det(k))

    def compute_spectrum(
        self, k_max: float, n_points: int = 10000
    ) -> NDArray[np.float64]:
        """Find all k in [0, k_max] where secular_equation(k) = 0.

        Uses dense sampling to find local minima of |det|, then
        refines each minimum and accepts it if |det| < threshold.
        """
        k_values = np.linspace(1e-6, k_max, n_points)
        abs_det = _dense_abs_det_sample(
            k_values, self._full_s, self._bond_lengths
        )

        # Threshold for accepting refined minima as eigenvalues.
        # minimize_scalar(method='bounded') has limited precision (~1e-5 in k),
        # so |det| at the refined point is bounded by |det'| * delta_k, not by
        # numerical det accuracy.  A flat 1e-4 catches true eigenvalues while
        # rejecting spurious minima (which have |det| >> 0.01).
        threshold = 1e-4

        eigenvalues: list[float] = []

        for i in range(1, len(abs_det) - 1):
            if abs_det[i] < abs_det[i - 1] and abs_det[i] < abs_det[i + 1]:
                try:
                    result = minimize_scalar(
                        self.secular_equation,
                        bounds=(k_values[i - 1], k_values[i + 1]),
                        method="bounded",
                    )
                    if result.fun < threshold:
                        eigenvalues.append(float(result.x))
                except (ValueError, RuntimeError):
                    continue

        return np.array(sorted(eigenvalues), dtype=np.float64)

    # --- Factory methods ---

    @classmethod
    def complete_graph(
        cls,
        n: int,
        base_length: float = 1.0,
        prime_lengths: bool = False,
    ) -> QuantumGraph:
        """Create a complete graph K_n with optional prime-based lengths."""
        adj = np.ones((n, n), dtype=np.int64) - np.eye(n, dtype=np.int64)
        n_edges = n * (n - 1) // 2
        lengths = _make_lengths(n_edges, base_length, prime_lengths)
        return cls(adj, lengths)

    @classmethod
    def cycle_graph(
        cls,
        n: int,
        prime_lengths: bool = False,
        base_length: float = 1.0,
    ) -> QuantumGraph:
        """Create a cycle graph C_n."""
        adj = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[(i + 1) % n, i] = 1
        lengths = _make_lengths(n, base_length, prime_lengths)
        return cls(adj, lengths)

    @classmethod
    def random_graph(
        cls,
        n_vertices: int,
        edge_prob: float = 0.5,
        length_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> QuantumGraph:
        """Create a random Erdos-Renyi graph ensuring connectivity."""
        rng = np.random.default_rng(seed)
        while True:
            g = nx.erdos_renyi_graph(n_vertices, edge_prob, seed=rng)
            if nx.is_connected(g):
                break
        adj = nx.to_numpy_array(g, dtype=np.int64)
        n_edges = g.number_of_edges()
        lengths = [
            float(length_scale * (0.5 + rng.random()))
            for _ in range(n_edges)
        ]
        return cls(adj, lengths)


def neumann_scattering(degree: int) -> NDArray[np.complex128]:
    """Neumann vertex scattering matrix: S_ij = 2/d - delta_ij."""
    if degree == 0:
        raise ValueError("neumann_scattering: degree must be >= 1")
    if degree == 1:
        # Degree-1 vertex: pure reflection (Dirichlet end)
        return np.array([[-1.0]], dtype=np.complex128)
    s = (2.0 / degree) * np.ones(
        (degree, degree), dtype=np.complex128
    ) - np.eye(degree, dtype=np.complex128)
    return s


def directed_scattering(
    degree: int, phase: float
) -> NDArray[np.complex128]:
    """Scattering matrix with complex phase for TRS breaking.

    Uses a unitary similarity transform D @ S_neumann @ D† where
    D = diag(1, e^{iθ}, e^{2iθ}, ...) to break time-reversal symmetry
    while preserving unitarity exactly.
    """
    base = neumann_scattering(degree)
    phases = np.array(
        [np.exp(1j * phase * k) for k in range(degree)],
        dtype=np.complex128,
    )
    d_mat = np.diag(phases)
    d_mat_dag = np.diag(phases.conj())
    return d_mat @ base @ d_mat_dag


# --- Private helpers ---


def _build_bond_pairs(
    adjacency: NDArray[np.int64],
) -> tuple[tuple[int, int], ...]:
    """Build directed bond list from adjacency matrix."""
    n = adjacency.shape[0]
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if adjacency[i, j] > 0:
                pairs.append((i, j))
    return tuple(pairs)


def _assign_bond_lengths(
    bond_pairs: tuple[tuple[int, int], ...],
    adjacency: NDArray[np.int64],
    edge_lengths: list[float],
) -> list[float]:
    """Map each directed bond to its undirected edge length."""
    n = adjacency.shape[0]
    edge_map: dict[tuple[int, int], float] = {}
    edge_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                edge_map[(i, j)] = edge_lengths[edge_idx]
                edge_map[(j, i)] = edge_lengths[edge_idx]
                edge_idx += 1
    return [edge_map[bp] for bp in bond_pairs]


def _make_lengths(
    n_edges: int, base_length: float, prime_lengths: bool
) -> list[float]:
    """Generate edge lengths, optionally using ln(prime) scaling."""
    if prime_lengths:
        primes = _first_n_primes(n_edges)
        return [base_length * np.log(p) for p in primes]
    return [base_length] * n_edges


def _first_n_primes(n: int) -> list[int]:
    """Return the first n prime numbers."""
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


def _dense_abs_det_sample(
    k_values: NDArray[np.float64],
    full_s: NDArray[np.complex128],
    bond_lengths: list[float],
) -> NDArray[np.float64]:
    """Evaluate |det(I - S*U(k))| at many k values for minimum finding.

    Vectorised: builds all matrices at once via broadcasting, then
    batches the determinant calls through np.linalg.det on a 3-D array.
    """
    n_bonds = full_s.shape[0]
    lengths_arr = np.array(bond_lengths, dtype=np.float64)

    # phases shape: (n_k, n_bonds)
    phases = np.exp(1j * np.outer(k_values, lengths_arr))

    # S * diag(phases) == S[None,:,:] * phases[:,None,:] via broadcasting
    # matrices shape: (n_k, n_bonds, n_bonds)
    identity = np.eye(n_bonds, dtype=np.complex128)
    matrices = identity[None, :, :] - full_s[None, :, :] * phases[:, None, :]

    # Batched determinant
    return np.abs(np.linalg.det(matrices))
