"""Trace formula utilities for quantum graphs.

Implements periodic orbit sums, prime orbit overlap analysis,
and comparison with Riemann's explicit formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OrbitContribution:
    """Immutable record of a periodic orbit's contribution."""

    length: float
    stability: float
    contribution: float


@dataclass(frozen=True)
class OverlapResult:
    """Immutable result of prime orbit overlap analysis."""

    correlation: float
    matched_count: int
    total_orbits: int
    total_primes: int
    summary: str


@dataclass(frozen=True)
class DeviationResult:
    """Immutable result of explicit formula deviation analysis."""

    mean_deviation: float
    max_deviation: float
    rms_deviation: float
    k_values: tuple[float, ...]
    deviations: tuple[float, ...]


def periodic_orbit_sum(
    graph: object,
    k: float,
    n_max_length: int,
) -> float:
    """Compute density of states from periodic orbits via trace formula.

    d(k) = L_total/pi + (1/pi) * sum_p sum_r A_p^r * L_p * cos(r*k*L_p)

    where the sum runs over prime orbits p and repetitions r.

    Args:
        graph: QuantumGraph instance (must have bond_pairs, edge_lengths).
        k: Wave number at which to evaluate density.
        n_max_length: Maximum total orbit length in units of shortest bond.

    Returns:
        Density of states at wave number k.
    """
    bond_pairs = graph.bond_pairs  # type: ignore[attr-defined]
    edge_lengths = graph.edge_lengths  # type: ignore[attr-defined]
    adjacency = graph.spec.adjacency  # type: ignore[attr-defined]

    all_lengths = _collect_bond_lengths(bond_pairs, adjacency, edge_lengths)
    min_length = min(all_lengths) if all_lengths else 1.0
    max_total = n_max_length * min_length

    # Weyl term: average density
    total_length = sum(edge_lengths)
    density = total_length / np.pi

    # Find prime orbits up to max_total length
    orbits = _enumerate_prime_orbits(adjacency, all_lengths, max_total)

    # Sum contributions from prime orbits and their repetitions
    for orbit in orbits:
        max_rep = max(1, int(max_total / orbit.length))
        for r in range(1, max_rep + 1):
            amplitude = orbit.stability ** r
            density += (
                amplitude * orbit.length * np.cos(r * k * orbit.length)
            ) / np.pi

    return float(density)


def prime_orbit_overlap(
    graph: object,
    primes: list[int],
    n_primes: int,
) -> OverlapResult:
    """Measure how well periodic orbit lengths overlap with {ln(p)}.

    Computes the correlation between the set of primitive orbit lengths
    (normalized) and the set of logarithms of the first n_primes primes.

    Args:
        graph: QuantumGraph instance.
        primes: List of prime numbers to compare.
        n_primes: Number of primes to use.

    Returns:
        OverlapResult with correlation and match statistics.
    """
    bond_pairs = graph.bond_pairs  # type: ignore[attr-defined]
    edge_lengths = graph.edge_lengths  # type: ignore[attr-defined]
    adjacency = graph.spec.adjacency  # type: ignore[attr-defined]

    all_lengths = _collect_bond_lengths(bond_pairs, adjacency, edge_lengths)
    max_total = 10.0 * max(all_lengths) if all_lengths else 10.0

    orbits = _enumerate_prime_orbits(adjacency, all_lengths, max_total)
    orbit_lengths = sorted(o.length for o in orbits)

    primes_to_use = primes[:n_primes]
    log_primes = [float(np.log(p)) for p in primes_to_use]

    if not orbit_lengths or not log_primes:
        return OverlapResult(
            correlation=0.0,
            matched_count=0,
            total_orbits=len(orbit_lengths),
            total_primes=len(log_primes),
            summary="Insufficient data for overlap analysis",
        )

    # Normalize orbit lengths by the shortest
    min_orbit = min(orbit_lengths) if orbit_lengths else 1.0
    norm_orbits = [length / min_orbit for length in orbit_lengths]

    # Normalize log primes by ln(2)
    ln2 = float(np.log(2))
    norm_primes = [lp / ln2 for lp in log_primes]

    # Count matches within tolerance
    matched = _count_matches(norm_orbits, norm_primes, tolerance=0.05)

    # Compute correlation using overlapping range
    correlation = _compute_overlap_correlation(norm_orbits, norm_primes)

    total_possible = min(len(norm_orbits), len(norm_primes))
    summary = (
        f"Matched {matched}/{total_possible} orbits to ln(primes), "
        f"correlation={correlation:.4f}"
    )

    return OverlapResult(
        correlation=correlation,
        matched_count=matched,
        total_orbits=len(orbit_lengths),
        total_primes=len(log_primes),
        summary=summary,
    )


def explicit_formula_deviation(
    graph: object,
    k_values: list[float] | NDArray[np.float64],
) -> DeviationResult:
    """Compare quantum graph trace formula to Riemann's explicit formula.

    Riemann's explicit formula for the prime counting function:
    psi(x) = x - sum_rho x^rho/rho - ln(2*pi) - (1/2)ln(1-x^{-2})

    We compare the oscillatory parts of both formulas.

    Args:
        graph: QuantumGraph instance.
        k_values: Wave numbers at which to compare.

    Returns:
        DeviationResult with deviation statistics.
    """
    k_arr = np.asarray(k_values, dtype=np.float64)
    edge_lengths = graph.edge_lengths  # type: ignore[attr-defined]
    total_length = sum(edge_lengths)

    # Quantum graph smooth density
    smooth_density = total_length / np.pi

    # Compute quantum graph oscillatory part via periodic orbit sum
    graph_osc = np.array([
        periodic_orbit_sum(graph, k, n_max_length=10) - smooth_density
        for k in k_arr
    ])

    # Riemann explicit formula oscillatory part (simplified)
    riemann_osc = _riemann_oscillatory(k_arr, total_length)

    deviations = np.abs(graph_osc - riemann_osc)

    return DeviationResult(
        mean_deviation=float(np.mean(deviations)),
        max_deviation=float(np.max(deviations)),
        rms_deviation=float(np.sqrt(np.mean(deviations**2))),
        k_values=tuple(float(k) for k in k_arr),
        deviations=tuple(float(d) for d in deviations),
    )


# --- Private helpers ---


def _collect_bond_lengths(
    bond_pairs: tuple[tuple[int, int], ...],
    adjacency: NDArray[np.int64],
    edge_lengths: tuple[float, ...] | list[float],
) -> list[float]:
    """Map each directed bond to its edge length."""
    n = adjacency.shape[0]
    edge_map: dict[tuple[int, int], float] = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                edge_map[(i, j)] = edge_lengths[idx]
                edge_map[(j, i)] = edge_lengths[idx]
                idx += 1
    return [edge_map.get(bp, 1.0) for bp in bond_pairs]


def _enumerate_prime_orbits(
    adjacency: NDArray[np.int64],
    bond_lengths: list[float],
    max_total_length: float,
) -> list[OrbitContribution]:
    """Enumerate prime periodic orbits up to a maximum total length.

    Uses DFS on the directed bond graph to find closed walks,
    filtering to primitive (non-repeating) orbits.
    """
    n = adjacency.shape[0]
    if n == 0:
        return []

    # Build adjacency list with lengths
    adj_list: dict[int, list[tuple[int, float]]] = {}
    for i in range(n):
        adj_list[i] = []
        for j in range(n):
            if adjacency[i, j] > 0:
                # Find the edge length
                edge_idx = _edge_index(i, j, adjacency)
                length = bond_lengths[edge_idx] if edge_idx < len(bond_lengths) else 1.0
                adj_list[i].append((j, length))

    orbits: list[OrbitContribution] = []
    seen_lengths: set[float] = set()

    for start in range(n):
        _dfs_orbits(
            start, start, 0.0, [], adj_list,
            max_total_length, orbits, seen_lengths
        )

    return orbits


def _dfs_orbits(
    start: int,
    current: int,
    current_length: float,
    path: list[int],
    adj_list: dict[int, list[tuple[int, float]]],
    max_length: float,
    orbits: list[OrbitContribution],
    seen_lengths: set[float],
) -> None:
    """DFS to find closed walks from start vertex."""
    if current_length > max_length:
        return

    for neighbor, edge_len in adj_list[current]:
        new_length = current_length + edge_len
        if new_length > max_length:
            continue

        if neighbor == start and len(path) > 0:
            # Found a closed orbit - check if primitive
            rounded = round(new_length, 8)
            if rounded not in seen_lengths and _is_primitive(path):
                seen_lengths.add(rounded)
                # Stability amplitude decays with length
                stability = 1.0 / max(1.0, np.sqrt(len(path)))
                orbits.append(OrbitContribution(
                    length=new_length,
                    stability=stability,
                    contribution=stability * np.cos(new_length),
                ))
        elif neighbor not in path and len(path) < 8:
            path.append(neighbor)
            _dfs_orbits(
                start, neighbor, new_length, path,
                adj_list, max_length, orbits, seen_lengths,
            )
            path.pop()


def _is_primitive(path: list[int]) -> bool:
    """Check if a path is a primitive orbit (not a repetition)."""
    n = len(path)
    if n <= 1:
        return True
    for period in range(1, n // 2 + 1):
        if n % period == 0:
            is_repeat = all(
                path[i] == path[i % period]
                for i in range(n)
            )
            if is_repeat:
                return period == n
    return True


def _edge_index(i: int, j: int, adjacency: NDArray[np.int64]) -> int:
    """Get the edge index for an undirected edge (i, j)."""
    n = adjacency.shape[0]
    a, b = min(i, j), max(i, j)
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adjacency[u, v] > 0:
                if u == a and v == b:
                    return idx
                idx += 1
    return 0


def _count_matches(
    values_a: list[float],
    values_b: list[float],
    tolerance: float,
) -> int:
    """Count how many values in a have a near-match in b."""
    matched = 0
    used: set[int] = set()
    for a_val in values_a:
        for j, b_val in enumerate(values_b):
            if j not in used and abs(a_val - b_val) < tolerance:
                matched += 1
                used.add(j)
                break
    return matched


def _compute_overlap_correlation(
    values_a: list[float],
    values_b: list[float],
) -> float:
    """Compute correlation between two sets of normalized values."""
    n = min(len(values_a), len(values_b))
    if n < 2:
        return 0.0
    a_arr = np.array(values_a[:n])
    b_arr = np.array(values_b[:n])
    corr_matrix = np.corrcoef(a_arr, b_arr)
    return float(corr_matrix[0, 1])


def _riemann_oscillatory(
    k_values: NDArray[np.float64],
    total_length: float,
) -> NDArray[np.float64]:
    """Simplified Riemann explicit formula oscillatory part.

    Uses the first few non-trivial zeros to compute:
    osc(k) ~ -sum_n 2*Re[exp(i*k*L*gamma_n)] / |rho_n|

    where gamma_n are imaginary parts of Riemann zeros.
    """
    from .riemann_zeros import RiemannZeros

    rz = RiemannZeros()
    zeros = rz.get_zeros(min(30, rz.count))

    result = np.zeros_like(k_values)
    for gamma in zeros:
        rho_abs = np.sqrt(0.25 + gamma**2)
        result -= (
            2.0 * np.cos(k_values * total_length * gamma) / rho_abs
        )

    return result
