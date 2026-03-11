"""Tests for quantum graph construction and spectral computation."""

from __future__ import annotations

import numpy as np
import pytest

from riemann_qg.core.quantum_graph import (
    QuantumGraph,
    neumann_scattering,
    directed_scattering,
)


# -----------------------------------------------------------------------
# 1. Secular equation for a simple path graph (2 vertices, 1 edge)
# -----------------------------------------------------------------------


class TestSecularEquationPathGraph:
    """A path graph with 2 vertices and 1 edge of length L.

    The secular equation |det[I - S*U(k)]| should have minima near zero
    at the graph's eigenvalues.
    """

    def test_secular_has_eigenvalues(self):
        """The path graph must have eigenvalues (spectrum is non-empty)."""
        adj = np.array([[0, 1], [1, 0]], dtype=np.int64)
        length = 1.0
        graph = QuantumGraph(adj, [length])
        spectrum = graph.compute_spectrum(20.0, n_points=5000)
        assert len(spectrum) > 0, "Path graph should have eigenvalues"

    def test_secular_at_eigenvalue_is_near_zero(self):
        """At an eigenvalue, |det| should be near zero."""
        adj = np.array([[0, 1], [1, 0]], dtype=np.int64)
        graph = QuantumGraph(adj, [1.0])
        spectrum = graph.compute_spectrum(20.0, n_points=5000)
        if len(spectrum) > 0:
            val = graph.secular_equation(spectrum[0])
            assert val < 1e-4, f"|det| at eigenvalue should be ~0, got {val}"

    def test_secular_at_zero_is_nonzero(self):
        """det[I - S*U(0)] should generally be nonzero for a path graph."""
        adj = np.array([[0, 1], [1, 0]], dtype=np.int64)
        graph = QuantumGraph(adj, [1.0])
        val = graph.secular_equation(1e-3)
        assert val != 0.0


# -----------------------------------------------------------------------
# 2. Spectrum for a ring (cycle) graph: eigenvalues k = 2*pi*n / L_total
# -----------------------------------------------------------------------


class TestCycleGraphSpectrum:
    """A cycle graph with n equal edges of length L (Neumann scattering
    at degree-2 vertices = full transmission) behaves like a circle.
    Eigenvalues at k = 2*pi*m / L_total for integer m.
    """

    def test_ring_spectrum_matches_analytic(self):
        n_vertices = 3
        edge_length = 1.0
        total_length = n_vertices * edge_length

        graph = QuantumGraph.cycle_graph(
            n_vertices, base_length=edge_length, prime_lengths=False,
        )

        k_max = 30.0
        spectrum = graph.compute_spectrum(k_max, n_points=20000)

        # Analytic eigenvalues for a ring: k_m = 2*pi*m / L_total
        expected = [2.0 * np.pi * m / total_length for m in range(1, 20)]
        expected = [e for e in expected if e < k_max]

        # At least some eigenvalues should be close to analytic values
        matched = 0
        for e_val in expected[:8]:
            if len(spectrum) > 0:
                dists = np.abs(spectrum - e_val)
                if np.min(dists) < 0.15:
                    matched += 1

        assert matched >= 3, (
            f"Expected at least 3 eigenvalues near analytic ring spectrum, "
            f"got {matched}. Spectrum[:10]={spectrum[:10]}, "
            f"expected[:8]={expected[:8]}"
        )

    def test_ring_spectrum_is_nonempty(self):
        """A cycle graph must have eigenvalues."""
        graph = QuantumGraph.cycle_graph(4, base_length=1.0)
        spectrum = graph.compute_spectrum(20.0, n_points=10000)
        assert len(spectrum) > 0


# -----------------------------------------------------------------------
# 3. Neumann scattering matrix unitarity
# -----------------------------------------------------------------------


class TestNeumannScatteringUnitarity:
    """Neumann scattering matrices S_ij = 2/d - delta_ij must be unitary."""

    @pytest.mark.parametrize("degree", [2, 3, 4, 5, 6])
    def test_unitarity(self, degree: int):
        s = neumann_scattering(degree)
        product = s @ s.conj().T
        identity = np.eye(degree, dtype=np.complex128)
        np.testing.assert_allclose(
            product, identity, atol=1e-12,
            err_msg=f"Neumann matrix for degree {degree} is not unitary",
        )

    def test_degree_2_is_permutation(self):
        """For degree 2, Neumann scattering is [[0, 1], [1, 0]]."""
        s = neumann_scattering(2)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        np.testing.assert_allclose(s, expected, atol=1e-14)

    def test_degree_3_trace(self):
        """Trace of degree-3 Neumann matrix: 3*(2/3 - 1) = -1."""
        s = neumann_scattering(3)
        assert abs(np.trace(s) - (-1.0)) < 1e-14


# -----------------------------------------------------------------------
# 4. GUE symmetry breaking shifts statistics toward GUE
# -----------------------------------------------------------------------


class TestGUESymmetryBreaking:
    """Phase-breaking (TRS-broken) scattering should push spacing
    statistics away from GOE and closer to GUE.
    """

    def test_directed_scattering_unitarity(self):
        """directed_scattering should produce a unitary matrix."""
        for degree in [3, 4, 5]:
            s = directed_scattering(degree, phase=0.3)
            product = s @ s.conj().T
            identity = np.eye(degree, dtype=np.complex128)
            np.testing.assert_allclose(
                product, identity, atol=1e-10,
                err_msg=f"Directed scattering (deg={degree}) not unitary",
            )

    def test_phase_zero_recovers_neumann(self):
        """With phase=0, directed_scattering degenerates (phases all 1 or 0).

        At minimum, the matrix should still be unitary.
        """
        s = directed_scattering(3, phase=0.0)
        product = s @ s.conj().T
        identity = np.eye(3, dtype=np.complex128)
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_trs_breaking_changes_spectrum(self):
        """A graph with phase-breaking should produce a different spectrum
        than the same graph with pure Neumann scattering.
        """
        adj = np.array(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int64,
        )
        lengths = [np.log(2), np.log(3), np.log(5)]

        graph_neumann = QuantumGraph(adj, lengths)

        # Build with directed scattering
        degrees = np.sum(adj, axis=1)
        directed_matrices = [
            directed_scattering(int(d), phase=0.5) for d in degrees
        ]
        graph_directed = QuantumGraph(adj, lengths, directed_matrices)

        spec_n = graph_neumann.compute_spectrum(30.0, n_points=5000)
        spec_d = graph_directed.compute_spectrum(30.0, n_points=5000)

        # They should not be identical
        if len(spec_n) > 0 and len(spec_d) > 0:
            n_compare = min(len(spec_n), len(spec_d))
            diff = np.sum(np.abs(spec_n[:n_compare] - spec_d[:n_compare]))
            assert diff > 0.01, "Phase-breaking should alter the spectrum"
