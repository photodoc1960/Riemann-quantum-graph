# Vertex Boundary Conditions Determine Spectral Correspondence in Quantum Graphs

A computational study of the spectral correspondence problem on finite quantum graphs, using the nontrivial zeros of the Riemann zeta function as a test target.

This repository accompanies the manuscript *"Vertex Boundary Conditions Determine Spectral Correspondence in Quantum Graphs"* (J. D. Slater, draft v3 — `Primes_in_the_Wires_v3.md`). All code, optimization results, and verification data supporting the manuscript's claims are reproducible from this repository.

---

## Key findings

1. **Neumann ceiling theorem.** Under standard Neumann boundary conditions on the cycle graph $C_n$, the scattering monodromy reduces algebraically to $\sigma_x^n$ with eigenvalues $\pm 1$ independent of edge lengths. The spectrum depends only on total cycle length and parity of $n$. The resulting hard correspondence ceiling $\mathcal{S} \approx 0.092$ at $N = 100$ Riemann zeros cannot be broken by any choice of topology, cycle size, or optimization effort. Adding per-vertex TRS-breaking phases raises the ceiling to $\mathcal{S} \approx 0.72$ but the ceiling remains size-independent. See `Theorem1_Neumann_Rigidity.md`.

2. **U(2) scattering breaks the ceiling.** Replacing Neumann with the most general self-adjoint boundary condition at degree-2 vertices — general U(2) scattering matrices — yields spectral correspondence with the first 100 Riemann zeros at MAD $= 0.10$–$0.15$ mean spacings. Best observed: $\mathcal{S} = 0.9258$ at MAD $= 0.097$ mean spacings on $C_{20}$ (verified bit-exact reproducible from saved parameters; not reproducible by independent CMA-ES initialization). Mean across optimization restarts: $0.907 \pm 0.010$ at $C_{20}$.

3. **Topology is approximately irrelevant under unitary scattering.** Replacing the cycle topology by a theta graph at matched parameter count yields scores within $\pm 0.005$ of the matched cycle across all parameter counts tested up to $96$ dimensions. The relevant degree of freedom is the dimension of the unitary group at vertices, not the graph topology.

4. **No algebraic structure in the optimal vertex matrices at 50 dp.** A PSLQ integer-relation search against bases of $\pi$, prime logarithms up to $\ln 61$, and standard mathematical constants returns zero relations with $|c| \leq 10$ at residual $< 10^{-20}$. If the optimal U(2) vertex conditions admit a closed-form characterization, it does not involve simple linear combinations of the natural constants at this precision.

---

## Repository contents

### Manuscript and theorem

| File | Description |
|------|-------------|
| `Primes_in_the_Wires_v3.md` | Current manuscript draft (Sections I–IV + references) |
| `Theorem1_Neumann_Rigidity.md` | Formal statement and proof of the Neumann ceiling theorem (Theorem 1) |
| `Primes_JPhysA_final_v2.pdf` | Earlier manuscript version (PRE / J Phys A submission) — retained for diff |

### Core library (`riemann_qg/`)

Reusable quantum-graph spectral computation and scoring.

- `core/quantum_graph.py` — `QuantumGraph` class, secular equation solver, U(d) vertex parameterizations
- `core/scoring.py` — `SpectralScorer` with nonlinear Weyl rescaling and 0.70/0.20/0.10 score weights
- `core/riemann_zeros.py` — first 100+ Riemann zeros at sufficient precision for scoring
- `core/trace_formula.py` — periodic orbit sums and prime overlap diagnostics
- `agents/`, `search/`, `analysis/` — earlier evolutionary-search infrastructure, retained for reproducing the topology search in `results/exhaustive_7v.jsonl`
- `tests/` — pytest suite (33 tests, all passing)

### Experiments (`experiments/`)

Standalone scripts that produce the manuscript's results. Each writes to a JSON or JSONL artifact in `results/` and is independently reproducible.

| Script | Produces | Manuscript section |
|--------|----------|-------------------|
| `scaling_experiment.py` | `scaling_results.csv`, `scaling_plot.png` | III.B (Neumann scaling) |
| `unitary_scattering_experiment.py` | `unitary_scattering.jsonl` | III.C (initial U(2) demonstration) |
| `unitary_deep.py` | `unitary_deep.jsonl`, `best_unitary_graph.json` | III.C ($C_7$ U(2) result, $\mathcal{S} = 0.897$) |
| `u2_scaling.py` | `u2_scaling_results.json`, `u2_scaling_plot.png` | III.D ($C_5$ through $C_{21}$ sweep) |
| `verify_c20_best.py` | `c20_verification.json` | III.D (bit-exact reproducibility of $\mathcal{S} = 0.9258$) |
| `c20_refinement.py` | `c20_refinement.jsonl` | III.D (24 Gaussian-perturbed restarts) |
| `theta_unitary.py` | `theta_unitary_results.json`, `theta_unitary_plot.png` | III.E (topology-independence) |
| `pslq_matrix_analysis.py` | `pslq_analysis.json` | III.F (algebraic structure negative result) |
| `scattering_analysis.py` | `scattering_analysis.txt` | III.C (matrix structure analysis) |

Earlier experiments retained for reproducibility of the topology search and cycle sweeps:
`exhaustive_7v.py`, `cycle_sweep.py`, `cycle_deep.py`, `theta_deep.py`.

### Results (`results/`)

All optimization runs save incrementally to `results/`. Key artifacts:

- `figure1_score_vs_edges.png`, `figure2_neumann_residuals.png` — manuscript figures
- `u2_scaling_plot.png`, `theta_unitary_plot.png` — supplementary figures
- `best_unitary_graph.json` — original $C_7$ U(2) best graph ($\mathcal{S} = 0.897$)
- `u2_scaling_results.json` — cycle scaling sweep, including the $C_{20}$ best at $\mathcal{S} = 0.9258$
- `theta_unitary_results.json` — theta graph U(3)+U(2) results $k = 2$ to $6$
- `c20_verification.json` — verification protocol output confirming bit-exact reproducibility
- `pslq_analysis.json` — algebraic structure search results
- `exhaustive_7v.jsonl` — 457 optimized 7-vertex graph topologies
- `scaling_experiment.jsonl`, `scaling_results.csv` — Neumann scaling data $C_3$ through $C_{28}$

Earlier evolutionary-search trajectory data (`trajectory.jsonl`, `residuals.jsonl`) is retained for reproducibility of the topology-search phase of the project.

---

## Installation

```bash
git clone https://github.com/photodoc1960/Riemann-quantum-graph.git
cd Riemann-quantum-graph
pip install -r requirements.txt
python -m pytest riemann_qg/tests/ -v
```

### Dependencies

- Python 3.10+
- numpy, scipy, sympy, networkx
- mpmath (high-precision arithmetic, PSLQ)
- cma (CMA-ES optimization)
- matplotlib (figures)
- rich (console output)
- pytest (test suite)

---

## Reproducing the manuscript's results

The principal results can be reproduced from saved parameters without rerunning the optimizations.

**Verify $\mathcal{S} = 0.9258$ at $C_{20}$:**
```bash
python experiments/verify_c20_best.py
```
This loads the saved parameters from `results/u2_scaling_results.json` and computes the score from scratch. Output should match `results/c20_verification.json`: score reproduces to 16-digit precision at $\mathcal{S} = 0.92580628$.

**Verify the PSLQ negative result:**
```bash
python experiments/pslq_matrix_analysis.py
```
Tests the saved $C_7$ U(2) matrices, eigenvalue phases, determinant phases, and edge lengths against bases of mathematical constants at 50 decimal places. Output: `results/pslq_analysis.json` (verdict: `PARTIALLY_STRUCTURED`, the only structure being the near-$(-1)$ monodromy property of the composition).

### Reproducing the optimizations

Reproducing the optimizations requires substantial compute. Approximate single-machine runtime on a 16-thread CPU:

| Experiment | Runtime |
|------------|---------|
| `unitary_deep.py` (single C_n) | 3–4 hours |
| `u2_scaling.py` (full sweep C_5 to C_21) | ~100 hours |
| `theta_unitary.py` (k=2 to 6) | ~55 hours |
| `c20_refinement.py` | ~25–70 hours depending on restart count |
| `pslq_matrix_analysis.py` | ~5 minutes |
| `verify_c20_best.py` | ~30 seconds |
| `exhaustive_7v.py` (457 graphs) | ~30 hours |

All optimization scripts checkpoint after each restart or cycle size and resume from the saved JSON or JSONL output if interrupted.

---

## Manuscript status

The accompanying manuscript has been submitted to and desk-rejected from Physical Review Letters, Physical Review E, and Journal of Physics A: Mathematical and Theoretical, on framing rather than scientific grounds. The current draft (`Primes_in_the_Wires_v3.md`) reframes the contribution as a small theorem (the Neumann ceiling) plus a numerical demonstration (U(2) breakthrough) plus a topology-independence finding, with the Hilbert-Pólya connection appearing only in the introduction and discussion sections. The intended primary submission target is *Communications in Number Theory and Physics*.

---

## Citation

If you use this code or build on these results, please cite the manuscript and this repository.

Manuscript citation will be added once a venue is finalized. For now:

> J. D. Slater. *Vertex Boundary Conditions Determine Spectral Correspondence in Quantum Graphs.* Manuscript v3, May 2026. Code and data: https://github.com/photodoc1960/Riemann-quantum-graph.

---

## Acknowledgements

Software implementation and analysis scripting were developed with assistance from Claude Code (Anthropic). Scientific interpretation, framing, and conclusions are the author's.

---

## License

This repository contains research code released without warranty. The intended use is academic reproducibility and scientific reuse. A formal license file is not yet attached; if you intend to use the code beyond reproduction of the manuscript's results, please contact the author.

---

## Mathematical background and earlier project notes

The original README contained an extended introduction to the Riemann hypothesis, the Hilbert-Pólya conjecture, GUE statistics, quantum graphs, and the trace formula, as well as detailed documentation of an earlier multi-agent evolutionary search architecture. That material described the project as it stood before the rewrite that produced the current results. It is retained in `docs/README_v1_archive.md` for historical reference. The current state of the project is the U(d) scattering framework described above; the evolutionary-search architecture remains in the code but is no longer the project's headline approach.
