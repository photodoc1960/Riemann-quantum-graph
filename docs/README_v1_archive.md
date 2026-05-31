# Riemann Quantum Graph Search System

A multi-agent evolutionary search system that hunts for quantum graphs whose
eigenvalue spectra match the non-trivial zeros of the Riemann zeta function.

This project treats the search as a principled scientific investigation,
not a brute-force optimizer. Each generation accumulates structural knowledge
about which graph topologies, edge-length patterns, and scattering matrices
produce spectra that resemble the Riemann zeros — building toward a
constructive spectral interpretation of the Riemann Hypothesis.

---

## Table of Contents

1. [Mathematical Background](#mathematical-background)
2. [Why Quantum Graphs?](#why-quantum-graphs)
3. [The Secular Equation](#the-secular-equation)
4. [The Prime Connection](#the-prime-connection)
5. [Scoring and Validation](#scoring-and-validation)
6. [System Architecture](#system-architecture)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Interpreting Results](#interpreting-results)

---

## Mathematical Background

### The Riemann Hypothesis

The Riemann zeta function is defined for Re(s) > 1 as:

    zeta(s) = sum_{n=1}^{infinity} 1/n^s = prod_p (1 - p^{-s})^{-1}

where the product (Euler product) runs over all primes p. Riemann showed this
function extends analytically to the entire complex plane (except s = 1) and
satisfies the functional equation:

    zeta(s) = 2^s pi^{s-1} sin(pi*s/2) Gamma(1-s) zeta(1-s)

The **non-trivial zeros** of zeta(s) lie in the critical strip 0 < Re(s) < 1.
The Riemann Hypothesis (RH) asserts that every non-trivial zero has
Re(s) = 1/2, meaning each zero takes the form s = 1/2 + i*gamma_n for real
numbers gamma_n. The first few gamma_n values are:

    14.1347, 21.0220, 25.0109, 30.4249, 32.9351, 37.5862, ...

These zeros encode the fine structure of how primes are distributed among the
integers. Proving RH is one of the central open problems in mathematics.

### The Hilbert-Polya Conjecture

In the early 20th century, Hilbert and Polya independently conjectured that
the gamma_n might be eigenvalues of some self-adjoint operator on an
appropriate Hilbert space. If such an operator exists, then its eigenvalues
are real, which would immediately prove RH.

This conjecture was given dramatic support by **Montgomery's pair correlation
conjecture** (1973): the statistical distribution of spacings between nearby
Riemann zeros matches the eigenvalue spacing distribution of large random
matrices from the **Gaussian Unitary Ensemble (GUE)**. Freeman Dyson
recognized this as the same statistics that appear in quantum chaotic systems.

This connection suggests: somewhere in physics, there should be a quantum
system whose energy levels are exactly the Riemann zeros.

### Random Matrix Theory and GUE Statistics

The GUE (Gaussian Unitary Ensemble) describes NxN Hermitian matrices with
entries drawn from a Gaussian distribution, under the constraint that the
distribution is invariant under unitary conjugation. Key predictions:

**Wigner surmise for nearest-neighbor spacing:**

    P_GUE(s) = (32/pi^2) * s^2 * exp(-4s^2/pi)

where s is the spacing between adjacent eigenvalues normalized to mean 1.
This distribution exhibits **level repulsion**: P(0) = 0, meaning eigenvalues
repel each other (unlike random uniform points).

**Two-point correlation function:**

    R_2(r) = 1 - (sin(pi*r) / (pi*r))^2

This describes the probability of finding two eigenvalues at distance r.

Our scoring system tests candidate graph spectra against both of these GUE
predictions, because matching GUE statistics is a necessary condition for
matching the Riemann zeros.

---

## Why Quantum Graphs?

### What Is a Quantum Graph?

A **quantum graph** is a metric graph (vertices connected by edges of specified
lengths) equipped with a differential operator (the Laplacian, -d^2/dx^2) on
each edge, plus matching conditions at each vertex that make the operator
self-adjoint. The eigenvalue problem is:

    -psi''(x) = k^2 psi(x)

on each edge, where k is the wavenumber (eigenvalue parameter). On each edge
of length L_e, the general solution is:

    psi_e(x) = A_e exp(ikx) + B_e exp(-ikx)

The **vertex scattering matrix** S_v at each vertex prescribes how incoming
waves scatter into outgoing waves. Self-adjointness requires S_v to be unitary.

### Why They Are Promising

Quantum graphs are attractive candidates for a Hilbert-Polya operator because:

1. **Controllable spectra**: By choosing topology, edge lengths, and scattering
   matrices, we have fine-grained control over the eigenvalue distribution.

2. **Trace formula**: Quantum graphs have an exact trace formula (the
   Gutzwiller-Tabor trace formula) relating the spectrum to periodic orbits
   on the graph. This is structurally analogous to the Riemann explicit
   formula, which relates the zeros of zeta to prime numbers:

       sum_n f(gamma_n) = f_hat(0) + sum_p sum_m [ln(p)/p^{m/2}] f_hat(m*ln(p))

   In both formulas, the left side sums over "eigenvalues" and the right side
   sums over "periodic orbits" (prime powers for zeta, closed walks on the
   graph for quantum graphs).

3. **GUE statistics**: Quantum graphs with time-reversal symmetry breaking
   produce eigenvalue statistics in the GUE universality class — matching
   the Montgomery-Odlyzko statistics of the Riemann zeros.

4. **Prime encoding**: If edge lengths are set to logarithms of primes,
   the periodic orbits of the graph naturally involve sums of log-primes,
   which are exactly the terms that appear in the Riemann explicit formula.

### The Key Hypothesis

This system tests whether there exists a quantum graph G with:
- Edge lengths L_e = alpha * ln(p_e) for primes p_e and some scaling alpha
- Vertex scattering matrices with appropriate phases (GUE symmetry class)
- A topology that produces the correct combinatorial weights

such that the eigenvalues k_n of G coincide with the Riemann zeros gamma_n.

---

## The Secular Equation

### Construction

The eigenvalues of a quantum graph are determined by the **secular equation**
(Kottos & Smilansky, 1997):

    det[I - S * U(k)] = 0

where:

- **U(k)** is the 2E x 2E diagonal **bond propagation matrix**. Each directed
  edge (bond) of length L_b contributes a phase factor exp(i*k*L_b) on the
  diagonal. For an undirected graph with E edges, there are 2E directed bonds
  (one in each direction).

- **S** is the 2E x 2E **vertex scattering matrix**, assembled block-diagonally
  from the per-vertex scattering matrices. At each vertex v of degree d_v,
  the scattering matrix S_v is a d_v x d_v unitary matrix that maps incoming
  bond amplitudes to outgoing bond amplitudes.

### Neumann Scattering (Default)

The standard choice is the **Neumann** (or "free" or "democratic") vertex
condition:

    S_v[i,j] = 2/d_v - delta_{i,j}

This matrix is unitary (S S^dagger = I), symmetric (preserves time-reversal
symmetry), and distributes incoming amplitude equally among all outgoing bonds
minus the reflection term. For degree 2, this reduces to S = [[0,1],[1,0]],
meaning full transmission with no reflection — the wave passes straight through.

### Time-Reversal Symmetry Breaking

To obtain GUE statistics (matching Riemann zeros), we need to break
time-reversal symmetry. We accomplish this via a **magnetic flux** —
a unitary similarity transform:

    S_directed = D * S_neumann * D^dagger

where D = diag(1, e^{i*theta}, e^{2i*theta}, ...) introduces direction-dependent
phases. This preserves unitarity exactly while making the scattering matrix
non-symmetric, breaking TRS and shifting the statistics from GOE toward GUE.

### Numerical Eigenvalue Finding

We find eigenvalues by computing |det(I - S*U(k))| on a dense grid of k
values and locating local minima near zero. Each minimum is refined using
bounded scalar minimization (scipy). This approach is more robust than
tracking sign changes of Re(det), which can produce spurious roots when the
complex determinant passes near (but not through) zero.

---

## The Prime Connection

### Edge Lengths as Prime Logarithms

The central mathematical insight is that if edge lengths are set to:

    L_e = alpha * ln(p_e)

for primes p_e, then the periodic orbits of the graph have lengths that are
sums of terms alpha * ln(p_i), which equal alpha * ln(p_i * p_j * ...).
These are exactly the terms that appear in Riemann's explicit formula for
the prime counting function.

### The Trace Formula Bridge

For a quantum graph, the density of states can be written as:

    d(k) = L_total/pi + (1/pi) * sum_p sum_r A_p^r * L_p * cos(r*k*L_p)

where L_p is the length of the p-th prime orbit and A_p is its stability
amplitude. This is the graph-theoretic analogue of the Riemann explicit
formula:

    psi(x) = x - sum_rho x^rho / rho - ln(2*pi) - (1/2)*ln(1-x^{-2})

The oscillatory terms in both formulas have the same structure: a sum over
"primes" (prime orbits / prime numbers) of cosine terms with arguments
proportional to log-primes. When the edge lengths ARE log-primes, these
two formulas can potentially be made to coincide term-by-term.

### PSLQ Integer Relation Detection

When the search finds a high-scoring graph, the symbolic agent uses the
**PSLQ algorithm** (Ferguson & Bailey) to determine whether each edge length
can be expressed as a rational linear combination of log-primes:

    L_e = sum_p a_p * ln(p),    a_p in rationals

PSLQ finds integer relations among real numbers to arbitrary precision.
If every edge length has such a decomposition with small integer coefficients,
this constitutes strong evidence that the graph is encoding prime arithmetic.

---

## Scoring and Validation

### Multi-Component Scoring

Each candidate graph receives a score from 0 to 1 based on three components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Absolute match | 0.40 | After spectral unfolding, mean absolute deviation between first N eigenvalues and Riemann zeros. Score = exp(-MAD/0.1) |
| Spacing distribution | 0.35 | Kolmogorov-Smirnov test of nearest-neighbor spacings against the GUE Wigner surmise. Score = 1 - KS_statistic |
| Pair correlation | 0.25 | Integrated squared deviation of the two-point correlation R_2(r) from the GUE prediction. Score = exp(-deviation) |

### Spectral Unfolding

Before comparison, both the graph spectrum and Riemann zeros undergo
**spectral unfolding**: a smooth polynomial is fitted to the staircase function
N(k) (number of eigenvalues below k), and eigenvalues are mapped through this
polynomial to achieve unit mean spacing. This removes the smooth (Weyl law)
component and isolates the oscillatory structure that encodes the primes.

### Validation Ladder

| Level | Score Range | Interpretation |
|-------|------------|----------------|
| 0 | < 0.50 | Random, no match |
| 1 | 0.50 - 0.65 | Better than random; GUE-like spacing |
| 2 | 0.65 - 0.75 | First ~10 zeros approximately matched |
| 3 | 0.75 - 0.85 | First ~20 zeros matched; prime-log edge lengths confirmed |
| 4 | 0.85 - 0.92 | First ~50 zeros matched; strong structural regularity |
| 5 | > 0.92 | First ~100 zeros matched; closed-form edge length rule |

Level 5 triggers a full scientific report and human review alert.

---

## System Architecture

```
riemann_qg/
├── agents/
│   ├── generative_agent.py    # Proposes candidate graph topologies
│   ├── spectral_agent.py      # Computes spectra & scores (parallel)
│   ├── pattern_agent.py       # Extracts patterns from top performers
│   ├── symbolic_agent.py      # PSLQ & symbolic regression on lengths
│   └── orchestrator.py        # Meta-agent coordinating all layers
├── core/
│   ├── quantum_graph.py       # QuantumGraph class & secular equation
│   ├── riemann_zeros.py       # 100+ high-precision Riemann zeros
│   ├── scoring.py             # Multi-component spectral scoring
│   └── trace_formula.py       # Periodic orbit sums & prime overlap
├── search/
│   ├── topology_search.py     # Graph enumeration & mutation
│   ├── optimizer.py           # Edge length / scattering optimization
│   └── evolutionary.py        # DEAP-based evolutionary algorithm
├── analysis/
│   ├── pattern_extractor.py   # Population statistics & enrichment
│   ├── symbolic_regression.py # PSLQ & grammar-based regression
│   └── reporter.py            # Rich console & Markdown reports
├── tests/                     # pytest suite (31 tests)
├── results/                   # Auto-created run outputs
├── main.py                    # CLI entry point
└── config.py                  # SearchConfig dataclass
```

### Agent Roles

**GenerativeAgent** — Proposes new candidate graphs using four strategies:
- *PrimeLengthStrategy*: Random topologies with edge lengths = alpha * ln(p)
- *MutationStrategy*: Perturb lengths, swap edges, adjust scattering phases
- *TemplateStrategy*: Complete graphs, cycles, expanders with prime lengths
- *CrossoverStrategy*: Combine two parent graphs' topologies and lengths

**SpectralAgent** — The computational engine. Computes the spectrum of each
graph via the secular equation and scores it against the Riemann zeros.
Supports parallel batch evaluation via ProcessPoolExecutor.

**PatternAgent** — Analyzes the top-performing fraction of each generation to
identify enriched features: which topologies, degree sequences, clustering
coefficients, and edge-length patterns appear disproportionately in winners.

**SymbolicAgent** — Runs PSLQ integer-relation detection and symbolic
regression on the edge lengths of top graphs, searching for closed-form
rules like L_e = ln(p_e) or L_e = ln(p_e * p_{e+1}).

**Orchestrator** — Coordinates the generational loop: evaluate → analyze →
breed → checkpoint. Manages elitism (top 5% survive unchanged), validation
level assessment, deep analysis triggers, and checkpointing.

### Evolutionary Search Loop

```
1. Initialize population: mix of prime-length templates + random graphs
2. For each generation:
   a. SpectralAgent evaluates all graphs in parallel
   b. Sort by score; elites preserved
   c. PatternAgent analyzes top 10%
   d. SymbolicAgent runs on top 5
   e. GenerativeAgent proposes next generation
   f. Log generation summary
   g. If score > target: trigger deep analysis
   h. Save periodic checkpoints
```

---

## Installation

```bash
# Clone and enter directory
cd Reimann_quantum_graph

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest riemann_qg/tests/ -v
```

### Requirements

- Python 3.10+
- numpy, scipy, sympy, networkx (core math)
- mpmath (high-precision arithmetic for zeta zeros and PSLQ)
- numba (JIT acceleration)
- deap (evolutionary algorithms)
- matplotlib, pandas (analysis)
- rich (console formatting)
- tqdm (progress bars)
- pytest (testing)

---

## Usage

### Fresh Search

```bash
# Default configuration (200 population, 100 generations)
python main.py run

# Quick test run
python main.py run --population-size 20 --n-generations 10 --n-workers 1

# Custom configuration
python main.py run --population-size 500 --n-generations 200 --k-max 120
```

### Resume from Checkpoint

```bash
python main.py resume --checkpoint results/checkpoint_gen_42.pkl
```

### Analyze a Saved Graph

```bash
python main.py analyze --graph results/best_graph.pkl
```

### Run Tests

```bash
python main.py test
# or directly:
python -m pytest riemann_qg/tests/ -v
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| population_size | 200 | Number of graphs per generation |
| n_generations | 100 | Number of evolutionary generations |
| k_max | 80.0 | Maximum wavenumber for spectrum computation |
| n_zeros_compare | 20 | Number of Riemann zeros to compare against |
| top_fraction | 0.10 | Fraction of population for pattern analysis |
| elite_fraction | 0.05 | Fraction preserved unchanged each generation |
| n_workers | -1 | Parallel workers (-1 = cpu_count - 1) |
| prime_length_bias | 0.7 | Probability new graphs use log-prime lengths |
| phase_breaking | True | Enable TRS breaking for GUE statistics |
| target_score | 0.85 | Score threshold for deep analysis |
| checkpoint_every | 10 | Generations between checkpoints |

---

## Interpreting Results

### What a High Score Means

A score above 0.85 (Level 4) means:
- The graph's eigenvalues, after spectral unfolding, closely match the first
  ~20-50 Riemann zeros
- The spacing distribution follows GUE statistics
- The pair correlation matches the GUE two-point function

This is mathematically significant because it suggests the graph's periodic
orbit structure is encoding prime-number information.

### What to Look For in Reports

The scientific report (generated at high scores) includes:

1. **Graph specification**: Adjacency matrix, edge lengths, and whether
   lengths decompose cleanly into log-prime combinations

2. **Spectral comparison table**: Side-by-side eigenvalues vs. Riemann zeros
   with per-value deviation

3. **GUE statistics**: KS test results and spacing distribution plots

4. **Trace formula analysis**: How well the graph's periodic orbit sum
   matches the Riemann explicit formula term-by-term

5. **Symbolic discoveries**: Any closed-form rules found for edge lengths

### What Would Constitute a "Proof"

Finding a graph with score ~1.0 would not itself prove RH, but it would:

1. Provide an explicit construction of a self-adjoint operator whose
   eigenvalues appear to coincide with the Riemann zeros

2. Suggest a specific Hilbert-Polya operator, which could then be analyzed
   rigorously using spectral theory

3. If the edge-length rule is a closed-form expression in terms of primes,
   this could potentially be verified to all orders using the trace formula,
   yielding a proof strategy

The extension point `FormalVerificationAgent` is stubbed out for future
connection to a Lean 4 proof assistant, which could formalize such arguments.

---

## Mathematical References

- **Kottos & Smilansky** (1997). "Quantum chaos on graphs." *Phys. Rev. Lett.* 79, 4794.
  — Introduced the secular equation for quantum graphs.

- **Montgomery** (1973). "The pair correlation of zeros of the zeta function."
  *Proc. Symp. Pure Math.* 24, 181-193.
  — Discovered the GUE connection.

- **Odlyzko** (1987). "On the distribution of spacings between zeros of the zeta function."
  *Math. Comp.* 48, 273-308.
  — Numerically confirmed Montgomery's conjecture to high precision.

- **Berry & Keating** (1999). "The Riemann zeros and eigenvalue asymptotics."
  *SIAM Review* 41, 236-266.
  — Proposed specific Hamiltonians for the Hilbert-Polya program.

- **Berkolaiko & Kuchment** (2013). *Introduction to Quantum Graphs.*
  AMS Mathematical Surveys and Monographs, Vol. 186.
  — Comprehensive reference on quantum graph spectral theory.

- **Ferguson & Bailey** (1999). "A polynomial time, numerically stable
  integer relation algorithm." RNR Technical Report.
  — The PSLQ algorithm used for symbolic analysis.
