# Claude Code Prompt: Multi-Agent Quantum Graph Riemann Zero Solver

## Mission

Build a multi-agent system in Python that searches for quantum graphs whose eigenvalue
spectra match the non-trivial zeros of the Riemann zeta function. The system should
treat this as a principled scientific search, accumulating structural knowledge at each
step, not merely a brute-force optimizer.

---

## Project Structure

Create the following directory layout:

```
riemann_qg/
├── agents/
│   ├── __init__.py
│   ├── generative_agent.py       # Proposes candidate graph topologies
│   ├── spectral_agent.py         # Computes quantum graph spectra
│   ├── pattern_agent.py          # Extracts structural patterns from winners
│   ├── symbolic_agent.py         # Symbolic regression on edge length patterns
│   └── orchestrator.py           # Meta-agent coordinating all layers
├── core/
│   ├── __init__.py
│   ├── quantum_graph.py          # Quantum graph data structures and secular equation
│   ├── riemann_zeros.py          # High-precision known Riemann zeros (first 100+)
│   ├── trace_formula.py          # Gutzwiller/Selberg trace formula utilities
│   └── scoring.py                # Spectral match scoring functions
├── search/
│   ├── __init__.py
│   ├── topology_search.py        # Graph connectivity enumeration and mutation
│   ├── optimizer.py              # Edge length and scattering matrix optimization
│   └── evolutionary.py           # Evolutionary algorithm over graph population
├── analysis/
│   ├── __init__.py
│   ├── pattern_extractor.py      # Identifies shared features of high-scoring graphs
│   ├── symbolic_regression.py    # Discovers closed-form edge length rules
│   └── reporter.py               # Generates structured scientific reports
├── tests/
│   ├── test_quantum_graph.py
│   ├── test_spectral_computation.py
│   ├── test_scoring.py
│   └── test_full_pipeline.py
├── results/                       # Auto-created; stores run outputs
├── main.py                        # Entry point
├── config.py                      # All tunable parameters
└── requirements.txt
```

---

## Requirements

```
# requirements.txt
numpy>=1.26
scipy>=1.11
sympy>=1.12
networkx>=3.2
mpmath>=1.3          # High-precision arithmetic for zeta zeros
matplotlib>=3.8
pandas>=2.1
scipy>=1.11
scikit-learn>=1.3
deap>=1.4            # Evolutionary algorithms
numba>=0.58          # JIT acceleration for spectral computation
tqdm>=4.66
pytest>=7.4
rich>=13.0           # Console output formatting
```

---

## Core Mathematical Implementation

### `core/riemann_zeros.py`

Implement the following:

1. **Hard-coded high-precision zeros**: Store the imaginary parts of the first 100
   non-trivial Riemann zeros to 15 decimal places. The first ten are:
   14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
   37.586178, 40.918719, 43.327073, 48.005151, 49.773832

2. **`RiemannZeros` class** with methods:
   - `get_zeros(n)` — return first n zeros as numpy array
   - `get_normalized_zeros(n)` — return zeros normalized to mean spacing 1
     (divide by average spacing = 2π/ln(t/2π) locally)
   - `nearest_zero(k)` — find the nearest known zero to value k
   - `spacing_statistics(n)` — compute GUE spacing distribution statistics
     for first n zeros (for validation against candidate graph spectra)

3. **`load_extended_zeros(filepath)`** — load additional zeros from file if
   provided (to support scaling up precision later)

---

### `core/quantum_graph.py`

Implement a `QuantumGraph` class representing a metric graph with:

**Attributes:**
- `adjacency`: NxN integer matrix (number of edges between vertices i and j)
- `edge_lengths`: list of positive reals, one per edge
- `vertex_scattering`: list of NxN unitary matrices, one per vertex
  (default: Neumann/free boundary conditions — democratic scattering)
- `n_vertices`, `n_edges`

**Key method — `secular_equation(k)`:**

Compute det[I - S·U(k)] = 0 where:
- k is the wavenumber (real positive number)
- U(k) is the 2E × 2E diagonal matrix with entries exp(i·k·L_e) for each
  directed edge e of length L_e
- S is the 2E × 2E block-diagonal vertex scattering matrix assembled from
  the per-vertex matrices using the standard quantum graph construction
  (see Kottos & Smilansky 1997 convention)

The secular equation should return a real scalar (the real part of the
determinant on the unit circle) whose zeros are the graph eigenvalues.

**Method — `compute_spectrum(k_max, n_points)`:**

Find all k in [0, k_max] where secular_equation(k) = 0 using:
1. Dense sampling to find sign changes
2. Brentq root finding between sign changes
3. Return sorted numpy array of eigenvalues

Use numba JIT where possible for the dense sampling loop.

**Method — `neumann_scattering(degree)`:**

Construct the standard Neumann scattering matrix for a vertex of given degree:
S_ij = 2/degree - delta_ij

This is the democratic, time-reversal-breaking-neutral choice. Also implement:

**Method — `directed_scattering(degree, phase)`:**

Add a complex phase to break time-reversal symmetry. This shifts the
statistics from GOE toward GUE, which is required to match Riemann zero
spacing statistics.

**Factory methods:**
- `complete_graph(n, base_length, prime_lengths)` — complete graph on n vertices
  with edge lengths set to logarithms of the first n(n-1)/2 primes scaled by base_length
- `cycle_graph(n, prime_lengths)` — cycle with prime-log edge lengths
- `random_graph(n_vertices, edge_prob, length_scale)` — random connected graph

---

### `core/scoring.py`

Implement `SpectralScorer` class:

**`score_spectrum(candidate_eigenvalues, riemann_zeros, n_compare)`:**

Compute a multi-component score measuring how well candidate_eigenvalues
match riemann_zeros. Components:

1. **Absolute match score** (weight 0.4):
   After normalizing both sequences to unit mean spacing, compute the
   mean absolute deviation between the first n_compare values.
   Score = exp(-mean_deviation / tolerance) where tolerance = 0.1

2. **Spacing distribution score** (weight 0.35):
   Compute the nearest-neighbor spacing distribution P(s) for the candidate
   spectrum and compare to the GUE Wigner surmise:
   P_GUE(s) = (32/π²) s² exp(-4s²/π)
   Use KS-test statistic; score = 1 - KS_statistic

3. **Pair correlation score** (weight 0.25):
   Compute the two-point correlation function R_2(r) for the candidate
   spectrum and compare to the GUE prediction. Use integrated squared
   deviation as the metric.

Return a `ScoringResult` dataclass with individual components, total score,
and a human-readable summary string.

**`normalize_spectrum(eigenvalues)`:**
Apply the local unfolding procedure to map eigenvalues to unit mean density.

---

### `core/trace_formula.py`

Implement utilities for the trace formula connection:

**`periodic_orbit_sum(graph, k, n_max_length)`:**
Compute the contribution to the density of states from periodic orbits
up to length n_max_length using the graph's topology and edge lengths.
This is the quantum graph analogue of Riemann's explicit formula.

**`prime_orbit_overlap(graph, primes, n_primes)`:**
Measure how well the graph's periodic orbit lengths (and their repetitions)
overlap with {ln p : p prime, p ≤ n_primes}. This is the key structural
diagnostic — high overlap means the graph is "encoding" primes correctly.

**`explicit_formula_deviation(graph, k_values)`:**
Compare the quantum graph trace formula to Riemann's explicit formula
term-by-term. Return a deviation score as a function of k.

---

## Agent Implementations

### `agents/generative_agent.py`

`GenerativeAgent` class responsible for proposing new candidate graphs.

**Strategies (implement all, cycle or choose by performance):**

1. **PrimeLengthStrategy**: Always set edge lengths to {α·ln(p) : p ∈ primes}
   for various α values. Search over topology (n_vertices, connectivity) and α.

2. **MutationStrategy**: Take a high-scoring graph and apply one of:
   - Add/remove edge
   - Perturb one edge length by ±ε
   - Swap two edge lengths
   - Change one vertex scattering matrix

3. **TemplateStrategy**: Start from known high-performing topologies
   (complete graphs, Cayley graphs of finite groups, expander graphs)
   and systematically vary parameters.

4. **CrossoverStrategy**: Take two high-scoring graphs and combine their
   topologies via a graph crossover operation.

**`propose_batch(population, scores, n_new, strategy)`:**
Return a list of n_new QuantumGraph candidates based on current population
and their scores. Higher-scoring graphs should be sampled more frequently
as parents.

---

### `agents/spectral_agent.py`

`SpectralAgent` class — the compute workhorse.

**`evaluate_graph(graph, scorer, k_max, n_zeros_compare)`:**
1. Compute spectrum of graph up to k_max
2. Score against Riemann zeros using scorer
3. Return (graph, scoring_result, spectrum)

**`evaluate_batch(graphs, scorer, k_max, n_zeros_compare, n_workers)`:**
Parallel evaluation using ProcessPoolExecutor. Return sorted list of
(graph, scoring_result, spectrum) tuples, best first.

Include progress bar via tqdm.

**`adaptive_k_max(n_zeros_target)`:**
Estimate the k_max needed to capture n_zeros_target eigenvalues using
Weyl's law: N(k) ≈ (total_length · k) / π

---

### `agents/pattern_agent.py`

`PatternAgent` — extracts what high-scoring graphs have in common.

**`analyze_population(graphs, scores, top_fraction)`:**
Take the top_fraction of graphs by score and compute:

1. **Topology statistics**: Distribution of n_vertices, n_edges, degree sequences,
   clustering coefficients, diameter. Report which topology features are
   enriched in top performers vs. random.

2. **Edge length statistics**: For top graphs, extract all edge lengths.
   Test whether they cluster near {ln p : p prime}. Compute the correlation
   between edge lengths and log-primes. Report best-fit regression.

3. **Scattering matrix statistics**: What phase structure in the scattering
   matrices do top performers share?

4. **Orbit length statistics**: Using trace_formula utilities, compute
   whether top graphs have more periodic orbits aligned with prime logarithms.

Return a `PopulationInsights` dataclass with all statistics and a
plain-language summary.

---

### `agents/symbolic_agent.py`

`SymbolicAgent` — attempts to discover closed-form rules for the winning
edge length patterns.

**`fit_prime_combination(edge_lengths, n_primes, max_coeff)`:**
Given a set of edge lengths from a high-scoring graph, attempt to express
each length as:
L = Σ_p a_p · ln(p), p ∈ {2,3,5,7,11,...,prime_n}, a_p ∈ rationals

Use integer relation algorithms (PSLQ via mpmath) to find rational
coefficient vectors. Report any exact or near-exact matches.

**`discover_length_rule(top_graphs, n_primes)`:**
Across all top graphs, find a single rule that predicts their edge lengths.
Try:
- L_e = ln(p_e) for the e-th prime
- L_e = ln(p_e · p_f) for pairs
- L_e = (1/2)·ln(p_e) (half-logarithms)
- L_e determined by a simple function of edge index

Report the best-fitting rule with its residual error.

**`symbolic_regression_lengths(top_graphs)`:**
Use a simple symbolic regression (grammar-based or polynomial) to find
a formula mapping edge index → edge length that fits the top graphs.

---

### `agents/orchestrator.py`

`Orchestrator` — the meta-agent coordinating the full search.

**Configuration parameters (from config.py):**
- population_size: 200
- n_generations: 100
- k_max: 80.0 (captures ~25 eigenvalues for typical graph sizes)
- n_zeros_compare: 20
- top_fraction: 0.1
- n_workers: (cpu_count - 1)
- elite_fraction: 0.05 (top graphs always survive to next generation)

**`run(config)`** — main search loop:

```
Initialize population (mix of prime-length templates and random graphs)
For each generation:
    1. SpectralAgent evaluates all graphs in parallel
    2. Sort by score; log top-5 to results/
    3. PatternAgent analyzes top 10%
    4. SymbolicAgent runs on top 5 graphs
    5. GenerativeAgent proposes next generation:
       - Keep elite_fraction unchanged
       - Fill rest via mutation/crossover/new templates
    6. Log generation summary to console (rich formatting)
    7. If top score > 0.90: trigger deep analysis and report
    8. Save checkpoint to results/checkpoint_gen_{n}.pkl
```

**`deep_analysis(graph, spectrum, insights)`:**
When a high-scoring graph is found, perform extended analysis:
- Extend spectrum to k_max × 3 and re-score against more zeros
- Run trace formula comparison
- Run full symbolic analysis
- Generate a detailed scientific report (see reporter.py)

**`resume(checkpoint_path)`:**
Load a checkpoint and continue the search.

---

## Analysis and Reporting

### `analysis/reporter.py`

`Reporter` class generating structured output.

**`generation_report(generation, top_graphs, insights)`:**
Rich console output showing:
- Generation number, best score, mean score
- Top graph topology summary
- Key pattern insights
- Symbolic discoveries (if any)
- Progress bar toward target score

**`scientific_report(graph, spectrum, scoring_result, insights)`:**
Generate a full Markdown report saved to results/ containing:
- Graph specification (adjacency, edge lengths in exact and ln(p) form)
- Spectral comparison table: candidate eigenvalues vs. Riemann zeros, deviation
- GUE statistics comparison with plots (save as PNG)
- Trace formula analysis
- Periodic orbit table
- Prime-length overlap analysis
- Symbolic regression findings
- Assessment: does this graph constitute meaningful mathematical evidence?
  What would be needed to formalize this as a proof scaffold?

---

## Tests

### `tests/test_quantum_graph.py`

1. **Test secular equation construction**: For a simple path graph with 2 edges,
   verify the secular equation matches the analytic formula.

2. **Test spectrum computation**: For a ring graph (cycle) with equal edge lengths L,
   verify eigenvalues are k = 2πn/L for integer n (analytic result).

3. **Test Neumann scattering matrix**: Verify unitarity and correct form for
   degree-2 and degree-3 vertices.

4. **Test GUE symmetry breaking**: Verify that adding phase to scattering matrices
   shifts spacing statistics toward GUE (KS test against GOE vs GUE prediction).

### `tests/test_spectral_computation.py`

1. **Test normalization**: Verify that normalized zeros have unit mean spacing.

2. **Test scoring**: Verify that a spectrum equal to Riemann zeros scores 1.0,
   and a random spectrum scores near 0.

3. **Test GUE spacing score**: Verify that a synthetically GUE-distributed
   spectrum scores higher than a GOE-distributed one.

### `tests/test_scoring.py`

1. **Test prime_orbit_overlap**: For a graph with edge lengths exactly {ln 2, ln 3, ln 5},
   verify high overlap score.

2. **Test symbolic fit**: For edge lengths [ln 2, ln 3, ln 5, ln 7], verify that
   PSLQ correctly identifies them as logarithms of primes.

### `tests/test_full_pipeline.py`

1. **Smoke test**: Run orchestrator for 3 generations with population_size=10.
   Verify it completes without error and produces output files.

2. **Score monotonicity test**: Verify that the elite population's best score
   is non-decreasing across generations.

3. **Checkpoint test**: Run 2 generations, save checkpoint, resume, run 2 more.
   Verify final state is consistent.

---

## `main.py`

```python
"""
Riemann Quantum Graph Multi-Agent Search System
Usage:
    python main.py run                    # Fresh run with default config
    python main.py run --config my.yaml   # Custom config
    python main.py resume --checkpoint results/checkpoint_gen_42.pkl
    python main.py analyze --graph results/best_graph.pkl
    python main.py test                   # Run test suite
"""
```

Implement CLI using argparse. On `run`, instantiate Orchestrator with config,
call run(), and on KeyboardInterrupt save a final checkpoint and print the
best graph found so far.

---

## `config.py`

```python
from dataclasses import dataclass

@dataclass
class SearchConfig:
    population_size: int = 200
    n_generations: int = 100
    k_max: float = 80.0
    n_zeros_compare: int = 20
    top_fraction: float = 0.10
    elite_fraction: float = 0.05
    n_workers: int = -1          # -1 = cpu_count - 1
    min_vertices: int = 3
    max_vertices: int = 12
    min_edges: int = 3
    max_edges: int = 30
    prime_length_bias: float = 0.7   # Probability new graphs use prime-log lengths
    phase_breaking: bool = True       # Enable time-reversal symmetry breaking
    target_score: float = 0.85       # Score that triggers deep analysis
    checkpoint_every: int = 10       # Save checkpoint every N generations
    results_dir: str = "results"
    verbose: bool = True
    random_seed: int = 42
```

---

## Implementation Notes and Mathematical Constraints

1. **Self-adjointness constraint**: The quantum graph Hamiltonian is self-adjoint
   iff the vertex scattering matrices are unitary. Always verify unitarity after
   construction and mutation.

2. **Connectedness**: Always verify the graph is connected before evaluation.
   Disconnected graphs have independent spectra — uninteresting.

3. **Time-reversal breaking**: To get GUE rather than GOE statistics, at least some
   vertex scattering matrices must have complex (not purely real) entries. Implement
   this via a magnetic flux: multiply scattering matrices by exp(iθ) for small θ.
   Make θ a tunable parameter.

4. **Weyl's law normalization**: Before comparing to Riemann zeros, always apply the
   smooth Weyl term subtraction to the graph spectrum:
   N_smooth(k) = (L_total · k) / π - (n_vertices - n_edges)
   This removes the trivial smooth background and isolates the oscillatory part.

5. **Numerical precision**: Use mpmath with 30 decimal places for computing the
   known Riemann zeros. Use standard float64 for spectral computation (acceptable
   for k < 100). For the secular equation determinant, monitor condition number
   and warn if it exceeds 1e12.

6. **Scale invariance**: The quantum graph spectrum scales as k → αk when all edge
   lengths scale as L → L/α. Always normalize to a canonical scale (sum of edge
   lengths = 2π) before comparison, then rescale Riemann zeros accordingly.

---

## Scientific Validation Ladder

Implement a `ValidationLevel` enum and have the orchestrator assess each
high-scoring graph against these levels:

- **Level 0** (score < 0.5): Random, no match
- **Level 1** (score 0.5–0.65): Better than random; GUE-like spacing
- **Level 2** (score 0.65–0.75): First ~10 zeros approximately matched
- **Level 3** (score 0.75–0.85): First ~20 zeros matched; prime-log edge lengths confirmed
- **Level 4** (score 0.85–0.92): First ~50 zeros matched; strong structural regularity
- **Level 5** (score > 0.92): First ~100 zeros matched; closed-form edge length rule found
  → Trigger full scientific report and human review alert

At Level 5, print a prominent alert:

```
╔══════════════════════════════════════════════════════════════╗
║  HIGH-VALUE CANDIDATE DETECTED — HUMAN REVIEW RECOMMENDED   ║
║  Score: {score:.4f} | Zeros matched: {n} | Gen: {gen}       ║
║  Report saved to: results/candidate_{timestamp}.md           ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Extension Points (implement interfaces, not yet functionality)

1. **FormalVerificationAgent**: Interface for sending high-scoring graph candidates
   to a Lean 4 proof assistant via subprocess. Stub out the connection; document
   what the Lean theorem statement would look like.

2. **NeuralTopologyProposer**: Interface for a GNN-based topology proposer trained
   on accumulated (graph, score) data. Stub with random proposer; document the
   training setup that would replace it after ~1000 evaluations.

3. **DistributedOrchestrator**: Interface for distributing search across multiple
   machines using ray or dask. Current single-machine Orchestrator should be
   refactored to implement this interface.

---

## Deliverables Checklist

After implementation, verify:

- [ ] `pytest tests/` passes with zero failures
- [ ] `python main.py run` runs for 5 generations without error
- [ ] A results/ directory is created with at least one checkpoint and one report
- [ ] The best graph found in 5 generations is printed with its score
- [ ] `python main.py resume` successfully continues from checkpoint
- [ ] All agent classes have docstrings explaining their mathematical role
- [ ] `requirements.txt` is complete and `pip install -r requirements.txt` succeeds
- [ ] Code includes a README.md explaining the mathematical motivation,
      system architecture, and how to interpret results scientifically
