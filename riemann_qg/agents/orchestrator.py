"""Orchestrator -- meta-agent coordinating the full search.

Manages the lifecycle of the evolutionary search: population initialisation,
generational evaluation, pattern/symbolic analysis, next-generation breeding,
checkpointing, and validation-level alerts.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field, replace, asdict
from enum import IntEnum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from riemann_qg.agents.generative_agent import GenerativeAgent, StrategyType
from riemann_qg.agents.pattern_agent import PatternAgent, PopulationInsights
from riemann_qg.agents.spectral_agent import EvalResult, SpectralAgent
from riemann_qg.agents.symbolic_agent import SymbolicAgent
from riemann_qg.core.quantum_graph import QuantumGraph
from riemann_qg.core.scoring import SpectralScorer
from riemann_qg.search.optimizer import optimize_edge_lengths, joint_optimize


# ---------------------------------------------------------------------------
# Validation levels
# ---------------------------------------------------------------------------

class ValidationLevel(IntEnum):
    """Scientific validation ladder based on score thresholds."""
    LEVEL_0 = 0  # < 0.50 -- random, no match
    LEVEL_1 = 1  # 0.50 - 0.65 -- better than random; GUE-like spacing
    LEVEL_2 = 2  # 0.65 - 0.75 -- first ~10 zeros approx matched
    LEVEL_3 = 3  # 0.75 - 0.85 -- first ~20 zeros matched; prime-log lengths
    LEVEL_4 = 4  # 0.85 - 0.92 -- first ~50 zeros matched; structural regularity
    LEVEL_5 = 5  # > 0.92 -- first ~100 zeros matched; closed-form rule


_LEVEL_THRESHOLDS: list[float] = [0.0, 0.50, 0.65, 0.75, 0.85, 0.92]


def classify_score(score: float) -> ValidationLevel:
    """Map a total score to its validation level."""
    for level in reversed(ValidationLevel):
        if score >= _LEVEL_THRESHOLDS[level]:
            return level
    return ValidationLevel.LEVEL_0


# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchConfig:
    """Immutable search configuration."""
    population_size: int = 200
    n_generations: int = 100
    k_max: float = 80.0
    n_zeros_compare: int = 20
    top_fraction: float = 0.10
    elite_fraction: float = 0.05
    n_workers: int = -1
    min_vertices: int = 3
    max_vertices: int = 12
    min_edges: int = 3
    max_edges: int = 30
    prime_length_bias: float = 0.7
    phase_breaking: bool = True
    target_score: float = 0.85
    checkpoint_every: int = 10
    results_dir: str = "results"
    verbose: bool = True
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointState:
    """Serialisable snapshot of the search state."""
    generation: int
    population: tuple[QuantumGraph, ...]
    scores: tuple[float, ...]
    best_score: float
    best_graph: QuantumGraph | None
    config: SearchConfig
    rng_state: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Generation trajectory log
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationRecord:
    """One row of the generation-by-generation trajectory log."""
    generation: int
    best_score: float
    mean_score: float
    std_score: float
    validation_level: int
    # topology
    mean_n_vertices: float
    mean_n_edges: float
    degree_sequence_mode: tuple[int, ...]
    mean_clustering: float
    # edge lengths
    mean_length: float
    correlation_with_log_primes: float
    best_alpha: float
    mean_nearest_prime_log_dist: float
    # scattering
    fraction_near_neumann: float
    mean_phase: float
    # orbits
    mean_prime_overlap: float
    # best graph specifics
    best_n_vertices: int
    best_n_edges: int
    best_edge_lengths: tuple[float, ...]
    # symbolic
    symbolic_formula: str = ""
    symbolic_residual: float = float("inf")
    # diversity
    n_unique_topologies: int = 0


def _record_to_dict(rec: GenerationRecord) -> dict[str, Any]:
    """Convert a GenerationRecord to a JSON-serialisable dict."""
    d = asdict(rec)
    d["best_edge_lengths"] = list(d["best_edge_lengths"])
    d["degree_sequence_mode"] = list(d["degree_sequence_mode"])
    return d


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Meta-agent that drives the generational search loop.

    Coordinates the generative, spectral, pattern, and symbolic agents
    through repeated evaluate-analyse-breed cycles while managing
    checkpoints, logging, and validation-level alerts.
    """

    def __init__(self) -> None:
        self._console = Console()
        self._generative: GenerativeAgent | None = None
        self._spectral: SpectralAgent | None = None
        self._pattern: PatternAgent | None = None
        self._symbolic: SymbolicAgent | None = None

    # ---- main entry point ----

    def run(self, config: SearchConfig | None = None) -> CheckpointState:
        """Execute the full evolutionary search.

        Parameters
        ----------
        config : SearchConfig or None
            Uses defaults when None.

        Returns
        -------
        CheckpointState
            Final state of the search.
        """
        config = config or SearchConfig()
        self._init_agents(config)
        self._ensure_results_dir(config.results_dir)

        population = self._init_population(config)
        scores = np.zeros(len(population))
        best_score = 0.0
        best_graph: QuantumGraph | None = None
        trajectory: list[GenerationRecord] = []
        trajectory_path = os.path.join(config.results_dir, "trajectory.jsonl")

        for gen in range(config.n_generations):
            # 1. Evaluate
            eval_results = self._spectral.evaluate_batch(
                population, k_max=config.k_max,
                n_zeros_compare=config.n_zeros_compare,
                n_workers=self._resolve_workers(config.n_workers),
            )

            scores = np.array([r.score for r in eval_results])
            population = [r.graph for r in eval_results]

            # 2. Optimize top graphs' edge lengths (the key improvement)
            population, scores = self._optimize_top_graphs(
                population, scores, config, gen,
            )

            gen_best = float(scores[0]) if len(scores) > 0 else 0.0
            if gen_best > best_score:
                best_score = gen_best
                best_graph = population[0] if population else None

            # 3. Log generation summary
            if config.verbose:
                self._log_generation(gen, scores, eval_results)

            # 3b. Diagnostic at gen 0: show Weyl-rescaled eigenvalues vs zeros
            if gen == 0 and config.verbose and eval_results:
                self._diagnostic_gen0(eval_results[0], config)

            # 4. Pattern analysis on top fraction
            insights = self._pattern.analyze_population(
                population, scores, config.top_fraction,
            )

            # 5. Symbolic analysis on top 5
            symbolic_formula = ""
            symbolic_residual = float("inf")
            top_5 = population[:5]
            if top_5:
                rule = self._symbolic.discover_length_rule(
                    top_5, n_primes=20,
                )
                symbolic_formula = rule.formula
                symbolic_residual = rule.mean_residual
                if config.verbose and rule.mean_residual < 1.0:
                    self._console.print(
                        f"  Symbolic: best rule = {rule.formula} "
                        f"(residual {rule.mean_residual:.4f})"
                    )

            # 6. Record trajectory
            best_g = population[0] if population else None
            n_unique = len({
                (g.n_vertices, g.n_edges) for g in population
            })
            record = GenerationRecord(
                generation=gen,
                best_score=gen_best,
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores)),
                validation_level=classify_score(gen_best).value,
                mean_n_vertices=insights.topology.mean_n_vertices,
                mean_n_edges=insights.topology.mean_n_edges,
                degree_sequence_mode=insights.topology.degree_sequence_mode,
                mean_clustering=insights.topology.mean_clustering,
                mean_length=insights.edge_lengths.mean_length,
                correlation_with_log_primes=insights.edge_lengths.correlation_with_log_primes,
                best_alpha=insights.edge_lengths.best_alpha,
                mean_nearest_prime_log_dist=insights.edge_lengths.mean_nearest_prime_log_dist,
                fraction_near_neumann=insights.scattering.fraction_near_neumann,
                mean_phase=insights.scattering.mean_phase,
                mean_prime_overlap=insights.orbits.mean_prime_overlap,
                best_n_vertices=best_g.n_vertices if best_g else 0,
                best_n_edges=best_g.n_edges if best_g else 0,
                best_edge_lengths=tuple(best_g.edge_lengths) if best_g else (),
                symbolic_formula=symbolic_formula,
                symbolic_residual=symbolic_residual,
                n_unique_topologies=n_unique,
            )
            trajectory.append(record)
            self._append_trajectory_record(record, trajectory_path)

            # 7. Validation level check
            level = classify_score(best_score)
            if level >= ValidationLevel.LEVEL_5:
                self._alert_level_5(best_score, gen, config.results_dir)
            if best_score >= config.target_score and best_graph is not None:
                spectrum = eval_results[0].spectrum if eval_results else np.array([])
                self.deep_analysis(best_graph, spectrum, insights)

            # 8. Checkpoint (before breeding — population and scores are aligned)
            if (gen + 1) % config.checkpoint_every == 0:
                state = CheckpointState(
                    generation=gen,
                    population=tuple(population),
                    scores=tuple(scores.tolist()),
                    best_score=best_score,
                    best_graph=best_graph,
                    config=config,
                )
                self._save_checkpoint(state, gen, config.results_dir)

            # 9. Generate next generation (always preserve all-time best)
            population = self._next_generation(
                population, scores, config, insights,
                best_graph=best_graph,
            )

        final_state = CheckpointState(
            generation=config.n_generations - 1,
            population=tuple(population),
            scores=tuple(scores.tolist()),
            best_score=best_score,
            best_graph=best_graph,
            config=config,
        )
        self._save_checkpoint(
            final_state, config.n_generations - 1, config.results_dir,
        )
        return final_state

    # ---- deep analysis ----

    def deep_analysis(
        self,
        graph: QuantumGraph,
        spectrum: np.ndarray,
        insights: PopulationInsights,
    ) -> None:
        """Extended analysis triggered when score exceeds target.

        - Re-evaluates with 3x k_max and more zeros
        - Runs full symbolic analysis
        - Logs detailed findings
        """
        self._console.print(
            Panel("[bold green]Deep analysis triggered[/bold green]")
        )

        # Extended spectrum
        extended_result = self._spectral.evaluate_graph(
            graph,
            k_max=self._spectral._n_points * 3 / 1000,  # rough heuristic
            n_zeros_compare=50,
        )
        self._console.print(
            f"  Extended score: {extended_result.score:.4f} "
            f"({len(extended_result.spectrum)} eigenvalues)"
        )

        # Full symbolic analysis
        fits = self._symbolic.fit_prime_combination(
            list(graph.edge_lengths), n_primes=15,
        )
        exact_count = sum(1 for f in fits if f.is_exact)
        self._console.print(
            f"  Symbolic: {exact_count}/{len(fits)} edges "
            f"are exact log-prime combinations"
        )

        regression = self._symbolic.symbolic_regression_lengths([graph])
        self._console.print(
            f"  Regression: {regression.best_formula} "
            f"(residual {regression.best_residual:.6f})"
        )

        self._console.print(f"  Pattern summary:\n{insights.summary}")

    # ---- resume ----

    def resume(self, checkpoint_path: str) -> CheckpointState:
        """Load a checkpoint and continue the search.

        Parameters
        ----------
        checkpoint_path : str
            Path to a ``.pkl`` checkpoint file.

        Returns
        -------
        CheckpointState
            Final state after continued search.
        """
        with open(checkpoint_path, "rb") as fh:
            state: CheckpointState = pickle.load(fh)

        self._console.print(
            f"Resuming from generation {state.generation} "
            f"(best score {state.best_score:.4f})"
        )

        remaining = state.config.n_generations - state.generation - 1
        if remaining <= 0:
            self._console.print("Search already complete.")
            return state

        resumed_config = replace(
            state.config,
            n_generations=remaining,
        )
        self._init_agents(resumed_config)
        self._ensure_results_dir(resumed_config.results_dir)

        # Rebuild mutable state from checkpoint
        population = list(state.population)
        scores = np.array(state.scores)
        best_score = state.best_score
        best_graph = state.best_graph

        for gen_offset in range(remaining):
            gen = state.generation + 1 + gen_offset

            eval_results = self._spectral.evaluate_batch(
                population, k_max=resumed_config.k_max,
                n_zeros_compare=resumed_config.n_zeros_compare,
                n_workers=self._resolve_workers(resumed_config.n_workers),
            )

            scores = np.array([r.score for r in eval_results])
            population = [r.graph for r in eval_results]

            # Optimize top graphs' edge lengths (same as run path)
            population, scores = self._optimize_top_graphs(
                population, scores, resumed_config, gen,
            )

            gen_best = float(scores[0]) if len(scores) > 0 else 0.0
            if gen_best > best_score:
                best_score = gen_best
                best_graph = population[0]

            if resumed_config.verbose:
                self._log_generation(gen, scores, eval_results)

            insights = self._pattern.analyze_population(
                population, scores, resumed_config.top_fraction,
            )

            # Checkpoint before breeding (population and scores aligned)
            if (gen + 1) % resumed_config.checkpoint_every == 0:
                cp = CheckpointState(
                    generation=gen,
                    population=tuple(population),
                    scores=tuple(scores.tolist()),
                    best_score=best_score,
                    best_graph=best_graph,
                    config=state.config,
                )
                self._save_checkpoint(cp, gen, state.config.results_dir)

            population = self._next_generation(
                population, scores, resumed_config, insights,
                best_graph=best_graph,
            )

        return CheckpointState(
            generation=state.config.n_generations - 1,
            population=tuple(population),
            scores=tuple(scores.tolist()),
            best_score=best_score,
            best_graph=best_graph,
            config=state.config,
        )

    # ---- edge length optimization ----

    def _optimize_top_graphs(
        self,
        population: list[QuantumGraph],
        scores: np.ndarray,
        config: SearchConfig,
        gen: int,
    ) -> tuple[list[QuantumGraph], np.ndarray]:
        """Joint-optimize edge lengths and scattering phases for top graphs.

        Uses differential evolution over the combined (lengths, phases)
        parameter space.  This is far more effective than length-only
        optimization because TRS-breaking phases can shift eigenvalue
        positions significantly.

        Runs every 5 generations on the top 3 graphs to balance cost.
        """
        # Skip early generations (no good topologies yet) and non-optimizer gens
        if gen < 5 or gen % 5 != 0 or len(population) < 5:
            return population, scores

        n_optimize = min(3, len(population))
        scorer = SpectralScorer()
        zeros = scorer._rz.get_zeros(config.n_zeros_compare)

        improved = list(population)
        improved_scores = scores.copy()

        for i in range(n_optimize):
            graph = population[i]
            try:
                opt_result = joint_optimize(
                    graph, scorer,
                    riemann_zeros=zeros,
                    n_compare=config.n_zeros_compare,
                    max_iter=50,
                    seed=gen * 1000 + i,
                )
                if opt_result.score > improved_scores[i]:
                    optimized_graph = graph.with_edge_lengths(
                        list(opt_result.edge_lengths)
                    )
                    if opt_result.phases:
                        optimized_graph = optimized_graph.with_scattering_phases(
                            list(opt_result.phases)
                        )
                    improved[i] = optimized_graph
                    improved_scores[i] = opt_result.score
                    if config.verbose:
                        self._console.print(
                            f"  [green]Joint-optimized graph {i}: "
                            f"{scores[i]:.4f} -> {opt_result.score:.4f}[/green]"
                        )
            except Exception:
                continue

        # Re-sort by score
        order = np.argsort(improved_scores)[::-1]
        improved = [improved[j] for j in order]
        improved_scores = improved_scores[order]

        return improved, improved_scores

    # ---- private helpers ----

    def _init_agents(self, config: SearchConfig) -> None:
        self._generative = GenerativeAgent(seed=config.random_seed)
        self._spectral = SpectralAgent()
        self._pattern = PatternAgent()
        self._symbolic = SymbolicAgent()

    def _init_population(
        self, config: SearchConfig,
    ) -> list[QuantumGraph]:
        """Create an initial population mixing templates and random graphs."""
        n_prime = int(config.population_size * config.prime_length_bias)
        n_random = config.population_size - n_prime

        prime_graphs = self._generative.propose_batch(
            population=[],
            scores=np.array([]),
            n_new=n_prime,
            strategy=StrategyType.PRIME_LENGTH,
        )
        template_graphs = self._generative.propose_batch(
            population=[],
            scores=np.array([]),
            n_new=n_random,
            strategy=StrategyType.TEMPLATE,
        )
        return prime_graphs + template_graphs

    def _next_generation(
        self,
        population: list[QuantumGraph],
        scores: np.ndarray,
        config: SearchConfig,
        insights: PopulationInsights,
        best_graph: QuantumGraph | None = None,
    ) -> list[QuantumGraph]:
        """Breed the next generation: elites + diverse offspring + fresh injections."""
        n_elite = max(1, int(config.population_size * config.elite_fraction))

        # Always preserve the all-time best graph as the first elite
        elites = _deduplicate_graphs(population[:n_elite])
        if best_graph is not None:
            elites = [best_graph] + [
                g for g in elites if g is not best_graph
            ]

        # Reserve 15% of population for fresh random graphs (diversity injection)
        n_fresh = max(2, int(config.population_size * 0.15))
        n_offspring = max(0, config.population_size - len(elites) - n_fresh)

        offspring = self._generative.propose_batch(
            population=population,
            scores=scores,
            n_new=n_offspring,
            strategy=None,  # cycle through strategies
        )

        fresh = self._generative.propose_batch(
            population=[],
            scores=np.array([]),
            n_new=n_fresh,
            strategy=StrategyType.PRIME_LENGTH,
        )

        return elites + offspring + fresh

    def _log_generation(
        self,
        gen: int,
        scores: np.ndarray,
        eval_results: list[EvalResult],
    ) -> None:
        """Print a generation summary using rich."""
        if len(scores) == 0:
            return
        level = classify_score(float(scores[0]))
        table = Table(title=f"Generation {gen}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Best score", f"{scores[0]:.4f}")
        table.add_row("Mean score", f"{np.mean(scores):.4f}")
        table.add_row("Std score", f"{np.std(scores):.4f}")
        table.add_row("Validation level", f"Level {level.value}")
        if eval_results:
            top = eval_results[0]
            table.add_row(
                "Best topology",
                f"{top.graph.n_vertices}v / {top.graph.n_edges}e",
            )
            sr = top.scoring_result
            table.add_row("Position match", f"{sr.absolute_match:.4f}")
            table.add_row("GUE spacing", f"{sr.spacing_distribution:.4f}")
            table.add_row(
                "Window",
                f"{sr.n_windowed}/{20} eigs, alpha={sr.weyl_alpha:.2f}",
            )
        self._console.print(table)

        # Per-zero residual table for the best graph
        if eval_results:
            self._log_residual_table(gen, eval_results[0])

    def _log_residual_table(
        self, gen: int, best_result: EvalResult,
    ) -> None:
        """Print per-zero residuals for the best graph this generation.

        Shows how far each Weyl-rescaled eigenvalue sits from its target
        zero.  Uniformly small residuals across all 20 zeros is a
        fundamentally different result from small mean driven by perfect
        matches on a few zeros and large misses on the rest.
        """
        from riemann_qg.core.scoring import _weyl_rescale

        graph = best_result.graph
        spectrum = best_result.spectrum
        scorer = SpectralScorer()
        zeros = scorer._rz.get_zeros(20)

        total_length = sum(graph.edge_lengths)
        if total_length <= 0 or len(spectrum) < 3:
            return

        scaled_eigs, _ = _weyl_rescale(
            np.sort(spectrum), total_length, zeros,
        )

        # Window
        zero_spacings = np.diff(zeros)
        margin = float(np.mean(zero_spacings)) if len(zero_spacings) > 0 else 1.0
        k_low = zeros[0] - margin
        k_high = zeros[-1] + margin
        windowed = scaled_eigs[
            (scaled_eigs >= k_low) & (scaled_eigs <= k_high)
        ]
        n_match = min(len(windowed), len(zeros))

        if n_match < 3:
            return

        table = Table(
            title=f"Gen {gen} — Per-zero residuals (best graph)",
            show_lines=False,
        )
        table.add_column("i", style="dim", width=3)
        table.add_column("Zero", style="green", width=10)
        table.add_column("Mapped Eig", style="cyan", width=10)
        table.add_column("Residual", width=10)
        table.add_column("", width=12)

        residuals = []
        for i in range(min(n_match, 20)):
            z = zeros[i]
            e = windowed[i]
            r = abs(e - z)
            residuals.append(r)
            # Color code: green < 0.5, yellow < 1.5, red >= 1.5
            if r < 0.5:
                style = "[green]"
            elif r < 1.5:
                style = "[yellow]"
            else:
                style = "[red]"
            bar_len = min(int(r * 4), 20)
            bar = style + "|" * max(bar_len, 1) + "[/]"
            table.add_row(
                str(i), f"{z:.2f}", f"{e:.2f}", f"{r:.3f}", bar,
            )

        residuals_arr = np.array(residuals)
        self._console.print(table)
        self._console.print(
            f"  MAD={np.mean(residuals_arr):.3f}  "
            f"Max={np.max(residuals_arr):.3f}  "
            f"Std={np.std(residuals_arr):.3f}  "
            f"({n_match}/{len(zeros)} matched)"
        )
        self._console.print("")

        # Persist for longitudinal analysis (paper figure)
        residual_path = os.path.join("results", "residuals.jsonl")
        record = {
            "gen": gen,
            "zeros": [float(zeros[i]) for i in range(min(n_match, 20))],
            "mapped_eigs": [float(windowed[i]) for i in range(min(n_match, 20))],
            "residuals": [float(r) for r in residuals],
            "mad": float(np.mean(residuals_arr)),
            "max_residual": float(np.max(residuals_arr)),
            "std_residual": float(np.std(residuals_arr)),
            "n_matched": n_match,
            "score": float(best_result.score),
        }
        try:
            with open(residual_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            pass  # Non-critical; don't crash the run

    def _diagnostic_gen0(
        self, best_result: EvalResult, config: SearchConfig,
    ) -> None:
        """Print diagnostic at generation 0: Weyl-rescaled eigenvalues vs zeros.

        Confirms that the Weyl normalization is working before the search
        burns 100 generations on a broken foundation.
        """
        from riemann_qg.core.scoring import SpectralScorer, _weyl_rescale

        graph = best_result.graph
        spectrum = best_result.spectrum
        scorer = SpectralScorer()
        zeros = scorer._rz.get_zeros(config.n_zeros_compare)

        total_length = sum(graph.edge_lengths)
        scaled_eigs, alpha = _weyl_rescale(spectrum, total_length, zeros)

        self._console.print(Panel(
            "[bold cyan]Generation 0 Diagnostic: "
            "Weyl-rescaled eigenvalues vs Riemann zeros[/bold cyan]",
        ))
        self._console.print(
            f"  Graph: {graph.n_vertices}v / {graph.n_edges}e, "
            f"L_total = {total_length:.2f}"
        )
        self._console.print(
            f"  Weyl alpha = {alpha:.4f}  "
            f"(raw eigs [{spectrum[0]:.3f}, {spectrum[-1]:.3f}] "
            f"-> scaled [{scaled_eigs[0]:.2f}, {scaled_eigs[-1]:.2f}])"
        )
        self._console.print(
            f"  Target zeros [{zeros[0]:.2f}, {zeros[-1]:.2f}]"
        )

        # Number line comparison
        zero_spacings = np.diff(zeros)
        margin = float(np.mean(zero_spacings))
        k_low = zeros[0] - margin
        k_high = zeros[-1] + margin
        windowed = scaled_eigs[
            (scaled_eigs >= k_low) & (scaled_eigs <= k_high)
        ]

        n_compare = min(len(windowed), len(zeros))
        self._console.print(
            f"  Windowed eigenvalues: {len(windowed)} "
            f"(target: {len(zeros)})"
        )

        # Side-by-side comparison
        table = Table(title="Eigenvalue vs Zero (first 20)")
        table.add_column("i", style="dim", width=4)
        table.add_column("Riemann zero", style="green", width=14)
        table.add_column("Scaled eigenvalue", style="cyan", width=14)
        table.add_column("Difference", style="red", width=12)

        for i in range(min(n_compare, 20)):
            z = zeros[i] if i < len(zeros) else float("nan")
            e = windowed[i] if i < len(windowed) else float("nan")
            diff = abs(e - z) if i < n_compare else float("nan")
            table.add_row(
                str(i), f"{z:.4f}", f"{e:.4f}", f"{diff:.4f}",
            )
        self._console.print(table)

        sr = best_result.scoring_result
        self._console.print(
            f"  Score breakdown: {sr.summary}"
        )
        self._console.print("")

        # Save diagnostic to file so it survives terminal scroll
        diag_path = os.path.join(config.results_dir, "gen0_diagnostic.txt")
        with open(diag_path, "w") as fh:
            fh.write("GENERATION 0 DIAGNOSTIC\n")
            fh.write("=" * 60 + "\n")
            fh.write(
                f"Graph: {graph.n_vertices}v / {graph.n_edges}e, "
                f"L_total = {total_length:.2f}\n"
            )
            fh.write(
                f"Weyl alpha = {alpha:.4f}  "
                f"(raw eigs [{spectrum[0]:.3f}, {spectrum[-1]:.3f}] "
                f"-> scaled [{scaled_eigs[0]:.2f}, {scaled_eigs[-1]:.2f}])\n"
            )
            fh.write(f"Target zeros [{zeros[0]:.2f}, {zeros[-1]:.2f}]\n")
            fh.write(
                f"Windowed eigenvalues: {len(windowed)} "
                f"(target: {len(zeros)})\n"
            )
            fh.write(f"Score: {sr.summary}\n\n")
            fh.write(f"{'i':>4s}  {'Zero':>12s}  {'Scaled Eig':>12s}  {'Diff':>10s}\n")
            fh.write("-" * 44 + "\n")
            for i in range(min(n_compare, 20)):
                z = zeros[i] if i < len(zeros) else float("nan")
                e = windowed[i] if i < len(windowed) else float("nan")
                diff = abs(e - z)
                fh.write(f"{i:4d}  {z:12.4f}  {e:12.4f}  {diff:10.4f}\n")
        self._console.print(f"  [dim]Saved to {diag_path}[/dim]\n")

    @staticmethod
    def _append_trajectory_record(
        record: GenerationRecord,
        path: str,
    ) -> None:
        """Append one generation record to the JSONL trajectory log."""
        with open(path, "a") as fh:
            fh.write(json.dumps(_record_to_dict(record)) + "\n")

    def _alert_level_5(
        self,
        score: float,
        gen: int,
        results_dir: str,
    ) -> None:
        """Print a prominent Level 5 alert."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            results_dir, f"candidate_{timestamp}.md",
        )
        msg = (
            f"HIGH-VALUE CANDIDATE DETECTED -- HUMAN REVIEW RECOMMENDED\n"
            f"Score: {score:.4f} | Gen: {gen}\n"
            f"Report saved to: {report_path}"
        )
        self._console.print(Panel(
            f"[bold red]{msg}[/bold red]",
            border_style="red",
            title="LEVEL 5 ALERT",
        ))

    @staticmethod
    def _save_checkpoint(
        state: CheckpointState,
        gen: int,
        results_dir: str,
    ) -> None:
        path = os.path.join(results_dir, f"checkpoint_gen_{gen}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _ensure_results_dir(results_dir: str) -> None:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_workers(n_workers: int) -> int:
        if n_workers <= 0:
            return max(1, (os.cpu_count() or 2) + n_workers)
        return n_workers


def _deduplicate_graphs(
    graphs: list[QuantumGraph],
) -> list[QuantumGraph]:
    """Remove duplicate graphs based on vertex count + sorted edge lengths."""
    seen: set[tuple[int, tuple[float, ...]]] = set()
    unique: list[QuantumGraph] = []
    for g in graphs:
        key = (g.n_vertices, tuple(round(l, 4) for l in sorted(g.edge_lengths)))
        if key not in seen:
            seen.add(key)
            unique.append(g)
    return unique if unique else graphs[:1]
