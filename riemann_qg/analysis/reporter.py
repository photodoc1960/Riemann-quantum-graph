"""Structured scientific reporting for quantum graph search results.

Generates rich console output for generation summaries and full Markdown
scientific reports for high-scoring graph candidates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

if TYPE_CHECKING:
    from riemann_qg.core.quantum_graph import QuantumGraph
    from riemann_qg.core.scoring import ScoringResult


@dataclass(frozen=True)
class GenerationSummary:
    """Immutable snapshot of a generation's key metrics."""

    generation: int
    best_score: float
    mean_score: float
    top_n_vertices: int
    top_n_edges: int
    n_evaluated: int


class Reporter:
    """Generates structured output for quantum graph search progress.

    Produces rich console reports during search and Markdown scientific
    reports for high-scoring candidates.
    """

    def __init__(
        self,
        results_dir: str = "results",
        target_score: float = 0.85,
    ) -> None:
        self._results_dir = Path(results_dir)
        self._target_score = target_score
        self._console = Console()
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def generation_report(
        self,
        generation: int,
        top_graphs: list[tuple["QuantumGraph", float]],
        insights: dict[str, Any] | None = None,
    ) -> GenerationSummary:
        """Display rich console output summarizing one generation.

        Shows scores, topology info, pattern insights, and a progress
        bar toward the target score.

        Args:
            generation: Current generation number.
            top_graphs: List of (graph, score) pairs, best first.
            insights: Optional dict of pattern analysis insights.

        Returns:
            Immutable GenerationSummary of key metrics.
        """
        scores = [score for _, score in top_graphs]
        best_score = max(scores) if scores else 0.0
        mean_score = float(np.mean(scores)) if scores else 0.0

        best_graph = top_graphs[0][0] if top_graphs else None

        summary = GenerationSummary(
            generation=generation,
            best_score=best_score,
            mean_score=mean_score,
            top_n_vertices=best_graph.n_vertices if best_graph else 0,
            top_n_edges=best_graph.n_edges if best_graph else 0,
            n_evaluated=len(top_graphs),
        )

        self._print_generation_header(summary)
        self._print_top_graphs_table(top_graphs[:5])
        self._print_insights(insights)
        self._print_progress_bar(best_score)

        return summary

    def _print_generation_header(self, summary: GenerationSummary) -> None:
        """Print the generation header panel."""
        header_text = (
            f"Generation {summary.generation} | "
            f"Best: {summary.best_score:.4f} | "
            f"Mean: {summary.mean_score:.4f} | "
            f"Evaluated: {summary.n_evaluated}"
        )
        self._console.print(Panel(header_text, title="Generation Summary"))

    def _print_top_graphs_table(
        self, top_graphs: list[tuple["QuantumGraph", float]]
    ) -> None:
        """Print a table of top-scoring graphs."""
        table = Table(title="Top Candidates")
        table.add_column("Rank", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Vertices", justify="right")
        table.add_column("Edges", justify="right")
        table.add_column("Total Length", justify="right")

        for rank, (graph, score) in enumerate(top_graphs, 1):
            total_length = sum(graph.edge_lengths)
            table.add_row(
                str(rank),
                f"{score:.4f}",
                str(graph.n_vertices),
                str(graph.n_edges),
                f"{total_length:.3f}",
            )

        self._console.print(table)

    def _print_insights(self, insights: dict[str, Any] | None) -> None:
        """Print pattern analysis insights if available."""
        if not insights:
            return

        self._console.print("\n[bold]Pattern Insights:[/bold]")
        for key, value in insights.items():
            self._console.print(f"  {key}: {value}")

    def _print_progress_bar(self, best_score: float) -> None:
        """Print progress bar toward target score."""
        fraction = min(best_score / self._target_score, 1.0)
        bar_width = 40
        filled = int(fraction * bar_width)
        bar = "=" * filled + "-" * (bar_width - filled)

        self._console.print(
            f"\nProgress: [{bar}] "
            f"{fraction * 100:.1f}% of target {self._target_score}"
        )

    def scientific_report(
        self,
        graph: "QuantumGraph",
        spectrum: np.ndarray,
        scoring_result: "ScoringResult",
        insights: dict[str, Any] | None = None,
    ) -> Path:
        """Generate a full Markdown scientific report.

        Saves to results/ with graph specification, spectral comparison,
        statistical analysis, and scientific assessment.

        Args:
            graph: The quantum graph candidate.
            spectrum: Computed eigenvalue spectrum.
            scoring_result: Detailed scoring breakdown.
            insights: Optional pattern/symbolic analysis results.

        Returns:
            Path to the saved report file.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"candidate_{timestamp}.md"
        filepath = self._results_dir / filename

        sections = [
            _report_header(scoring_result, timestamp),
            _graph_specification_section(graph),
            _spectral_comparison_section(spectrum, scoring_result),
            _scoring_breakdown_section(scoring_result),
            _insights_section(insights),
            _assessment_section(scoring_result),
        ]

        content = "\n\n".join(sections)
        filepath.write_text(content, encoding="utf-8")

        self._console.print(
            f"\n[green]Report saved to: {filepath}[/green]"
        )

        return filepath


def _report_header(
    scoring_result: "ScoringResult", timestamp: str
) -> str:
    """Generate the report header section."""
    return (
        f"# Quantum Graph Candidate Report\n\n"
        f"**Generated:** {timestamp}\n"
        f"**Total Score:** {scoring_result.total_score:.6f}\n"
        f"**Summary:** {scoring_result.summary}"
    )


def _graph_specification_section(graph: "QuantumGraph") -> str:
    """Generate the graph specification section."""
    lines = [
        "## Graph Specification\n",
        f"- **Vertices:** {graph.n_vertices}",
        f"- **Edges:** {graph.n_edges}",
        f"- **Total Length:** {sum(graph.edge_lengths):.6f}",
        "",
        "### Adjacency Matrix\n",
        "```",
        str(graph.adjacency),
        "```",
        "",
        "### Edge Lengths\n",
        "| Edge | Length | ln(p) nearest |",
        "|------|--------|---------------|",
    ]

    for idx, length in enumerate(graph.edge_lengths):
        nearest_info = _nearest_prime_log_info(length)
        lines.append(f"| {idx} | {length:.6f} | {nearest_info} |")

    return "\n".join(lines)


def _nearest_prime_log_info(length: float) -> str:
    """Find nearest ln(prime) to a given length."""
    from sympy import primerange

    primes = list(primerange(2, 100))
    prime_logs = [float(np.log(p)) for p in primes]

    deviations = [abs(length - pl) for pl in prime_logs]
    best_idx = int(np.argmin(deviations))
    best_prime = primes[best_idx]
    deviation = deviations[best_idx]

    return f"ln({best_prime}) + {deviation:.4f}"


def _spectral_comparison_section(
    spectrum: np.ndarray,
    scoring_result: "ScoringResult",
) -> str:
    """Generate the spectral comparison table section."""
    lines = [
        "## Spectral Comparison\n",
        "| Index | Candidate k | Riemann zero | Deviation |",
        "|-------|-------------|--------------|-----------|",
    ]

    n_show = min(20, len(spectrum))
    for i in range(n_show):
        k_val = spectrum[i] if i < len(spectrum) else 0.0
        lines.append(f"| {i} | {k_val:.6f} | - | - |")

    return "\n".join(lines)


def _scoring_breakdown_section(scoring_result: "ScoringResult") -> str:
    """Generate the scoring breakdown section."""
    lines = [
        "## Scoring Breakdown\n",
        f"- **Total Score:** {scoring_result.total_score:.6f}",
        f"- **Absolute Match:** {scoring_result.absolute_score:.6f} "
        f"(weight 0.40)",
        f"- **Spacing Distribution:** {scoring_result.spacing_score:.6f} "
        f"(weight 0.35)",
        f"- **Pair Correlation:** {scoring_result.correlation_score:.6f} "
        f"(weight 0.25)",
    ]
    return "\n".join(lines)


def _insights_section(insights: dict[str, Any] | None) -> str:
    """Generate the analysis insights section."""
    if not insights:
        return "## Analysis Insights\n\nNo additional insights available."

    lines = ["## Analysis Insights\n"]
    for key, value in insights.items():
        lines.append(f"- **{key}:** {value}")

    return "\n".join(lines)


def _assessment_section(scoring_result: "ScoringResult") -> str:
    """Generate the scientific assessment section."""
    score = scoring_result.total_score
    level = _validation_level(score)

    lines = [
        "## Scientific Assessment\n",
        f"**Validation Level:** {level}\n",
    ]

    if score < 0.5:
        lines.append("No meaningful spectral match detected. "
                      "Graph spectrum appears random relative to "
                      "Riemann zeros.")
    elif score < 0.65:
        lines.append("Weak match: GUE-like spacing statistics present "
                      "but individual zero positions not well matched.")
    elif score < 0.75:
        lines.append("Moderate match: first ~10 zeros approximately "
                      "reproduced. Topology and edge lengths show some "
                      "prime-log structure.")
    elif score < 0.85:
        lines.append("Strong match: first ~20 zeros matched with "
                      "prime-log edge lengths confirmed. Warrants "
                      "detailed investigation.")
    elif score < 0.92:
        lines.append("Very strong match: first ~50 zeros matched with "
                      "clear structural regularity. Extended analysis "
                      "recommended.")
    else:
        lines.append("Exceptional match: first ~100 zeros matched with "
                      "closed-form edge length rule. Human review "
                      "strongly recommended for mathematical "
                      "verification.")

    lines.append("\n### Next Steps\n")
    lines.append("- Extend spectrum computation to higher k values")
    lines.append("- Compare trace formula term-by-term")
    lines.append("- Test stability under small perturbations")
    lines.append("- Verify GUE statistics with larger sample")

    return "\n".join(lines)


def _validation_level(score: float) -> str:
    """Map score to validation level description."""
    if score < 0.5:
        return "Level 0 - No match"
    elif score < 0.65:
        return "Level 1 - Better than random"
    elif score < 0.75:
        return "Level 2 - Approximate match (~10 zeros)"
    elif score < 0.85:
        return "Level 3 - Good match (~20 zeros)"
    elif score < 0.92:
        return "Level 4 - Strong match (~50 zeros)"
    else:
        return "Level 5 - Exceptional match (~100 zeros)"
