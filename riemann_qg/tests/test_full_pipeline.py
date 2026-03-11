"""Full pipeline integration tests: orchestrator smoke test, score
monotonicity, and checkpoint round-trip.

These tests use tiny populations and few generations to stay fast.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from riemann_qg.agents.orchestrator import (
    CheckpointState,
    Orchestrator,
    SearchConfig,
    classify_score,
    ValidationLevel,
)


# -----------------------------------------------------------------------
# Shared tiny config
# -----------------------------------------------------------------------

def _tiny_config(results_dir: str, n_generations: int = 3) -> SearchConfig:
    """Return a minimal config for fast testing."""
    return SearchConfig(
        population_size=10,
        n_generations=n_generations,
        k_max=30.0,
        n_zeros_compare=5,
        top_fraction=0.3,
        elite_fraction=0.2,
        n_workers=1,
        min_vertices=3,
        max_vertices=5,
        min_edges=3,
        max_edges=10,
        prime_length_bias=0.5,
        phase_breaking=False,
        target_score=0.99,  # unreachable → no deep analysis
        checkpoint_every=1,
        results_dir=results_dir,
        verbose=False,
        random_seed=42,
    )


# -----------------------------------------------------------------------
# 1. Smoke test: run 3 generations, verify completion
# -----------------------------------------------------------------------


class TestSmokeRun:
    """The orchestrator should complete a short run without errors."""

    def test_runs_to_completion(self, tmp_path: Path):
        config = _tiny_config(str(tmp_path), n_generations=3)
        orchestrator = Orchestrator()

        final_state = orchestrator.run(config)

        assert isinstance(final_state, CheckpointState)
        assert final_state.generation == 2  # 0-indexed: 3 gens → last is 2
        assert final_state.best_score >= 0.0
        assert len(final_state.population) == config.population_size
        assert len(final_state.scores) == config.population_size

    def test_classify_score_levels(self):
        """classify_score should return expected validation levels."""
        assert classify_score(0.0) == ValidationLevel.LEVEL_0
        assert classify_score(0.55) == ValidationLevel.LEVEL_1
        assert classify_score(0.70) == ValidationLevel.LEVEL_2
        assert classify_score(0.80) == ValidationLevel.LEVEL_3
        assert classify_score(0.90) == ValidationLevel.LEVEL_4
        assert classify_score(0.95) == ValidationLevel.LEVEL_5


# -----------------------------------------------------------------------
# 2. Score monotonicity: elite best score non-decreasing
# -----------------------------------------------------------------------


class TestScoreMonotonicity:
    """Because elites are preserved, the best score across generations
    should be non-decreasing.
    """

    def test_best_score_non_decreasing(self, tmp_path: Path):
        config = _tiny_config(str(tmp_path), n_generations=5)
        orchestrator = Orchestrator()

        final_state = orchestrator.run(config)

        # Collect best scores from checkpoints
        checkpoint_files = sorted(tmp_path.glob("checkpoint_gen_*.pkl"))
        best_scores: list[float] = []
        for cp_file in checkpoint_files:
            with open(cp_file, "rb") as fh:
                cp: CheckpointState = pickle.load(fh)
            best_scores.append(cp.best_score)

        # best_score in the checkpoint state is tracked as running max,
        # so it should be non-decreasing
        for i in range(1, len(best_scores)):
            assert best_scores[i] >= best_scores[i - 1] - 1e-10, (
                f"Best score decreased from gen checkpoint {i-1} "
                f"({best_scores[i-1]:.4f}) to {i} ({best_scores[i]:.4f})"
            )


# -----------------------------------------------------------------------
# 3. Checkpoint round-trip: run 2, save, resume, run 2 more
# -----------------------------------------------------------------------


class TestCheckpointResume:
    """Save a checkpoint after 2 generations, resume, run 2 more."""

    def test_checkpoint_and_resume(self, tmp_path: Path):
        # Phase 1: run 2 generations
        config_phase1 = _tiny_config(str(tmp_path), n_generations=2)
        orchestrator1 = Orchestrator()
        state1 = orchestrator1.run(config_phase1)

        assert state1.generation == 1
        checkpoint_path = tmp_path / "checkpoint_gen_1.pkl"
        assert checkpoint_path.exists(), "Checkpoint for gen 1 should exist"

        # Modify the saved checkpoint to allow 4 total generations
        with open(checkpoint_path, "rb") as fh:
            saved_state: CheckpointState = pickle.load(fh)

        # Patch config to have n_generations=4 so resume runs 2 more
        patched_config = SearchConfig(
            population_size=saved_state.config.population_size,
            n_generations=4,
            k_max=saved_state.config.k_max,
            n_zeros_compare=saved_state.config.n_zeros_compare,
            top_fraction=saved_state.config.top_fraction,
            elite_fraction=saved_state.config.elite_fraction,
            n_workers=saved_state.config.n_workers,
            min_vertices=saved_state.config.min_vertices,
            max_vertices=saved_state.config.max_vertices,
            min_edges=saved_state.config.min_edges,
            max_edges=saved_state.config.max_edges,
            prime_length_bias=saved_state.config.prime_length_bias,
            phase_breaking=saved_state.config.phase_breaking,
            target_score=saved_state.config.target_score,
            checkpoint_every=saved_state.config.checkpoint_every,
            results_dir=saved_state.config.results_dir,
            verbose=saved_state.config.verbose,
            random_seed=saved_state.config.random_seed,
        )
        patched_state = CheckpointState(
            generation=saved_state.generation,
            population=saved_state.population,
            scores=saved_state.scores,
            best_score=saved_state.best_score,
            best_graph=saved_state.best_graph,
            config=patched_config,
        )
        with open(checkpoint_path, "wb") as fh:
            pickle.dump(patched_state, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # Phase 2: resume
        orchestrator2 = Orchestrator()
        state2 = orchestrator2.resume(str(checkpoint_path))

        assert state2.generation == 3  # 4 total gens, 0-indexed → last is 3
        assert state2.best_score >= state1.best_score - 1e-10, (
            f"Resumed best score {state2.best_score:.4f} should not be worse "
            f"than phase-1 best {state1.best_score:.4f}"
        )
        assert len(state2.population) > 0
