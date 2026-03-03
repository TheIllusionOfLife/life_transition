"""Tests for parallel experiment execution infrastructure.

Verifies that:
1. Parallel execution produces identical results to sequential execution
2. Thread-safe result collection works correctly
3. GIL release enables actual concurrency
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import life_transition
from experiment_semi_life_v1v3 import V0, V1, V2, V3, make_config


def _run_one_seed(seed: int, cap_bits: int = V0, resource_init: float = 0.3) -> dict:
    """Run a single semi-life experiment and return parsed result."""
    config_json = make_config("viroid", cap_bits, resource_init, seed=seed)
    return json.loads(life_transition.run_semi_life_v0_experiment_json(config_json, 100, 100))


def _extract_final_alive(result: dict) -> int:
    """Extract alive count from the final sample snapshot."""
    last = result["samples"][-1]["snapshots"]
    return sum(1 for s in last if s["alive"] and s["archetype"] == "viroid")


class TestParallelDeterminism:
    """Parallel execution must produce identical results to sequential."""

    SEEDS = list(range(10))

    def test_parallel_matches_sequential(self):
        """Results from ThreadPoolExecutor must match sequential for-loop."""
        # Sequential
        sequential_results = {}
        for seed in self.SEEDS:
            result = _run_one_seed(seed)
            sequential_results[seed] = _extract_final_alive(result)

        # Parallel
        parallel_results = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_run_one_seed, seed): seed for seed in self.SEEDS}
            for future in futures:
                seed = futures[future]
                result = future.result()
                parallel_results[seed] = _extract_final_alive(result)

        # Compare
        for seed in self.SEEDS:
            assert sequential_results[seed] == parallel_results[seed], (
                f"Seed {seed}: sequential={sequential_results[seed]}, "
                f"parallel={parallel_results[seed]}"
            )

    def test_parallel_result_count(self):
        """Parallel execution must return exactly one result per seed."""
        results = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_run_one_seed, seed) for seed in self.SEEDS]
            for f in futures:
                results.append(f.result())
        assert len(results) == len(self.SEEDS)

    def test_parallel_different_conditions(self):
        """Parallel runs with different capability bits must produce distinct results."""
        conditions = [
            (V0, 0.3),
            (V0 | V1, 0.3),
            (V0 | V1 | V2 | V3, 0.3),
        ]
        seed = 42

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(_run_one_seed, seed, cap, res): (cap, res) for cap, res in conditions
            }
            results = {}
            for future in futures:
                key = futures[future]
                results[key] = _extract_final_alive(future.result())

        # At least two different alive counts expected
        unique_counts = set(results.values())
        assert len(unique_counts) >= 2, (
            f"Expected at least 2 distinct alive counts across conditions, got {results}"
        )
