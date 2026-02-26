"""Tests for mean_energy supplementary analysis in the capability ladder script.

Validates the new get_mean_energy_at_final() extractor and the
analyze_mean_energy_supplement() function that produces exploratory
effect-size comparisons for the finer-grained mean_energy metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_semi_life_capability_ladder import (
    analyze_mean_energy_supplement,
    get_mean_energy_at_final,
)

_HARSHNESS_LEVELS = ["rich", "medium", "sparse", "scarce"]


def _make_rows(
    condition: str,
    harshness: str,
    seeds: range,
    step: int,
    alive: int,
    mean_energy: float,
) -> list[dict]:
    """Build synthetic TSV rows for testing."""
    rows = []
    for seed in seeds:
        rows.append(
            {
                "condition": condition,
                "harshness": harshness,
                "seed": str(seed),
                "step": str(step),
                "alive": str(alive),
                "mean_energy": str(mean_energy),
                "mean_ii": "0.0",
                "total_replications": "0",
                "total_failed": "0",
                "world_replications_total": "0",
                "archetype": "viroid",
                "capability_bits": "1",
            }
        )
    return rows


def _build_pair(
    cond_a: str,
    energy_a: float,
    cond_b: str,
    energy_b: float,
    harshness: str = "rich",
) -> list[dict]:
    """Build rows for two conditions at a single harshness level."""
    rows = _make_rows(
        cond_a,
        harshness,
        range(100, 120),
        step=500,
        alive=10,
        mean_energy=energy_a,
    )
    rows += _make_rows(
        cond_b,
        harshness,
        range(100, 120),
        step=500,
        alive=10,
        mean_energy=energy_b,
    )
    return rows


@pytest.fixture()
def v0_rich_rows():
    """Standard viroid_v0 rows at rich harshness for extractor tests."""
    return _make_rows(
        "viroid_v0",
        "rich",
        range(100, 105),
        step=500,
        alive=10,
        mean_energy=0.55,
    )


@pytest.fixture()
def full_supplement_rows():
    """Rows covering all harshness × both H1/H2 condition pairs."""
    rows: list[dict] = []
    for h in _HARSHNESS_LEVELS:
        rows += _build_pair("viroid_v0", 0.6, "viroid_v0v1", 0.4, h)
        rows += _build_pair(
            "viroid_v0v1v2v3",
            0.8,
            "viroid_v0v1v2",
            0.3,
            h,
        )
    return rows


class TestGetMeanEnergyAtFinal:
    def test_extracts_correct_values(self, v0_rich_rows):
        result = get_mean_energy_at_final(v0_rich_rows, "viroid_v0", "rich")
        assert len(result) == 5
        assert all(v == pytest.approx(0.55) for v in result)

    def test_returns_empty_for_missing_condition(self, v0_rich_rows):
        assert get_mean_energy_at_final(v0_rich_rows, "viroid_v0v1", "rich") == []

    def test_returns_empty_for_missing_harshness(self, v0_rich_rows):
        assert get_mean_energy_at_final(v0_rich_rows, "viroid_v0", "sparse") == []

    def test_selects_final_step_only(self):
        rows = _make_rows(
            "viroid_v0",
            "rich",
            range(100, 102),
            step=250,
            alive=10,
            mean_energy=0.30,
        )
        rows += _make_rows(
            "viroid_v0",
            "rich",
            range(100, 102),
            step=500,
            alive=10,
            mean_energy=0.55,
        )
        result = get_mean_energy_at_final(rows, "viroid_v0", "rich")
        assert len(result) == 2
        assert all(v == pytest.approx(0.55) for v in result)


class TestAnalyzeMeanEnergySupplement:
    def test_h1_returns_results_for_all_harshness(self):
        rows: list[dict] = []
        for h in _HARSHNESS_LEVELS:
            rows += _build_pair("viroid_v0", 0.6, "viroid_v0v1", 0.4, h)
        results = analyze_mean_energy_supplement(rows)
        h1 = [r for r in results if r["hypothesis"] == "H1_energy"]
        assert len(h1) == 4

    def test_h2_returns_results_for_all_harshness(self):
        rows: list[dict] = []
        for h in _HARSHNESS_LEVELS:
            rows += _build_pair(
                "viroid_v0v1v2v3",
                0.8,
                "viroid_v0v1v2",
                0.3,
                h,
            )
        results = analyze_mean_energy_supplement(rows)
        h2 = [r for r in results if r["hypothesis"] == "H2_energy"]
        assert len(h2) == 4

    def test_metric_field_is_mean_energy(self, full_supplement_rows):
        results = analyze_mean_energy_supplement(full_supplement_rows)
        assert all(r["metric"] == "mean_energy" for r in results)

    def test_clear_separation_gives_large_delta(self):
        rows: list[dict] = []
        for h in _HARSHNESS_LEVELS:
            rows += _build_pair("viroid_v0", 0.9, "viroid_v0v1", 0.1, h)
            rows += _build_pair(
                "viroid_v0v1v2v3",
                0.9,
                "viroid_v0v1v2",
                0.1,
                h,
            )
        results = analyze_mean_energy_supplement(rows)
        h1_rich = [
            r for r in results if r["hypothesis"] == "H1_energy" and r["harshness"] == "rich"
        ][0]
        assert h1_rich["cliffs_delta"] is not None
        assert abs(h1_rich["cliffs_delta"]) > 0.8

    def test_label_is_exploratory(self, full_supplement_rows):
        results = analyze_mean_energy_supplement(full_supplement_rows)
        assert all(r.get("analysis_type") == "exploratory" for r in results)


class TestMeanEnergyColumnGuard:
    """Verify fail-soft behavior when mean_energy column is absent."""

    def test_missing_column_raises_key_error_in_extractor(self):
        """get_mean_energy_at_final crashes if column absent — the guard
        in main() should prevent this from being called."""
        rows = [
            {
                "condition": "viroid_v0",
                "harshness": "rich",
                "seed": "100",
                "step": "500",
                "alive": "10",
                "mean_ii": "0.0",
                "total_replications": "0",
                "total_failed": "0",
                "world_replications_total": "0",
                "archetype": "viroid",
                "capability_bits": "1",
            }
        ]
        with pytest.raises(KeyError, match="mean_energy"):
            get_mean_energy_at_final(rows, "viroid_v0", "rich")

    def test_main_guard_skips_when_column_absent(self, tmp_path):
        """main() should complete H1-H4 even if mean_energy is absent."""
        from analyze_semi_life_capability_ladder import main

        tsv = tmp_path / "no_energy.tsv"
        header = (
            "condition\tharshness\tseed\tstep\tarchetype\t"
            "capability_bits\talive\tmean_ii\t"
            "total_replications\ttotal_failed\t"
            "world_replications_total"
        )
        # Minimal row — enough for load_tsv to succeed
        row = "viroid_v0\trich\t100\t500\tviroid\t1\t10\t0.0\t0\t0\t0"
        tsv.write_text(f"{header}\n{row}\n")

        # Override experiments dir so main() writes to tmp
        import analyze_semi_life_capability_ladder as mod

        orig_dir = mod._EXPERIMENTS_DIR
        mod._EXPERIMENTS_DIR = tmp_path
        try:
            main([str(tsv)])
        finally:
            mod._EXPERIMENTS_DIR = orig_dir

        # H1-H4 stats should be written
        assert (tmp_path / "semi_life_capability_stats.json").exists()
        # Supplement should NOT be written
        assert not (tmp_path / "semi_life_mean_energy_supplement.json").exists()
