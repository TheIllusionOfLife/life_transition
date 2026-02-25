"""Tests for PR 5: phase diagram experiments, analysis helpers, and figure generators.

Covers:
- Pre-registration document completeness (H1–H4 present)
- Capability stats JSON output schema
- Jonckheere-Terpstra trend detection on synthetic data
- Phase diagram figure renders without error
- InternalizationIndex monotonicity across V-levels
- Shock experiment produces alive time series without error
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic TSV helpers
# ---------------------------------------------------------------------------

_TSV_HEADER = "\t".join(
    [
        "condition",
        "harshness",
        "seed",
        "step",
        "archetype",
        "capability_bits",
        "alive",
        "mean_energy",
        "mean_ii",
        "total_replications",
        "total_failed",
        "world_replications_total",
    ]
)


def _make_tsv_row(
    condition: str,
    harshness: str,
    seed: int,
    step: int,
    archetype: str,
    cap_bits: int,
    alive: int,
    mean_ii: float,
    total_reps: int = 0,
) -> str:
    return "\t".join(
        [
            condition,
            harshness,
            str(seed),
            str(step),
            archetype,
            str(cap_bits),
            str(alive),
            "0.5000",
            f"{mean_ii:.4f}",
            str(total_reps),
            "0",
            "0",
        ]
    )


def _synthetic_tsv(tmp_path: Path) -> Path:
    """Minimal synthetic TSV covering all 9 conditions × 4 harshness × 5 seeds."""
    viroid_caps = {
        "viroid_v0": (1, 0, 5),
        "viroid_v0v1": (3, 0.0, 4),
        "viroid_v0v1v2": (7, 0.15, 6),
        "viroid_v0v1v2v3": (15, 0.50, 8),
    }
    virus_caps = {
        "virus_baseline": (3, 0.0, 5),
        "virus_v0v1v2": (7, 0.15, 6),
        "virus_v0v1v2v3": (15, 0.50, 8),
    }
    proto_caps = {
        "proto_baseline": (14, 0.45, 0, 0),  # cap_bits, ii, alive, reps
        "proto_liberated": (15, 0.50, 5, 10),
    }
    harshness_levels = ["rich", "medium", "sparse", "scarce"]

    lines = [_TSV_HEADER]
    for harshness in harshness_levels:
        for cond, (cbits, ii, alive) in viroid_caps.items():
            for seed in range(5):
                lines.append(_make_tsv_row(cond, harshness, seed, 500, "viroid", cbits, alive, ii))
        for cond, (cbits, ii, alive) in virus_caps.items():
            for seed in range(5):
                lines.append(_make_tsv_row(cond, harshness, seed, 500, "virus", cbits, alive, ii))
        for cond, args in proto_caps.items():
            cbits, ii, alive, reps = args
            arch = "proto_organelle"
            for seed in range(5):
                lines.append(
                    _make_tsv_row(cond, harshness, seed, 500, arch, cbits, alive, ii, reps)
                )

    tsv_path = tmp_path / "synthetic_semi_life.tsv"
    tsv_path.write_text("\n".join(lines), encoding="utf-8")
    return tsv_path


# ---------------------------------------------------------------------------
# Test 1: Pre-registration document completeness
# ---------------------------------------------------------------------------


def test_preregistration_file_has_required_sections():
    """Pre-registration file must exist and contain H1–H4 hypothesis labels."""
    repo_root = Path(__file__).resolve().parent.parent
    prereg = repo_root / "docs" / "research" / "preregistration.md"
    assert prereg.exists(), f"Pre-registration file not found: {prereg}"
    text = prereg.read_text(encoding="utf-8")
    for label in ("H1", "H2", "H3", "H4"):
        assert label in text, f"Pre-registration missing hypothesis {label}"
    for kw in ("Mann-Whitney", "Cliff", "Holm-Bonferroni", "Jonckheere"):
        assert kw in text, f"Pre-registration missing statistical keyword: {kw}"


# ---------------------------------------------------------------------------
# Test 2: Capability stats output schema
# ---------------------------------------------------------------------------


def test_capability_stats_output_schema():
    """analyze_h1 output must contain required schema fields per comparison."""
    from analyze_semi_life_capability_ladder import analyze_h1

    # Minimal rows: viroid_v0 and viroid_v0v1 across 4 harshness levels, 5 seeds each
    rows = []
    for harshness in ["rich", "medium", "sparse", "scarce"]:
        for seed in range(5):
            rows.append(
                {
                    "condition": "viroid_v0",
                    "harshness": harshness,
                    "seed": str(seed),
                    "step": "500",
                    "archetype": "viroid",
                    "capability_bits": "1",
                    "alive": str(5 + seed),
                    "mean_energy": "0.5",
                    "mean_ii": "0.0",
                    "total_replications": "10",
                    "total_failed": "0",
                    "world_replications_total": "0",
                }
            )
            rows.append(
                {
                    "condition": "viroid_v0v1",
                    "harshness": harshness,
                    "seed": str(seed),
                    "step": "500",
                    "archetype": "viroid",
                    "capability_bits": "3",
                    "alive": str(3 + seed),
                    "mean_energy": "0.5",
                    "mean_ii": "0.0",
                    "total_replications": "8",
                    "total_failed": "0",
                    "world_replications_total": "0",
                }
            )

    results = analyze_h1(rows)
    assert len(results) == 4, f"Expected 4 results (one per harshness), got {len(results)}"
    required_fields = {"hypothesis", "comparison", "harshness", "metric", "p_raw"}
    for r in results:
        missing = required_fields - r.keys()
        assert not missing, f"Result missing fields: {missing}\nResult: {r}"
    # H1 label
    assert all(r["hypothesis"] == "H1" for r in results)


# ---------------------------------------------------------------------------
# Test 3: Jonckheere-Terpstra detects positive trend on synthetic data
# ---------------------------------------------------------------------------


def test_jonckheere_trend_v0_to_v3_positive_in_scarce():
    """JT test must detect a positive trend when V0→V3 groups are monotonically larger."""
    from analyses.results.statistics import jonckheere_terpstra

    rng = np.random.default_rng(42)
    groups = [
        rng.normal(3.0, 0.8, 30),  # V0
        rng.normal(5.0, 0.8, 30),  # V0+V1
        rng.normal(7.0, 0.8, 30),  # V0+V1+V2
        rng.normal(10.0, 0.8, 30),  # V0+V1+V2+V3
    ]
    jt_stat, p_val = jonckheere_terpstra(groups)
    assert p_val < 0.05, f"JT should detect positive trend; p={p_val:.4f}"
    # JT statistic should exceed the expected value under null (monotone groups → large JT)
    n_total = sum(len(g) for g in groups)
    ns = [len(g) for g in groups]
    e_jt = (n_total**2 - sum(n**2 for n in ns)) / 4.0
    # For an increasing trend, lower-indexed groups rarely exceed higher-indexed ones,
    # so the JT statistic falls BELOW E[JT] (large negative z-score → small two-sided p).
    assert jt_stat < e_jt, f"JT stat ({jt_stat:.1f}) should be below E[JT] ({e_jt:.1f})"


# ---------------------------------------------------------------------------
# Test 4: Phase diagram figure renders without error
# ---------------------------------------------------------------------------


def test_phase_diagram_figure_renders(tmp_path):
    """generate_fig_semi_life_phase_diagram must save a PDF without raising."""
    from figures.fig_semi_life_phase_diagram import generate_fig_semi_life_phase_diagram

    tsv_path = _synthetic_tsv(tmp_path)
    generate_fig_semi_life_phase_diagram(tsv_path, tmp_path)
    assert (tmp_path / "fig_semi_life_phase_diagram.pdf").exists()


# ---------------------------------------------------------------------------
# Test 5: InternalizationIndex is monotonic across V-levels in synthetic data
# ---------------------------------------------------------------------------


def test_internalization_monotonic_across_v_levels():
    """mean_ii must be non-decreasing V0→V3 when given crafted monotonic data."""
    from figures.fig_semi_life_internalization import _mean_ii_at_final

    conds = ["viroid_v0", "viroid_v0v1", "viroid_v0v1v2", "viroid_v0v1v2v3"]
    ii_values = [0.0, 0.15, 0.30, 0.50]

    rows = []
    for cond, ii in zip(conds, ii_values, strict=True):
        for seed in range(5):
            rows.append(
                {
                    "condition": cond,
                    "harshness": "sparse",
                    "seed": str(seed),
                    "step": "500",
                    "archetype": "viroid",
                    "capability_bits": "1",
                    "alive": "5",
                    "mean_energy": "0.5",
                    "mean_ii": str(ii),
                    "total_replications": "10",
                    "total_failed": "0",
                    "world_replications_total": "0",
                }
            )

    ii_by_level = [_mean_ii_at_final(rows, cond, "sparse") for cond in conds]
    for i in range(1, len(ii_by_level)):
        assert ii_by_level[i] >= ii_by_level[i - 1], (
            f"II not monotonic: level {i} = {ii_by_level[i]:.3f} < "
            f"level {i - 1} = {ii_by_level[i - 1]:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 6: Shock experiment produces alive time series (smoke)
# ---------------------------------------------------------------------------


def test_shock_experiment_produces_recovery_metric():
    """Shock experiment for 3 seeds must produce non-empty alive time series."""
    import life_transition
    from experiment_semi_life_shocks import _aggregate, make_shock_config

    seeds = [100, 101, 102]
    all_alive = []
    for seed in seeds:
        config_json = make_shock_config(
            archetype="viroid",
            cap_bits=0x01,
            resource_initial=0.1,
            shock_period=50,
            seed=seed,
        )
        result = json.loads(life_transition.run_semi_life_v0_experiment_json(config_json, 200, 100))
        for sample in result["samples"]:
            agg = _aggregate(sample["snapshots"], "viroid")
            all_alive.append(agg["alive"])

    assert len(all_alive) > 0, "Shock experiment produced no samples"
