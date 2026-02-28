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
import pytest

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


# ---------------------------------------------------------------------------
# Test 7: analyze_h2, analyze_h3, analyze_h4 output schema
# ---------------------------------------------------------------------------


def _make_rows_for_h234(harshness_levels: list[str]) -> list[dict]:
    """Build minimal rows covering H2, H3, and H4 conditions."""
    rows = []
    for harshness in harshness_levels:
        for seed in range(5):
            base = {
                "step": "500",
                "mean_energy": "0.5",
                "mean_ii": "0.1",
                "total_failed": "0",
                "world_replications_total": "0",
                "harshness": harshness,
                "seed": str(seed),
            }
            for cond, arch, cbits, alive, reps in [
                ("viroid_v0", "viroid", "1", str(3 + seed), "8"),
                ("viroid_v0v1", "viroid", "3", str(4 + seed), "9"),
                ("viroid_v0v1v2", "viroid", "7", str(5 + seed), "10"),
                ("viroid_v0v1v2v3", "viroid", "15", str(7 + seed), "12"),
                ("proto_baseline", "proto_organelle", "14", "3", "0"),
                ("proto_liberated", "proto_organelle", "15", "5", "10"),
            ]:
                rows.append(
                    {
                        **base,
                        "condition": cond,
                        "archetype": arch,
                        "capability_bits": cbits,
                        "alive": alive,
                        "total_replications": reps,
                    }
                )
    return rows


def test_analyze_h2_h3_output_schema():
    """analyze_h2 and analyze_h3 must return 4 results with required schema fields."""
    from analyze_semi_life_capability_ladder import analyze_h2, analyze_h3

    rows = _make_rows_for_h234(["rich", "medium", "sparse", "scarce"])
    required = {"hypothesis", "comparison", "harshness", "metric", "p_raw"}

    for fn, label in [(analyze_h2, "H2"), (analyze_h3, "H3")]:
        results = fn(rows)
        assert len(results) == 4, f"{label}: expected 4 results, got {len(results)}"
        for r in results:
            missing = required - r.keys()
            assert not missing, f"{label} result missing fields: {missing}"
        assert all(r["hypothesis"] == label for r in results)


def test_analyze_h4_schema_includes_cliffs_delta_fields():
    """analyze_h4 results must include cliffs_delta/ci fields for uniform schema."""
    from analyze_semi_life_capability_ladder import analyze_h4

    rows = _make_rows_for_h234(["rich", "medium", "sparse", "scarce"])
    results = analyze_h4(rows)
    assert len(results) == 4, f"Expected 4 H4 results, got {len(results)}"
    for r in results:
        assert r["hypothesis"] == "H4"
        assert "cliffs_delta" in r, "H4 result missing cliffs_delta field"
        assert "ci_low" in r, "H4 result missing ci_low field"
        assert "ci_high" in r, "H4 result missing ci_high field"


# ---------------------------------------------------------------------------
# Test 7b: analyze_h8 output schema (V2 overconsumption regulation)
# ---------------------------------------------------------------------------


def test_analyze_h8_output_schema():
    """analyze_h8 must return 4 results with required schema fields."""
    from analyze_semi_life_capability_ladder import analyze_h8

    rows = _make_rows_for_h234(["rich", "medium", "sparse", "scarce"])
    required = {"hypothesis", "comparison", "harshness", "metric", "p_raw"}
    results = analyze_h8(rows)
    assert len(results) == 4, f"H8: expected 4 results, got {len(results)}"
    for r in results:
        missing = required - r.keys()
        assert not missing, f"H8 result missing fields: {missing}"
    assert all(r["hypothesis"] == "H8" for r in results)
    assert all(r["metric"] == "alive" for r in results)
    assert all("viroid_v0v1v2 vs viroid_v0v1" in r["comparison"] for r in results)


# ---------------------------------------------------------------------------
# Test 8: Holm-Bonferroni correction never reduces p-values
# ---------------------------------------------------------------------------


def test_holm_bonferroni_never_reduces_p_values():
    """apply_holm_bonferroni must produce p_corrected >= p_raw for all 32 results."""
    from analyze_semi_life_capability_ladder import apply_holm_bonferroni

    # Build exactly 32 results to match pre-registered family size (H1–H8 × 4).
    results = [{"hypothesis": f"H{i}", "p_raw": 0.01 + i * 0.001} for i in range(32)]
    corrected = apply_holm_bonferroni(results)
    for r in corrected:
        assert r["p_corrected"] >= r["p_raw"], (
            f"HB correction lowered p: raw={r['p_raw']}, corrected={r['p_corrected']}"
        )


def test_holm_bonferroni_rejects_wrong_family_size():
    """apply_holm_bonferroni must reject if family size != 32."""
    from analyze_semi_life_capability_ladder import apply_holm_bonferroni

    results = [
        {"hypothesis": "H1", "p_raw": 0.01},
        {"hypothesis": "H2", "p_raw": 0.04},
        {"hypothesis": "H3", "p_raw": 0.10},
    ]
    with pytest.raises(ValueError, match="Expected 32"):
        apply_holm_bonferroni(results)


# ---------------------------------------------------------------------------
# Test 9: compute_recovery_time detects known recovery event
# ---------------------------------------------------------------------------


def test_compute_recovery_time_detects_recovery():
    """Recovery time must be detected when alive crosses 80% threshold post-shock."""
    from analyze_semi_life_shocks import compute_recovery_time

    # Shock at step 50: population drops, then recovers at step 80
    step_alives = {
        0: [10.0, 10.0],  # pre-shock baseline: mean = 10.0
        50: [10.0, 10.0],  # shock occurs here; last pre-shock observation is step 0
        60: [2.0, 2.0],  # post-shock trough: mean = 2.0 < threshold (8.0)
        80: [9.0, 9.0],  # recovered: mean = 9.0 >= 80% × 10.0 = 8.0
    }
    recovery_times = compute_recovery_time(step_alives, shock_period=50, recovery_target=0.80)
    assert len(recovery_times) == 1, f"Expected 1 shock event, got {len(recovery_times)}"
    assert recovery_times[0] == 30.0, (
        f"Expected recovery at step 30 post-shock, got {recovery_times[0]}"
    )
