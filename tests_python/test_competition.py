"""Unit tests for the SemiLife competition experiment.

Tests cover:
- frequency_ratio computation (pure function, no simulation)
- TSV column completeness
- Multi-archetype config structure validation (smoke test)
"""

from __future__ import annotations

import json

import pytest
from analyze_competition import (
    _get_final_step_rows,
    get_frequency_ratio_at_final,
    run_mannwhitney,
)
from experiment_semi_life_competition import (
    TSV_COLUMNS,
    V0,
    V1,
    V2,
    V3,
    compute_frequency_ratio,
    make_competition_config,
)

# ---------------------------------------------------------------------------
# frequency_ratio computation
# ---------------------------------------------------------------------------


def test_frequency_ratio_equal() -> None:
    """Equal populations → plasmid frequency is 0.5."""
    result = compute_frequency_ratio(plasmid_alive=5, viroid_alive=5)
    assert result == pytest.approx(0.5)


def test_frequency_ratio_plasmid_dominates() -> None:
    """Viroid extinct, plasmid survives → frequency is 1.0."""
    result = compute_frequency_ratio(plasmid_alive=10, viroid_alive=0)
    assert result == pytest.approx(1.0)


def test_frequency_ratio_both_zero() -> None:
    """Both extinct → returns None (not a number)."""
    result = compute_frequency_ratio(plasmid_alive=0, viroid_alive=0)
    assert result is None


def test_frequency_ratio_viroid_only() -> None:
    """Plasmid extinct, viroid survives → plasmid frequency is 0.0."""
    result = compute_frequency_ratio(plasmid_alive=0, viroid_alive=10)
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TSV column completeness
# ---------------------------------------------------------------------------


def test_tsv_columns_complete() -> None:
    """TSV_COLUMNS must contain all required output fields."""
    required = {
        "condition",
        "harshness",
        "seed",
        "step",
        "viroid_alive",
        "plasmid_alive",
        "viroid_replications",
        "plasmid_replications",
        "frequency_ratio",
    }
    assert required.issubset(set(TSV_COLUMNS)), f"Missing columns: {required - set(TSV_COLUMNS)}"


# ---------------------------------------------------------------------------
# Config structure smoke test
# ---------------------------------------------------------------------------


def test_make_competition_config_structure() -> None:
    """make_competition_config returns valid JSON with both archetypes enabled."""
    config_json = make_competition_config(
        viroid_caps=V0,
        plasmid_caps=V0 | V1 | V2 | V3,
        resource_initial=1.0,
        seed=0,
    )
    config = json.loads(config_json)

    assert config["enable_semi_life"] is True
    sl = config["semi_life_config"]
    assert "viroid" in sl["enabled_archetypes"]
    assert "plasmid" in sl["enabled_archetypes"]
    overrides = sl["capability_overrides"]
    assert overrides["viroid"] == V0
    assert overrides["plasmid"] == V0 | V1 | V2 | V3


def test_make_competition_config_ablation_structure() -> None:
    """Ablation config correctly withholds V3 from plasmid."""
    config_json = make_competition_config(
        viroid_caps=V0,
        plasmid_caps=V0 | V1 | V2,
        resource_initial=0.1,
        seed=42,
    )
    config = json.loads(config_json)

    overrides = config["semi_life_config"]["capability_overrides"]
    assert overrides["viroid"] == V0
    assert overrides["plasmid"] == V0 | V1 | V2
    # V3 bit must NOT be set for plasmid in the ablation condition
    assert (overrides["plasmid"] & V3) == 0


# ---------------------------------------------------------------------------
# analyze_competition core logic
# ---------------------------------------------------------------------------

_SAMPLE_ROWS = [
    {
        "condition": "primary",
        "harshness": "rich",
        "seed": "0",
        "step": "450",
        "viroid_alive": "5",
        "plasmid_alive": "8",
        "viroid_replications": "10",
        "plasmid_replications": "20",
        "frequency_ratio": "0.615385",
    },
    {
        "condition": "primary",
        "harshness": "rich",
        "seed": "0",
        "step": "500",
        "viroid_alive": "3",
        "plasmid_alive": "9",
        "viroid_replications": "6",
        "plasmid_replications": "22",
        "frequency_ratio": "0.750000",
    },
    {
        "condition": "primary",
        "harshness": "rich",
        "seed": "1",
        "step": "500",
        "viroid_alive": "0",
        "plasmid_alive": "10",
        "viroid_replications": "0",
        "plasmid_replications": "25",
        "frequency_ratio": "1.000000",
    },
    {
        "condition": "primary",
        "harshness": "rich",
        "seed": "2",
        "step": "500",
        "viroid_alive": "0",
        "plasmid_alive": "0",
        "viroid_replications": "0",
        "plasmid_replications": "0",
        "frequency_ratio": "nan",
    },
]


def test_get_final_step_rows_selects_max_step() -> None:
    """Only rows at the highest step value are returned."""
    final = _get_final_step_rows(_SAMPLE_ROWS, "primary", "rich")
    steps = {r["step"] for r in final}
    assert steps == {"500"}
    assert len(final) == 3  # seeds 0, 1, 2 all have step=500


def test_get_final_step_rows_empty_on_no_match() -> None:
    """Returns empty list when condition/harshness has no rows."""
    final = _get_final_step_rows(_SAMPLE_ROWS, "ablation", "rich")
    assert final == []


def test_get_frequency_ratio_excludes_nan() -> None:
    """'nan' string values are excluded; result contains only numeric ratios."""
    ratios = get_frequency_ratio_at_final(_SAMPLE_ROWS, "primary", "rich")
    # seed 2 has frequency_ratio='nan' and must be excluded
    assert len(ratios) == 2
    assert all(isinstance(v, float) for v in ratios)
    assert pytest.approx(1.0) in ratios


def test_run_mannwhitney_small_n_returns_none_fields() -> None:
    """n < 2 in either group → all statistical fields are None (not errors)."""
    result = run_mannwhitney([0.5], [0.8, 0.9])
    assert result["U"] is None
    assert result["p_raw"] is None
    assert result["cliffs_delta"] is None
    assert result["n_a"] == 1


def test_run_mannwhitney_valid_inputs_returns_floats() -> None:
    """Valid inputs return numeric U, p_raw, and cliffs_delta."""
    a = [0.1, 0.2, 0.3, 0.4]
    b = [0.7, 0.8, 0.9, 1.0]
    result = run_mannwhitney(a, b)
    assert result["U"] is not None
    assert isinstance(result["p_raw"], float)
    assert isinstance(result["cliffs_delta"], float)
    assert result["n_a"] == 4
    assert result["n_b"] == 4
