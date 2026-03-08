"""Unit tests for the SemiLife competition experiment.

Tests cover:
- frequency_ratio computation (pure function, no simulation)
- TSV column completeness
- Multi-archetype config structure validation (smoke test)
"""

from __future__ import annotations

import json

import pytest
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
