"""Tests for experiment_semi_life_calibrate.py.

Covers combo generation, scoring logic, config construction, and config
persistence without running expensive full-sweep simulations.
"""

from __future__ import annotations

import json

import life_transition
import pytest
from experiment_semi_life_calibrate import (
    _BASE_SEMI_LIFE_CONFIG,
    N_REEVAL,
    REEVAL_SEEDS,
    SCREEN_SEEDS,
    _build_config,
    _genomic_combos,
    _proto_organelle_combos,
    _save_archetype_config,
    _score,
    _virus_combos,
)

# ---------------------------------------------------------------------------
# Combo generation
# ---------------------------------------------------------------------------


def test_genomic_combos_count():
    """Viroid sweep: 3 maintenance_cost × 3 replication_threshold = 9 combos."""
    combos = _genomic_combos("viroid")
    assert len(combos) == 9


def test_virus_combos_count():
    """Virus sweep adds boundary_decay_rate axis: 3×3×2 = 18 combos."""
    combos = _virus_combos()
    assert len(combos) == 18


def test_proto_organelle_combos_count():
    """ProtoOrganelle sweep: 3 maintenance_cost × 3 internal_conversion_rate = 9 combos."""
    combos = _proto_organelle_combos()
    assert len(combos) == 9


def test_genomic_combo_has_replication_cost_less_than_threshold():
    """replication_cost must be strictly less than replication_threshold for all combos."""
    for combo in _genomic_combos("viroid") + _virus_combos():
        rc = combo["replication_cost"]
        rt = combo["replication_threshold"]
        assert rc < rt, f"replication_cost {rc} >= replication_threshold {rt}"


def test_virus_combos_have_boundary_decay_axis():
    """Virus combos must include boundary_decay_rate as a second sweep axis."""
    combos = _virus_combos()
    bdr_values = {c["boundary_decay_rate"] for c in combos}
    assert len(bdr_values) >= 2, "Expected at least 2 distinct boundary_decay_rate values"
    assert all("boundary_decay_rate" in c for c in combos)


def test_combos_have_required_sweep_keys():
    """All combos must carry sweep_param / sweep_value metadata keys."""
    all_combos = _genomic_combos("viroid") + _virus_combos() + _proto_organelle_combos()
    for combo in all_combos:
        assert "sweep_param" in combo, f"Missing sweep_param in {combo}"
        assert "sweep_value" in combo, f"Missing sweep_value in {combo}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_lexicographic_order():
    """Higher survival_rate wins over higher median_alive."""
    high_surv = {"survival_rate": 1.0, "median_alive": 3.0, "median_final_energy": 0.5}
    low_surv = {"survival_rate": 0.7, "median_alive": 10.0, "median_final_energy": 0.9}
    assert _score(high_surv) > _score(low_surv)


def test_score_tiebreak_by_median_alive():
    """Equal survival_rate → higher median_alive wins."""
    a = {"survival_rate": 0.8, "median_alive": 8.0, "median_final_energy": 0.4}
    b = {"survival_rate": 0.8, "median_alive": 5.0, "median_final_energy": 0.9}
    assert _score(a) > _score(b)


def test_score_tiebreak_by_energy():
    """Equal survival_rate and median_alive → higher median_final_energy wins."""
    a = {"survival_rate": 1.0, "median_alive": 10.0, "median_final_energy": 0.7}
    b = {"survival_rate": 1.0, "median_alive": 10.0, "median_final_energy": 0.3}
    assert _score(a) > _score(b)


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------


def test_build_config_produces_valid_json():
    """_build_config() output must be valid JSON parseable as a dict."""
    config_str = _build_config(
        "viroid",
        seed=0,
        overrides={"maintenance_cost": 0.001, "replication_threshold": 0.7},
    )
    parsed = json.loads(config_str)
    assert isinstance(parsed, dict)
    assert parsed["enable_semi_life"] is True
    assert parsed["semi_life_config"]["enabled_archetypes"] == ["viroid"]


def test_build_config_passes_rust_validation():
    """Config produced by _build_config() must pass the Rust-side validator."""
    config_str = _build_config(
        "viroid",
        seed=0,
        overrides={
            "maintenance_cost": 0.001,
            "replication_threshold": 0.7,
            "replication_cost": 0.315,
        },
    )
    assert life_transition.validate_config_json(config_str) is True


def test_build_config_overrides_take_precedence():
    """Explicit overrides must override the base SemiLife config defaults."""
    overrides = {"maintenance_cost": 0.0001, "replication_threshold": 0.6, "replication_cost": 0.27}
    config_str = _build_config("viroid", seed=5, overrides=overrides)
    sl_cfg = json.loads(config_str)["semi_life_config"]
    assert sl_cfg["maintenance_cost"] == pytest.approx(0.0001)
    assert sl_cfg["replication_threshold"] == pytest.approx(0.6)


def test_build_config_single_archetype_isolation():
    """Single-archetype mode: only the specified archetype is in enabled_archetypes."""
    for archetype in ("viroid", "virus", "proto_organelle"):
        cfg = json.loads(_build_config(archetype, seed=0, overrides={}))
        assert cfg["semi_life_config"]["enabled_archetypes"] == [archetype]


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def test_save_archetype_config_creates_file(tmp_path, monkeypatch):
    """_save_archetype_config() must create the expected JSON file under _CONFIGS_DIR."""
    import experiment_semi_life_calibrate as cal

    monkeypatch.setattr(cal, "_CONFIGS_DIR", tmp_path)
    overrides = {"maintenance_cost": 0.001, "replication_threshold": 0.7, "replication_cost": 0.315}
    _save_archetype_config("viroid", overrides)

    path = tmp_path / "semi_life_viroid.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["maintenance_cost"] == pytest.approx(0.001)
    assert data["replication_threshold"] == pytest.approx(0.7)
    assert data["enabled_archetypes"] == ["viroid"]


def test_save_archetype_config_has_metadata_keys(tmp_path, monkeypatch):
    """Saved config must contain '_generated_by' and '_target' metadata."""
    import experiment_semi_life_calibrate as cal

    monkeypatch.setattr(cal, "_CONFIGS_DIR", tmp_path)
    _save_archetype_config("viroid", {})

    data = json.loads((tmp_path / "semi_life_viroid.json").read_text())
    assert "_generated_by" in data
    assert "_target" in data
    assert "_screen_seeds" in data
    assert "_reeval_seeds" in data


def test_save_archetype_config_has_all_base_fields(tmp_path, monkeypatch):
    """Saved config must contain every field from _BASE_SEMI_LIFE_CONFIG."""
    import experiment_semi_life_calibrate as cal

    monkeypatch.setattr(cal, "_CONFIGS_DIR", tmp_path)
    _save_archetype_config("proto_organelle", {})

    data = json.loads((tmp_path / "semi_life_proto_organelle.json").read_text())
    for key in _BASE_SEMI_LIFE_CONFIG:
        assert key in data, f"Missing base field '{key}' in saved config"


# ---------------------------------------------------------------------------
# Seed split invariant
# ---------------------------------------------------------------------------


def test_screen_and_reeval_seeds_are_disjoint():
    """Screen seeds (phase 1) and re-eval seeds (phase 2) must not overlap."""
    assert set(SCREEN_SEEDS).isdisjoint(set(REEVAL_SEEDS)), (
        "SCREEN_SEEDS and REEVAL_SEEDS must be disjoint to avoid data contamination"
    )


def test_n_reeval_within_phase1_combo_count():
    """N_REEVAL must not exceed the Phase-1 combo count for any archetype."""
    min_combos = min(
        len(_genomic_combos("viroid")),
        len(_virus_combos()),
        len(_proto_organelle_combos()),
    )
    assert N_REEVAL <= min_combos, (
        f"N_REEVAL={N_REEVAL} exceeds smallest archetype combo count ({min_combos})"
    )
