"""Tests for experiment_semi_life_v1v3.py and the PR 3 config additions.

Covers:
- resource_initial_value config field (validation, backward compat, effect on pool)
- capability_overrides config field (validation, active_capabilities propagation)
- Capability ladder dynamics (II=0 for V0, II>0 for V3, boundary cost for V1)
- V1v3 experiment script helpers (config building, isolation mode)
"""

from __future__ import annotations

import json

import life_transition
import pytest
from experiment_semi_life_v1v3 import (
    ARCHETYPE_CONDITIONS,
    RESOURCE_INITIAL_VALUES,
    V0,
    V1,
    V2,
    V3,
    V4,
    V5,
    make_config,
)

# ---------------------------------------------------------------------------
# Capability bitmask constants
# ---------------------------------------------------------------------------


def test_capability_bitmask_values():
    """V0–V5 constants must match Rust capability bitmasks exactly."""
    assert V0 == 0x01
    assert V1 == 0x02
    assert V2 == 0x04
    assert V3 == 0x08
    assert V4 == 0x10
    assert V5 == 0x20


# ---------------------------------------------------------------------------
# resource_initial_value config field
# ---------------------------------------------------------------------------


def test_resource_initial_value_default_is_one():
    """Default config must have resource_initial_value == 1.0."""
    cfg_str = make_config("viroid", V0, 1.0, seed=0)
    cfg = json.loads(cfg_str)
    assert cfg["resource_initial_value"] == pytest.approx(1.0)


def test_resource_initial_value_passes_validation():
    """Various resource_initial_value values must pass Rust validation."""
    for val in [1.0, 0.5, 0.1, 0.05, 0.0]:
        cfg_str = make_config("viroid", V0, val, seed=0)
        assert life_transition.validate_config_json(cfg_str) is True, f"Failed for val={val}"


def test_resource_initial_value_zero_is_valid():
    """resource_initial_value=0.0 is valid (empty world on start)."""
    cfg_str = make_config("viroid", V0, 0.0, seed=0)
    assert life_transition.validate_config_json(cfg_str) is True


def test_resource_initial_value_affects_population():
    """Low resource_initial_value must produce a smaller population at step 500."""

    def run_alive(resource_init: float) -> int:
        cfg_str = make_config("viroid", V0, resource_init, seed=0)
        result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 500, 500))
        last = result["samples"][-1]["snapshots"]
        return sum(1 for s in last if s["alive"] and s["archetype"] == "viroid")

    alive_rich = run_alive(1.0)
    alive_scarce = run_alive(0.05)
    assert alive_rich > 0, (
        f"Rich-resource run went extinct (alive_rich=={alive_rich}); check model/seed/steps"
    )
    assert alive_scarce < alive_rich, (
        f"Scarce pool ({alive_scarce}) should produce fewer entities than rich ({alive_rich})"
    )


# ---------------------------------------------------------------------------
# capability_overrides config field
# ---------------------------------------------------------------------------


def test_capability_overrides_empty_is_default():
    """No overrides → enabled viroid uses V0 baseline (active_capabilities=1)."""
    cfg_str = make_config("viroid", None, 0.1, seed=0)
    cfg = json.loads(cfg_str)
    assert cfg["semi_life_config"].get("capability_overrides", {}) == {}


def test_capability_overrides_appears_in_config():
    """cap_bits override must appear in semi_life_config.capability_overrides."""
    cfg_str = make_config("viroid", V0 | V1, 0.1, seed=0)
    cfg = json.loads(cfg_str)
    overrides = cfg["semi_life_config"]["capability_overrides"]
    assert overrides == {"viroid": V0 | V1}


def test_capability_overrides_passes_rust_validation():
    """All valid V0–V5 capability combinations must pass Rust validation."""
    for bits in [
        V0, V0 | V1, V0 | V1 | V2, V0 | V1 | V2 | V3,
        V0 | V1 | V2 | V3 | V4, V0 | V1 | V2 | V3 | V4 | V5,
    ]:
        cfg_str = make_config("viroid", bits, 0.1, seed=0)
        assert life_transition.validate_config_json(cfg_str) is True, f"Failed for bits={bits}"


def test_capability_overrides_active_capabilities_in_snapshot():
    """Entities in a V0+V1 run must report active_capabilities == 3."""
    cfg_str = make_config("viroid", V0 | V1, 0.1, seed=0)
    result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 100, 100))
    last = result["samples"][-1]["snapshots"]
    alive = [s for s in last if s["alive"] and s["archetype"] == "viroid"]
    assert alive, "No alive viroid entities — cannot check active_capabilities"
    cap_values = {s["active_capabilities"] for s in alive}
    assert cap_values == {V0 | V1}, (
        f"Expected all active_capabilities == {V0 | V1}, got {cap_values}"
    )


def test_v0_only_has_zero_internalization_index():
    """V0-only viroid entities must report II == 0.0 (no internal metabolism)."""
    cfg_str = make_config("viroid", V0, 0.1, seed=0)
    result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 100, 100))
    last = result["samples"][-1]["snapshots"]
    alive = [s for s in last if s["alive"] and s["archetype"] == "viroid"]
    assert alive, "No alive viroid entities"
    assert all(s["internalization_index"] == pytest.approx(0.0) for s in alive), (
        "V0-only entities must have II=0 (no V3 metabolism)"
    )


def test_v3_produces_positive_internalization_index():
    """Entities with V3 metabolism must show II > 0 at steady state."""
    cfg_str = make_config("viroid", V0 | V1 | V2 | V3, 0.1, seed=0)
    result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 200, 200))
    last = result["samples"][-1]["snapshots"]
    alive = [s for s in last if s["alive"] and s["archetype"] == "viroid"]
    assert alive, "No alive viroid entities"
    mean_ii = sum(s["internalization_index"] for s in alive) / len(alive)
    assert mean_ii > 0.0, f"V3 entities must have II > 0; got mean_ii={mean_ii}"


def test_v1_boundary_reduces_population_in_scarce():
    """V0+V1 must produce a smaller population than V0 in scarce resources.

    Boundary maintenance adds per-step energy drain; in resource-scarce environments
    this overhead reduces the equilibrium population compared with V0-only.
    """

    def alive_at_500(cap_bits: int) -> int:
        cfg_str = make_config("viroid", cap_bits, 0.1, seed=0)
        result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 500, 500))
        last = result["samples"][-1]["snapshots"]
        return sum(1 for s in last if s["alive"] and s["archetype"] == "viroid")

    alive_v0 = alive_at_500(V0)
    alive_v0v1 = alive_at_500(V0 | V1)
    assert alive_v0v1 < alive_v0, (
        f"V0+V1 ({alive_v0v1}) should have fewer entities than V0 ({alive_v0}) "
        "in scarce resources due to boundary overhead"
    )


def test_v3_metabolism_boosts_population_in_scarce():
    """V0+V1+V2+V3 must produce a larger population than V0+V1+V2 in scarce resources.

    Internal metabolism acts as an energy buffer, allowing higher population density
    than non-metabolising entities in the same environment.

    Tested across 3 seeds; at least 2/3 must show the expected direction (majority-vote
    guards against single-seed noise in a stochastic simulation).
    """
    _SEEDS = [0, 1, 2]

    def alive_at_500(cap_bits: int, seed: int) -> int:
        cfg_str = make_config("viroid", cap_bits, 0.1, seed=seed)
        result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 500, 500))
        last = result["samples"][-1]["snapshots"]
        return sum(1 for s in last if s["alive"] and s["archetype"] == "viroid")

    wins = sum(alive_at_500(V0 | V1 | V2 | V3, s) > alive_at_500(V0 | V1 | V2, s) for s in _SEEDS)
    assert wins >= 2, (
        f"V3 metabolism boost expected in majority of seeds; "
        f"only {wins}/{len(_SEEDS)} seeds showed V3 > V0+V1+V2"
    )


# ---------------------------------------------------------------------------
# ProtoOrganelle liberation
# ---------------------------------------------------------------------------


def test_proto_organelle_baseline_has_no_replications():
    """ProtoOrganelle (V1+V2+V3, no V0) must not replicate at step 200."""
    cfg_str = make_config("proto_organelle", None, 0.1, seed=0)
    result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 200, 200))
    last = result["samples"][-1]["snapshots"]
    proto = [s for s in last if s["archetype"] == "proto_organelle"]
    total_reps = sum(s["replications"] for s in proto)
    assert total_reps == 0, f"ProtoOrganelle baseline must not replicate (no V0); got {total_reps}"


def test_proto_organelle_liberation_replicates():
    """ProtoOrganelle with V0 override must replicate in adequate resources (liberation).

    Uses resource_initial_value=0.3 (medium pool) to ensure sufficient energy
    for the replication_threshold=0.8 to be reached. The biological claim is that
    adding V0 replication capability ENABLES replication — not that it works at
    all resource levels (resource sensitivity is measured by the experiment script).
    """
    cfg_str = make_config("proto_organelle", V0 | V1 | V2 | V3, 0.3, seed=0)
    result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 200, 200))
    last = result["samples"][-1]["snapshots"]
    proto = [s for s in last if s["archetype"] == "proto_organelle"]
    total_reps = sum(s["replications"] for s in proto)
    assert total_reps > 0, "Liberated ProtoOrganelle (V0+V1+V2+V3) must replicate at res=0.3"


# ---------------------------------------------------------------------------
# Experiment script structure
# ---------------------------------------------------------------------------


def test_archetype_conditions_has_all_viroid_levels():
    """ARCHETYPE_CONDITIONS must include all six Viroid capability levels."""
    viroid_bits = {bits for label, arch, bits in ARCHETYPE_CONDITIONS if arch == "viroid"}
    assert V0 in viroid_bits
    assert V0 | V1 in viroid_bits
    assert V0 | V1 | V2 in viroid_bits
    assert V0 | V1 | V2 | V3 in viroid_bits
    assert V0 | V1 | V2 | V3 | V4 in viroid_bits
    assert V0 | V1 | V2 | V3 | V4 | V5 in viroid_bits


def test_archetype_conditions_has_liberation_pair():
    """ARCHETYPE_CONDITIONS must include both ProtoOrganelle baseline and liberated."""
    proto_entries = [
        (label, bits) for label, arch, bits in ARCHETYPE_CONDITIONS if arch == "proto_organelle"
    ]
    labels = [e[0] for e in proto_entries]
    bits_set = {e[1] for e in proto_entries}
    assert None in bits_set, "ProtoOrganelle baseline (no override) must be present"
    assert V0 | V1 | V2 | V3 in bits_set, "Liberated ProtoOrganelle must be present"
    assert any("liberat" in lbl for lbl in labels)


def testmake_config_single_archetype_isolation():
    """make_config must only list the target archetype in enabled_archetypes."""
    for _, archetype, cap_bits in ARCHETYPE_CONDITIONS:
        cfg = json.loads(make_config(archetype, cap_bits, 0.1, seed=0))
        assert cfg["semi_life_config"]["enabled_archetypes"] == [archetype], (
            f"Isolation mode violated for archetype={archetype}"
        )


def testmake_config_uses_correct_resource_initial():
    """make_config must pass resource_initial_value to the top-level config."""
    for val in RESOURCE_INITIAL_VALUES.values():
        cfg = json.loads(make_config("viroid", V0, val, seed=0))
        assert cfg["resource_initial_value"] == pytest.approx(val)


# ---------------------------------------------------------------------------
# V4 — Response to stimuli
# ---------------------------------------------------------------------------


def test_v4_snapshot_has_policy_magnitude():
    """V4 entities must have non-zero policy_magnitude in snapshot."""
    cfg_str = make_config("viroid", V0 | V1 | V2 | V3 | V4, 0.3, seed=0)
    result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 100, 100))
    last = result["samples"][-1]["snapshots"]
    alive = [s for s in last if s["alive"] and s["archetype"] == "viroid"]
    assert alive, "No alive V4 viroid entities"
    assert all("policy_magnitude" in s for s in alive), "Missing policy_magnitude field"
    mean_pm = sum(s["policy_magnitude"] for s in alive) / len(alive)
    assert mean_pm > 0.0, f"V4 entities must have policy_magnitude > 0; got {mean_pm}"


def test_v4_active_capabilities_in_snapshot():
    """V4 entities must report correct active_capabilities bitmask."""
    bits = V0 | V1 | V2 | V3 | V4
    cfg_str = make_config("viroid", bits, 0.3, seed=0)
    result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 50, 50))
    last = result["samples"][-1]["snapshots"]
    alive = [s for s in last if s["alive"] and s["archetype"] == "viroid"]
    assert alive, "No alive V4 viroid entities"
    cap_values = {s["active_capabilities"] for s in alive}
    assert cap_values == {bits}, f"Expected active_capabilities == {bits}, got {cap_values}"


def test_v4_without_v3_has_zero_ii():
    """V4 without V3 must have II=0 (response adds no internal metabolism)."""
    cfg_str = make_config("viroid", V0 | V1 | V2 | V4, 0.3, seed=0)
    result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 100, 100))
    last = result["samples"][-1]["snapshots"]
    alive = [s for s in last if s["alive"] and s["archetype"] == "viroid"]
    assert alive, "No alive viroid entities"
    assert all(s["internalization_index"] == pytest.approx(0.0) for s in alive)


# ---------------------------------------------------------------------------
# V5 — Staged lifecycle
# ---------------------------------------------------------------------------


def test_v5_snapshot_has_stage_field():
    """V5 entities must have a 'stage' field in snapshot."""
    cfg_str = make_config("viroid", V0 | V1 | V2 | V3 | V4 | V5, 0.3, seed=0)
    result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 100, 100))
    last = result["samples"][-1]["snapshots"]
    alive = [s for s in last if s["alive"] and s["archetype"] == "viroid"]
    assert alive, "No alive V5 viroid entities"
    assert all("stage" in s for s in alive), "Missing stage field"
    stages = {s["stage"] for s in alive}
    valid_stages = {"dormant", "active", "dispersal"}
    assert stages.issubset(valid_stages), f"Invalid stages: {stages - valid_stages}"


def test_v5_without_v5_has_null_stage():
    """Entities without V5 must have stage=null in snapshot."""
    cfg_str = make_config("viroid", V0 | V1 | V2 | V3, 0.3, seed=0)
    result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 50, 50))
    last = result["samples"][-1]["snapshots"]
    alive = [s for s in last if s["alive"] and s["archetype"] == "viroid"]
    assert alive, "No alive viroid entities"
    assert all(s["stage"] is None for s in alive), "Non-V5 entities must have stage=null"


def test_v5_full_ladder_runs_without_crash():
    """Full V0..V5 ladder must complete 200 steps without errors."""
    cfg_str = make_config("viroid", V0 | V1 | V2 | V3 | V4 | V5, 0.3, seed=42)
    result = json.loads(life_transition.run_semi_life_v0_experiment_json(cfg_str, 200, 200))
    assert result["kind"] == "semi_life_v0"
    assert result["steps"] == 200
    last = result["samples"][-1]["snapshots"]
    assert len(last) > 0, "No entities in final snapshot"
