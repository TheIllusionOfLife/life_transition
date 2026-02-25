"""SemiLife parameter calibration sweep.

Sweeps key parameters for each archetype in isolation (single archetype per run)
to find stable baseline parameters before the full V0–V5 ladder experiments.

Sweep design (Phase 1 — coarse screen, 10 seeds):
  - Viroid (V0 only):           maintenance_cost × replication_threshold (3×3 = 9 combos)
  - Virus (V0+V1):              maintenance_cost × replication_threshold × boundary_decay_rate
                                (3×3×2 = 18 combos; boundary axis captures V1-specific drain)
  - ProtoOrganelle (V1+V2+V3):  maintenance_cost × internal_conversion_rate (3×3 = 9 combos)

Phase 2 — fine re-eval: top N_REEVAL combos per archetype re-run with REEVAL_SEEDS (30 seeds).
Final config is selected from Phase 2 results only, using lexicographic scoring.

Target: each archetype maintains stable population for ≥500 steps (survival_rate ≥ 0.8,
median alive ≥ 5 for genomic archetypes; survival_rate = 1.0 for ProtoOrganelle).

Saves best params to configs/semi_life_<archetype>.json.

Usage:
    uv run python scripts/experiment_semi_life_calibrate.py

Output:
    TSV to stdout with per-combo statistics (Phase 1 + Phase 2).
    Best config per archetype saved to configs/semi_life_<archetype>.json.
"""

import json
from itertools import product
from pathlib import Path
from statistics import mean, median

import life_transition
from experiment_common import log, make_config_dict

STEPS = 500
SAMPLE_EVERY = 50
# Phase 1: coarse screen (seeds 0–9)
SCREEN_SEEDS = list(range(10))
# Phase 2: fine re-eval on top combos (seeds 10–39, non-overlapping with screen)
REEVAL_SEEDS = list(range(10, 40))
# Number of top Phase-1 combos to re-evaluate in Phase 2
N_REEVAL = 3

# Rich resource environment for stable baseline calibration
RICH_RESOURCE_RATE = 0.01
# Minimum survival rate required before any config is accepted as "best".
# Matches the docstring target (≥ 0.8 for genomic, 1.0 for ProtoOrganelle).
# WARNING is emitted when the best combo falls below this floor.
MIN_SURVIVAL_RATE = 0.8

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

TSV_COLUMNS = [
    "phase",
    "archetype",
    "maintenance_cost",
    "sweep_param",
    "sweep_value",
    "sweep_param2",
    "sweep_value2",
    "survival_rate",
    "median_alive",
    "mean_alive",
    "median_final_energy",
    "n_seeds",
]

# Fixed SemiLife config fields shared across all archetypes (single-archetype isolation mode).
# Only the swept parameters are overridden per combo.
_BASE_SEMI_LIFE_CONFIG: dict = {
    "num_per_archetype": 10,
    "initial_energy": 0.5,
    "energy_capacity": 1.0,
    "replication_spawn_radius": 3.0,
    "resource_uptake_rate": 0.02,
    # V1 boundary (defaults; Virus sweep overrides boundary_decay_rate)
    "boundary_decay_rate": 0.002,
    "boundary_repair_rate": 0.01,
    "boundary_death_threshold": 0.1,
    "boundary_replication_min": 0.5,
    # V2 homeostasis
    "regulator_init": 1.0,
    "regulator_uptake_scale": 1.0,
    "regulator_cost_per_step": 0.0005,
    # V3 metabolism
    "internal_pool_init_fraction": 0.5,
    "internal_pool_capacity": 1.0,
    "internal_pool_uptake_rate": 0.01,
    # Prion (unused for genomic archetypes, but required by SemiLifeConfig)
    "prion_contact_radius": 5.0,
    "prion_conversion_prob": 0.05,
    "prion_fragmentation_loss": 0.01,
    "prion_dilution_death_energy": 0.0,
    "prion_contact_gain": 0.01,
    # Safe replication defaults (overridden per sweep for genomic archetypes)
    "replication_threshold": 0.8,
    "replication_cost": 0.3,
    "maintenance_cost": 0.001,
    "internal_conversion_rate": 0.05,
}


def _build_config(archetype: str, seed: int, overrides: dict) -> str:
    """Build a full config JSON string for a single-archetype calibration run."""
    config = make_config_dict(seed=seed, overrides={})
    config["enable_semi_life"] = True
    config["resource_regeneration_rate"] = RICH_RESOURCE_RATE
    config["semi_life_config"] = {
        **_BASE_SEMI_LIFE_CONFIG,
        "enabled_archetypes": [archetype],
        **overrides,
    }
    return json.dumps(config)


def _run_one(archetype: str, seed: int, overrides: dict) -> tuple[int, float]:
    """Run one combo/seed; return (alive_count, mean_final_energy) at the final step."""
    config_json = _build_config(archetype, seed, overrides)
    result = json.loads(
        life_transition.run_semi_life_v0_experiment_json(config_json, STEPS, SAMPLE_EVERY)
    )
    last_snapshots = result["samples"][-1]["snapshots"]
    alive = [s for s in last_snapshots if s["alive"]]
    alive_count = len(alive)
    mean_energy = mean(s["maintenance_energy"] for s in alive) if alive else 0.0
    return alive_count, mean_energy


def _eval_combo(archetype: str, overrides: dict, seeds: list[int]) -> dict:
    """Evaluate one parameter combo across *seeds*.

    Returns a stats dict with keys:
        survival_rate, median_alive, mean_alive, median_final_energy, n_seeds

    Note: median_final_energy is computed only over seeds where alive > 0
    (survivorship bias by design — energy of extinct populations is undefined).
    Two configs with equal survival_rate and median_alive but different extinction
    patterns can be separated by this tiebreaker, favouring configs with higher
    energy among survivors.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")

    counts = []
    energies = []
    for seed in seeds:
        alive_count, mean_e = _run_one(archetype, seed, overrides)
        counts.append(alive_count)
        if alive_count > 0:
            energies.append(mean_e)

    surv = sum(1 for c in counts if c > 0) / len(counts)
    med_alive = float(median(counts))
    mean_alive = mean(counts)
    med_energy = float(median(energies)) if energies else 0.0
    return {
        "survival_rate": surv,
        "median_alive": med_alive,
        "mean_alive": mean_alive,
        "median_final_energy": med_energy,
        "n_seeds": len(seeds),
    }


def _score(stats: dict) -> tuple[float, float, float]:
    """Lexicographic score tuple: (survival_rate, median_alive, median_final_energy).

    Higher is better on all three axes. survival_rate is the primary axis to avoid
    selecting near-collapse configs that happen to have a high median count from
    a single long-lived survivor.
    """
    return (stats["survival_rate"], stats["median_alive"], stats["median_final_energy"])


def _tsv_row(phase: str, archetype: str, combo: dict, stats: dict) -> str:
    """Format one TSV row for a (phase, archetype, combo, stats) tuple."""
    p1k = combo.get("sweep_param", "")
    p1v = combo.get("sweep_value", "")
    p2k = combo.get("sweep_param2", "")
    p2v = combo.get("sweep_value2", "")
    return "\t".join(
        [
            phase,
            archetype,
            str(combo.get("maintenance_cost", "")),
            p1k,
            str(p1v),
            p2k,
            str(p2v),
            f"{stats['survival_rate']:.4f}",
            f"{stats['median_alive']:.1f}",
            f"{stats['mean_alive']:.2f}",
            f"{stats['median_final_energy']:.4f}",
            str(stats["n_seeds"]),
        ]
    )


# ---------------------------------------------------------------------------
# Sweep generators
# ---------------------------------------------------------------------------


def _genomic_combos(archetype: str) -> list[dict]:
    """Generate combo dicts for Viroid (V0 only) sweep."""
    combos = []
    for mc, rt in product([0.0005, 0.001, 0.002], [0.6, 0.7, 0.8]):
        combos.append(
            {
                "maintenance_cost": mc,
                "replication_threshold": rt,
                # Fix cost/threshold ratio to isolate threshold level, not energy economics.
                # Sensitivity to this ratio (0.35/0.45/0.55) should be checked separately
                # for winning candidates before publishing final configs.
                "replication_cost": round(rt * 0.45, 4),
                "sweep_param": "replication_threshold",
                "sweep_value": rt,
                "sweep_param2": "",
                "sweep_value2": "",
            }
        )
    return combos


def _virus_combos() -> list[dict]:
    """Generate combo dicts for Virus (V0+V1) sweep.

    Adds boundary_decay_rate as a second sweep axis to capture the V1-specific
    energy drain that Viroid does not experience. Cross-archetype comparability
    is preserved by keeping the same maintenance_cost × replication_threshold grid.
    """
    combos = []
    for mc, rt, bdr in product(
        [0.0005, 0.001, 0.002],
        [0.6, 0.7, 0.8],
        [0.001, 0.003],  # low vs high boundary decay
    ):
        combos.append(
            {
                "maintenance_cost": mc,
                "replication_threshold": rt,
                "replication_cost": round(rt * 0.45, 4),
                "boundary_decay_rate": bdr,
                "sweep_param": "replication_threshold",
                "sweep_value": rt,
                "sweep_param2": "boundary_decay_rate",
                "sweep_value2": bdr,
            }
        )
    return combos


def _proto_organelle_combos() -> list[dict]:
    """Generate combo dicts for ProtoOrganelle (V1+V2+V3, no V0) sweep.

    ProtoOrganelle cannot replicate: calibrate for 100% survival of initial entities.
    Sweep internal_conversion_rate (V3) to find the minimum metabolism that sustains
    energy against maintenance drain.
    """
    combos = []
    for mc, cr in product([0.0005, 0.001, 0.002], [0.02, 0.05, 0.1]):
        combos.append(
            {
                "maintenance_cost": mc,
                "internal_conversion_rate": cr,
                "sweep_param": "internal_conversion_rate",
                "sweep_value": cr,
                "sweep_param2": "",
                "sweep_value2": "",
            }
        )
    return combos


# ---------------------------------------------------------------------------
# Phase 1 + Phase 2 sweep driver
# ---------------------------------------------------------------------------


def _sweep_archetype(
    archetype: str, combos: list[dict]
) -> tuple[dict, list[tuple[str, dict, dict]]]:
    """Run Phase 1 screen then Phase 2 re-eval for one archetype.

    Returns (best_overrides, tsv_rows) where tsv_rows is a list of
    (phase, combo, stats) tuples ready for printing.
    """
    log(f"  Phase 1 screen: {len(combos)} combos × {len(SCREEN_SEEDS)} seeds")

    phase1: list[tuple[dict, dict]] = []
    for combo in combos:
        overrides = {k: v for k, v in combo.items() if not k.startswith("sweep_")}
        stats = _eval_combo(archetype, overrides, SCREEN_SEEDS)
        phase1.append((combo, stats))
        p2_suffix = (
            f"  {combo['sweep_param2']}={combo['sweep_value2']}"
            if combo.get("sweep_param2")
            else ""
        )
        log(
            f"    mc={combo['maintenance_cost']:.4f}"
            f"  {combo['sweep_param']}={combo['sweep_value']}"
            + p2_suffix
            + f"  → surv={stats['survival_rate']:.0%}"
            f"  med={stats['median_alive']:.0f}"
            f"  e={stats['median_final_energy']:.3f}"
        )

    # Sort Phase 1 by score descending; pick top N_REEVAL for Phase 2.
    # Python sort is stable: ties are broken by original insertion order from
    # product(...), which follows the parameter grid definition order. This is
    # deterministic but may exclude statistically equivalent combos. Acceptable
    # for a coarse screen whose purpose is only to narrow the re-eval candidate set.
    phase1.sort(key=lambda t: _score(t[1]), reverse=True)
    top_combos = phase1[:N_REEVAL]

    log(f"  Phase 2 re-eval: top {N_REEVAL} combos × {len(REEVAL_SEEDS)} seeds")

    phase2: list[tuple[dict, dict]] = []
    for combo, _p1_stats in top_combos:
        overrides = {k: v for k, v in combo.items() if not k.startswith("sweep_")}
        stats = _eval_combo(archetype, overrides, REEVAL_SEEDS)
        phase2.append((combo, stats))
        p2_suffix2 = (
            f"  {combo['sweep_param2']}={combo['sweep_value2']}"
            if combo.get("sweep_param2")
            else ""
        )
        log(
            f"    mc={combo['maintenance_cost']:.4f}"
            f"  {combo['sweep_param']}={combo['sweep_value']}"
            + p2_suffix2
            + f"  → surv={stats['survival_rate']:.0%}"
            f"  med={stats['median_alive']:.0f}"
            f"  e={stats['median_final_energy']:.3f}"
        )

    # Final selection: best Phase-2 combo by lexicographic score; require MIN_SURVIVAL_RATE
    phase2.sort(key=lambda t: _score(t[1]), reverse=True)
    best_combo, best_stats = phase2[0]
    if best_stats["survival_rate"] < MIN_SURVIVAL_RATE:
        log(
            f"  WARNING: best survival_rate={best_stats['survival_rate']:.0%} < "
            f"{MIN_SURVIVAL_RATE:.0%} threshold — calibration may be degenerate. "
            "Consider adjusting resource_rate or sweep ranges."
        )

    best_overrides = {k: v for k, v in best_combo.items() if not k.startswith("sweep_")}

    # Build TSV rows: all Phase 1, then re-eval Phase 2
    tsv_rows: list[tuple[str, dict, dict]] = []
    for combo, stats in phase1:
        tsv_rows.append(("screen", combo, stats))
    for combo, stats in phase2:
        tsv_rows.append(("reeval", combo, stats))

    return best_overrides, tsv_rows


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def _save_archetype_config(archetype: str, overrides: dict) -> None:
    """Save best calibrated params as configs/semi_life_<archetype>.json.

    The saved file is a complete SemiLifeConfig-compatible dict (drop-in for
    the semi_life_config key in main experiment configs), plus '_'-prefixed
    metadata keys (stripped by experiment_common pattern).
    """
    _CONFIGS_DIR.mkdir(exist_ok=True)
    path = _CONFIGS_DIR / f"semi_life_{archetype}.json"
    config = {
        "_generated_by": "experiment_semi_life_calibrate.py",
        "_archetype": archetype,
        "_screen_seeds": SCREEN_SEEDS,
        "_reeval_seeds": REEVAL_SEEDS,
        "_resource_rate": RICH_RESOURCE_RATE,
        "_target": (
            "survival_rate >= 0.8, median_alive >= 5 at step 500 (genomic); "
            "100% survival (ProtoOrganelle)"
        ),
        **_BASE_SEMI_LIFE_CONFIG,
        "enabled_archetypes": [archetype],
        **overrides,
    }
    path.write_text(json.dumps(config, indent=2) + "\n")
    log(f"  → Saved {path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    log(f"Life Transition v{life_transition.version()}")
    log(
        f"SemiLife calibration: {STEPS} steps, sample_every={SAMPLE_EVERY}"
        f", screen={len(SCREEN_SEEDS)} seeds, reeval={len(REEVAL_SEEDS)} seeds"
    )
    log(f"Environment: rich (resource_rate={RICH_RESOURCE_RATE})")
    log("")

    print("\t".join(TSV_COLUMNS))

    archetypes_combos: list[tuple[str, list[dict]]] = [
        ("viroid", _genomic_combos("viroid")),
        ("virus", _virus_combos()),
        ("proto_organelle", _proto_organelle_combos()),
    ]

    for archetype, combos in archetypes_combos:
        log(f"\n[{archetype.upper()}]")
        best_overrides, tsv_rows = _sweep_archetype(archetype, combos)
        log(f"  Best overrides: {best_overrides}")
        _save_archetype_config(archetype, best_overrides)
        for phase, combo, stats in tsv_rows:
            print(_tsv_row(phase, archetype, combo, stats))

    log("\nCalibration complete.")


if __name__ == "__main__":
    main()
