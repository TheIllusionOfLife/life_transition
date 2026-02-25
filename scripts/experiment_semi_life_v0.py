"""SemiLife V0 survival sweep experiment.

Sweeps resource_regeneration_rate across 4 harshness levels for 3 archetypes
(Viroid, Virus, ProtoOrganelle) with 50 calibration seeds each.

Usage:
    uv run python scripts/experiment_semi_life_v0.py > experiments/semi_life_v0_data.tsv

Output:
    TSV to stdout (one row per archetype × step × seed × condition).
    JSON sidecar per condition to experiments/semi_life_v0_<condition>_<seed>.json.
"""

import json
from pathlib import Path

import life_transition
from experiment_common import log, make_config_dict

# Experiment parameters (calibration set, seeds 0–49)
STEPS = 500
SAMPLE_EVERY = 25
SEEDS = list(range(50))

# Resource harshness: map density label → resource_regeneration_rate
RESOURCE_DENSITIES = {
    "rich": 0.01,
    "medium": 0.006,
    "sparse": 0.003,
    "scarce": 0.001,
}

# Archetypes to test (subset of the 5-archetype full set for calibration)
ARCHETYPES = ["viroid", "virus", "proto_organelle"]

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

TSV_COLUMNS = [
    "condition",
    "seed",
    "step",
    "archetype",
    "alive",
    "mean_energy",
    "mean_ii",
    "snapshot_replications",  # per-archetype sum from current snapshots (may miss pruned)
    "total_failed",
    "world_replications_total",  # monotonic world-level cumulative count (never decrements)
]


def make_semi_life_config(resource_rate: float) -> dict:
    """Build a config dict with SemiLife enabled and given resource rate."""
    config = make_config_dict(seed=0, overrides={})
    config["enable_semi_life"] = True
    config["resource_regeneration_rate"] = resource_rate
    # Run with all 3 calibration archetypes simultaneously
    config["semi_life_config"] = {
        "enabled_archetypes": ARCHETYPES,
        "num_per_archetype": 10,
        "initial_energy": 0.5,
        "energy_capacity": 1.0,
        "maintenance_cost": 0.001,
        "replication_threshold": 0.8,
        "replication_cost": 0.3,
        "replication_spawn_radius": 3.0,
        "resource_uptake_rate": 0.02,
        "boundary_decay_rate": 0.002,
        "boundary_repair_rate": 0.01,
        "boundary_death_threshold": 0.1,
        "boundary_replication_min": 0.5,
        "regulator_init": 1.0,
        "regulator_uptake_scale": 1.0,
        "regulator_cost_per_step": 0.0005,
        "internal_pool_init_fraction": 0.5,
        "internal_pool_capacity": 1.0,
        "internal_conversion_rate": 0.05,
        "internal_pool_uptake_rate": 0.01,
        "prion_contact_radius": 5.0,
        "prion_conversion_prob": 0.05,
        "prion_fragmentation_loss": 0.01,
        "prion_dilution_death_energy": 0.0,
    }
    return config


def aggregate_by_archetype(snapshots: list[dict]) -> dict[str, dict]:
    """Compute per-archetype alive count, mean energy, mean II, and replication totals."""
    by_arch: dict[str, list[dict]] = {}
    for snap in snapshots:
        arch = snap["archetype"]
        by_arch.setdefault(arch, []).append(snap)

    result: dict[str, dict] = {}
    for arch, entities in by_arch.items():
        alive = [e for e in entities if e["alive"]]
        result[arch] = {
            "alive": len(alive),
            "mean_energy": (
                sum(e["maintenance_energy"] for e in alive) / len(alive) if alive else 0.0
            ),
            "mean_ii": (
                sum(e["internalization_index"] for e in alive) / len(alive) if alive else 0.0
            ),
            "total_replications": sum(e["replications"] for e in entities),
            "total_failed": sum(e["failed_replications"] for e in entities),
        }
    return result


def run_condition(condition: str, resource_rate: float, seed: int) -> None:
    """Run one condition/seed pair and print TSV rows."""
    config = make_semi_life_config(resource_rate)
    config["seed"] = seed
    config_json = json.dumps(config)

    result_json = life_transition.run_semi_life_v0_experiment_json(config_json, STEPS, SAMPLE_EVERY)
    result = json.loads(result_json)

    # Save JSON sidecar
    _EXPERIMENTS_DIR.mkdir(exist_ok=True)
    sidecar = _EXPERIMENTS_DIR / f"semi_life_v0_{condition}_{seed}.json"
    sidecar.write_text(result_json)

    for sample in result["samples"]:
        step = sample["step"]
        world_replications_total = sample.get("replications_total", 0)
        agg = aggregate_by_archetype(sample["snapshots"])
        for arch in ARCHETYPES:
            stats = agg.get(
                arch,
                {
                    "alive": 0,
                    "mean_energy": 0.0,
                    "mean_ii": 0.0,
                    "total_replications": 0,
                    "total_failed": 0,
                },
            )
            row = [
                condition,
                str(seed),
                str(step),
                arch,
                str(stats["alive"]),
                f"{stats['mean_energy']:.4f}",
                f"{stats['mean_ii']:.4f}",
                str(stats["total_replications"]),
                str(stats["total_failed"]),
                str(world_replications_total),
            ]
            print("\t".join(row))


def main() -> None:
    log(f"Life Transition v{life_transition.version()}")
    log(
        f"SemiLife V0 sweep: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds 0–{len(SEEDS) - 1} (n={len(SEEDS)})"
    )
    log(f"Resource densities: {list(RESOURCE_DENSITIES.keys())}")
    log(f"Archetypes: {ARCHETYPES}")
    log("")

    print("\t".join(TSV_COLUMNS))

    total = len(RESOURCE_DENSITIES) * len(SEEDS)
    done = 0
    for condition, resource_rate in RESOURCE_DENSITIES.items():
        for seed in SEEDS:
            run_condition(condition, resource_rate, seed)
            done += 1
            if done % 10 == 0 or done == total:
                log(f"  {done}/{total} conditions done")

    log("Done.")


if __name__ == "__main__":
    main()
