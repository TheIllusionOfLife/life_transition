"""SemiLife capability ladder experiment: V0 → V0+V1 → V0+V1+V2 → V0+V1+V2+V3.

For each archetype, runs capability levels in isolation (single archetype per run,
one capability level per run) across multiple resource harshness levels.

Primary comparisons:
  Viroid:        V0 | V0+V1 | V0+V1+V2 | V0+V1+V2+V3
  ProtoOrganelle: V1+V2+V3 (baseline) | V0+V1+V2+V3 ("liberation")
  Virus:         V0+V1 (baseline) | V0+V1+V2 | V0+V1+V2+V3

Resource harshness axis: resource_initial_value ∈ {1.0, 0.3, 0.1, 0.05}.
Lower values reduce the starting resource pool, creating immediate scarcity
(complementary to resource_regeneration_rate, which controls long-run replenishment).

Phase 1 uses calibration seeds (0–49). Final test seeds (100–199) are reserved for PR 5.

Usage:
    uv run python scripts/experiment_semi_life_v1v3.py > experiments/semi_life_v1v3_data.tsv

Output:
    TSV to stdout (one row per condition × seed × step).
    JSON sidecar per condition × seed to experiments/semi_life_v1v3_<cond>_<seed>.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import life_transition
from experiment_common import log, make_config_dict

STEPS = 500
SAMPLE_EVERY = 50
SEEDS = list(range(50))

# Capability bitmask constants (must match crate::semi_life::capability).
V0 = 0x01
V1 = 0x02
V2 = 0x04
V3 = 0x08

# Harshness axis: resource_initial_value per cell.
# 1.0 = full pool (10 000 total units); 0.05 = scarcest (500 total units).
RESOURCE_INITIAL_VALUES = {
    "rich": 1.0,
    "medium": 0.3,
    "sparse": 0.1,
    "scarce": 0.05,
}

# Regeneration rate is kept fixed across harshness levels so pool size is the
# only varying dimension in this sweep (cross-product with PR 5's shock axis).
RESOURCE_REGEN_RATE = 0.003

# Per-archetype conditions: list of (label_suffix, archetype_name, cap_bits | None).
# None means no capability_overrides entry → use archetype's baseline_capabilities.
ARCHETYPE_CONDITIONS: list[tuple[str, str, int | None]] = [
    # Viroid capability ladder
    ("viroid_v0", "viroid", V0),
    ("viroid_v0v1", "viroid", V0 | V1),
    ("viroid_v0v1v2", "viroid", V0 | V1 | V2),
    ("viroid_v0v1v2v3", "viroid", V0 | V1 | V2 | V3),
    # ProtoOrganelle: baseline (V1+V2+V3) vs liberation (V0 added)
    ("proto_baseline", "proto_organelle", None),
    ("proto_liberated", "proto_organelle", V0 | V1 | V2 | V3),
    # Virus: baseline (V0+V1) vs adding V2 and V3
    ("virus_baseline", "virus", None),
    ("virus_v0v1v2", "virus", V0 | V1 | V2),
    ("virus_v0v1v2v3", "virus", V0 | V1 | V2 | V3),
]

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

TSV_COLUMNS = [
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


def _load_archetype_config(archetype: str) -> dict:
    """Load calibrated SemiLife parameters for the given archetype.

    Falls back to defaults if no calibrated config exists.
    Strips '_'-prefixed metadata keys.
    """
    path = _CONFIGS_DIR / f"semi_life_{archetype}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    log(f"  WARNING: no calibrated config for {archetype}; using defaults")
    return {}


def _make_config(
    archetype: str,
    cap_bits: int | None,
    resource_initial: float,
    seed: int,
) -> str:
    """Build a full config JSON string for one condition/seed."""
    base_params = _load_archetype_config(archetype)
    config = make_config_dict(seed=seed, overrides={})
    config["enable_semi_life"] = True
    config["resource_regeneration_rate"] = RESOURCE_REGEN_RATE
    config["resource_initial_value"] = resource_initial

    sl: dict = {**base_params}
    sl["enabled_archetypes"] = [archetype]
    if cap_bits is not None:
        sl["capability_overrides"] = {archetype: cap_bits}
    else:
        sl.pop("capability_overrides", None)
    config["semi_life_config"] = sl
    return json.dumps(config)


def _aggregate(snapshots: list[dict], archetype: str) -> dict:
    """Compute per-archetype aggregate stats from one sample's snapshots."""
    entities = [s for s in snapshots if s["archetype"] == archetype]
    alive = [e for e in entities if e["alive"]]
    return {
        "alive": len(alive),
        "mean_energy": (sum(e["maintenance_energy"] for e in alive) / len(alive) if alive else 0.0),
        "mean_ii": (sum(e["internalization_index"] for e in alive) / len(alive) if alive else 0.0),
        "total_replications": sum(e["replications"] for e in entities),
        "total_failed": sum(e["failed_replications"] for e in entities),
        "capability_bits": alive[0]["active_capabilities"] if alive else 0,
    }


def run_one(
    condition: str,
    harshness: str,
    archetype: str,
    cap_bits: int | None,
    resource_initial: float,
    seed: int,
) -> None:
    """Run one condition/seed pair and print TSV rows."""
    config_json = _make_config(archetype, cap_bits, resource_initial, seed)
    result = json.loads(
        life_transition.run_semi_life_v0_experiment_json(config_json, STEPS, SAMPLE_EVERY)
    )

    _EXPERIMENTS_DIR.mkdir(exist_ok=True)
    sidecar = _EXPERIMENTS_DIR / f"semi_life_v1v3_{condition}_{harshness}_{seed}.json"
    sidecar.write_text(json.dumps(result))

    for sample in result["samples"]:
        step = sample["step"]
        world_rep = sample.get("replications_total", 0)
        agg = _aggregate(sample["snapshots"], archetype)
        row = [
            condition,
            harshness,
            str(seed),
            str(step),
            archetype,
            str(agg["capability_bits"]),
            str(agg["alive"]),
            f"{agg['mean_energy']:.4f}",
            f"{agg['mean_ii']:.4f}",
            str(agg["total_replications"]),
            str(agg["total_failed"]),
            str(world_rep),
        ]
        print("\t".join(row))


def main() -> None:
    log(f"Life Transition v{life_transition.version()}")
    log(
        f"SemiLife V1-V3 capability ladder: {STEPS} steps, sample every {SAMPLE_EVERY},"
        f" {len(SEEDS)} seeds"
    )
    log(f"Conditions: {len(ARCHETYPE_CONDITIONS)} archetype×capability combos")
    log(f"Harshness: {list(RESOURCE_INITIAL_VALUES.keys())} (resource_initial_value)")
    total = len(ARCHETYPE_CONDITIONS) * len(RESOURCE_INITIAL_VALUES) * len(SEEDS)
    log(f"Total runs: {total}")
    log("")

    print("\t".join(TSV_COLUMNS))

    done = 0
    for condition, archetype, cap_bits in ARCHETYPE_CONDITIONS:
        for harshness, resource_initial in RESOURCE_INITIAL_VALUES.items():
            for seed in SEEDS:
                run_one(condition, harshness, archetype, cap_bits, resource_initial, seed)
                done += 1
                if done % 50 == 0 or done == total:
                    log(f"  {done}/{total} runs done")

    log("\nDone.")


if __name__ == "__main__":
    main()
