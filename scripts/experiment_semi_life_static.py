"""EXPLORATORY: Static vs dynamic resource field comparison.

Factorial design testing the interaction between resource field dynamics
and capability level. Key prediction: V4 chemotaxis is differentially
useful in dynamic (organism-driven gradients) vs static (no gradients) worlds.

Design:
  - Factor A: world_type ∈ {dynamic, static} (2 levels)
  - Factor B: archetype_condition (13 conditions from main experiment)
  - Factor C: harshness ∈ {rich, medium, sparse, scarce} (4 levels)
  - Seeds 100–199 (test range)
  - Total: 2 × 13 × 4 × 100 = 10,400 runs

This analysis is labeled EXPLORATORY per pre-registration Amendment 4.

Usage:
    uv run python scripts/experiment_semi_life_static.py \
        > experiments/semi_life_static_comparison.tsv
"""

from __future__ import annotations

import json
from pathlib import Path

import life_transition
from experiment_common import DEFAULT_MAX_WORKERS, log, make_config_dict, run_parallel

STEPS = 500
SAMPLE_EVERY = 500  # Only final step needed for comparison
SEEDS = list(range(100, 200))

V0 = 0x01
V1 = 0x02
V2 = 0x04
V3 = 0x08
V4 = 0x10
V5 = 0x20

RESOURCE_REGEN_RATE = 0.003

HARSHNESS = {
    "rich": 1.0,
    "medium": 0.3,
    "sparse": 0.1,
    "scarce": 0.05,
}

# Same conditions as main experiment
ARCHETYPE_CONDITIONS: list[tuple[str, str, int | None]] = [
    ("viroid_v0", "viroid", V0),
    ("viroid_v0v1", "viroid", V0 | V1),
    ("viroid_v0v1v2", "viroid", V0 | V1 | V2),
    ("viroid_v0v1v2v3", "viroid", V0 | V1 | V2 | V3),
    ("viroid_v0v1v2v3v4", "viroid", V0 | V1 | V2 | V3 | V4),
    ("viroid_v0v1v2v3v4v5", "viroid", V0 | V1 | V2 | V3 | V4 | V5),
    ("viroid_v4_sham", "viroid", V0 | V1 | V2 | V3),
    ("viroid_v5_sham", "viroid", V0 | V1 | V2 | V3 | V4),
    ("proto_baseline", "proto_organelle", None),
    ("proto_liberated", "proto_organelle", V0 | V1 | V2 | V3),
    ("virus_baseline", "virus", None),
    ("virus_v0v1v2", "virus", V0 | V1 | V2),
    ("virus_v0v1v2v3", "virus", V0 | V1 | V2 | V3),
]

TSV_COLUMNS = [
    "world_type",
    "condition",
    "harshness",
    "seed",
    "alive",
    "mean_energy",
    "mean_ii",
    "total_replications",
]

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def _load_archetype_config(archetype: str) -> dict:
    """Load calibrated SemiLife parameters for the given archetype."""
    path = _CONFIGS_DIR / f"semi_life_{archetype}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}


def run_one(
    world_type: str,
    condition: str,
    harshness: str,
    archetype: str,
    cap_bits: int | None,
    resource_initial: float,
    seed: int,
) -> list[str]:
    """Run one condition and return TSV row strings."""
    base_params = _load_archetype_config(archetype)
    config = make_config_dict(seed=seed, overrides={})
    config["enable_semi_life"] = True
    config["resource_regeneration_rate"] = RESOURCE_REGEN_RATE
    config["resource_initial_value"] = resource_initial
    config["resource_field_mode"] = world_type  # "dynamic" or "static"

    sl: dict = {**base_params}
    sl["enabled_archetypes"] = [archetype]
    if cap_bits is not None:
        sl["capability_overrides"] = {archetype: cap_bits}
    else:
        sl.pop("capability_overrides", None)
    config["semi_life_config"] = sl

    result = json.loads(
        life_transition.run_semi_life_v0_experiment_json(json.dumps(config), STEPS, SAMPLE_EVERY)
    )

    snaps = result["samples"][-1]["snapshots"]
    entities = [s for s in snaps if s["archetype"] == archetype]
    alive = [e for e in entities if e["alive"]]
    n = len(alive)
    mean_e = sum(s["maintenance_energy"] for s in alive) / n if n else 0.0
    mean_ii = sum(s["internalization_index"] for s in alive) / n if n else 0.0
    total_rep = sum(s["replications"] for s in entities)

    row = [
        world_type,
        condition,
        harshness,
        str(seed),
        str(n),
        f"{mean_e:.4f}",
        f"{mean_ii:.4f}",
        str(total_rep),
    ]
    return ["\t".join(row)]


def main() -> None:
    log(f"Life Transition v{life_transition.version()}")
    total = 2 * len(ARCHETYPE_CONDITIONS) * len(HARSHNESS) * len(SEEDS)
    log(f"EXPLORATORY: Static vs dynamic resource field — {total} runs")
    log(
        f"  2 world types × {len(ARCHETYPE_CONDITIONS)} conditions"
        f" × {len(HARSHNESS)} harshness × {len(SEEDS)} seeds"
    )
    log(f"Workers: {DEFAULT_MAX_WORKERS}")
    log("")

    print("\t".join(TSV_COLUMNS))
    _EXPERIMENTS_DIR.mkdir(exist_ok=True)

    tasks: list[tuple] = []
    for world_type in ["dynamic", "static"]:
        for condition, archetype, cap_bits in ARCHETYPE_CONDITIONS:
            for harshness, resource_initial in HARSHNESS.items():
                for seed in SEEDS:
                    tasks.append(
                        (
                            world_type,
                            condition,
                            harshness,
                            archetype,
                            cap_bits,
                            resource_initial,
                            seed,
                        )
                    )

    all_results = run_parallel(tasks, run_one, description="static field comparison runs")

    for rows in all_results:
        for row in rows:
            print(row)

    log("\nDone.")


if __name__ == "__main__":
    main()
