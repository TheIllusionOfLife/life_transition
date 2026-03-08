"""SemiLife competition experiment: Viroid (V0) vs Plasmid (V0+V1+V2+V3) in the same world.

Both archetypes inhabit a single shared environment simultaneously. The experiment
measures whether higher capability (V0+V1+V2+V3 plasmid) out-competes the minimal
replicator (V0 viroid) and whether the advantage is attributable specifically to V3.

Primary condition:  viroid=V0,       plasmid=V0+V1+V2+V3
Ablation condition: viroid=V0,       plasmid=V0+V1+V2  (V3 withheld)

Design:
  2 conditions × 4 harshness levels × 100 seeds = 800 runs (calibration seeds 0–99)

Usage:
    uv run python scripts/experiment_semi_life_competition.py \
        > experiments/semi_life_competition.tsv

Output:
    TSV to stdout (one row per condition × harshness × seed × step).
    JSON sidecar per run to experiments/competition_{condition}_{harshness}_{seed}.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import life_transition
from experiment_common import DEFAULT_MAX_WORKERS, log, make_config_dict, run_parallel

STEPS = 500
SAMPLE_EVERY = 50
SEEDS = list(range(100))

# Capability bitmask constants (must match crate::semi_life::capability).
V0 = 0x01
V1 = 0x02
V2 = 0x04
V3 = 0x08

# Harshness axis: resource_initial_value per cell.
RESOURCE_INITIAL_VALUES = {
    "rich": 1.0,
    "medium": 0.3,
    "sparse": 0.1,
    "scarce": 0.05,
}

RESOURCE_REGEN_RATE = 0.003

# Competition conditions: (condition_label, viroid_caps, plasmid_caps).
COMPETITION_CONDITIONS: list[tuple[str, int, int]] = [
    # Primary: asymmetric — viroid minimal vs plasmid fully capable
    ("primary", V0, V0 | V1 | V2 | V3),
    # Ablation: viroid minimal vs plasmid without V3 (isolates V3 advantage)
    ("ablation", V0, V0 | V1 | V2),
]

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

TSV_COLUMNS = [
    "condition",
    "harshness",
    "seed",
    "step",
    "viroid_alive",
    "plasmid_alive",
    "viroid_replications",
    "plasmid_replications",
    "frequency_ratio",  # plasmid_alive / (viroid_alive + plasmid_alive), nan if both 0
]


def _load_archetype_config(archetype: str) -> dict:
    """Load calibrated SemiLife parameters for the given archetype.

    Falls back to empty dict (simulation defaults) if no calibrated config exists.
    Strips '_'-prefixed metadata keys.
    """
    path = _CONFIGS_DIR / f"semi_life_{archetype}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    log(f"  WARNING: no calibrated config for {archetype}; using defaults")
    return {}


def compute_frequency_ratio(plasmid_alive: int, viroid_alive: int) -> float | None:
    """Return plasmid frequency in the combined population.

    Returns plasmid_alive / (plasmid_alive + viroid_alive), or None if both are zero.
    """
    total = plasmid_alive + viroid_alive
    if total == 0:
        return None
    return plasmid_alive / total


def make_competition_config(
    viroid_caps: int,
    plasmid_caps: int,
    resource_initial: float,
    seed: int,
) -> str:
    """Build a full config JSON string for one competition condition/seed.

    Both archetypes share the same world. Viroid's calibrated params are used
    as the base SemiLife configuration (they were calibrated for stability).
    """
    viroid_params = _load_archetype_config("viroid")
    config = make_config_dict(seed=seed, overrides={})
    config["enable_semi_life"] = True
    config["resource_regeneration_rate"] = RESOURCE_REGEN_RATE
    config["resource_initial_value"] = resource_initial

    sl: dict = {**viroid_params}
    sl["enabled_archetypes"] = ["viroid", "plasmid"]
    sl["capability_overrides"] = {"viroid": viroid_caps, "plasmid": plasmid_caps}
    config["semi_life_config"] = sl
    return json.dumps(config)


def _aggregate_archetype(snapshots: list[dict], archetype: str) -> dict:
    """Extract alive count and cumulative replications for one archetype."""
    entities = [s for s in snapshots if s["archetype"] == archetype]
    alive = [e for e in entities if e["alive"]]
    total_replications = sum(e["replications"] for e in entities)
    return {
        "alive": len(alive),
        "total_replications": total_replications,
    }


def run_one(
    condition: str,
    harshness: str,
    viroid_caps: int,
    plasmid_caps: int,
    resource_initial: float,
    seed: int,
) -> list[str]:
    """Run one competition condition/seed pair and return TSV row strings."""
    config_json = make_competition_config(viroid_caps, plasmid_caps, resource_initial, seed)
    result = json.loads(
        life_transition.run_semi_life_v0_experiment_json(config_json, STEPS, SAMPLE_EVERY)
    )

    sidecar = _EXPERIMENTS_DIR / f"competition_{condition}_{harshness}_{seed}.json"
    sidecar.write_text(json.dumps(result), encoding="utf-8")

    rows: list[str] = []
    for sample in result["samples"]:
        step = sample["step"]
        snapshots = sample["snapshots"]

        viroid_agg = _aggregate_archetype(snapshots, "viroid")
        plasmid_agg = _aggregate_archetype(snapshots, "plasmid")

        ratio = compute_frequency_ratio(plasmid_agg["alive"], viroid_agg["alive"])
        ratio_str = "nan" if ratio is None else f"{ratio:.6f}"

        row = [
            condition,
            harshness,
            str(seed),
            str(step),
            str(viroid_agg["alive"]),
            str(plasmid_agg["alive"]),
            str(viroid_agg["total_replications"]),
            str(plasmid_agg["total_replications"]),
            ratio_str,
        ]
        rows.append("\t".join(row))
    return rows


def main() -> None:
    log(f"Life Transition v{life_transition.version()}")
    log(f"SemiLife Competition: {STEPS} steps, sample every {SAMPLE_EVERY}, {len(SEEDS)} seeds")
    log(
        "Conditions: primary (viroid=V0 vs plasmid=V0+V1+V2+V3),"
        " ablation (viroid=V0 vs plasmid=V0+V1+V2)"
    )
    log(f"Harshness: {list(RESOURCE_INITIAL_VALUES.keys())} (resource_initial_value)")
    total = len(COMPETITION_CONDITIONS) * len(RESOURCE_INITIAL_VALUES) * len(SEEDS)
    log(f"Total runs: {total}")
    log(f"Workers: {DEFAULT_MAX_WORKERS}")
    log("")

    print("\t".join(TSV_COLUMNS))

    _EXPERIMENTS_DIR.mkdir(exist_ok=True)

    tasks: list[tuple] = []
    for condition, viroid_caps, plasmid_caps in COMPETITION_CONDITIONS:
        for harshness, resource_initial in RESOURCE_INITIAL_VALUES.items():
            for seed in SEEDS:
                tasks.append(
                    (condition, harshness, viroid_caps, plasmid_caps, resource_initial, seed)
                )

    all_results = run_parallel(tasks, run_one, description="competition runs")

    for rows in all_results:
        for row in rows:
            print(row)

    log("\nDone.")


if __name__ == "__main__":
    main()
