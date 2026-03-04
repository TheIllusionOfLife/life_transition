"""EXPLORATORY: V1 protection regime discovery sweep.

Sweeps env_damage_probability × env_damage_amount × harshness to find
conditions where V1 boundary provides net survival benefit over V0 alone.

Design:
  - env_damage_probability ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}
  - env_damage_amount ∈ {0.05, 0.10, 0.15, 0.25}
  - harshness ∈ {sparse, scarce, medium}
  - Comparison: Viroid V0 vs Viroid V0+V1 at each grid point
  - Seeds 0–39 for sweep phase

Theoretical crossover: composite hazard H = p_damage × damage_amount.
  - Low H: No V1 benefit (cost outweighs protection)
  - Mid H: V1 positive delta (protection regime)
  - High H: Both populations near floor (delta collapses)

Decision rule: Claim "V1 protects" only if positive delta is consistent
across seeds with CI excluding 0 in a contiguous hazard band.

Floor-resistant outcomes: alive-count AUC + time-to-extinction in addition
to alive at step 500.

Usage:
    uv run python scripts/experiment_semi_life_v1_sweep.py \\
        > experiments/semi_life_v1_sweep.tsv
"""

from __future__ import annotations

import json
from pathlib import Path

import life_transition
from experiment_common import log, make_config_dict, run_parallel

STEPS = 500
SAMPLE_EVERY = 50  # Need intermediate steps for AUC
SEEDS = list(range(40))  # Calibration seeds

V0 = 0x01
V1 = 0x02

# Damage parameter grid
ENV_DAMAGE_PROBS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
ENV_DAMAGE_AMOUNTS = [0.05, 0.10, 0.15, 0.25]

# Harshness levels where V1 protection matters most
HARSHNESS = {
    "medium": 0.3,
    "sparse": 0.1,
    "scarce": 0.05,
}

RESOURCE_REGEN_RATE = 0.003

TSV_COLUMNS = [
    "condition",
    "harshness",
    "env_damage_prob",
    "env_damage_amount",
    "composite_hazard",
    "seed",
    "alive_final",
    "alive_auc",
    "time_to_extinction",
    "total_replications",
    "mean_energy",
    "per_capita_replication_rate",
]

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def _load_archetype_config() -> dict:
    """Load calibrated Viroid parameters."""
    path = _CONFIGS_DIR / "semi_life_viroid.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def run_one(
    condition: str,
    cap_bits: int,
    harshness: str,
    resource_initial: float,
    env_damage_prob: float,
    env_damage_amount: float,
    seed: int,
) -> list[str]:
    """Run one condition and return TSV row strings."""
    base_params = _load_archetype_config()
    config = make_config_dict(seed=seed, overrides={})
    config["enable_semi_life"] = True
    config["resource_regeneration_rate"] = RESOURCE_REGEN_RATE
    config["resource_initial_value"] = resource_initial

    sl: dict = {**base_params}
    sl["enabled_archetypes"] = ["viroid"]
    sl["capability_overrides"] = {"viroid": cap_bits}
    sl["env_damage_probability"] = env_damage_prob
    sl["env_damage_amount"] = env_damage_amount
    config["semi_life_config"] = sl

    result = json.loads(
        life_transition.run_semi_life_v0_experiment_json(json.dumps(config), STEPS, SAMPLE_EVERY)
    )

    # Extract alive counts at each sampled step for AUC
    alive_series = []
    steps_series = []
    for sample in result["samples"]:
        snaps = sample["snapshots"]
        alive = sum(1 for s in snaps if s["alive"] and s["archetype"] == "viroid")
        alive_series.append(alive)
        steps_series.append(sample["step"])

    # Final step metrics
    final_snaps = result["samples"][-1]["snapshots"]
    alive_final = sum(1 for s in final_snaps if s["alive"] and s["archetype"] == "viroid")
    alive_entities = [s for s in final_snaps if s["alive"] and s["archetype"] == "viroid"]
    mean_energy = (
        sum(s["maintenance_energy"] for s in alive_entities) / len(alive_entities)
        if alive_entities
        else 0.0
    )
    total_rep = sum(s["replications"] for s in final_snaps if s["archetype"] == "viroid")

    # AUC via trapezoidal rule
    alive_auc = 0.0
    for i in range(1, len(steps_series)):
        dt = steps_series[i] - steps_series[i - 1]
        alive_auc += 0.5 * (alive_series[i - 1] + alive_series[i]) * dt

    # Time to extinction: first step where alive hits 0 (or STEPS if persists)
    time_to_extinction = STEPS
    for i, a in enumerate(alive_series):
        if a == 0:
            time_to_extinction = steps_series[i]
            break

    # Per-capita replication rate
    eps = 1e-6
    per_capita_rep = total_rep / (alive_auc + eps)

    composite_hazard = env_damage_prob * env_damage_amount

    row = [
        condition,
        harshness,
        f"{env_damage_prob:.2f}",
        f"{env_damage_amount:.2f}",
        f"{composite_hazard:.4f}",
        str(seed),
        str(alive_final),
        f"{alive_auc:.1f}",
        str(time_to_extinction),
        str(total_rep),
        f"{mean_energy:.4f}",
        f"{per_capita_rep:.6f}",
    ]
    return ["\t".join(row)]


def main() -> None:
    log(f"Life Transition v{life_transition.version()}")
    n_grid = len(ENV_DAMAGE_PROBS) * len(ENV_DAMAGE_AMOUNTS) * len(HARSHNESS)
    total = n_grid * 2 * len(SEEDS)  # 2 conditions: V0 and V0+V1
    log(f"EXPLORATORY: V1 protection regime sweep — {total} runs")
    log(
        f"  {len(ENV_DAMAGE_PROBS)} probs × {len(ENV_DAMAGE_AMOUNTS)} amounts"
        f" × {len(HARSHNESS)} harshness × 2 conditions × {len(SEEDS)} seeds"
    )
    log("")

    print("\t".join(TSV_COLUMNS))
    _EXPERIMENTS_DIR.mkdir(exist_ok=True)

    tasks: list[tuple] = []
    for env_prob in ENV_DAMAGE_PROBS:
        for env_amt in ENV_DAMAGE_AMOUNTS:
            for harshness, resource_initial in HARSHNESS.items():
                for cap_label, cap_bits in [("v0", V0), ("v0v1", V0 | V1)]:
                    for seed in SEEDS:
                        tasks.append(
                            (
                                cap_label,
                                cap_bits,
                                harshness,
                                resource_initial,
                                env_prob,
                                env_amt,
                                seed,
                            )
                        )

    all_results = run_parallel(tasks, run_one, description="V1 sweep runs")
    for rows in all_results:
        for row in rows:
            print(row)

    log("\nDone.")


if __name__ == "__main__":
    main()
