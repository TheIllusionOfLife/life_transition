"""SemiLife shock/perturbation experiment: periodic resource crashes.

Evaluates recovery resilience of V0–V3 capability levels under periodic
environmental stress. Runs the same 9 archetype×capability conditions as
experiment_semi_life_v1v3.py, but with cyclic resource modulation
(environment_cycle_period / environment_cycle_low_rate).

Shock parameters:
  shock_period ∈ {200, 50}  — environment_cycle_period (steps per cycle)
  shock_magnitude = 0.8     — low phase uses 20% of normal regen rate
  Primary harshness: sparse (resource_initial_value = 0.1)

No-shock baseline: use TSV from experiment_semi_life_v1v3.py (no cycle).

Usage:
    uv run python scripts/experiment_semi_life_shocks.py > experiments/semi_life_shocks.tsv

Output:
    TSV to stdout (v1v3 columns + shock_period).
    JSON sidecar per condition to experiments/semi_life_shocks_<cond>_<period>_<seed>.json.
"""

from __future__ import annotations

import json

import life_transition
from experiment_common import log
from experiment_semi_life_v1v3 import (
    _EXPERIMENTS_DIR,
    ARCHETYPE_CONDITIONS,
    RESOURCE_REGEN_RATE,
    SAMPLE_EVERY,
    SEEDS,
    STEPS,
    _aggregate,
    make_config,
)

# Shock parameters
SHOCK_PERIODS = [200, 50]  # environment_cycle_period values
SHOCK_MAGNITUDE = 0.8  # 80% resource reduction during low phase
PRIMARY_HARSHNESS = "sparse"
PRIMARY_RESOURCE_INITIAL = 0.1

# Low-phase regen rate: normal rate × (1 − magnitude)
SHOCK_LOW_RATE = RESOURCE_REGEN_RATE * (1.0 - SHOCK_MAGNITUDE)

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
    "shock_period",
]


def make_shock_config(
    archetype: str,
    cap_bits: int | None,
    resource_initial: float,
    shock_period: int,
    seed: int,
) -> str:
    """Build config JSON with cyclic resource modulation for shock experiments."""
    base_json = make_config(archetype, cap_bits, resource_initial, seed)
    config = json.loads(base_json)
    config["environment_cycle_period"] = shock_period
    config["environment_cycle_low_rate"] = SHOCK_LOW_RATE
    return json.dumps(config)


def run_one(
    condition: str,
    archetype: str,
    cap_bits: int | None,
    shock_period: int,
    seed: int,
) -> None:
    """Run one shock condition/seed pair and print TSV rows."""
    config_json = make_shock_config(
        archetype, cap_bits, PRIMARY_RESOURCE_INITIAL, shock_period, seed
    )
    result = json.loads(
        life_transition.run_semi_life_v0_experiment_json(config_json, STEPS, SAMPLE_EVERY)
    )

    sidecar = _EXPERIMENTS_DIR / f"semi_life_shocks_{condition}_{shock_period}_{seed}.json"
    sidecar.write_text(json.dumps(result), encoding="utf-8")

    for sample in result["samples"]:
        step = sample["step"]
        world_rep = sample.get("replications_total", 0)
        agg = _aggregate(sample["snapshots"], archetype)
        row = [
            condition,
            PRIMARY_HARSHNESS,
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
            str(shock_period),
        ]
        print("\t".join(row), flush=True)


def main() -> None:
    log("Life Transition — SemiLife shock experiment")
    log(f"  Steps={STEPS}, sample_every={SAMPLE_EVERY}, seeds={len(SEEDS)}")
    log(f"  Harshness: {PRIMARY_HARSHNESS} (resource_initial_value={PRIMARY_RESOURCE_INITIAL})")
    log(f"  Shock periods: {SHOCK_PERIODS}, magnitude={SHOCK_MAGNITUDE}")
    log(f"  Low-phase regen rate: {SHOCK_LOW_RATE:.5f}")
    total = len(ARCHETYPE_CONDITIONS) * len(SHOCK_PERIODS) * len(SEEDS)
    log(f"  Total runs: {total}")
    log("")

    print("\t".join(TSV_COLUMNS))

    _EXPERIMENTS_DIR.mkdir(exist_ok=True)
    done = 0
    for condition, archetype, cap_bits in ARCHETYPE_CONDITIONS:
        for shock_period in SHOCK_PERIODS:
            for seed in SEEDS:
                run_one(condition, archetype, cap_bits, shock_period, seed)
                done += 1
                if done % 50 == 0 or done == total:
                    log(f"  {done}/{total} runs done")

    log("\nDone.")


if __name__ == "__main__":
    main()
