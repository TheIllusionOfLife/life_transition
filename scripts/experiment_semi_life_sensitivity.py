"""EXPLORATORY: Parameter sensitivity sweep for SemiLife model.

One-at-a-time sweep: 6 parameters × 5 multipliers (0.5×, 0.75×, 1.0×, 1.5×, 2.0×),
Viroid archetype, 2 harshness levels (rich + sparse), 30 seeds (0–29).
Total: 6 × 5 × 2 × 30 = 1,800 runs.

This analysis is labeled EXPLORATORY per pre-registration Amendment 3.

Usage:
    uv run python scripts/experiment_semi_life_sensitivity.py \
        > experiments/semi_life_sensitivity_data.tsv
"""

from __future__ import annotations

import json
from pathlib import Path

import life_transition
from experiment_common import log, make_config_dict, run_parallel

STEPS = 500
SAMPLE_EVERY = 500  # Only final step needed for sensitivity
SEEDS = list(range(30))  # Calibration seeds 0-29

V0 = 0x01
V1 = 0x02
V2 = 0x04
V3 = 0x08
V4 = 0x10
V5 = 0x20

# Primary sweep: V0+V1+V2+V3 (full ladder through V3) — this is the key condition
# where all new mechanisms (leakage, damage, waste, metabolism) interact.
CAPABILITY_BITS = V0 | V1 | V2 | V3

# Extended sweep: V0..V5 for V4/V5-specific parameters.
CAPABILITY_BITS_FULL = V0 | V1 | V2 | V3 | V4 | V5

HARSHNESS = {
    "rich": 1.0,
    "sparse": 0.1,
}

RESOURCE_REGEN_RATE = 0.003

# Parameters to sweep, with their config key and default value.
# (param_name, config_key, default_value, capability_bits)
# V0–V3 params use CAPABILITY_BITS; V4/V5 params use CAPABILITY_BITS_FULL.
SWEEP_PARAMS: list[tuple[str, str, float, int]] = [
    ("energy_leakage_rate", "energy_leakage_rate", 0.005, CAPABILITY_BITS),
    ("boundary_decay_rate", "boundary_decay_rate", 0.002, CAPABILITY_BITS),
    ("env_damage_probability", "env_damage_probability", 0.05, CAPABILITY_BITS),
    ("overconsumption_waste_fraction", "overconsumption_waste_fraction", 0.3, CAPABILITY_BITS),
    ("regulator_cost_per_step", "regulator_cost_per_step", 0.0005, CAPABILITY_BITS),
    ("internal_conversion_rate", "internal_conversion_rate", 0.05, CAPABILITY_BITS),
    # V4/V5 params (Amendment 4): require full ladder to be active.
    ("v4_move_cost", "v4_move_cost", 0.01, CAPABILITY_BITS_FULL),
    ("v5_dormant_decay_mult", "v5_dormant_decay_mult", 0.2, CAPABILITY_BITS_FULL),
]

MULTIPLIERS = [0.5, 0.75, 1.0, 1.5, 2.0]

TSV_COLUMNS = [
    "param_name",
    "multiplier",
    "harshness",
    "seed",
    "alive",
    "mean_energy",
    "mean_ii",
    "total_replications",
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
    param_name: str,
    param_key: str,
    value: float,
    multiplier: float,
    harshness: str,
    resource_initial: float,
    seed: int,
    cap_bits: int = CAPABILITY_BITS,
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
    sl[param_key] = value
    config["semi_life_config"] = sl

    result = json.loads(
        life_transition.run_semi_life_v0_experiment_json(json.dumps(config), STEPS, SAMPLE_EVERY)
    )

    snaps = result["samples"][-1]["snapshots"]
    alive = [s for s in snaps if s["alive"] and s["archetype"] == "viroid"]
    n = len(alive)
    mean_e = sum(s["maintenance_energy"] for s in alive) / n if n else 0.0
    mean_ii = sum(s["internalization_index"] for s in alive) / n if n else 0.0
    total_rep = sum(s["replications"] for s in snaps if s["archetype"] == "viroid")

    row = [
        param_name,
        f"{multiplier:.2f}",
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
    total = len(SWEEP_PARAMS) * len(MULTIPLIERS) * len(HARSHNESS) * len(SEEDS)
    log(f"EXPLORATORY: Sensitivity sweep — {total} runs")
    log(
        f"  {len(SWEEP_PARAMS)} params × {len(MULTIPLIERS)} multipliers"
        f" × {len(HARSHNESS)} harshness × {len(SEEDS)} seeds"
    )
    log("")

    print("\t".join(TSV_COLUMNS))
    _EXPERIMENTS_DIR.mkdir(exist_ok=True)

    tasks: list[tuple] = []
    for param_name, param_key, default_val, cap_bits in SWEEP_PARAMS:
        for mult in MULTIPLIERS:
            value = default_val * mult
            for harshness, resource_initial in HARSHNESS.items():
                for seed in SEEDS:
                    tasks.append(
                        (
                            param_name,
                            param_key,
                            value,
                            mult,
                            harshness,
                            resource_initial,
                            seed,
                            cap_bits,
                        )
                    )

    all_results = run_parallel(tasks, run_one, description="sensitivity sweep runs")
    for rows in all_results:
        for row in rows:
            print(row)

    log("\nDone.")


if __name__ == "__main__":
    main()
