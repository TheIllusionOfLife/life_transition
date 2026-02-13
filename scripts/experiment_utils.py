"""Shared utilities for experiment scripts."""

import json
import sys

import digital_life

TUNED_BASELINE = {
    "boundary_decay_base_rate": 0.001,
    "boundary_repair_rate": 0.05,
    "metabolic_viability_floor": 0.1,
    "crowding_neighbor_threshold": 50.0,
    "homeostasis_decay_rate": 0.01,
    "growth_maturation_steps": 200,
    "growth_immature_metabolic_efficiency": 0.3,
    "resource_regeneration_rate": 0.01,
}

CONDITIONS = {
    "normal": {},
    "no_metabolism": {"enable_metabolism": False},
    "no_boundary": {"enable_boundary_maintenance": False},
    "no_homeostasis": {"enable_homeostasis": False},
    "no_response": {"enable_response": False},
    "no_reproduction": {"enable_reproduction": False},
    "no_evolution": {"enable_evolution": False},
    "no_growth": {"enable_growth": False},
}


def log(msg: str) -> None:
    """Write a message to stderr for progress reporting."""
    print(msg, file=sys.stderr)


def make_config(seed: int, *override_dicts: dict) -> str:
    """Build a JSON config string with tuned baseline, seed, and any number of override dicts."""
    config = json.loads(digital_life.default_config_json())
    config["seed"] = seed
    config.update(TUNED_BASELINE)
    for overrides in override_dicts:
        config.update(overrides)
    return json.dumps(config)


def run_single(seed: int, steps: int, sample_every: int, *override_dicts: dict) -> dict:
    """Run a single experiment and return parsed results."""
    config_json = make_config(seed, *override_dicts)
    result_json = digital_life.run_experiment_json(config_json, steps, sample_every)
    return json.loads(result_json)
