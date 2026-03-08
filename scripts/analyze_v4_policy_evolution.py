"""Analyze V4 policy weight evolution from SemiLife experiment JSON sidecars.

Reads per-seed sidecar JSONs produced by experiment_semi_life_v1v3.py for the
viroid_v0v1v2v3v4 condition (seeds 100–199).  At the final step (step 500),
alive entities with V4 active are collected; their evolved policy vectors are
compared to the initial policy to quantify how much chemotaxis weights have
drifted over the run.

Usage:
    uv run python scripts/analyze_v4_policy_evolution.py

Output:
    experiments/semi_life_v4_policy_evolution.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

# V4 capability bitmask (must match Rust definition)
_V4_BIT: int = 0x10

# Initial policy weight vector (8 weights, index 0-7)
_INITIAL_POLICY: list[float] = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_HARSHNESS_LEVELS: list[str] = ["rich", "medium", "sparse", "scarce"]

_SEED_START: int = 100
_SEED_END: int = 199  # inclusive
_FINAL_STEP: int = 500


def log(msg: str) -> None:
    """Write a diagnostic message to stderr."""
    print(msg, file=sys.stderr)


def _sidecar_path(harshness: str, seed: int) -> Path:
    """Return the expected path for a single seed sidecar JSON."""
    filename = f"semi_life_v1v3_viroid_v0v1v2v3v4_{harshness}_{seed}.json"
    return _EXPERIMENTS_DIR / filename


def _extract_alive_v4_policies(sidecar: dict) -> list[list[float]]:
    """Extract policy vectors from alive V4 entities at the final step.

    Args:
        sidecar: Parsed JSON object from a seed sidecar file.

    Returns:
        List of policy vectors (each a list of 8 floats) from alive V4 entities
        at the last recorded step.  Returns an empty list if no eligible entities
        are found.
    """
    samples: list[dict] = sidecar.get("samples", [])
    if not samples:
        return []

    # Use the chronologically last sample (highest step value)
    final_sample = max(samples, key=lambda s: s.get("step", 0))
    if final_sample.get("step", 0) != _FINAL_STEP:
        # File may contain a shorter run — skip it to avoid mixing data
        return []

    policies: list[list[float]] = []
    for snap in final_sample.get("snapshots", []):
        if not snap.get("alive", False):
            continue
        if (snap.get("active_capabilities", 0) & _V4_BIT) == 0:
            continue
        policy = snap.get("policy")
        if policy is None:
            continue
        if len(policy) != len(_INITIAL_POLICY):
            # Defensive guard against schema changes
            continue
        policies.append([float(w) for w in policy])

    return policies


def compute_weight_delta(
    policy_vectors: list[list[float]],
    initial: list[float],
) -> dict:
    """Compute per-weight deviation statistics relative to an initial policy.

    For each weight index i, delta_w[i] = evolved[i] - initial[i].  Descriptive
    statistics (mean and std) are computed across all entities in
    ``policy_vectors``.  When only a single entity is present the std is defined
    as 0.0 rather than NaN to keep downstream consumers simple.

    Args:
        policy_vectors: List of evolved policy weight vectors.  Each inner list
            must have the same length as ``initial``.
        initial: Reference (un-evolved) policy weight vector.

    Returns:
        dict with keys:
            - ``weight_mean_delta``: mean deviation per weight (list of floats)
            - ``weight_std_delta``: std of deviation per weight (list of floats)
            - ``n_entities``: number of policy vectors supplied
    """
    n = len(policy_vectors)
    n_weights = len(initial)

    if n == 0:
        return {
            "weight_mean_delta": [0.0] * n_weights,
            "weight_std_delta": [0.0] * n_weights,
            "n_entities": 0,
        }

    # Build a 2-D array: shape (n_entities, n_weights)
    arr = np.array(policy_vectors, dtype=float)
    init_arr = np.array(initial, dtype=float)
    delta = arr - init_arr  # broadcast: (n, n_weights) - (n_weights,)

    mean_delta = np.mean(delta, axis=0).tolist()
    # ddof=0 (population std); single-entity case → 0.0, not NaN
    std_delta = np.std(delta, axis=0, ddof=0).tolist()

    return {
        "weight_mean_delta": mean_delta,
        "weight_std_delta": std_delta,
        "n_entities": n,
    }


def validate_output_schema(result: dict) -> bool:
    """Validate that the output JSON contains all required top-level keys.

    Args:
        result: The analysis output dict to validate.

    Returns:
        True if all required keys are present, False otherwise.
    """
    required_keys = {
        "analysis",
        "seeds",
        "initial_policy",
        "harshness_results",
    }
    return required_keys.issubset(result.keys())


def _analyze_harshness(harshness: str) -> dict:
    """Collect V4 policy data for all test seeds at a given harshness level.

    Args:
        harshness: One of "rich", "medium", "sparse", "scarce".

    Returns:
        Dict with n_entities, n_seeds, per-weight delta stats, and magnitude stats.
    """
    all_policies: list[list[float]] = []
    all_magnitudes: list[float] = []
    n_seeds_with_data = 0

    for seed in range(_SEED_START, _SEED_END + 1):
        path = _sidecar_path(harshness, seed)
        if not path.exists():
            continue
        try:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log(f"  [WARN] Could not read {path.name}: {exc}")
            continue

        policies = _extract_alive_v4_policies(sidecar)
        if not policies:
            continue

        n_seeds_with_data += 1
        for policy in policies:
            all_policies.append(policy)
            magnitude = math.sqrt(sum(w * w for w in policy))
            all_magnitudes.append(magnitude)

    delta_stats = compute_weight_delta(all_policies, _INITIAL_POLICY)

    initial_magnitude = math.sqrt(sum(w * w for w in _INITIAL_POLICY))
    if all_magnitudes:
        mag_arr = np.array(all_magnitudes, dtype=float)
        magnitude_mean = float(np.mean(mag_arr))
        magnitude_std = float(np.std(mag_arr, ddof=0))
    else:
        magnitude_mean = initial_magnitude
        magnitude_std = 0.0

    return {
        "n_entities": delta_stats["n_entities"],
        "n_seeds": n_seeds_with_data,
        "weight_mean_delta": delta_stats["weight_mean_delta"],
        "weight_std_delta": delta_stats["weight_std_delta"],
        "magnitude_mean": magnitude_mean,
        "magnitude_std": magnitude_std,
    }


def main() -> None:
    """Run V4 policy evolution analysis and write results to experiments/."""
    log("=== V4 policy evolution analysis ===")
    log(f"Seeds: {_SEED_START}–{_SEED_END}")
    log(f"Harshness levels: {_HARSHNESS_LEVELS}")
    log(f"Final step filter: {_FINAL_STEP}")
    log(f"Initial policy: {_INITIAL_POLICY}")

    harshness_results: dict[str, dict] = {}

    for harshness in _HARSHNESS_LEVELS:
        log(f"\nProcessing harshness={harshness} ...")
        result = _analyze_harshness(harshness)
        harshness_results[harshness] = result
        log(
            f"  n_entities={result['n_entities']}, n_seeds={result['n_seeds']}, "
            f"magnitude_mean={result['magnitude_mean']:.4f}"
        )
        mean_d = result["weight_mean_delta"]
        log(f"  weight_mean_delta: {[f'{v:.4f}' for v in mean_d]}")

    output = {
        "analysis": "v4_policy_evolution",
        "seeds": f"{_SEED_START}-{_SEED_END}",
        "initial_policy": _INITIAL_POLICY,
        "harshness_results": harshness_results,
    }

    assert validate_output_schema(output), "Output failed schema validation — internal error"

    out_path = _EXPERIMENTS_DIR / "semi_life_v4_policy_evolution.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"\nWrote {out_path}")

    total_entities = sum(v["n_entities"] for v in harshness_results.values())
    log(f"Total alive V4 entities analysed: {total_entities}")


if __name__ == "__main__":
    main()
