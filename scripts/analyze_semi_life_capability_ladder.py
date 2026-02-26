"""Statistical analysis of the SemiLife capability ladder experiment.

Reads the test-seed TSV (experiment_semi_life_v1v3.py, seeds 100–199) and
runs pre-registered hypothesis tests H1–H4.

Usage:
    uv run python scripts/analyze_semi_life_capability_ladder.py \\
        experiments/semi_life_v1v3_test.tsv

Output:
    JSON to experiments/semi_life_capability_stats.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from analyses.results.statistics import (
    bootstrap_cliffs_delta_ci,
    cliffs_delta,
    holm_bonferroni,
    jonckheere_terpstra,
)
from experiment_semi_life_v1v3 import RESOURCE_INITIAL_VALUES
from scipy import stats as scipy_stats

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def load_tsv(path: Path) -> list[dict]:
    """Load SemiLife experiment TSV into list of row dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def get_alive_at_final(rows: list[dict], condition: str, harshness: str) -> list[float]:
    """Extract alive count at the final step for each seed."""
    cond_rows = [r for r in rows if r["condition"] == condition and r["harshness"] == harshness]
    if not cond_rows:
        return []
    max_step = max(float(r["step"]) for r in cond_rows)
    return [float(r["alive"]) for r in cond_rows if float(r["step"]) == max_step]


def get_total_replications_at_final(
    rows: list[dict], condition: str, harshness: str
) -> list[float]:
    """Extract total_replications at the final step (cumulative) per seed."""
    cond_rows = [r for r in rows if r["condition"] == condition and r["harshness"] == harshness]
    if not cond_rows:
        return []
    max_step = max(float(r["step"]) for r in cond_rows)
    return [float(r["total_replications"]) for r in cond_rows if float(r["step"]) == max_step]


def get_mean_ii_at_final(rows: list[dict], condition: str, harshness: str) -> list[float]:
    """Extract mean_ii at the final step per seed."""
    cond_rows = [r for r in rows if r["condition"] == condition and r["harshness"] == harshness]
    if not cond_rows:
        return []
    max_step = max(float(r["step"]) for r in cond_rows)
    return [float(r["mean_ii"]) for r in cond_rows if float(r["step"]) == max_step]


def get_mean_energy_at_final(rows: list[dict], condition: str, harshness: str) -> list[float]:
    """Extract mean_energy at the final step per seed."""
    cond_rows = [r for r in rows if r["condition"] == condition and r["harshness"] == harshness]
    if not cond_rows:
        return []
    max_step = max(float(r["step"]) for r in cond_rows)
    return [float(r["mean_energy"]) for r in cond_rows if float(r["step"]) == max_step]


def run_mannwhitney(a: list[float], b: list[float]) -> dict:
    """Mann-Whitney U test with Cliff's delta and bootstrap CI."""
    arr_a = np.array(a)
    arr_b = np.array(b)
    n_a, n_b = len(arr_a), len(arr_b)
    if n_a < 2 or n_b < 2:
        return {
            "n_a": n_a,
            "n_b": n_b,
            "U": None,
            "p_raw": None,
            "cliffs_delta": None,
            "ci_low": None,
            "ci_high": None,
        }
    u_stat, p_raw = scipy_stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
    cd = cliffs_delta(arr_a, arr_b)
    ci_low, ci_high = bootstrap_cliffs_delta_ci(arr_a, arr_b)
    return {
        "n_a": n_a,
        "n_b": n_b,
        "U": float(u_stat),
        "p_raw": float(p_raw),
        "cliffs_delta": float(cd),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def analyze_h1(rows: list[dict]) -> list[dict]:
    """H1: Viroid V0 vs V0+V1 — boundary overhead reduces alive in scarce env."""
    results = []
    for harshness in RESOURCE_INITIAL_VALUES:
        a = get_alive_at_final(rows, "viroid_v0", harshness)
        b = get_alive_at_final(rows, "viroid_v0v1", harshness)
        result = run_mannwhitney(a, b)
        result.update(
            {
                "hypothesis": "H1",
                "archetype": "viroid",
                "harshness": harshness,
                "comparison": "viroid_v0 vs viroid_v0v1",
                "metric": "alive",
                "pre_registered_direction": "V0 > V0+V1 (boundary overhead in scarce)",
            }
        )
        results.append(result)
    return results


def analyze_h2(rows: list[dict]) -> list[dict]:
    """H2: Viroid V0+V1+V2+V3 vs V0+V1+V2 — V3 metabolism boosts survival."""
    results = []
    for harshness in RESOURCE_INITIAL_VALUES:
        a = get_alive_at_final(rows, "viroid_v0v1v2v3", harshness)
        b = get_alive_at_final(rows, "viroid_v0v1v2", harshness)
        result = run_mannwhitney(a, b)
        result.update(
            {
                "hypothesis": "H2",
                "archetype": "viroid",
                "harshness": harshness,
                "comparison": "viroid_v0v1v2v3 vs viroid_v0v1v2",
                "metric": "alive",
                "pre_registered_direction": "V3 > V2 (metabolism buffers energy)",
            }
        )
        results.append(result)
    return results


def analyze_h3(rows: list[dict]) -> list[dict]:
    """H3: ProtoOrganelle baseline vs liberated — V0 enables replication."""
    results = []
    for harshness in RESOURCE_INITIAL_VALUES:
        a = get_total_replications_at_final(rows, "proto_liberated", harshness)
        b = get_total_replications_at_final(rows, "proto_baseline", harshness)
        result = run_mannwhitney(a, b)
        result.update(
            {
                "hypothesis": "H3",
                "archetype": "proto_organelle",
                "harshness": harshness,
                "comparison": "proto_liberated vs proto_baseline",
                "metric": "total_replications",
                "pre_registered_direction": "liberated > baseline (V0 enables replication)",
            }
        )
        results.append(result)
    return results


def analyze_h4(rows: list[dict]) -> list[dict]:
    """H4: JT monotonic trend V0→V3 × alive × all harshness levels."""
    viroid_order = ["viroid_v0", "viroid_v0v1", "viroid_v0v1v2", "viroid_v0v1v2v3"]
    results = []
    for harshness in RESOURCE_INITIAL_VALUES:
        groups = [np.array(get_alive_at_final(rows, cond, harshness)) for cond in viroid_order]
        # Require all 4 groups to have data; skip if any is too sparse for a valid trend test.
        if any(len(g) < 2 for g in groups):
            results.append(
                {
                    "hypothesis": "H4",
                    "archetype": "viroid",
                    "harshness": harshness,
                    "comparison": "V0→V3 trend (JT)",
                    "metric": "alive",
                    "JT_statistic": None,
                    "p_raw": None,
                    "cliffs_delta": None,
                    "ci_low": None,
                    "ci_high": None,
                    "pre_registered_direction": "monotonic increase V0→V3",
                }
            )
            continue
        jt_stat, p_val = jonckheere_terpstra(groups)
        results.append(
            {
                "hypothesis": "H4",
                "archetype": "viroid",
                "harshness": harshness,
                "comparison": "V0→V3 trend (JT)",
                "metric": "alive",
                "JT_statistic": float(jt_stat),
                "p_raw": float(p_val),
                "cliffs_delta": None,
                "ci_low": None,
                "ci_high": None,
                "pre_registered_direction": "monotonic increase V0→V3",
            }
        )
    return results


_ENERGY_COMPARISONS: list[tuple[str, str, str]] = [
    ("H1_energy", "viroid_v0", "viroid_v0v1"),
    ("H2_energy", "viroid_v0v1v2v3", "viroid_v0v1v2"),
]


def analyze_mean_energy_supplement(rows: list[dict]) -> list[dict]:
    """Exploratory: H1/H2 comparisons using mean_energy instead of alive count.

    Addresses the δ=1.00 ceiling/floor concern by providing a finer-grained
    continuous metric.  These are explicitly labelled 'exploratory'.
    """
    results = []
    for harshness in RESOURCE_INITIAL_VALUES:
        for hypothesis, cond_a, cond_b in _ENERGY_COMPARISONS:
            a = get_mean_energy_at_final(rows, cond_a, harshness)
            b = get_mean_energy_at_final(rows, cond_b, harshness)
            result = run_mannwhitney(a, b)
            result.update(
                {
                    "hypothesis": hypothesis,
                    "archetype": "viroid",
                    "harshness": harshness,
                    "comparison": f"{cond_a} vs {cond_b}",
                    "metric": "mean_energy",
                    "analysis_type": "exploratory",
                }
            )
            results.append(result)
    return results


def apply_holm_bonferroni(all_results: list[dict]) -> list[dict]:
    """Apply Holm-Bonferroni correction across all 16 pre-registered tests."""
    with_p = [r for r in all_results if r.get("p_raw") is not None]
    p_values = [r["p_raw"] for r in with_p]
    corrected = holm_bonferroni(p_values)
    for result, p_corr in zip(with_p, corrected, strict=True):
        result["p_corrected"] = float(p_corr)
    return all_results


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    tsv_path = Path(argv[0]) if argv else _EXPERIMENTS_DIR / "semi_life_v1v3_test.tsv"

    if not tsv_path.exists():
        print(f"ERROR: TSV not found: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {tsv_path} ...", file=sys.stderr)
    rows = load_tsv(tsv_path)
    print(f"  {len(rows)} rows loaded", file=sys.stderr)

    all_results: list[dict] = []
    all_results.extend(analyze_h1(rows))
    all_results.extend(analyze_h2(rows))
    all_results.extend(analyze_h3(rows))
    all_results.extend(analyze_h4(rows))
    all_results = apply_holm_bonferroni(all_results)

    out_path = _EXPERIMENTS_DIR / "semi_life_capability_stats.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"  Total comparisons: {len(all_results)}", file=sys.stderr)

    # Exploratory: mean_energy supplement (addresses δ=1.00 concern)
    if rows and "mean_energy" in rows[0]:
        supplement = analyze_mean_energy_supplement(rows)
        supp_path = _EXPERIMENTS_DIR / "semi_life_mean_energy_supplement.json"
        supp_path.write_text(json.dumps(supplement, indent=2), encoding="utf-8")
        print(f"Wrote {supp_path}", file=sys.stderr)
        print(f"  Exploratory mean_energy comparisons: {len(supplement)}", file=sys.stderr)
    else:
        print("Skipping mean_energy supplement: column not in TSV", file=sys.stderr)


if __name__ == "__main__":
    main()
