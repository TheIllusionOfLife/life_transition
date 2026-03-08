"""Statistical analysis of the SemiLife competition experiment.

Reads the competition TSV (experiment_semi_life_competition.py, seeds 0–99) and
runs Mann-Whitney U tests on frequency_ratio and viroid/plasmid alive counts.

Usage:
    uv run python scripts/analyze_competition.py experiments/semi_life_competition.tsv

Output:
    JSON to experiments/semi_life_competition.json
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
)
from scipy import stats as scipy_stats

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

HARSHNESS_LEVELS = ["rich", "medium", "sparse", "scarce"]


def load_tsv(path: Path) -> list[dict]:
    """Load competition experiment TSV into list of row dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def _get_final_step_rows(rows: list[dict], condition: str, harshness: str) -> list[dict]:
    """Return rows at the final sample step for a given condition and harshness."""
    cond_rows = [r for r in rows if r["condition"] == condition and r["harshness"] == harshness]
    if not cond_rows:
        return []
    max_step = max(float(r["step"]) for r in cond_rows)
    return [r for r in cond_rows if float(r["step"]) == max_step]


def get_frequency_ratio_at_final(rows: list[dict], condition: str, harshness: str) -> list[float]:
    """Extract frequency_ratio at the final step, excluding nan values."""
    final = _get_final_step_rows(rows, condition, harshness)
    result = []
    for r in final:
        val = r["frequency_ratio"]
        if val != "nan":
            result.append(float(val))
    return result


def get_alive_at_final(
    rows: list[dict], condition: str, harshness: str, archetype: str
) -> list[float]:
    """Extract alive count for a specific archetype at the final step."""
    final = _get_final_step_rows(rows, condition, harshness)
    col = f"{archetype}_alive"
    return [float(r[col]) for r in final]


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


def _summary_stats(values: list[float]) -> dict:
    """Return mean, std, and n for a list of floats."""
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "n": len(arr),
    }


def analyze_harshness(rows: list[dict], harshness: str) -> dict:
    """Run all comparisons for a single harshness level."""
    primary_ratios = get_frequency_ratio_at_final(rows, "primary", harshness)
    ablation_ratios = get_frequency_ratio_at_final(rows, "ablation", harshness)

    primary_viroid = get_alive_at_final(rows, "primary", harshness, "viroid")
    primary_plasmid = get_alive_at_final(rows, "primary", harshness, "plasmid")

    primary_vs_ablation = run_mannwhitney(primary_ratios, ablation_ratios)
    viroid_vs_plasmid = run_mannwhitney(primary_viroid, primary_plasmid)

    return {
        "primary_frequency_ratio": _summary_stats(primary_ratios),
        "ablation_frequency_ratio": _summary_stats(ablation_ratios),
        "primary_vs_ablation_mw": primary_vs_ablation,
        "viroid_vs_plasmid_alive_mw": viroid_vs_plasmid,
    }


def _collect_p_values(harshness_results: dict) -> list[tuple[str, str, float]]:
    """Collect raw p-values from all Mann-Whitney results for Holm-Bonferroni.

    Returns list of (harshness, comparison_key, p_raw) triples.
    """
    collected = []
    comparison_keys = ["primary_vs_ablation_mw", "viroid_vs_plasmid_alive_mw"]
    for harshness, result in harshness_results.items():
        for key in comparison_keys:
            p = result[key].get("p_raw")
            if p is not None:
                collected.append((harshness, key, p))
    return collected


def apply_holm_bonferroni(harshness_results: dict) -> dict:
    """Apply Holm-Bonferroni correction across all valid Mann-Whitney p-values.

    Adds p_corrected in-place to each Mann-Whitney result dict.
    """
    triples = _collect_p_values(harshness_results)
    if not triples:
        return harshness_results

    p_values = [t[2] for t in triples]
    corrected = holm_bonferroni(p_values)

    for (harshness, key, _), p_corr in zip(triples, corrected, strict=True):
        harshness_results[harshness][key]["p_corrected"] = float(p_corr)

    return harshness_results


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    tsv_path = Path(argv[0]) if argv else _EXPERIMENTS_DIR / "semi_life_competition.tsv"

    if not tsv_path.exists():
        print(f"ERROR: TSV not found: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {tsv_path} ...", file=sys.stderr)
    rows = load_tsv(tsv_path)
    print(f"  {len(rows)} rows loaded", file=sys.stderr)

    harshness_results: dict[str, dict] = {}
    for harshness in HARSHNESS_LEVELS:
        harshness_results[harshness] = analyze_harshness(rows, harshness)
        print(f"  Analyzed harshness={harshness}", file=sys.stderr)

    harshness_results = apply_holm_bonferroni(harshness_results)

    output = {
        "analysis": "competition",
        "seeds": "0-99",
        "harshness_results": harshness_results,
    }

    out_path = _EXPERIMENTS_DIR / "semi_life_competition.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
