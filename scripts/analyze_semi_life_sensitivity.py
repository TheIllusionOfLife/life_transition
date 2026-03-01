"""EXPLORATORY: Analyze parameter sensitivity sweep results.

Computes Cliff's δ for key metrics (alive, mean_energy, mean_ii) at each
multiplier level relative to the baseline (1.0×), grouped by parameter
and harshness.

Usage:
    uv run python scripts/analyze_semi_life_sensitivity.py \
        experiments/semi_life_sensitivity_data.tsv

Output:
    JSON to experiments/semi_life_sensitivity_stats.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from analyses.results.statistics import bootstrap_cliffs_delta_ci, cliffs_delta

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

METRICS = ["alive", "mean_energy", "mean_ii"]
BASELINE_MULTIPLIER = "1.00"


def load_tsv(path: Path) -> list[dict]:
    """Load sensitivity sweep TSV into list of row dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def _get_values(
    rows: list[dict], param: str, multiplier: str, harshness: str, metric: str
) -> list[float]:
    """Extract metric values for a specific param × multiplier × harshness."""
    return [
        float(r[metric])
        for r in rows
        if r["param_name"] == param
        and r["multiplier"] == multiplier
        and r["harshness"] == harshness
    ]


def analyze_sensitivity(rows: list[dict]) -> list[dict]:
    """Compute Cliff's δ for each param × multiplier × harshness vs baseline."""
    params = sorted(set(r["param_name"] for r in rows))
    multipliers = sorted(set(r["multiplier"] for r in rows))
    harshness_levels = sorted(set(r["harshness"] for r in rows))

    results = []
    for param in params:
        for harshness in harshness_levels:
            baseline_vals = {
                m: _get_values(rows, param, BASELINE_MULTIPLIER, harshness, m) for m in METRICS
            }
            for mult in multipliers:
                for metric in METRICS:
                    base = baseline_vals[metric]
                    test = _get_values(rows, param, mult, harshness, metric)
                    if len(base) < 2 or len(test) < 2:
                        results.append(
                            {
                                "param_name": param,
                                "multiplier": mult,
                                "harshness": harshness,
                                "metric": metric,
                                "cliffs_delta": None,
                                "ci_low": None,
                                "ci_high": None,
                                "n_baseline": len(base),
                                "n_test": len(test),
                                "analysis_type": "exploratory",
                            }
                        )
                        continue
                    arr_base = np.array(base)
                    arr_test = np.array(test)
                    cd = cliffs_delta(arr_test, arr_base)
                    ci_low, ci_high = bootstrap_cliffs_delta_ci(arr_test, arr_base)
                    results.append(
                        {
                            "param_name": param,
                            "multiplier": mult,
                            "harshness": harshness,
                            "metric": metric,
                            "cliffs_delta": float(cd),
                            "ci_low": float(ci_low),
                            "ci_high": float(ci_high),
                            "n_baseline": len(base),
                            "n_test": len(test),
                            "analysis_type": "exploratory",
                        }
                    )
    return results


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    tsv_path = Path(argv[0]) if argv else _EXPERIMENTS_DIR / "semi_life_sensitivity_data.tsv"

    if not tsv_path.exists():
        print(f"ERROR: TSV not found: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {tsv_path} ...", file=sys.stderr)
    rows = load_tsv(tsv_path)
    print(f"  {len(rows)} rows loaded", file=sys.stderr)

    results = analyze_sensitivity(rows)
    out_path = _EXPERIMENTS_DIR / "semi_life_sensitivity_stats.json"
    _EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"  Total comparisons: {len(results)}", file=sys.stderr)


if __name__ == "__main__":
    main()
