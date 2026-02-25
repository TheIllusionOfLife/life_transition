"""Statistical analysis of the SemiLife shock/perturbation experiment.

Computes recovery rates from the shock TSV and compares against the no-shock
baseline from experiment_semi_life_v1v3.py.

Usage:
    uv run python scripts/analyze_semi_life_shocks.py \\
        experiments/semi_life_shocks.tsv \\
        experiments/semi_life_v1v3_test.tsv

Output:
    JSON to experiments/semi_life_shock_stats.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from analyses.results.statistics import bootstrap_cliffs_delta_ci, cliffs_delta
from experiment_semi_life_v1v3 import ARCHETYPE_CONDITIONS

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

RECOVERY_TARGET = 0.80  # recover to 80% of pre-shock population


def load_tsv(path: Path) -> list[dict]:
    """Load experiment TSV into list of row dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def get_step_alives(rows: list[dict], condition: str) -> dict[int, list[float]]:
    """Build step → list[alive] mapping for a condition (all seeds)."""
    step_alives: dict[int, list[float]] = {}
    for r in rows:
        if r["condition"] == condition:
            step = int(float(r["step"]))
            step_alives.setdefault(step, []).append(float(r["alive"]))
    return step_alives


def compute_recovery_time(
    step_alives: dict[int, list[float]],
    shock_period: int,
    recovery_target: float = RECOVERY_TARGET,
) -> list[float]:
    """Recovery time (steps) per shock event.

    For each shock at step t = k × shock_period:
    - Pre-shock level: mean alive just before the shock
    - Recovery time: steps after shock until mean alive ≥ target × pre_shock_mean

    Returns list of recovery times (right-censored at final step if no recovery).
    """
    steps = sorted(step_alives.keys())
    if not steps or shock_period <= 0:
        return []

    shock_steps = [s for s in steps if s > 0 and s % shock_period == 0]
    if not shock_steps:
        return []

    max_step = max(steps)
    recovery_times = []
    for shock_step in shock_steps:
        pre_steps = [s for s in steps if s < shock_step]
        if not pre_steps:
            continue
        pre_shock_alives = step_alives.get(max(pre_steps), [])
        if not pre_shock_alives:
            continue
        pre_shock_mean = float(np.mean(pre_shock_alives))
        threshold = pre_shock_mean * recovery_target

        post_steps = sorted(s for s in steps if s > shock_step)
        for post_step in post_steps:
            post_alives = step_alives.get(post_step, [])
            if post_alives and float(np.mean(post_alives)) >= threshold:
                recovery_times.append(float(post_step - shock_step))
                break
        else:
            # Right-censored: did not recover within observation window
            recovery_times.append(float(max_step - shock_step))

    return recovery_times


def analyze_condition(
    shock_rows: list[dict],
    baseline_rows: list[dict],
    condition: str,
    shock_period: int,
) -> dict:
    """Per-condition recovery statistics for one shock period."""
    shock_filtered = [
        r
        for r in shock_rows
        if r["condition"] == condition and str(r.get("shock_period", "")) == str(shock_period)
    ]
    shock_ts = get_step_alives(shock_filtered, condition)

    baseline_filtered = [r for r in baseline_rows if r["condition"] == condition]
    baseline_ts = get_step_alives(baseline_filtered, condition)

    recovery_times = compute_recovery_time(shock_ts, shock_period)

    baseline_steps = sorted(baseline_ts.keys())
    baseline_final = baseline_ts.get(max(baseline_steps), []) if baseline_steps else []

    shock_steps_sorted = sorted(shock_ts.keys())
    shock_final = shock_ts.get(max(shock_steps_sorted), []) if shock_steps_sorted else []

    return {
        "condition": condition,
        "shock_period": shock_period,
        "n_recovery_events": len(recovery_times),
        "mean_recovery_time": float(np.mean(recovery_times)) if recovery_times else None,
        "median_recovery_time": float(np.median(recovery_times)) if recovery_times else None,
        "baseline_final_mean_alive": float(np.mean(baseline_final)) if baseline_final else None,
        "shock_final_mean_alive": float(np.mean(shock_final)) if shock_final else None,
    }


def compute_v0_vs_v3_effect(shock_rows: list[dict], shock_period: int) -> dict:
    """Cliff's δ comparing V0 vs V0+V1+V2+V3 alive counts at final step (EXPLORATORY)."""

    def get_final_alive(cond: str) -> list[float]:
        filtered = [
            r
            for r in shock_rows
            if r["condition"] == cond and str(r.get("shock_period", "")) == str(shock_period)
        ]
        if not filtered:
            return []
        max_step = max(float(r["step"]) for r in filtered)
        return [float(r["alive"]) for r in filtered if float(r["step"]) == max_step]

    v0_alive = np.array(get_final_alive("viroid_v0"))
    v3_alive = np.array(get_final_alive("viroid_v0v1v2v3"))

    if len(v0_alive) < 2 or len(v3_alive) < 2:
        return {
            "shock_period": shock_period,
            "comparison": "viroid_v0v1v2v3 vs viroid_v0",
            "metric": "alive_at_final_step",
            "note": "EXPLORATORY",
            "cliffs_delta": None,
            "ci_low": None,
            "ci_high": None,
        }

    cd = cliffs_delta(v3_alive, v0_alive)
    ci_low, ci_high = bootstrap_cliffs_delta_ci(v3_alive, v0_alive)
    return {
        "shock_period": shock_period,
        "comparison": "viroid_v0v1v2v3 vs viroid_v0",
        "metric": "alive_at_final_step",
        "note": "EXPLORATORY",
        "cliffs_delta": float(cd),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    shock_path = Path(argv[0]) if len(argv) >= 1 else _EXPERIMENTS_DIR / "semi_life_shocks.tsv"
    baseline_path = (
        Path(argv[1]) if len(argv) >= 2 else _EXPERIMENTS_DIR / "semi_life_v1v3_test.tsv"
    )

    for p in (shock_path, baseline_path):
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading shock TSV: {shock_path}", file=sys.stderr)
    shock_rows = load_tsv(shock_path)
    print(f"Loading baseline TSV: {baseline_path}", file=sys.stderr)
    baseline_rows = load_tsv(baseline_path)

    shock_periods = sorted({int(r["shock_period"]) for r in shock_rows if r.get("shock_period")})
    conditions = [c for c, _, _ in ARCHETYPE_CONDITIONS]

    per_condition: list[dict] = []
    for condition in conditions:
        for shock_period in shock_periods:
            per_condition.append(
                analyze_condition(shock_rows, baseline_rows, condition, shock_period)
            )

    effect_sizes: list[dict] = [compute_v0_vs_v3_effect(shock_rows, sp) for sp in shock_periods]

    output = {"per_condition": per_condition, "v0_vs_v3_effect_sizes": effect_sizes}

    out_path = _EXPERIMENTS_DIR / "semi_life_shock_stats.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
