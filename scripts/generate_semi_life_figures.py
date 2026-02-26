"""Generate all four Semi-Life paper figures.

Requires:
  experiments/semi_life_v1v3_test.tsv   -- capability ladder test-seed data
  experiments/semi_life_shocks.tsv      -- shock experiment data

Usage:
    uv run python scripts/generate_semi_life_figures.py

Output:
    paper/figures/fig_semi_life_phase_diagram.pdf
    paper/figures/fig_semi_life_internalization.pdf
    paper/figures/fig_semi_life_tradeoffs.pdf
    paper/figures/fig_semi_life_recovery.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is on the path when called directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

from figures import (
    generate_fig_semi_life_internalization,
    generate_fig_semi_life_phase_diagram,
    generate_fig_semi_life_recovery,
    generate_fig_semi_life_tradeoffs,
)

_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS = _ROOT / "experiments"
_FIG_DIR = _ROOT / "paper" / "figures"

_V1V3_TSV = _EXPERIMENTS / "semi_life_v1v3_test.tsv"
_SHOCKS_TSV = _EXPERIMENTS / "semi_life_shocks.tsv"


def main() -> None:
    """Generate all Semi-Life figures; skip gracefully if data is missing."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not _V1V3_TSV.exists():
        print(f"ERROR: capability ladder data not found: {_V1V3_TSV}", file=sys.stderr)
        print(
            f"  Run: uv run python scripts/experiment_semi_life_v1v3.py > {_V1V3_TSV}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading capability ladder data from {_V1V3_TSV.name} ...")

    print("Figure 1: Phase diagram")
    generate_fig_semi_life_phase_diagram(_V1V3_TSV, _FIG_DIR)

    print("Figure 2: InternalizationIndex")
    generate_fig_semi_life_internalization(_V1V3_TSV, _FIG_DIR)

    print("Figure 3: Replication-persistence tradeoff")
    generate_fig_semi_life_tradeoffs(_V1V3_TSV, _FIG_DIR)

    if _SHOCKS_TSV.exists():
        print("Figure 4: Shock recovery")
        generate_fig_semi_life_recovery(_SHOCKS_TSV, _V1V3_TSV, _FIG_DIR)
    else:
        print(f"SKIP Figure 4: shock data not found ({_SHOCKS_TSV.name})")

    print("Done.")


if __name__ == "__main__":
    main()
