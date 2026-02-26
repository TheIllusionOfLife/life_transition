"""Recovery figure: alive count time series through shock events per capability level.

Two-panel: with shocks (period=50) vs. no-shock baseline.
Shows V3 entities maintain higher populations through periodic resource crashes.
"""

import numpy as np
from figures._shared import *

_VIROID_CONDITIONS = ["viroid_v0", "viroid_v0v1", "viroid_v0v1v2", "viroid_v0v1v2v3"]

_CAP_LABELS = {
    "viroid_v0": "V0",
    "viroid_v0v1": "V0+V1",
    "viroid_v0v1v2": "V0+V1+V2",
    "viroid_v0v1v2v3": "V0+V1+V2+V3",
}

_CAP_COLORS = {
    "viroid_v0": "#E69F00",
    "viroid_v0v1": "#56B4E9",
    "viroid_v0v1v2": "#009E73",
    "viroid_v0v1v2v3": "#000000",
}


def _mean_alive_by_step(rows: list[dict], condition: str) -> tuple[list[int], list[float]]:
    """Mean alive per step across all seeds for a condition."""
    step_alives: dict[int, list[float]] = {}
    for r in rows:
        if r["condition"] == condition:
            step = int(float(r["step"]))
            step_alives.setdefault(step, []).append(float(r["alive"]))
    steps = sorted(step_alives.keys())
    means = [float(np.mean(step_alives[s])) for s in steps]
    return steps, means


def generate_fig_semi_life_recovery(shock_tsv: Path, baseline_tsv: Path, out_dir: Path) -> None:
    """Two-panel: alive time series with shock (period=50) vs. no-shock baseline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shock_rows = parse_semi_life_tsv(shock_tsv)
    baseline_rows = parse_semi_life_tsv(baseline_tsv)

    # Use shock period 50 (more frequent shocks → clearer dynamics)
    shock_p50 = [r for r in shock_rows if str(r.get("shock_period", "")) == "50"]
    # Baseline: sparse harshness matches shock experiment primary harshness
    baseline_sparse = [r for r in baseline_rows if r.get("harshness") == "sparse"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    panels = [
        (shock_p50, "With Shocks (cycle period = 50)"),
        (baseline_sparse, "No Shocks (baseline, sparse)"),
    ]

    for ax, (panel_rows, title) in zip(axes, panels, strict=True):
        for cond in _VIROID_CONDITIONS:
            steps, means = _mean_alive_by_step(panel_rows, cond)
            if not steps:
                continue
            ax.plot(
                steps,
                means,
                color=_CAP_COLORS[cond],
                label=_CAP_LABELS[cond],
                linewidth=1.2,
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Step", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean Alive Count", fontsize=8)
    axes[0].legend(title="Capability", fontsize=7, title_fontsize=7)

    fig.suptitle("Recovery Under Periodic Resource Shocks (Viroid, sparse)", fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "fig_semi_life_recovery.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")
