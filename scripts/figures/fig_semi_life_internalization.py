"""Multi-channel Internalization Index across capability levels (Viroid).

Two-panel figure:
  Left: Stacked bar chart showing per-channel II contributions (rich harshness).
  Right: Composite II lines across harshness levels (as before, with CI bands).

Demonstrates that II rises gradually V0→V5, not as a V3 step function.
"""

import numpy as np
from figures._shared import *

_VIROID_CONDITIONS = [
    "viroid_v0",
    "viroid_v0v1",
    "viroid_v0v1v2",
    "viroid_v0v1v2v3",
    "viroid_v0v1v2v3v4",
    "viroid_v0v1v2v3v4v5",
]
_CAP_LABELS = ["V0", "V0+V1", "V0..V2", "V0..V3", "V0..V4", "V0..V5"]

_HARSHNESS_ORDER = ["rich", "medium", "sparse", "scarce"]
_HARSHNESS_COLORS = {
    "rich": "#000000",
    "medium": "#0072B2",
    "sparse": "#D55E00",
    "scarce": "#CC79A7",
}

# Per-channel II columns and visual config
_II_CHANNELS = [
    ("mean_ii_energy", "Energy (V3)", "#E69F00"),
    ("mean_ii_regulation", "Regulation (V2)", "#56B4E9"),
    ("mean_ii_behavior", "Behavior (V4)", "#009E73"),
    ("mean_ii_lifecycle", "Lifecycle (V5)", "#CC79A7"),
]


def _get_final_values(rows: list[dict], condition: str, harshness: str, column: str) -> list[float]:
    """Extract column values at the final step for each seed."""
    cond_rows = [r for r in rows if r["condition"] == condition and r["harshness"] == harshness]
    if not cond_rows:
        return []
    max_step = max(float(r["step"]) for r in cond_rows)
    return [float(r[column]) for r in cond_rows if float(r["step"]) == max_step and column in r]


def _has_channel_columns(rows: list[dict]) -> bool:
    """Check if TSV has multi-channel II columns."""
    if not rows:
        return False
    return "mean_ii_energy" in rows[0]


def generate_fig_semi_life_internalization(data_tsv: Path, out_dir: Path) -> None:
    """Multi-channel II figure: stacked bars (left) + composite lines with CI (right)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_semi_life_tsv(data_tsv)
    has_channels = _has_channel_columns(rows)

    if has_channels:
        fig, (ax_bars, ax_lines) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    else:
        fig, ax_lines = plt.subplots(figsize=(4.0, 3.0))

    x = np.arange(len(_VIROID_CONDITIONS))

    # --- Left panel: stacked bars (rich harshness, per-channel breakdown) ---
    if has_channels:
        harshness = "rich"
        bottoms = np.zeros(len(_VIROID_CONDITIONS))
        for col, label, color in _II_CHANNELS:
            means = []
            for cond in _VIROID_CONDITIONS:
                vals = _get_final_values(rows, cond, harshness, col)
                means.append(float(np.mean(vals)) if vals else 0.0)
            ax_bars.bar(
                x,
                means,
                bottom=bottoms,
                width=0.6,
                color=color,
                label=label,
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += np.array(means)

        ax_bars.set_xticks(x)
        ax_bars.set_xticklabels(_CAP_LABELS, rotation=20, ha="right", fontsize=7)
        ax_bars.set_ylabel("II Channel Contribution", fontsize=9)
        ax_bars.set_xlabel("Capability Level", fontsize=9)
        ax_bars.set_title("Per-channel II (rich)", fontsize=9)
        ax_bars.set_ylim(bottom=0, top=1.05)
        ax_bars.legend(fontsize=6, loc="upper left")
        ax_bars.spines["top"].set_visible(False)
        ax_bars.spines["right"].set_visible(False)

    # --- Right panel: composite II lines with CI bands ---
    for harshness in _HARSHNESS_ORDER:
        means = []
        ci_low = []
        ci_high = []
        for cond in _VIROID_CONDITIONS:
            vals = _get_final_values(rows, cond, harshness, "mean_ii")
            if vals:
                arr = np.array(vals)
                m = float(np.mean(arr))
                se = float(np.std(arr) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
                means.append(m)
                ci_low.append(m - 1.96 * se)
                ci_high.append(m + 1.96 * se)
            else:
                means.append(0.0)
                ci_low.append(0.0)
                ci_high.append(0.0)

        ax_lines.plot(
            x,
            means,
            marker="o",
            markersize=5,
            color=_HARSHNESS_COLORS[harshness],
            label=harshness.capitalize(),
            linewidth=1.2,
        )
        ax_lines.fill_between(
            x,
            ci_low,
            ci_high,
            color=_HARSHNESS_COLORS[harshness],
            alpha=0.15,
        )

    ax_lines.set_xticks(x)
    ax_lines.set_xticklabels(_CAP_LABELS, rotation=20, ha="right", fontsize=7)
    ax_lines.set_ylabel("Composite II (mean ± 95% CI)", fontsize=9)
    ax_lines.set_xlabel("Capability Level", fontsize=9)
    ax_lines.set_title("Internalization Index across V-levels", fontsize=9)
    ax_lines.set_ylim(bottom=-0.02, top=1.05)
    ax_lines.legend(title="Harshness", fontsize=7, title_fontsize=7)
    ax_lines.spines["top"].set_visible(False)
    ax_lines.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / "fig_semi_life_internalization.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")
