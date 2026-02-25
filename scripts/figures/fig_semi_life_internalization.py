"""InternalizationIndex vs capability level: one line per harshness level (Viroid).

Shows II rises monotonically V0→V3 (structural result independent of survival metrics).
"""

import numpy as np
from figures._shared import *

_VIROID_CONDITIONS = ["viroid_v0", "viroid_v0v1", "viroid_v0v1v2", "viroid_v0v1v2v3"]
_CAP_LABELS = ["V0", "V0+V1", "V0+V1+V2", "V0+V1+V2+V3"]

_HARSHNESS_ORDER = ["rich", "medium", "sparse", "scarce"]
_HARSHNESS_COLORS = {
    "rich": "#000000",
    "medium": "#0072B2",
    "sparse": "#D55E00",
    "scarce": "#CC79A7",
}


def _mean_ii_at_final(rows: list[dict], condition: str, harshness: str) -> float:
    cond_rows = [r for r in rows if r["condition"] == condition and r["harshness"] == harshness]
    if not cond_rows:
        return 0.0
    max_step = max(float(r["step"]) for r in cond_rows)
    final = [float(r["mean_ii"]) for r in cond_rows if float(r["step"]) == max_step]
    return float(np.mean(final)) if final else 0.0


def generate_fig_semi_life_internalization(data_tsv: Path, out_dir: Path) -> None:
    """Line plot: capability level vs mean II, one line per harshness (Viroid)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_semi_life_tsv(data_tsv)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    x = np.arange(len(_VIROID_CONDITIONS))

    for harshness in _HARSHNESS_ORDER:
        y = [_mean_ii_at_final(rows, cond, harshness) for cond in _VIROID_CONDITIONS]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=5,
            color=_HARSHNESS_COLORS[harshness],
            label=harshness.capitalize(),
            linewidth=1.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(_CAP_LABELS, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean InternalizationIndex (II)", fontsize=9)
    ax.set_xlabel("Capability Level", fontsize=9)
    ax.set_title("Internalization Index across V-levels (Viroid)", fontsize=9)
    ax.set_ylim(bottom=-0.02, top=1.05)
    ax.legend(title="Harshness", fontsize=7, title_fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / "fig_semi_life_internalization.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")
