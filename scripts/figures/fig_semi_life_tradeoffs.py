"""Replication rate vs. persistence tradeoff scatter.

One point per condition × harshness. Color = archetype; marker shape = capability level.
"""

import numpy as np
from figures._shared import *

_ARCHETYPE_COLORS = {
    "viroid": "#E69F00",
    "virus": "#56B4E9",
    "proto_organelle": "#009E73",
}

# Marker shape by condition (grouped by number of active capabilities)
_COND_MARKER = {
    "viroid_v0": "o",  # V0 only
    "viroid_v0v1": "s",  # 2 capabilities
    "virus_baseline": "s",  # V0+V1
    "viroid_v0v1v2": "^",  # 3 capabilities
    "virus_v0v1v2": "^",
    "proto_baseline": "P",  # V1+V2+V3 (no V0)
    "viroid_v0v1v2v3": "D",  # 4 capabilities
    "virus_v0v1v2v3": "D",
    "proto_liberated": "D",
    "viroid_v0v1v2v3v4": "h",  # 5 capabilities (hexagon)
    "viroid_v0v1v2v3v4v5": "*",  # 6 capabilities (star)
}

_HARSHNESS_ALPHA = {"rich": 0.95, "medium": 0.7, "sparse": 0.5, "scarce": 0.3}

_ALL_CONDITIONS = [
    ("viroid_v0", "viroid"),
    ("viroid_v0v1", "viroid"),
    ("viroid_v0v1v2", "viroid"),
    ("viroid_v0v1v2v3", "viroid"),
    ("viroid_v0v1v2v3v4", "viroid"),
    ("viroid_v0v1v2v3v4v5", "viroid"),
    ("proto_baseline", "proto_organelle"),
    ("proto_liberated", "proto_organelle"),
    ("virus_baseline", "virus"),
    ("virus_v0v1v2", "virus"),
    ("virus_v0v1v2v3", "virus"),
]

_HARSHNESS_ORDER = ["rich", "medium", "sparse", "scarce"]


def _aggregate_condition(rows: list[dict], condition: str, harshness: str) -> dict | None:
    cond_rows = [r for r in rows if r["condition"] == condition and r["harshness"] == harshness]
    if not cond_rows:
        return None
    max_step = max(float(r["step"]) for r in cond_rows)
    final = [r for r in cond_rows if float(r["step"]) == max_step]
    if not final:
        return None
    alive_vals = [float(r["alive"]) for r in final]
    rep_vals = [float(r["total_replications"]) for r in final]
    archetype = str(final[0].get("archetype", ""))
    return {
        "condition": condition,
        "harshness": harshness,
        "mean_alive": float(np.mean(alive_vals)),
        "mean_replications": float(np.mean(rep_vals)),
        "archetype": archetype,
    }


def generate_fig_semi_life_tradeoffs(data_tsv: Path, out_dir: Path) -> None:
    """Scatter: total_replications vs alive at step 500; color=archetype, shape=capability."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_semi_life_tsv(data_tsv)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    plotted_archetypes: set[str] = set()
    for condition, _archetype in _ALL_CONDITIONS:
        for harshness in _HARSHNESS_ORDER:
            agg = _aggregate_condition(rows, condition, harshness)
            if agg is None:
                continue
            color = _ARCHETYPE_COLORS.get(agg["archetype"], "#999999")
            marker = _COND_MARKER.get(condition, "o")
            alpha = _HARSHNESS_ALPHA.get(harshness, 0.6)
            label = agg["archetype"] if agg["archetype"] not in plotted_archetypes else None
            ax.scatter(
                agg["mean_replications"],
                agg["mean_alive"],
                c=color,
                marker=marker,
                s=40,
                alpha=alpha,
                edgecolors="white",
                linewidths=0.5,
                label=label,
                zorder=5,
            )
            plotted_archetypes.add(agg["archetype"])

    ax.set_xlabel("Total Replications (rate proxy)", fontsize=9)
    ax.set_ylabel("Mean Alive at Step 500 (persistence)", fontsize=9)
    ax.set_title("Replication Rate vs. Persistence Tradeoff", fontsize=9)
    ax.legend(title="Archetype", fontsize=7, title_fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / "fig_semi_life_tradeoffs.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")
