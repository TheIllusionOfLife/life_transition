"""Phase diagram: capability level × harshness → mean alive at step 500.

3-panel grid (Viroid, Virus, ProtoOrganelle). Color encodes mean alive count;
dashed contour marks the 50% survival phase boundary.
"""

import numpy as np
from figures._shared import *

# ---------------------------------------------------------------------------
# Archetype panel definitions
# ---------------------------------------------------------------------------
_PANELS = {
    "viroid": {
        "conditions": [
            "viroid_v0",
            "viroid_v0v1",
            "viroid_v0v1v2",
            "viroid_v0v1v2v3",
            "viroid_v0v1v2v3v4",
            "viroid_v0v1v2v3v4v5",
        ],
        "cap_labels": ["V0", "V0+V1", "V0..V2", "V0..V3", "V0..V4", "V0..V5"],
        "title": "Viroid",
    },
    "virus": {
        "conditions": ["virus_baseline", "virus_v0v1v2", "virus_v0v1v2v3"],
        "cap_labels": ["V0+V1\n(base)", "V0+V1+V2", "V0+V1+V2+V3"],
        "title": "Virus",
    },
    "proto_organelle": {
        "conditions": ["proto_baseline", "proto_liberated"],
        "cap_labels": ["V1+V2+V3\n(base)", "V0+V1+V2+V3"],
        "title": "ProtoOrganelle",
    },
}

_HARSHNESS_ORDER = ["rich", "medium", "sparse", "scarce"]
_HARSHNESS_LABELS = {
    "rich": "Rich\n(1.0)",
    "medium": "Med.\n(0.3)",
    "sparse": "Sparse\n(0.1)",
    "scarce": "Scarce\n(0.05)",
}

# Initial population per archetype from calibrated configs
_INITIAL_POP = 10


def _alive_at_final(rows: list[dict], condition: str, harshness: str) -> tuple[float, float]:
    """Return (mean, std) of alive count at the final step."""
    cond_rows = [r for r in rows if r["condition"] == condition and r["harshness"] == harshness]
    if not cond_rows:
        return 0.0, 0.0
    max_step = max(float(r["step"]) for r in cond_rows)
    final = [float(r["alive"]) for r in cond_rows if float(r["step"]) == max_step]
    if not final:
        return 0.0, 0.0
    return float(np.mean(final)), float(np.std(final))


def generate_fig_semi_life_phase_diagram(data_tsv: Path, out_dir: Path) -> None:
    """3-panel phase diagram: capability × harshness → mean alive at step 500."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_semi_life_tsv(data_tsv)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))

    global_max = max(
        _alive_at_final(rows, cond, h)[0]
        for panel in _PANELS.values()
        for cond in panel["conditions"]
        for h in _HARSHNESS_ORDER
    )
    vmax = max(_INITIAL_POP, global_max)

    for ax, panel_cfg in zip(axes, _PANELS.values(), strict=True):
        conditions = panel_cfg["conditions"]
        cap_labels = panel_cfg["cap_labels"]
        n_caps = len(conditions)
        n_harsh = len(_HARSHNESS_ORDER)

        # data[i, j] = mean alive; std_data for annotation
        data = np.zeros((n_harsh, n_caps))
        std_data = np.zeros((n_harsh, n_caps))
        for j, cond in enumerate(conditions):
            for i, harshness in enumerate(_HARSHNESS_ORDER):
                mean, std = _alive_at_final(rows, cond, harshness)
                data[i, j] = mean
                std_data[i, j] = std

        im = ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
        )

        # Annotate cells with mean ± std
        for i in range(n_harsh):
            for j in range(n_caps):
                val = data[i, j]
                sd = std_data[i, j]
                # Use dark text on light cells, white on dark
                text_color = "white" if val > vmax * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.0f}\n±{sd:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=text_color,
                    fontweight="bold",
                )

        # Phase boundary: 50% of initial population
        if data.max() > 0 and n_caps >= 2 and n_harsh >= 2:
            try:
                ax.contour(
                    data,
                    levels=[_INITIAL_POP * 0.5],
                    colors=["#0072B2"],
                    linewidths=1.2,
                    linestyles="--",
                )
            except (TypeError, ValueError):
                pass  # contour silently skipped if data too sparse for level

        ax.set_xticks(range(n_caps))
        ax.set_xticklabels(cap_labels, rotation=25, ha="right", fontsize=7)
        ax.set_yticks(range(n_harsh))
        ax.set_yticklabels([_HARSHNESS_LABELS[h] for h in _HARSHNESS_ORDER], fontsize=7)
        ax.set_title(panel_cfg["title"], fontsize=9)
        ax.set_xlabel("Capability Level", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Resource Harshness", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean Alive")

    fig.suptitle("Phase Diagram: Capability × Harshness → Survival (step 500)", fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "fig_semi_life_phase_diagram.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")
