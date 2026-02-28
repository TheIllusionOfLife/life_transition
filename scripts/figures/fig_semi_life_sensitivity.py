"""Sensitivity tornado plot: Cliff's δ (alive) vs parameter multiplier.

Shows how each parameter's perturbation affects the alive count relative
to the 1.0× baseline. One subplot per harshness level. EXPLORATORY.
"""

import json

from figures._shared import *

_PARAM_LABELS = {
    "energy_leakage_rate": "Leakage rate",
    "boundary_decay_rate": "Boundary decay",
    "env_damage_probability": "Damage prob.",
    "overconsumption_waste_fraction": "Waste fraction",
    "regulator_cost_per_step": "Regulator cost",
    "internal_conversion_rate": "Conversion rate",
    "v4_move_cost": "V4 move cost",
    "v5_dormant_decay_mult": "V5 dormancy mult",
}

# Okabe-Ito inspired, 8 distinguishable colors
_PARAM_COLORS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


def generate_fig_semi_life_sensitivity(stats_json: Path, out_dir: Path) -> None:
    """Tornado plot: δ(alive) vs multiplier, one panel per harshness."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(stats_json, encoding="utf-8") as f:
        results = json.load(f)

    # Filter to alive metric only
    alive_results = [r for r in results if r["metric"] == "alive"]
    if not alive_results:
        print("  WARNING: no alive-metric results in sensitivity stats; skipping figure")
        return

    harshness_levels = sorted(set(r["harshness"] for r in alive_results))
    params = sorted(set(r["param_name"] for r in alive_results))
    param_colors = {p: _PARAM_COLORS[i % len(_PARAM_COLORS)] for i, p in enumerate(params)}

    n_panels = len(harshness_levels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, harshness in zip(axes, harshness_levels, strict=True):
        for param in params:
            param_data = [
                r
                for r in alive_results
                if r["param_name"] == param
                and r["harshness"] == harshness
                and r["cliffs_delta"] is not None
            ]
            if not param_data:
                continue
            mults = [float(r["multiplier"]) for r in param_data]
            deltas = [r["cliffs_delta"] for r in param_data]
            ci_lo = [r.get("ci_low", r["cliffs_delta"]) for r in param_data]
            ci_hi = [r.get("ci_high", r["cliffs_delta"]) for r in param_data]

            label_text = _PARAM_LABELS.get(param, param)
            color = param_colors[param]
            ax.plot(
                mults,
                deltas,
                marker="o",
                markersize=4,
                color=color,
                label=label_text,
                linewidth=1.0,
            )
            ax.fill_between(mults, ci_lo, ci_hi, color=color, alpha=0.12)

        ax.axhline(0, color="#999999", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Multiplier (×baseline)", fontsize=8)
        ax.set_title(f"{harshness.capitalize()}", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Cliff's δ (alive vs 1.0× baseline)", fontsize=8)
    # Place legend outside rightmost panel
    axes[-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.suptitle("Parameter Sensitivity (EXPLORATORY)", fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "fig_semi_life_sensitivity.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")
