"""Generate LaTeX macros from experiment data.

Reads the TSV experiment output and stats JSON to produce auto_macros.tex
with \\newcommand definitions for every in-text number, eliminating the
hardcoded-number class of bugs.

Usage:
    uv run python scripts/generate_latex_macros.py [TSV_PATH] [STATS_PATH]

Output:
    paper/auto_macros.tex
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
_PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"


def load_tsv(path: Path) -> list[dict]:
    """Load SemiLife experiment TSV into list of row dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def get_alive_at_final(rows: list[dict], condition: str, harshness: str) -> np.ndarray:
    """Extract alive count at the final step for each seed."""
    cond_rows = [r for r in rows if r["condition"] == condition and r["harshness"] == harshness]
    if not cond_rows:
        return np.array([])
    max_step = max(float(r["step"]) for r in cond_rows)
    return np.array([float(r["alive"]) for r in cond_rows if float(r["step"]) == max_step])


_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def _safe_name(s: str) -> str:
    """Convert condition/harshness to a LaTeX-safe macro name component.

    LaTeX command names must contain only ASCII letters — no digits allowed.
    """
    s = s.replace("_", "").replace("+", "").replace(".", "")
    # Replace each digit with its word form
    for digit, word in _DIGIT_WORDS.items():
        s = s.replace(digit, word)
    return s


def _fmt(val: float, decimals: int = 1) -> str:
    """Format a float for LaTeX."""
    return f"{val:.{decimals}f}"


def generate_phase_diagram_macros(rows: list[dict]) -> list[str]:
    """Generate macros for phase diagram numbers (Section 5.1)."""
    macros = []
    macros.append("% === Phase diagram alive counts (mean at step 500) ===")

    conditions = [
        ("viroid_v0", "Vzero"),
        ("viroid_v0v1", "VzeroVone"),
        ("viroid_v0v1v2", "VzeroVoneVtwo"),
        ("viroid_v0v1v2v3", "VzeroVoneVtwoVthree"),
        ("viroid_v0v1v2v3v4", "VzeroToVfour"),
        ("viroid_v0v1v2v3v4v5", "VzeroToVfive"),
        ("proto_baseline", "ProtoBase"),
        ("proto_liberated", "ProtoLib"),
        ("virus_baseline", "VirusBase"),
        ("virus_v0v1v2", "VirusVtwo"),
        ("virus_v0v1v2v3", "VirusVthree"),
    ]
    harshness_levels = ["rich", "medium", "sparse", "scarce"]

    for cond_name, macro_name in conditions:
        for h in harshness_levels:
            alive = get_alive_at_final(rows, cond_name, h)
            if len(alive) > 0:
                mean_val = float(np.mean(alive))
                std_val = float(np.std(alive))
                h_cap = h.capitalize()
                macros.append(f"\\newcommand{{\\alive{macro_name}{h_cap}}}{{{_fmt(mean_val)}}}")
                macros.append(f"\\newcommand{{\\aliveSD{macro_name}{h_cap}}}{{{_fmt(std_val)}}}")
    return macros


def generate_stats_macros(stats: list[dict]) -> list[str]:
    """Generate macros for hypothesis test statistics (Table 2)."""
    macros = []
    macros.append("")
    macros.append("% === Hypothesis test statistics (Table 2) ===")

    for result in stats:
        hyp = result.get("hypothesis", "")
        h = result.get("harshness", "")
        if not hyp or not h:
            continue

        h_cap = h.capitalize()
        prefix = f"{_safe_name(hyp)}{h_cap}"

        # U statistic
        u_val = result.get("U")
        if u_val is not None:
            u_str = f"{int(u_val):,}".replace(",", "\\,")
            macros.append(f"\\newcommand{{\\statU{prefix}}}{{{u_str}}}")

        # p-value (corrected)
        p_corr = result.get("p_corrected")
        if p_corr is not None:
            if p_corr < 1e-100:
                p_str = "<\\!10^{-100}"
            elif p_corr < 0.001:
                exp = int(np.floor(np.log10(p_corr)))
                p_str = f"<\\!10^{{{exp}}}"
            else:
                p_str = f"{p_corr:.3f}"
            macros.append(f"\\newcommand{{\\statP{prefix}}}{{{p_str}}}")

        # Cliff's delta
        delta = result.get("cliffs_delta")
        if delta is not None:
            macros.append(f"\\newcommand{{\\statDelta{prefix}}}{{{_fmt(delta, 2)}}}")

        # CI
        ci_low = result.get("ci_low")
        ci_high = result.get("ci_high")
        if ci_low is not None and ci_high is not None:
            macros.append(
                f"\\newcommand{{\\statCI{prefix}}}{{[{_fmt(ci_low, 2)}, {_fmt(ci_high, 2)}]}}"
            )

    return macros


def generate_ii_macros(rows: list[dict]) -> list[str]:
    """Generate macros for II channel values at final step."""
    macros = []
    macros.append("")
    macros.append("% === Internalization Index values ===")

    conditions = [
        ("viroid_v0v1v2", "VzeroVoneVtwo"),
        ("viroid_v0v1v2v3", "VzeroVoneVtwoVthree"),
        ("viroid_v0v1v2v3v4", "VzeroToVfour"),
        ("viroid_v0v1v2v3v4v5", "VzeroToVfive"),
        ("proto_baseline", "ProtoBase"),
        ("proto_liberated", "ProtoLib"),
    ]
    channels = [
        "mean_ii",
        "mean_ii_energy",
        "mean_ii_regulation",
        "mean_ii_behavior",
        "mean_ii_lifecycle",
    ]
    channel_names = ["II", "IIE", "IIR", "IIB", "IIL"]

    for cond_name, macro_name in conditions:
        for h in ["rich", "medium", "sparse", "scarce"]:
            cond_rows = [r for r in rows if r["condition"] == cond_name and r["harshness"] == h]
            if not cond_rows:
                continue
            max_step = max(float(r["step"]) for r in cond_rows)
            final = [r for r in cond_rows if float(r["step"]) == max_step]
            h_cap = h.capitalize()
            for ch, ch_name in zip(channels, channel_names, strict=True):
                if ch in final[0]:
                    vals = np.array([float(r[ch]) for r in final])
                    mean_val = float(np.mean(vals))
                    macros.append(
                        f"\\newcommand{{\\ii{ch_name}{macro_name}{h_cap}}}{{{_fmt(mean_val, 2)}}}"
                    )
    return macros


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    tsv_path = Path(argv[0]) if len(argv) > 0 else _EXPERIMENTS_DIR / "semi_life_v1v5_test.tsv"
    stats_path = (
        Path(argv[1]) if len(argv) > 1 else _EXPERIMENTS_DIR / "semi_life_capability_stats.json"
    )

    if not tsv_path.exists():
        print(f"ERROR: TSV not found: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {tsv_path} ...", file=sys.stderr)
    rows = load_tsv(tsv_path)
    print(f"  {len(rows)} rows loaded", file=sys.stderr)

    all_macros: list[str] = []
    all_macros.append("% Auto-generated by generate_latex_macros.py — DO NOT EDIT")
    all_macros.append(f"% Source: {tsv_path.name}")
    all_macros.append("")

    # Phase diagram macros
    all_macros.extend(generate_phase_diagram_macros(rows))

    # Stats macros (if available)
    if stats_path.exists():
        print(f"Loading {stats_path} ...", file=sys.stderr)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        all_macros.extend(generate_stats_macros(stats))
    else:
        print(f"Stats JSON not found: {stats_path} (skipping)", file=sys.stderr)

    # II macros
    all_macros.extend(generate_ii_macros(rows))

    out_path = _PAPER_DIR / "auto_macros.tex"
    out_path.write_text("\n".join(all_macros) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(all_macros)} lines)", file=sys.stderr)


if __name__ == "__main__":
    main()
