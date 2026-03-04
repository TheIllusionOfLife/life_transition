"""Check manuscript-reported parameters against manifest sources."""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAPER = PROJECT_ROOT / "paper" / "main.tex"
DEFAULT_MANIFEST = PROJECT_ROOT / "docs" / "research" / "manifest_reference.json"
DEFAULT_BINDINGS = PROJECT_ROOT / "docs" / "research" / "result_manifest_bindings.json"
DEFAULT_STATS = PROJECT_ROOT / "experiments" / "semi_life_capability_stats.json"
EXPECTED_PREREGISTERED_TESTS = 32
EXPECTED_HYPOTHESES = tuple(f"H{i}" for i in range(1, 9))
EXPERIMENT_SCRIPTS = [
    PROJECT_ROOT / "scripts" / "experiment_semi_life_v1v3.py",
    PROJECT_ROOT / "scripts" / "experiment_semi_life_shocks.py",
]


def _read_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc


def _extract_reported_timing(tex: str) -> tuple[int | None, int | None]:
    pattern = re.compile(
        r"(\d+)\s*~?\s*timesteps\s*[,;\s]\s*(?:with\s+population\s+)?sampl(?:ed|ing)\s+every\s+(\d+)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(tex)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _extract_script_paper_refs(paths: list[Path]) -> set[str]:
    refs: set[str] = set()
    pattern = re.compile(r'"paper_ref"\s*:\s*"([^"]+)"')
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text()
        refs.update(pattern.findall(text))
    return refs


def _check_files_exist(paths: dict[str, Path]) -> list[str]:
    issues: list[str] = []
    for name, path in paths.items():
        if not path.resolve().exists():
            issues.append(f"missing {name}: {path}")
    return issues


def _check_timing(tex: str, manifest: dict) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    checks: list[str] = []
    reported_steps, reported_sample_every = _extract_reported_timing(tex)

    if reported_steps is not None:
        checks.append("timing steps")
        manifest_steps = manifest.get("steps")
        try:
            manifest_steps_int = int(manifest_steps)
        except (TypeError, ValueError):
            issues.append(f"steps invalid in manifest: {manifest_steps}")
        else:
            if manifest_steps_int != reported_steps:
                issues.append(f"steps mismatch: paper={reported_steps} manifest={manifest_steps}")
    else:
        issues.append("could not parse timing steps from paper")

    if reported_sample_every is not None:
        checks.append("timing sample_every")
        manifest_sample_every = manifest.get("sample_every")
        try:
            manifest_sample_every_int = int(manifest_sample_every)
        except (TypeError, ValueError):
            issues.append(f"sample_every invalid in manifest: {manifest_sample_every}")
        else:
            if manifest_sample_every_int != reported_sample_every:
                issues.append(
                    "sample_every mismatch: "
                    f"paper={reported_sample_every} manifest={manifest_sample_every}"
                )
    else:
        issues.append("could not parse sample_every from paper")

    return issues, checks


def _check_base_config(manifest: dict) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    checks: list[str] = []
    base_cfg = manifest.get("base_config", {})
    if "mutation_point_rate" in base_cfg:
        checks.append("manifest base_config mutation_point_rate")
    else:
        issues.append("manifest missing base_config.mutation_point_rate")

    if "mutation_scale" in base_cfg or "mutation_point_scale" in base_cfg:
        checks.append("manifest base_config mutation_scale")
    else:
        issues.append("manifest missing base_config.mutation_scale (or mutation_point_scale)")
    return issues, checks


def _check_reference_manifest(manifest: dict) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    checks: list[str] = []
    for key in ["source_git_commit", "source_generated_at_utc"]:
        if manifest.get(key):
            checks.append(f"reference manifest {key}")
        else:
            issues.append(f"reference manifest missing {key}")
    return issues, checks


def _check_bindings(registry: dict, tex: str) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    checks: list[str] = []
    bindings = registry.get("bindings")
    if not isinstance(bindings, list) or len(bindings) == 0:
        issues.append("bindings registry is empty")
        return issues, checks

    checks.append("bindings registry non-empty")
    paper_labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    registry_refs: set[str] = set()
    for idx, binding in enumerate(bindings):
        paper_ref = binding.get("paper_ref")
        manifest_ref = binding.get("manifest")
        if not paper_ref:
            issues.append(f"binding[{idx}] missing paper_ref")
        elif paper_ref not in paper_labels:
            issues.append(f"binding[{idx}] paper_ref not found in paper labels: {paper_ref}")
        else:
            registry_refs.add(str(paper_ref))
        if not manifest_ref:
            issues.append(f"binding[{idx}] missing manifest")

    script_refs = _extract_script_paper_refs(EXPERIMENT_SCRIPTS)
    checks.append("experiment script paper_ref labels parsed")
    for ref in sorted(script_refs):
        if ref not in paper_labels:
            issues.append(f"experiment script paper_ref not found in paper labels: {ref}")
        if ref not in registry_refs:
            issues.append(f"experiment script paper_ref missing from registry: {ref}")

    return issues, checks


_GENERATED_ARTIFACTS = [
    PROJECT_ROOT / "experiments" / "semi_life_capability_stats.json",
    PROJECT_ROOT / "experiments" / "semi_life_shock_stats.json",
]


def _check_freshness(manifest: dict, manifest_path: Path) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    checks: list[str] = []

    should_check_freshness = manifest_path.resolve() == DEFAULT_MANIFEST.resolve()
    if not should_check_freshness:
        return issues, checks

    any_found = False
    for artifact in _GENERATED_ARTIFACTS:
        if artifact.exists():
            any_found = True
            try:
                data = _read_json(artifact)
            except ValueError as exc:
                issues.append(str(exc))
                continue
            checks.append(f"generated artifact valid JSON: {artifact.name}")
            # Basic schema validation: must be non-empty list or dict
            if isinstance(data, list) and len(data) == 0:
                issues.append(f"generated artifact is empty: {artifact.name}")
            elif isinstance(data, dict) and len(data) == 0:
                issues.append(f"generated artifact is empty: {artifact.name}")

    if not any_found:
        checks.append("no generated artifacts found (freshness check skipped)")

    return issues, checks


def _load_documents(
    paper_path: Path, manifest_path: Path, registry_path: Path
) -> tuple[str | None, dict | None, dict | None, list[str]]:
    """Load all required documents, returning (tex, manifest, registry, issues)."""
    tex = None
    manifest = None
    registry = None
    issues: list[str] = []

    try:
        tex = paper_path.read_text()
    except OSError as exc:
        issues.append(f"failed to read paper file {paper_path}: {exc}")
        return tex, manifest, registry, issues

    try:
        manifest = _read_json(manifest_path)
        registry = _read_json(registry_path)
    except ValueError as exc:
        issues.append(str(exc))
        return tex, manifest, registry, issues

    return tex, manifest, registry, issues


def _extract_test_family_sizes(tex: str) -> set[int]:
    sizes: set[int] = set()
    patterns = [
        r"(\d+)\s*[-–]?\s*test\s+family",
        r"across\s+(\d+)\s+tests",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, tex, re.IGNORECASE):
            try:
                sizes.add(int(match.group(1)))
            except (TypeError, ValueError):
                continue
    return sizes


def _should_check_hypothesis_family(paper_path: Path, registry_path: Path) -> bool:
    return (
        paper_path.resolve() == DEFAULT_PAPER.resolve()
        and registry_path.resolve() == DEFAULT_BINDINGS.resolve()
    )


def _check_paper_hypothesis_family(tex: str) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    checks: list[str] = []
    paper_sizes = _extract_test_family_sizes(tex)
    if EXPECTED_PREREGISTERED_TESTS in paper_sizes:
        checks.append("paper reports 32-test hypothesis family")
    else:
        issues.append(
            f"paper hypothesis family mismatch: expected {EXPECTED_PREREGISTERED_TESTS}-test family"
        )
    return issues, checks


def _check_bindings_hypothesis_family(registry: dict) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    checks: list[str] = []
    bindings = registry.get("bindings")
    if not isinstance(bindings, list):
        issues.append("bindings registry malformed while checking hypothesis family")
        return issues, checks

    target = next(
        (b for b in bindings if b.get("result_id") == "semi_life_hypothesis_tests"),
        None,
    )
    if target is None:
        issues.append("bindings missing result_id=semi_life_hypothesis_tests")
        return issues, checks

    note = str(target.get("notes", ""))
    if "H1-H8" in note:
        checks.append("bindings hypothesis note includes H1-H8")
    else:
        issues.append("bindings hypothesis note missing H1-H8")

    if "32-test" in note:
        checks.append("bindings hypothesis note includes 32-test family")
    else:
        issues.append("bindings hypothesis note missing 32-test family")

    return issues, checks


def _load_stats_rows() -> tuple[list[dict] | None, list[str], list[str]]:
    issues: list[str] = []
    checks: list[str] = []
    if not DEFAULT_STATS.exists():
        checks.append("hypothesis-family stats file missing (check skipped)")
        return None, issues, checks

    try:
        stats = _read_json(DEFAULT_STATS)
    except ValueError as exc:
        return None, [str(exc)], checks

    if not isinstance(stats, list):
        return None, ["stats file is not a JSON array"], checks

    rows = [row for row in stats if isinstance(row, dict)]
    checks.append("hypothesis-family stats file loaded")
    return rows, issues, checks


def _extract_preregistered_rows(stats_rows: list[dict]) -> list[dict]:
    return [row for row in stats_rows if str(row.get("hypothesis", "")) in EXPECTED_HYPOTHESES]


def _check_stats_hypothesis_family(stats_rows: list[dict]) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    checks: list[str] = []

    prereg = _extract_preregistered_rows(stats_rows)
    if len(prereg) == EXPECTED_PREREGISTERED_TESTS:
        checks.append("stats include 32 pre-registered H1-H8 tests")
    else:
        issues.append(
            "stats pre-registered test count mismatch: "
            f"expected {EXPECTED_PREREGISTERED_TESTS} got {len(prereg)}"
        )

    by_hypothesis: dict[str, int] = {h: 0 for h in EXPECTED_HYPOTHESES}
    for row in prereg:
        by_hypothesis[str(row.get("hypothesis", ""))] += 1

    for hypothesis in sorted(EXPECTED_HYPOTHESES):
        n = by_hypothesis[hypothesis]
        if n != 4:
            issues.append(f"stats hypothesis count mismatch: {hypothesis} expected 4 got {n}")
    checks.append("stats hypothesis multiplicity checked (4 harshness levels each)")

    missing_pcorr = [
        row for row in prereg if row.get("p_raw") is not None and row.get("p_corrected") is None
    ]
    if missing_pcorr:
        issues.append(
            f"stats missing p_corrected for {len(missing_pcorr)} pre-registered comparisons"
        )
    else:
        checks.append("stats p_corrected present for all pre-registered comparisons")

    return issues, checks


def _check_hypothesis_family(
    tex: str, registry: dict, paper_path: Path, registry_path: Path
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    checks: list[str] = []

    if not _should_check_hypothesis_family(paper_path, registry_path):
        return issues, checks

    p_issues, p_checks = _check_paper_hypothesis_family(tex)
    issues.extend(p_issues)
    checks.extend(p_checks)

    b_issues, b_checks = _check_bindings_hypothesis_family(registry)
    issues.extend(b_issues)
    checks.extend(b_checks)

    stats_rows, s_load_issues, s_load_checks = _load_stats_rows()
    checks.extend(s_load_checks)
    if s_load_issues:
        issues.extend(s_load_issues)
        return issues, checks
    if stats_rows is None:
        return issues, checks

    s_issues, s_checks = _check_stats_hypothesis_family(stats_rows)
    issues.extend(s_issues)
    checks.extend(s_checks)

    return issues, checks


def run_checks(paper_path: Path, manifest_path: Path, registry_path: Path) -> dict:
    """Run consistency checks and return a machine-readable report."""
    # 1. Check file existence
    file_issues = _check_files_exist(
        {
            "paper file": paper_path,
            "manifest file": manifest_path,
            "bindings registry": registry_path,
        }
    )
    if file_issues:
        return {"ok": False, "issues": file_issues, "checks": []}

    all_issues: list[str] = []
    all_checks: list[str] = []

    # 2. Load Documents
    tex, manifest, registry, load_issues = _load_documents(paper_path, manifest_path, registry_path)
    if load_issues:
        return {"ok": False, "issues": load_issues, "checks": all_checks}

    # Since we checked for load_issues, tex, manifest, and registry must be valid here
    assert tex is not None
    assert manifest is not None
    assert registry is not None

    # 3. Timing Checks
    t_issues, t_checks = _check_timing(tex, manifest)
    all_issues.extend(t_issues)
    all_checks.extend(t_checks)

    # 4. Base Config Checks
    bc_issues, bc_checks = _check_base_config(manifest)
    all_issues.extend(bc_issues)
    all_checks.extend(bc_checks)

    # 5. Reference Manifest Checks
    rm_issues, rm_checks = _check_reference_manifest(manifest)
    all_issues.extend(rm_issues)
    all_checks.extend(rm_checks)

    # 6. Bindings Registry Checks
    b_issues, b_checks = _check_bindings(registry, tex)
    all_issues.extend(b_issues)
    all_checks.extend(b_checks)

    # 7. Freshness Checks
    f_issues, f_checks = _check_freshness(manifest, manifest_path)
    all_issues.extend(f_issues)
    all_checks.extend(f_checks)

    # 8. Hypothesis-family consistency checks
    h_issues, h_checks = _check_hypothesis_family(tex, registry, paper_path, registry_path)
    all_issues.extend(h_issues)
    all_checks.extend(h_checks)

    return {"ok": len(all_issues) == 0, "issues": all_issues, "checks": all_checks}


def main() -> int:
    report = run_checks(DEFAULT_PAPER, DEFAULT_MANIFEST, DEFAULT_BINDINGS)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
