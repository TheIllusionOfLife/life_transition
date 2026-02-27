# Artifact Publication Policy — Semi-Life

## Publication Split

Artifact publication splits across two channels:

1. **GitHub repository** (lightweight, reviewable):
   - Paper source (`paper/main.tex`) and compiled PDF
   - Experiment scripts and analysis code (`scripts/`)
   - Run configs (`configs/semi_life_*.json`)
   - Provenance manifest (`experiments/result_manifest_bindings.json`)

2. **Zenodo record** (heavy, immutable, citable):
   - Raw test-seed experiment data (`semi_life_v1v5_test.tsv`, 4.0 MB)
   - Shock recovery data (`semi_life_shocks.tsv`, 1.1 MB)
   - Statistical outputs (`semi_life_capability_stats.json`, `semi_life_shock_stats.json`)
   - Effect size robustness check (`semi_life_mean_energy_supplement.json`)
   - Calibration data (`calibration_results.tsv`)

## Paper-Binding Artifacts (6 total)

| Paper reference | Source file | Channel |
|----------------|-------------|---------|
| `fig:phase` (Figure 1) | `semi_life_v1v5_test.tsv` | Zenodo |
| `fig:ii` (Figure 2) | `semi_life_v1v5_test.tsv` | Zenodo |
| `fig:tradeoff` (Figure 3) | `semi_life_v1v5_test.tsv` | Zenodo |
| `fig:recovery` (Figure 4) | `semi_life_shocks.tsv` | Zenodo |
| `tab:stats` (Table 2) | `semi_life_capability_stats.json` | Zenodo |
| `sec:results-energy` (Section 5.6) | `semi_life_mean_energy_supplement.json` | Zenodo |

## What Must Not Be Committed to Git

- Raw per-seed JSON outputs (`experiments/*.json` except manifests)
- TSV/CSV exports (`experiments/*.tsv`)
- Gzipped archives (`experiments/*.gz`, `zenodo_staging/`)
- Any single file exceeding ~5 MB

## Prerequisites

- **Python**: `requests` in `pyproject.toml` dependencies
- **`ZENODO_TOKEN`**: personal access token with `deposit:write` and
  `deposit:actions` scopes
  - Create at <https://zenodo.org/account/settings/applications/>
  - Export: `export ZENODO_TOKEN="your_token_here"` in shell profile

## Execution Runbook

### Step 1: Stage archives

```bash
mkdir -p zenodo_staging
tar -czf zenodo_staging/semi_life_capability_data.tar.gz \
  experiments/semi_life_v1v5_test.tsv \
  experiments/semi_life_capability_stats.json \
  experiments/semi_life_mean_energy_supplement.json
tar -czf zenodo_staging/semi_life_shock_data.tar.gz \
  experiments/semi_life_shocks.tsv \
  experiments/semi_life_shock_stats.json
tar -czf zenodo_staging/semi_life_calibration_data.tar.gz \
  experiments/calibration_results.tsv
```

### Step 2: Generate metadata

```bash
uv run python scripts/prepare_zenodo_metadata.py \
  zenodo_staging/*.tar.gz \
  --experiment-name semi_life_capability_ladder \
  --steps 500 --seed-start 100 --seed-end 199 \
  --paper-binding "fig:phase=experiments/semi_life_v1v5_test.tsv" \
  --paper-binding "fig:ii=experiments/semi_life_v1v5_test.tsv" \
  --paper-binding "fig:tradeoff=experiments/semi_life_v1v5_test.tsv" \
  --paper-binding "fig:recovery=experiments/semi_life_shocks.tsv" \
  --paper-binding "tab:stats=experiments/semi_life_capability_stats.json" \
  --paper-binding "sec:results-energy=experiments/semi_life_mean_energy_supplement.json" \
  --output docs/zenodo_metadata.json
```

### Step 3: Upload draft to Zenodo

```bash
uv run python scripts/upload_zenodo.py \
  --metadata docs/zenodo_metadata.json \
  --title "Semi-Life Capability Ladder: Experiment Data" \
  --description "Raw TSV data, statistical JSON outputs, and calibration data" \
  --creator "Anonymous" \
  --version v1.0 \
  --keyword "artificial life" --keyword "semi-life" \
  --conference-title "ALIFE 2026" \
  --language eng
```

### Step 4: Publish (post-acceptance only)

```bash
uv run python scripts/upload_zenodo.py \
  --metadata docs/zenodo_metadata.json \
  --title "Semi-Life Capability Ladder: Experiment Data" \
  --description "Raw TSV data, statistical JSON outputs, and calibration data" \
  --creator "Last, First; Affiliation; ORCID" \
  --version v1.0 \
  --keyword "artificial life" --keyword "semi-life" \
  --conference-title "ALIFE 2026" \
  --language eng \
  --publish
```

## Submission Sequence (post-acceptance)

1. Publish Zenodo dataset draft → get final DOI
2. Update paper Data Availability with DOI
3. Fill `\author{}` with real names + affiliations
4. Fill `[ANONYMOUS]` with GitHub repo URL
5. Update `.zenodo.json` with real author names
6. Create PR with camera-ready changes, merge to main
7. `git tag v1.0 && git push origin v1.0`
8. `gh release create v1.0 --title "ALIFE 2026"`
9. Verify code record at zenodo.org
10. Add DOI badges to README
11. Submit camera-ready manuscript
