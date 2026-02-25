# Pre-registration: SemiLife Capability Ladder Experiment

**Filed**: 2026-02-27 (before test-seed data collection)
**Test seeds**: 100–199 (reserved, unused until this pre-registration is committed)
**Calibration seeds**: 0–49 (used in PRs #3–#4 for parameter calibration only)

---

## Pre-registered Hypotheses

### H1: Boundary overhead in scarce environments (Viroid)

**Comparison**: Viroid V0 vs. Viroid V0+V1
**Environment**: scarce (resource_initial_value = 0.05)
**Direction**: V0+V1 produces *fewer* alive entities at step 500 than V0 alone
**Rationale**: Boundary maintenance (V1) requires per-step energy; in scarce resources this
overhead exceeds the protective benefit, reducing equilibrium population.
**Test**: Mann-Whitney U, two-tailed, α = 0.05

### H2: Metabolism boosts survival (Viroid)

**Comparison**: Viroid V0+V1+V2+V3 vs. Viroid V0+V1+V2
**Environments**: sparse + scarce (resource_initial_value ∈ {0.1, 0.05})
**Direction**: V3 produces *more* alive entities at step 500
**Rationale**: Internal metabolism (V3) provides energy buffering via internal_pool,
decoupling immediate resource uptake from replication threshold.
**Test**: Mann-Whitney U, two-tailed, α = 0.05

### H3: V0 addition enables ProtoOrganelle replication (liberation)

**Comparison**: ProtoOrganelle baseline (V1+V2+V3) vs. liberated (V0+V1+V2+V3)
**Environment**: all harshness levels
**Direction**: Liberated ProtoOrganelle shows *more* total_replications at step 500
**Rationale**: Without V0, ProtoOrganelle cannot replicate regardless of metabolic state;
adding V0 "liberates" the replication capability already latent in V1+V2+V3.
**Test**: Mann-Whitney U on total_replications, two-tailed, α = 0.05

### H4: Monotonic trend V0→V3 in scarce environments (Jonckheere-Terpstra)

**Groups**: Viroid V0 | V0+V1 | V0+V1+V2 | V0+V1+V2+V3
**Environment**: scarce (resource_initial_value = 0.05)
**Direction**: Alive count at step 500 increases monotonically across V-levels
**Test**: Jonckheere-Terpstra trend test, α = 0.05

---

## Statistical Plan

- **Primary test**: Mann-Whitney U (two-tailed, α = 0.05)
- **Effect size**: Cliff's δ (thresholds: |δ| < 0.147 negligible, < 0.33 small,
  < 0.474 medium, ≥ 0.474 large)
- **Bootstrap CIs**: 2000 resamples, 95% percentile interval
- **Multiple comparisons**: Holm-Bonferroni correction across all 16 pre-registered tests
  (4 hypotheses × 4 harshness levels)
- **Trend test**: Jonckheere-Terpstra for H4 (ordered groups, included in 16-test family)
- **Exploratory analyses**: clearly labeled "EXPLORATORY" in all output and manuscript sections

---

## Seed Plan

| Split | Seeds | Purpose |
|-------|-------|---------|
| Calibration | 0–49 | Parameter sweep (PRs #3–#4) |
| Buffer | 50–99 | Reserved (unused) |
| Test | 100–199 | Pre-registered hypothesis testing (this PR) |

*Test seeds are never used for parameter decisions before this pre-registration is committed.*

---

## Non-Hypotheses (Pre-registered Nulls)

- Sham controls (compute capability but no state effect) are NOT expected to differ from
  capability-absent conditions — if they do, flag as anomaly before writing results.
- V1 boundary *benefit* in *rich* environments (resource_initial_value = 1.0) is exploratory
  only — we pre-register no direction for this comparison.
- Recovery advantage of V3 over V0 under shocks is exploratory — not in H1–H4 family.
