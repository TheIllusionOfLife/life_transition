# Pre-registration: SemiLife Capability Ladder Experiment

**Filed**: 2026-02-27 (before test-seed data collection)
**Test seeds**: 100–199 (reserved, unused until this pre-registration is committed)
**Calibration seeds**: 0–49 (used in PRs #3–#4 for parameter calibration only)

---

## Pre-registered Hypotheses

### H1: Boundary cost–benefit tradeoff (Viroid)

**Comparison**: Viroid V0 vs. Viroid V0+V1
**Environments**: all 4 harshness levels (rich/medium/sparse/scarce)
**Direction**: V0+V1 produces *fewer* alive entities at step 500 than V0 alone in
resource-abundant environments (rich), where leakage/damage pressure is low relative to
boundary repair costs; the effect may reverse in harsh environments where environmental
damage is proportionally more costly without boundary protection
**Rationale**: V1 boundary protects against energy leakage and stochastic environmental
damage, but incurs per-step repair cost. In rich environments the leakage/damage savings
are small relative to repair overhead (net cost); in harsh environments the damage
protection may outweigh repair cost (net benefit). This creates an environment-dependent
tradeoff rather than a universal cost.
**Test**: Mann-Whitney U, two-tailed, α = 0.05

### H2: Metabolism boosts survival (Viroid)

**Comparison**: Viroid V0+V1+V2+V3 vs. Viroid V0+V1+V2
**Environments**: all 4 harshness levels (rich/medium/sparse/scarce)
**Direction**: V3 produces *more* alive entities at step 500; strongest effect predicted in
sparse + scarce where internal energy buffering matters most
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

### H4: Monotonic trend V0→V3 across harshness levels (Jonckheere-Terpstra)

**Groups**: Viroid V0 | V0+V1 | V0+V1+V2 | V0+V1+V2+V3
**Environments**: all 4 harshness levels (rich/medium/sparse/scarce)
**Direction**: Alive count at step 500 increases monotonically across V-levels; strongest
effect predicted in scarce where marginal capability gains matter most
**Test**: Jonckheere-Terpstra trend test, α = 0.05

### H5: V4 (response to stimuli) improves survival over V3-only in sparse/scarce (Viroid)

**Comparison**: Viroid V0+V1+V2+V3+V4 vs. Viroid V0+V1+V2+V3
**Environments**: all 4 harshness levels (rich/medium/sparse/scarce)
**Direction**: V4 produces *more* alive entities at step 500; strongest effect predicted in
sparse + scarce where chemotaxis toward resource gradients matters most
**Rationale**: Response to stimuli (V4) enables gradient-following movement, allowing entities
to locate and exploit resource-rich patches rather than relying on passive resource uptake at
their current location.
**Test**: Mann-Whitney U, two-tailed, α = 0.05

### H6: V5 (staged lifecycle) improves survival over V4 in sparse/scarce (Viroid)

**Comparison**: Viroid V0+V1+V2+V3+V4+V5 vs. Viroid V0+V1+V2+V3+V4
**Environments**: all 4 harshness levels (rich/medium/sparse/scarce)
**Direction**: V5 produces *more* alive entities at step 500; strongest effect predicted in
sparse + scarce where the dormancy energy-conservation strategy matters most
**Rationale**: Staged lifecycle (V5) enables dormancy during energy scarcity and dispersal
for resource prospecting, providing survival advantages over purely active entities.
**Test**: Mann-Whitney U, two-tailed, α = 0.05

### H7: Monotonic trend V0→V5 across harshness levels (Jonckheere-Terpstra)

**Groups**: Viroid V0 | V0+V1 | V0+V1+V2 | V0+V1+V2+V3 | V0+V1+V2+V3+V4 | V0+V1+V2+V3+V4+V5
**Environments**: all 4 harshness levels (rich/medium/sparse/scarce)
**Direction**: Alive count at step 500 increases monotonically across all 6 V-levels;
extends H4 with the full capability ladder
**Test**: Jonckheere-Terpstra trend test, α = 0.05

### H8: V2 overconsumption regulation benefit (Viroid)

**Comparison**: Viroid V0+V1+V2 vs. Viroid V0+V1
**Environments**: all 4 harshness levels (rich/medium/sparse/scarce)
**Direction**: V0+V1+V2 produces *more* alive entities at step 500 than V0+V1 alone;
strongest effect predicted in rich/medium where resource availability creates
overconsumption opportunities
**Rationale**: Without regulation (V2), rapid resource uptake exceeding an optimal rate
incurs energy waste from metabolic overflow. V2 homeostasis throttles uptake to match
processing capacity, reducing waste. The benefit is greatest where resources are abundant
enough to trigger overconsumption but the waste penalty is non-trivial.
**Test**: Mann-Whitney U, two-tailed, α = 0.05

---

## Statistical Plan

- **Primary test**: Mann-Whitney U (two-tailed, α = 0.05)
- **Effect size**: Cliff's δ (thresholds: |δ| < 0.147 negligible, < 0.33 small,
  < 0.474 medium, ≥ 0.474 large)
- **Bootstrap CIs**: 2000 resamples, 95% percentile interval
- **Multiple comparisons**: Holm-Bonferroni correction across all 32 pre-registered tests
  (H1 × 4 + H2 × 4 + H3 × 4 + H4 × 4 + H5 × 4 + H6 × 4 + H7 × 4 + H8 × 4 = 32)
- **Trend test**: Jonckheere-Terpstra for H4 and H7 (ordered groups, included in 32-test family)
- **Exploratory analyses**: clearly labeled "EXPLORATORY" in all output and manuscript sections

> **Amendment (2026-02-27, before test-seed data collection)**: H1, H2, and H4 scopes
> expanded to all 4 harshness levels (H1 from scarce-only, H2 from sparse+scarce, H4 from
> scarce-only). All changes were made for Holm-Bonferroni family consistency (16 tests
> total) and before any test-seed (100–199) data was collected. Directional predictions
> remain as stated above.

> **Amendment 2 (2026-02-27, before V4/V5 test-seed data collection)**: Added H5–H7 for V4
> (response to stimuli) and V5 (staged lifecycle) capabilities. Holm-Bonferroni family expanded
> from 16 to 28 tests. H5 and H6 use Mann-Whitney U; H7 extends H4's JT trend test to the full
> V0→V5 ladder. All amendments made before any V4/V5 test-seed (100–199) data was collected.

> **Amendment 3 (2026-02-28, after peer review, before any new simulation data with revised
> model)**: Model redesign in response to peer review feedback. Changes made before any
> test-seed (100–199) data was collected with the revised model. All prior test-seed results
> are invalidated by the model changes and will be re-collected.
>
> **Model changes**:
> - V1 boundary now provides protective benefit: entities without V1 suffer per-step energy
>   leakage (`energy_leakage_rate`, default 0.005) and stochastic environmental damage
>   (`env_damage_probability` 0.05, `env_damage_amount` 0.05). V1 boundary absorbs damage
>   proportional to integrity (`boundary_damage_absorption` 0.8) at an integrity cost
>   (`boundary_damage_integrity_cost` 0.02).
> - V2 homeostasis now mitigates overconsumption waste: resource uptake exceeding
>   `optimal_uptake_rate` (default 0.015) incurs waste at `overconsumption_waste_fraction`
>   (default 0.3). V2 regulator state reduces waste by up to 80%.
> - Internalization Index revised from single-channel (energy only) to multi-channel
>   composite: mean of active channels across energy (V3), regulation (V2), behavior (V4),
>   and lifecycle (V5). Per-channel values reported alongside composite.
>
> **Hypothesis changes**:
> - H1 revised: from "V1 always reduces survival" to "V1 reduces survival in resource-abundant
>   environments where leakage/damage pressure is low relative to repair costs." Directional
>   prediction weakened to environment-dependent tradeoff.
> - H8 added: "V0+V1+V2 produces more alive entities than V0+V1" (V2 overconsumption
>   regulation benefit, 4 harshness levels).
> - Holm-Bonferroni family expanded from 28 to 32 tests (H8 × 4 added).
>
> **Exploratory analyses** (not in confirmatory family):
> - Parameter sensitivity sweep: 6 parameters × 5 multipliers × 2 harshness × 30 seeds
>   (V4/V5 parameters excluded because sweep runs V0+V1+V2+V3 only)
> - Robustness runs: n_init=50, T=2000 for key conditions
> - V4 policy weight drift analysis across generations

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
  This applies to V4 sham (computes policy + deducts energy but does not move) and
  V5 sham (tracks stage transitions but applies no behavioral multipliers).
- Recovery advantage of V3 over V0 under shocks is exploratory — not in H1–H8 family.
- V4/V5 survival advantage in *rich* environments is exploratory only — gradient-following
  and lifecycle staging are predicted to matter primarily in resource-scarce conditions.
- Parameter sensitivity sweep (6 params × 5 multipliers) is EXPLORATORY — tests qualitative
  robustness of conclusions, not directional hypotheses.
- Robustness runs (n_init=50, T=2000) are EXPLORATORY — tests scale sensitivity.
- V4 policy weight drift analysis is EXPLORATORY — tests whether evolution occurs at
  experimental timescales.
