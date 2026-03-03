use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

/// Bitmask bit positions for the V0–V5 capability ladder.
pub mod capability {
    /// V0: Base replication — genomic copy production.
    pub const V0_REPLICATION: u8 = 1 << 0;
    /// V1: Boundary / capsid-like integrity maintenance.
    pub const V1_BOUNDARY: u8 = 1 << 1;
    /// V2: Homeostasis — replication throttling and internal regulation.
    pub const V2_HOMEOSTASIS: u8 = 1 << 2;
    /// V3: Metabolism — internal resource conversion (pool → energy).
    pub const V3_METABOLISM: u8 = 1 << 3;
    /// V4: Response to stimuli — sensing and action selection.
    pub const V4_RESPONSE: u8 = 1 << 4;
    /// V5: Growth/development — staged lifecycle.
    pub const V5_LIFECYCLE: u8 = 1 << 5;
}

/// Compact u8 bitmask of the V0–V5 capability ladder.
///
/// Each bit position corresponds to a capability constant in [`capability`].
/// Separate `baseline_capabilities` and `active_capabilities` in
/// [`SemiLifeRuntime`] allow capability ablation without losing archetype identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet(pub u8);

impl CapabilitySet {
    pub const NONE: Self = Self(0);

    /// Returns `true` if the given capability flag is set.
    #[inline]
    pub fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    /// Sets the given capability flag.
    #[inline]
    pub fn set(&mut self, flag: u8) {
        self.0 |= flag;
    }

    /// Clears the given capability flag (for ablation).
    #[inline]
    pub fn clear(&mut self, flag: u8) {
        self.0 &= !flag;
    }
}

/// Biological archetype of a semi-life entity, determining baseline capabilities,
/// dependency mode, and the key research question it addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemiLifeArchetype {
    /// Conformational propagation — no genome, contact-dependent spreading.
    /// Starts with NO capabilities; propagation is non-genomic.
    Prion,
    /// Minimal genomic replicator, resource-field dependent. Starts at V0.
    Viroid,
    /// Genomic replicator with boundary maintenance. Starts at V0+V1.
    Virus,
    /// Genomic replicator with different ecological profile from Viroid. Starts at V0.
    Plasmid,
    /// Proto-endosymbiont: boundary + homeostasis + metabolism, but NO replication.
    /// Starts at V1+V2+V3 (gains liberation when V0 is added in experiments).
    ProtoOrganelle,
}

impl SemiLifeArchetype {
    /// Returns the baseline [`CapabilitySet`] this archetype starts with.
    pub fn baseline_capabilities(self) -> CapabilitySet {
        use capability::*;
        match self {
            Self::Prion => CapabilitySet::NONE,
            Self::Viroid => CapabilitySet(V0_REPLICATION),
            Self::Virus => CapabilitySet(V0_REPLICATION | V1_BOUNDARY),
            Self::Plasmid => CapabilitySet(V0_REPLICATION),
            Self::ProtoOrganelle => CapabilitySet(V1_BOUNDARY | V2_HOMEOSTASIS | V3_METABOLISM),
        }
    }

    /// Returns the default [`DependencyMode`] for this archetype.
    pub fn default_dependency_mode(self) -> DependencyMode {
        match self {
            Self::Prion => DependencyMode::HostContact,
            _ => DependencyMode::ResourceField,
        }
    }

    /// Returns the archetype name as a string for logging/serialization.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prion => "prion",
            Self::Viroid => "viroid",
            Self::Virus => "virus",
            Self::Plasmid => "plasmid",
            Self::ProtoOrganelle => "proto_organelle",
        }
    }
}

/// How a SemiLife entity draws external resources from the environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyMode {
    /// Draw energy from the spatial resource grid (Viroid, Virus, Plasmid, ProtoOrganelle).
    ResourceField,
    /// Require contact with a nearby entity to propagate (Prion).
    HostContact,
    /// Both resource grid and contact — parasitic mode.
    Both,
}

/// Staged lifecycle state for V5 (currently a placeholder; populated in PR 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemiLifeStage {
    Dormant,
    Active,
    Dispersal,
}

/// Runtime state for a single SemiLife entity (one of the borderline-life archetypes).
///
/// Capability-gated optional fields are `Some(...)` when the corresponding bit is set
/// in `active_capabilities`, and `None` otherwise. Maintaining separate
/// `baseline_capabilities` vs `active_capabilities` allows ablation without losing
/// archetype identity.
#[derive(Debug)]
pub struct SemiLifeRuntime {
    /// Local array index (may be remapped after compaction).
    pub id: u16,
    /// Unique persistent identifier — never reassigned.
    pub stable_id: u64,
    /// Biological archetype determining research question and default capabilities.
    pub archetype: SemiLifeArchetype,
    /// Capabilities this archetype starts with (immutable reference for ablation studies).
    pub baseline_capabilities: CapabilitySet,
    /// Currently running capabilities (can be ablated or shammed for controls).
    pub active_capabilities: CapabilitySet,

    pub age_steps: usize,
    pub alive: bool,

    /// Universal energy pool — NOT tied to V0; ProtoOrganelle needs energy without replication.
    pub maintenance_energy: f32,

    // Capability-gated optional fields (allocated on first capability activation)
    /// V1 — boundary/capsid integrity in [0, 1].
    pub boundary_integrity: Option<f32>,
    /// V2 — homeostasis regulator state in [0, 1].
    pub regulator_state: Option<f32>,
    /// V3 — internal metabolite pool.
    pub internal_pool: Option<f32>,
    /// V4 — policy weights for response to stimuli.
    pub policy: Option<[f32; 8]>,
    /// V5 — staged lifecycle state.
    pub stage: Option<SemiLifeStage>,
    /// V5 — ticks spent in current stage (reset on stage transition).
    pub stage_ticks: u32,

    /// How this entity draws external resources.
    pub dependency_mode: DependencyMode,

    // Replication counters (always tracked for debugging + reviewer questions)
    pub replications: u32,
    pub failed_replications: u32,
    pub steps_without_replication: u32,

    // Per-step energy flow accumulators (reset each step) — used for InternalizationIndex.
    /// Energy obtained from internal conversion (V3+) this step.
    pub energy_from_internal: f32,
    /// Energy obtained from external resource field this step.
    pub energy_from_external: f32,

    // Multi-channel II accumulators (reset each step).
    /// V2 regulation channel: waste reduction achieved by regulator this step.
    pub regulation_internal: f32,
    /// V2 regulation channel: total unregulated waste (event-gated; >0 only when overconsumption fires).
    pub regulation_total: f32,
    /// V4 behavior channel: policy-driven movement fraction this step.
    pub behavior_internal: f32,
    /// V4 behavior channel: baseline (1.0 per step if V4 active, for normalization).
    pub behavior_total: f32,
    /// V5 lifecycle channel: internal-state-driven stage transitions this step.
    pub lifecycle_internal: f32,
    /// V5 lifecycle channel: baseline (1.0 per step if V5 active, for normalization).
    pub lifecycle_total: f32,

    /// Per-entity deterministic RNG stream (seeded from world seed XOR stable_id).
    pub rng: ChaCha12Rng,

    /// IDs of agents belonging to this entity (1 agent per entity for V0).
    pub agent_ids: Vec<u32>,
}

impl SemiLifeRuntime {
    /// Construct a new SemiLifeRuntime with the given archetype's baseline capabilities.
    ///
    /// `regulator_init` and `internal_pool_init` must come from [`SemiLifeConfig`] so that
    /// config parameter sweeps over those fields produce distinct simulation outcomes.
    pub fn new(
        id: u16,
        stable_id: u64,
        archetype: SemiLifeArchetype,
        initial_energy: f32,
        world_seed: u64,
        regulator_init: f32,
        internal_pool_init: f32,
    ) -> Self {
        let baseline = archetype.baseline_capabilities();
        let active = baseline;
        let dependency_mode = archetype.default_dependency_mode();

        // Gate optional fields on active capabilities; use config-supplied init values.
        let boundary_integrity = active.has(capability::V1_BOUNDARY).then_some(1.0f32);
        let regulator_state = active
            .has(capability::V2_HOMEOSTASIS)
            .then_some(regulator_init);
        let internal_pool = active
            .has(capability::V3_METABOLISM)
            .then_some(internal_pool_init);

        Self {
            id,
            stable_id,
            archetype,
            baseline_capabilities: baseline,
            active_capabilities: active,
            age_steps: 0,
            alive: true,
            maintenance_energy: initial_energy,
            boundary_integrity,
            regulator_state,
            internal_pool,
            policy: None,
            stage: None,
            stage_ticks: 0,
            dependency_mode,
            replications: 0,
            failed_replications: 0,
            steps_without_replication: 0,
            energy_from_internal: 0.0,
            energy_from_external: 0.0,
            regulation_internal: 0.0,
            regulation_total: 0.0,
            behavior_internal: 0.0,
            behavior_total: 0.0,
            lifecycle_internal: 0.0,
            lifecycle_total: 0.0,
            rng: ChaCha12Rng::seed_from_u64(world_seed ^ stable_id),
            agent_ids: Vec::new(),
        }
    }

    /// Compute the per-channel Internalization Index values.
    ///
    /// Returns `(ii_energy, ii_regulation, ii_behavior, ii_lifecycle)`.
    /// Each channel is the ratio of internal contribution to total for that dimension.
    /// A channel returns 0.0 if its total is below ε (capability not active or no activity).
    pub fn ii_channels(&self) -> (f32, f32, f32, f32) {
        let ratio = |internal: f32, total: f32| -> f32 {
            if total <= f32::EPSILON {
                0.0
            } else {
                (internal / total).clamp(0.0, 1.0)
            }
        };
        let ii_energy = {
            let total = self.energy_from_internal + self.energy_from_external;
            ratio(self.energy_from_internal, total)
        };
        let ii_regulation = ratio(self.regulation_internal, self.regulation_total);
        let ii_behavior = ratio(self.behavior_internal, self.behavior_total);
        let ii_lifecycle = ratio(self.lifecycle_internal, self.lifecycle_total);
        (ii_energy, ii_regulation, ii_behavior, ii_lifecycle)
    }

    /// Compute the composite multi-channel InternalizationIndex for this step.
    ///
    /// **Fixed 4-channel formula** (Amendment 4): `(IIE + IIR + IIB + IIL) / 4`.
    /// Inactive channels contribute 0 to the numerator and 4 to the denominator,
    /// making II comparable across capability levels.
    /// Returns 0.0 when no channels are active (V0-only entities).
    /// **Independence**: survival metrics (lifespan, persistence) are measured
    /// from agent counts, not from II, to avoid circularity.
    pub fn internalization_index(&self) -> f32 {
        let (e, r, b, l) = self.ii_channels();
        (e + r + b + l) / 4.0
    }

    /// Compute the composite II using the old active-channel-mean formula.
    ///
    /// Retained for supplementary reporting (side-by-side transparency).
    /// Composite = mean of channels with total > ε; returns 0.0 if none active.
    pub fn internalization_index_active_mean(&self) -> f32 {
        let (e, r, b, l) = self.ii_channels();
        let channels = [
            (e, self.energy_from_internal + self.energy_from_external),
            (r, self.regulation_total),
            (b, self.behavior_total),
            (l, self.lifecycle_total),
        ];
        let (sum, count) = channels
            .iter()
            .filter(|(_, total)| *total > f32::EPSILON)
            .fold((0.0f32, 0u32), |(s, c), (val, _)| (s + val, c + 1));
        if count == 0 {
            0.0
        } else {
            sum / count as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_entity(caps: u8) -> SemiLifeRuntime {
        let mut rt = SemiLifeRuntime::new(0, 1, SemiLifeArchetype::Viroid, 0.5, 42, 1.0, 0.5);
        rt.active_capabilities = CapabilitySet(caps);
        rt
    }

    #[test]
    fn ii_fixed_v3_only_returns_quarter() {
        // V3-only: ii_energy=0.6, others=0 → fixed composite = 0.6/4 = 0.15
        let mut rt = make_test_entity(capability::V0_REPLICATION | capability::V3_METABOLISM);
        rt.energy_from_internal = 0.6;
        rt.energy_from_external = 0.4;
        let ii = rt.internalization_index();
        let expected = 0.6 / 4.0; // 0.15
        assert!(
            (ii - expected).abs() < 1e-5,
            "V3-only fixed 4-channel: expected {expected}, got {ii}"
        );
    }

    #[test]
    fn ii_fixed_all_four_channels_active() {
        // All 4 channels active: composite = mean of all 4
        let mut rt = make_test_entity(
            capability::V0_REPLICATION
                | capability::V1_BOUNDARY
                | capability::V2_HOMEOSTASIS
                | capability::V3_METABOLISM
                | capability::V4_RESPONSE
                | capability::V5_LIFECYCLE,
        );
        rt.energy_from_internal = 0.8;
        rt.energy_from_external = 0.2;
        rt.regulation_internal = 0.6;
        rt.regulation_total = 1.0;
        rt.behavior_internal = 0.4;
        rt.behavior_total = 1.0;
        rt.lifecycle_internal = 0.2;
        rt.lifecycle_total = 1.0;
        let ii = rt.internalization_index();
        let expected = (0.8 + 0.6 + 0.4 + 0.2) / 4.0; // 0.5
        assert!(
            (ii - expected).abs() < 1e-5,
            "All channels active: expected {expected}, got {ii}"
        );
    }

    #[test]
    fn ii_fixed_v0_only_returns_zero() {
        // V0-only: no channels active → all contribute 0 → composite = 0
        let rt = make_test_entity(capability::V0_REPLICATION);
        assert_eq!(rt.internalization_index(), 0.0);
    }

    #[test]
    fn ii_active_mean_preserves_old_behavior() {
        // Active-channel mean (old formula) for V3-only: 0.6/1 = 0.6
        let mut rt = make_test_entity(capability::V0_REPLICATION | capability::V3_METABOLISM);
        rt.energy_from_internal = 0.6;
        rt.energy_from_external = 0.4;
        let ii_active = rt.internalization_index_active_mean();
        assert!(
            (ii_active - 0.6).abs() < 1e-5,
            "Active-mean V3-only: expected 0.6, got {ii_active}"
        );
    }

    #[test]
    fn ii_fixed_is_comparable_across_levels() {
        // Key property: adding capabilities with low II should DECREASE composite
        // under old formula but correctly reflect total internalization under fixed.
        let mut v3_only = make_test_entity(capability::V0_REPLICATION | capability::V3_METABOLISM);
        v3_only.energy_from_internal = 0.6;
        v3_only.energy_from_external = 0.4;

        let mut v3_v4 = make_test_entity(
            capability::V0_REPLICATION | capability::V3_METABOLISM | capability::V4_RESPONSE,
        );
        v3_v4.energy_from_internal = 0.6;
        v3_v4.energy_from_external = 0.4;
        v3_v4.behavior_internal = 0.1;
        v3_v4.behavior_total = 1.0;

        // Fixed formula: V3+V4 should be higher than V3-only (added behavior channel)
        assert!(
            v3_v4.internalization_index() > v3_only.internalization_index(),
            "Fixed formula: V3+V4 ({}) should exceed V3-only ({})",
            v3_v4.internalization_index(),
            v3_only.internalization_index()
        );
    }
}

/// Dedicated Prion propagation parameters — NOT a V0 branch.
///
/// Prion replication is fundamentally different from genomic replication:
/// it proceeds through conformational conversion of nearby susceptible entities,
/// with decay through dilution and fragmentation rather than starvation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrionPropagation {
    /// Radius within which contact propagation can occur (world units).
    pub contact_radius: f64,
    /// Per-contact probability of conformational conversion per step.
    pub conversion_prob: f32,
    /// Per-step fractional mass loss (prevents runaway growth; bounded runaway guard).
    pub fragmentation_loss: f32,
    /// Prion population fraction below which the entity is considered diluted to death.
    pub dilution_death_threshold: f32,
}

impl Default for PrionPropagation {
    fn default() -> Self {
        Self {
            contact_radius: 5.0,
            conversion_prob: 0.05,
            fragmentation_loss: 0.01,
            dilution_death_threshold: 0.1,
        }
    }
}
