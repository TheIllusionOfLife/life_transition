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
    /// Composite = mean of active channels (channels with total > ε).
    /// Returns 0.0 when no channels are active (V0-only entities).
    /// **Independence**: survival metrics (lifespan, persistence) are measured
    /// from agent counts, not from II, to avoid circularity.
    pub fn internalization_index(&self) -> f32 {
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
