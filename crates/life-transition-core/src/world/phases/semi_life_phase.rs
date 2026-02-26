use super::super::World;
use crate::agent::{Agent, OwnerType};
use crate::config::{SemiLifeConfig, SimConfig};
use crate::semi_life::{
    capability, CapabilitySet, DependencyMode, SemiLifeArchetype, SemiLifeRuntime, SemiLifeStage,
};
use crate::spatial;
use crate::spatial::AgentLocation;
use rand::Rng;
use rstar::RTree;
use std::f64::consts::PI;

/// Resolve active capabilities for an archetype, applying any config override.
///
/// Falls back to `archetype.baseline_capabilities()` when no override is present,
/// preserving the biological default for runs that do not need capability ladder experiments.
fn resolve_capabilities(archetype: SemiLifeArchetype, cfg: &SemiLifeConfig) -> CapabilitySet {
    if let Some(&bits) = cfg.capability_overrides.get(archetype.as_str()) {
        CapabilitySet(bits)
    } else {
        archetype.baseline_capabilities()
    }
}

/// Initialise optional runtime fields based on an active CapabilitySet.
///
/// Must be called after overriding `active_capabilities` on a [`SemiLifeRuntime`] to ensure
/// the capability-gated optional fields (`boundary_integrity`, `regulator_state`,
/// `internal_pool`) reflect the actual active set rather than the archetype baseline.
fn apply_capability_fields(sl: &mut SemiLifeRuntime, cfg: &SemiLifeConfig) {
    sl.boundary_integrity = sl
        .active_capabilities
        .has(capability::V1_BOUNDARY)
        .then_some(1.0f32);
    sl.regulator_state = sl
        .active_capabilities
        .has(capability::V2_HOMEOSTASIS)
        .then_some(cfg.regulator_init);
    sl.internal_pool = sl
        .active_capabilities
        .has(capability::V3_METABOLISM)
        .then_some(cfg.internal_pool_init_fraction * cfg.internal_pool_capacity);
    sl.policy = sl
        .active_capabilities
        .has(capability::V4_RESPONSE)
        .then_some(cfg.v4_policy_init);
    sl.stage = sl
        .active_capabilities
        .has(capability::V5_LIFECYCLE)
        .then_some(SemiLifeStage::Dormant);
}

/// Compute V5 behavior multipliers for energy decay, replication, and movement.
///
/// Returns `(decay_mult, repl_mult, speed_mult)`.
/// Entities without V5 get (1.0, 1.0, 1.0) — no behavioral change.
fn v5_multipliers(sl: &SemiLifeRuntime, cfg: &SemiLifeConfig) -> (f32, f32, f32) {
    match sl.stage {
        Some(SemiLifeStage::Dormant) => (cfg.v5_dormant_decay_mult, 0.0, 0.0),
        Some(SemiLifeStage::Active) => (1.0, 1.0, 1.0),
        Some(SemiLifeStage::Dispersal) => (
            cfg.v5_dispersal_decay_mult,
            0.0,
            cfg.v5_dispersal_speed_mult,
        ),
        None => (1.0, 1.0, 1.0),
    }
}

impl World {
    /// Step all SemiLife entities through one simulation timestep.
    ///
    /// Phase order per entity:
    ///      0.5. V5 stage transition (Dormant→Active→Dispersal→Dormant)
    ///      1.   Compute positions (immutable scan)
    ///      1.5. V4 movement pass (policy-driven chemotaxis)
    /// 2.   Maintenance cost (with V5 decay multiplier)
    /// 3.   Resource uptake / Prion contact propagation
    /// 4.   V3 internal pool → energy conversion + pool refill
    /// 5.   V2 regulator cost
    /// 6.   V1 boundary decay/repair
    /// 7.   Energy cap + death check
    /// 8.   Collect V0 replication candidates (with V5 replication multiplier)
    /// 9.   Spawn children (after the main loop to avoid borrow conflicts)
    pub(in crate::world) fn step_semi_life_phase(&mut self, tree: &RTree<AgentLocation>) {
        if !self.config.enable_semi_life || self.semi_lives.is_empty() {
            return;
        }
        // Clone config to avoid re-borrowing self.config inside the loop.
        let cfg = self.config.semi_life_config.clone();
        let dt = self.config.dt as f32;
        let world_size = self.config.world_size;
        let n = self.semi_lives.len();

        // --- Pass 0.5: V5 stage transitions ---
        for sl in self.semi_lives.iter_mut() {
            if !sl.alive {
                continue;
            }
            if let Some(stage) = sl.stage {
                sl.stage_ticks = sl.stage_ticks.saturating_add(1);
                let new_stage = match stage {
                    SemiLifeStage::Dormant => {
                        if sl.maintenance_energy > cfg.v5_activation_threshold {
                            Some(SemiLifeStage::Active)
                        } else {
                            None
                        }
                    }
                    SemiLifeStage::Active => {
                        if sl.stage_ticks >= cfg.v5_dispersal_age {
                            Some(SemiLifeStage::Dispersal)
                        } else {
                            None
                        }
                    }
                    SemiLifeStage::Dispersal => {
                        if sl.stage_ticks >= cfg.v5_dispersal_duration
                            || sl.maintenance_energy < 0.2
                        {
                            Some(SemiLifeStage::Dormant)
                        } else {
                            None
                        }
                    }
                };
                if let Some(next) = new_stage {
                    sl.stage = Some(next);
                    sl.stage_ticks = 0;
                }
            }
        }

        // --- Pass 1: Compute positions (immutable scan) ---
        // Stores None for entities whose agent is missing (shouldn't happen in normal operation).
        let positions: Vec<Option<[f64; 2]>> = (0..n)
            .map(|i| {
                let id = self.semi_lives[i].id;
                self.agents
                    .iter()
                    .find(|a| a.owner_type == OwnerType::SemiLife && a.organism_id == id)
                    .map(|a| a.position)
            })
            .collect();

        // --- Pass 1.5: V4 movement (policy-driven chemotaxis) ---
        #[allow(clippy::needless_range_loop)] // `i` indexes both `positions` and `self.semi_lives`
        for i in 0..n {
            if !self.semi_lives[i].alive {
                continue;
            }
            let pos = match positions[i] {
                Some(p) => p,
                None => continue,
            };
            if !self.semi_lives[i]
                .active_capabilities
                .has(capability::V4_RESPONSE)
            {
                continue;
            }
            let policy = match self.semi_lives[i].policy {
                Some(p) => p,
                None => continue,
            };

            // Compute sensory input vector.
            let cell_size = self.resource_field.cell_size();
            let res_right = self.resource_field.get(pos[0] + cell_size, pos[1]);
            let res_left = self.resource_field.get(pos[0] - cell_size, pos[1]);
            let res_up = self.resource_field.get(pos[0], pos[1] + cell_size);
            let res_down = self.resource_field.get(pos[0], pos[1] - cell_size);
            let grad_x = (res_right - res_left).clamp(-1.0, 1.0);
            let grad_y = (res_up - res_down).clamp(-1.0, 1.0);

            let energy_norm = self.semi_lives[i].maintenance_energy.clamp(0.0, 1.0);
            let boundary_norm = self.semi_lives[i].boundary_integrity.unwrap_or(1.0);

            // Neighbor count: count SemiLife agents nearby (simple O(n) scan; acceptable for
            // small entity counts; R-tree lookup would be overkill for 10-50 entities).
            let neighbor_count = self
                .agents
                .iter()
                .filter(|a| {
                    a.owner_type == OwnerType::SemiLife
                        && a.organism_id != self.semi_lives[i].id
                        && {
                            let dx = (a.position[0] - pos[0])
                                .abs()
                                .min(world_size - (a.position[0] - pos[0]).abs());
                            let dy = (a.position[1] - pos[1])
                                .abs()
                                .min(world_size - (a.position[1] - pos[1]).abs());
                            dx * dx + dy * dy < 25.0 // radius 5.0 squared
                        }
                })
                .count();
            let neighbor_norm = (neighbor_count as f32 / 8.0).clamp(0.0, 1.0);

            let age_norm = (self.semi_lives[i].age_steps as f32 / 500.0).clamp(0.0, 1.0);

            let input = [
                grad_x,
                grad_y,
                energy_norm,
                boundary_norm,
                neighbor_norm,
                age_norm,
                0.0,
                0.0,
            ];

            // Dot product: policy · input → velocity delta.
            let dot: f32 = policy.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

            // V5 speed multiplier.
            let (_, _, speed_mult) = v5_multipliers(&self.semi_lives[i], &cfg);
            let effective_max_speed = cfg.v4_max_speed * speed_mult;

            // Movement: scale gradient direction by dot product magnitude.
            let grad_mag = (grad_x * grad_x + grad_y * grad_y).sqrt().max(f32::EPSILON);
            let speed = dot.clamp(-effective_max_speed, effective_max_speed);
            let dx = (speed * grad_x / grad_mag) as f64;
            let dy = (speed * grad_y / grad_mag) as f64;

            // Energy cost of movement.
            let distance = ((dx * dx + dy * dy) as f32).sqrt();
            self.semi_lives[i].maintenance_energy -= cfg.v4_move_cost * distance;

            // Apply position change to the entity's agent.
            if let Some(agent) = self.agents.iter_mut().find(|a| {
                a.owner_type == OwnerType::SemiLife && a.organism_id == self.semi_lives[i].id
            }) {
                agent.position[0] = (agent.position[0] + dx).rem_euclid(world_size);
                agent.position[1] = (agent.position[1] + dy).rem_euclid(world_size);
            }
        }

        // --- Pass 2: Energy + capability updates, collect replication candidates ---
        // Re-read positions after V4 movement.
        let positions: Vec<Option<[f64; 2]>> = (0..self.semi_lives.len())
            .map(|i| {
                let id = self.semi_lives[i].id;
                self.agents
                    .iter()
                    .find(|a| a.owner_type == OwnerType::SemiLife && a.organism_id == id)
                    .map(|a| a.position)
            })
            .collect();
        let mut to_replicate: Vec<usize> = Vec::new();

        for (i, pos_opt) in positions.iter().enumerate() {
            if !self.semi_lives[i].alive {
                continue;
            }
            let pos = match *pos_opt {
                Some(p) => p,
                None => continue,
            };

            // Reset per-step energy-flow accumulators (used for InternalizationIndex).
            self.semi_lives[i].energy_from_internal = 0.0;
            self.semi_lives[i].energy_from_external = 0.0;

            match self.semi_lives[i].dependency_mode {
                DependencyMode::HostContact => {
                    Self::step_prion_entity_static(
                        &mut self.semi_lives[i],
                        pos,
                        &cfg,
                        dt,
                        tree,
                        world_size,
                    );
                }
                DependencyMode::ResourceField | DependencyMode::Both => {
                    Self::step_genomic_entity_static(
                        i,
                        pos,
                        &cfg,
                        dt,
                        &mut self.semi_lives,
                        &mut self.resource_field,
                    );
                }
            }

            if !self.semi_lives[i].alive {
                self.semi_life_deaths_last_step += 1;
                continue;
            }

            // Collect V0 replication candidates (V5 replication multiplier gates this).
            let (_, repl_mult, _) = v5_multipliers(&self.semi_lives[i], &cfg);
            if repl_mult > 0.0
                && self.semi_lives[i]
                    .active_capabilities
                    .has(capability::V0_REPLICATION)
            {
                self.semi_lives[i].steps_without_replication = self.semi_lives[i]
                    .steps_without_replication
                    .saturating_add(1);
                let energy_ok = self.semi_lives[i].maintenance_energy >= cfg.replication_threshold;
                let boundary_ok = if self.semi_lives[i]
                    .active_capabilities
                    .has(capability::V1_BOUNDARY)
                {
                    self.semi_lives[i]
                        .boundary_integrity
                        .is_none_or(|b| b >= cfg.boundary_replication_min)
                } else {
                    true
                };
                if energy_ok && boundary_ok {
                    to_replicate.push(i);
                }
            }
        }

        // --- Pass 3: Spawn children ---
        // parent_pos is always Some here — entities with None positions were skipped in Pass 2.
        for parent_idx in to_replicate {
            if let Some(parent_pos) = positions[parent_idx] {
                self.spawn_semi_life_child(parent_idx, parent_pos, world_size, &cfg);
            }
        }
    }

    /// Energy update for a genomic archetype entity (Viroid, Virus, Plasmid, ProtoOrganelle).
    ///
    /// Takes explicit split references to `semi_lives` and `resource_field` so the
    /// borrow checker can verify they are disjoint.
    fn step_genomic_entity_static(
        i: usize,
        pos: [f64; 2],
        cfg: &SemiLifeConfig,
        dt: f32,
        semi_lives: &mut [SemiLifeRuntime],
        resource_field: &mut crate::resource::ResourceField,
    ) {
        let sl = &mut semi_lives[i];

        // 1. Maintenance cost (scaled by V5 decay multiplier).
        let (decay_mult, _, _) = v5_multipliers(sl, cfg);
        sl.maintenance_energy -= cfg.maintenance_cost * dt * decay_mult;

        // 2. V3 — Internal pool → energy conversion.
        if sl.active_capabilities.has(capability::V3_METABOLISM) {
            if let Some(pool) = sl.internal_pool {
                let conversion = (cfg.internal_conversion_rate * pool * dt).min(pool);
                sl.maintenance_energy += conversion;
                sl.internal_pool = Some((pool - conversion).max(0.0));
                sl.energy_from_internal += conversion;
            }
        }

        // 3. Compute uptake scale from V2 regulator (no cost yet; cost applied below).
        let uptake_scale = if sl.active_capabilities.has(capability::V2_HOMEOSTASIS) {
            sl.regulator_state.map_or(cfg.regulator_uptake_scale, |r| {
                r * cfg.regulator_uptake_scale
            })
        } else {
            1.0
        };

        // NLL releases the sl borrow here (last use above); resource_field access below is safe.
        let _ = sl;

        // 4. Resource field uptake (energy).
        let want_energy = cfg.resource_uptake_rate * uptake_scale * dt;
        let got_energy = resource_field.take(pos[0], pos[1], want_energy);
        semi_lives[i].maintenance_energy += got_energy;
        semi_lives[i].energy_from_external += got_energy;

        // 5. V3 — Pool refill from resource field.
        if semi_lives[i]
            .active_capabilities
            .has(capability::V3_METABOLISM)
        {
            if let Some(pool) = semi_lives[i].internal_pool {
                let want_pool = cfg.internal_pool_uptake_rate * dt;
                let got_pool = resource_field.take(pos[0], pos[1], want_pool);
                semi_lives[i].internal_pool =
                    Some((pool + got_pool).min(cfg.internal_pool_capacity));
            }
        }

        // 6. V2 — Regulator maintenance cost.
        if semi_lives[i]
            .active_capabilities
            .has(capability::V2_HOMEOSTASIS)
        {
            semi_lives[i].maintenance_energy -= cfg.regulator_cost_per_step * dt;
        }

        // 7. V1 — Boundary decay + repair.
        if semi_lives[i]
            .active_capabilities
            .has(capability::V1_BOUNDARY)
        {
            if let Some(integrity) = semi_lives[i].boundary_integrity {
                let decay = cfg.boundary_decay_rate * dt;
                let repair = if semi_lives[i].maintenance_energy > cfg.boundary_repair_rate * dt {
                    semi_lives[i].maintenance_energy -= cfg.boundary_repair_rate * dt;
                    cfg.boundary_repair_rate * dt
                } else {
                    0.0
                };
                let new_integrity = (integrity - decay + repair).clamp(0.0, 1.0);
                semi_lives[i].boundary_integrity = Some(new_integrity);
                if new_integrity <= cfg.boundary_death_threshold {
                    semi_lives[i].alive = false;
                    return;
                }
            }
        }

        // 8. Energy cap + death check.
        semi_lives[i].maintenance_energy =
            semi_lives[i].maintenance_energy.min(cfg.energy_capacity);
        if semi_lives[i].maintenance_energy <= 0.0 {
            semi_lives[i].alive = false;
        }
        semi_lives[i].age_steps += 1;
    }

    /// Energy update for a Prion entity (contact-based propagation, no resource field).
    ///
    /// Prions are NOT V0 replicators; their propagation mechanism is fundamentally
    /// different from genomic replication. This is a dedicated module per the research plan.
    fn step_prion_entity_static(
        sl: &mut SemiLifeRuntime,
        pos: [f64; 2],
        cfg: &SemiLifeConfig,
        dt: f32,
        tree: &RTree<AgentLocation>,
        world_size: f64,
    ) {
        // 1. Fragmentation loss (bounded-runaway guard — prevents exponential takeover).
        sl.maintenance_energy *= 1.0 - cfg.prion_fragmentation_loss * dt;

        // 2. Death by dilution/degradation.
        if sl.maintenance_energy <= cfg.prion_dilution_death_energy {
            sl.alive = false;
            return;
        }

        // 3. Contact propagation: find nearby Organism agents in R-tree.
        //    Uses spatial::count_neighbors for toroidal-correct circular radius (not AABB square).
        //    build_index_active only contains Organism agents, so all results are hosts.
        //    u32::MAX is used as self_id since Prion agents are SemiLife type, never in this tree.
        let contacts =
            spatial::count_neighbors(tree, pos, cfg.prion_contact_radius, u32::MAX, world_size)
                as u32;

        if contacts > 0 {
            // Aggregate conversion: p(at least one conversion) ≈ 1 - (1-p)^contacts.
            // Simplified: uniform probability * contacts, capped at 1.0.
            let p = (cfg.prion_conversion_prob * contacts as f32 * dt).min(1.0);
            if sl.rng.random::<f32>() < p {
                sl.maintenance_energy =
                    (sl.maintenance_energy + cfg.prion_contact_gain).min(cfg.energy_capacity);
                sl.energy_from_external += cfg.prion_contact_gain;
            }
        }

        sl.age_steps += 1;
    }

    /// Spawn a child SemiLife entity from the given parent.
    ///
    /// `parent_pos` is pre-computed in Pass 1 to avoid an O(N_agents) scan per replication.
    /// Deducts `replication_cost` from parent energy, creates a new runtime and agent.
    fn spawn_semi_life_child(
        &mut self,
        parent_idx: usize,
        parent_pos: [f64; 2],
        world_size: f64,
        cfg: &SemiLifeConfig,
    ) {
        if self.semi_lives[parent_idx].maintenance_energy < cfg.replication_cost {
            self.semi_lives[parent_idx].failed_replications += 1;
            return;
        }

        // Guard global agent cap before any allocation.
        if self.agents.len() >= SimConfig::MAX_TOTAL_AGENTS {
            self.semi_lives[parent_idx].failed_replications += 1;
            return;
        }

        // Check child id space.
        let child_id = match u16::try_from(self.semi_lives.len()) {
            Ok(id) => id,
            Err(_) => {
                self.semi_lives[parent_idx].failed_replications += 1;
                return;
            }
        };

        // Allocate a new agent ID.
        let Some(agent_id) = self.next_agent_id_checked() else {
            self.semi_lives[parent_idx].failed_replications += 1;
            return;
        };

        // Random position near parent.
        let theta = self.rng.random::<f64>() * 2.0 * PI;
        let radius = self.rng.random::<f64>().sqrt() * cfg.replication_spawn_radius;
        let (st, ct) = theta.sin_cos();
        let child_pos = [
            (parent_pos[0] + radius * ct).rem_euclid(world_size),
            (parent_pos[1] + radius * st).rem_euclid(world_size),
        ];
        let child_agent = Agent::for_semi_life(agent_id, child_id, child_pos);
        self.agents.push(child_agent);

        // Create child runtime.
        let parent_archetype = self.semi_lives[parent_idx].archetype;
        let child_stable_id = self.next_semi_life_stable_id;
        self.next_semi_life_stable_id = self.next_semi_life_stable_id.saturating_add(1);

        let mut child = SemiLifeRuntime::new(
            child_id,
            child_stable_id,
            parent_archetype,
            cfg.replication_cost,
            self.config.seed,
            cfg.regulator_init,
            cfg.internal_pool_init_fraction * cfg.internal_pool_capacity,
        );
        // Inherit the same capability override as the parent so all generations are consistent.
        child.active_capabilities = resolve_capabilities(parent_archetype, cfg);
        apply_capability_fields(&mut child, cfg);

        // V4: Inherit parent policy with Gaussian mutation noise.
        if let Some(parent_policy) = self.semi_lives[parent_idx].policy {
            let mut child_policy = parent_policy;
            for w in child_policy.iter_mut() {
                // Box-Muller transform for Gaussian noise (avoids rand_distr dependency).
                let u1: f32 = child.rng.random::<f32>().max(f32::EPSILON);
                let u2: f32 = child.rng.random::<f32>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                *w += z * cfg.v4_mutation_sigma;
            }
            child.policy = Some(child_policy);
        }

        child.agent_ids.push(agent_id);
        self.semi_lives.push(child);

        // Deduct cost and record.
        self.semi_lives[parent_idx].maintenance_energy -= cfg.replication_cost;
        self.semi_lives[parent_idx].replications += 1;
        self.semi_lives[parent_idx].steps_without_replication = 0;
        self.semi_life_births_last_step += 1;
        self.semi_life_replications_total += 1;
    }

    /// Prune dead SemiLife entities and remap their agents.
    ///
    /// Mirrors `prune_dead_entities` for organisms — keeps arrays compact to avoid
    /// unbounded growth during long experiments.
    pub(in crate::world) fn prune_dead_semi_lives(&mut self) {
        if self.semi_lives.iter().all(|sl| sl.alive) {
            return;
        }

        let old_semi_lives = std::mem::take(&mut self.semi_lives);
        let mut remap = vec![None::<u16>; old_semi_lives.len()];
        let mut new_semi_lives: Vec<SemiLifeRuntime> = Vec::with_capacity(old_semi_lives.len());

        for (old_idx, mut sl) in old_semi_lives.into_iter().enumerate() {
            if !sl.alive {
                continue;
            }
            let new_id = match u16::try_from(new_semi_lives.len()) {
                Ok(id) => id,
                Err(_) => break, // ID space exhausted — leave remaining entities dropped.
            };
            remap[old_idx] = Some(new_id);
            sl.id = new_id;
            sl.agent_ids.clear();
            new_semi_lives.push(sl);
        }

        // Remap SemiLife agents; drop agents belonging to dead entities.
        let old_agents = std::mem::take(&mut self.agents);
        let mut new_agents: Vec<Agent> = Vec::with_capacity(old_agents.len());
        for mut agent in old_agents {
            match agent.owner_type {
                OwnerType::Organism => new_agents.push(agent),
                OwnerType::SemiLife => {
                    if let Some(&Some(new_id)) = remap.get(agent.organism_id as usize) {
                        agent.organism_id = new_id;
                        new_agents.push(agent);
                    }
                    // else: agent belongs to dead SemiLife — drop it.
                }
            }
        }

        self.semi_lives = new_semi_lives;
        self.agents = new_agents;

        // Rebuild agent_ids lists.
        for agent in &self.agents {
            if agent.owner_type == OwnerType::SemiLife {
                self.semi_lives[agent.organism_id as usize]
                    .agent_ids
                    .push(agent.id);
            }
        }
    }

    /// Returns the number of alive SemiLife entities.
    pub fn semi_life_alive_count(&self) -> usize {
        self.semi_lives.iter().filter(|sl| sl.alive).count()
    }

    /// Returns the total number of successful SemiLife replications since world creation.
    ///
    /// Unlike per-entity `replications` counters, this is never decremented by pruning —
    /// making it reliable for cumulative time-series analysis in experiment scripts.
    pub fn semi_life_replications_total(&self) -> u64 {
        self.semi_life_replications_total
    }

    /// Returns snapshots of all SemiLife entities (alive and recently dead).
    pub fn semi_life_snapshots(&self) -> Vec<crate::metrics::SemiLifeSnapshot> {
        self.semi_lives
            .iter()
            .map(crate::metrics::SemiLifeSnapshot::from_runtime)
            .collect()
    }

    /// Initialise SemiLife entities from config and push them into the world.
    pub(in crate::world) fn init_semi_lives(&mut self) {
        if !self.config.enable_semi_life {
            return;
        }
        let cfg = self.config.semi_life_config.clone();
        let world_size = self.config.world_size;

        for &archetype in &cfg.enabled_archetypes.clone() {
            for _ in 0..cfg.num_per_archetype {
                if self.agents.len() >= SimConfig::MAX_TOTAL_AGENTS {
                    break;
                }
                let sl_id = match u16::try_from(self.semi_lives.len()) {
                    Ok(id) => id,
                    Err(_) => break,
                };
                let Some(agent_id) = self.next_agent_id_checked() else {
                    break;
                };
                let stable_id = self.next_semi_life_stable_id;
                self.next_semi_life_stable_id = self.next_semi_life_stable_id.saturating_add(1);

                let pos_x = self.rng.random::<f64>() * world_size;
                let pos_y = self.rng.random::<f64>() * world_size;

                let mut sl = SemiLifeRuntime::new(
                    sl_id,
                    stable_id,
                    archetype,
                    cfg.initial_energy,
                    self.config.seed,
                    cfg.regulator_init,
                    cfg.internal_pool_init_fraction * cfg.internal_pool_capacity,
                );
                // Apply capability override (if any) and re-gate optional fields.
                sl.active_capabilities = resolve_capabilities(archetype, &cfg);
                apply_capability_fields(&mut sl, &cfg);
                sl.agent_ids.push(agent_id);

                let agent = Agent::for_semi_life(agent_id, sl_id, [pos_x, pos_y]);
                self.agents.push(agent);
                self.semi_lives.push(sl);
            }
        }
    }
}
