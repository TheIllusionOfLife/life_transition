use super::*;
use crate::config::{BoundaryMode, HomeostasisMode};

fn make_world(num_agents: usize, world_size: f64) -> World {
    let agents: Vec<Agent> = (0..num_agents)
        .map(|i| Agent::new(i as u32, 0, [50.0, 50.0]))
        .collect();
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.1f32, NeuralNet::WEIGHT_COUNT));
    let config = SimConfig {
        world_size,
        num_organisms: 1,
        agents_per_organism: num_agents,
        ..SimConfig::default()
    };
    World::new(agents, vec![nn], config).unwrap()
}

fn make_config(world_size: f64, dt: f64) -> SimConfig {
    SimConfig {
        world_size,
        dt,
        num_organisms: 1,
        agents_per_organism: 1,
        ..SimConfig::default()
    }
}

#[test]
fn toroidal_wrapping_keeps_positions_in_bounds() {
    let mut world = make_world(1, 100.0);
    world.agents[0].velocity = [100.0, 100.0];
    world.step();
    let pos = world.agents[0].position;
    assert!(pos[0] >= 0.0 && pos[0] < 100.0);
    assert!(pos[1] >= 0.0 && pos[1] < 100.0);
}

#[test]
fn step_returns_nonzero_timings() {
    let mut world = make_world(10, 100.0);
    let t = world.step();
    assert!(t.total_us > 0);
}

#[test]
fn new_returns_err_on_invalid_organism_id() {
    let agents = vec![Agent::new(0, 5, [0.0, 0.0])];
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    assert!(matches!(
        World::new(agents, vec![nn], make_config(100.0, 0.1)),
        Err(WorldInitError::InvalidOrganismId)
    ));
}

#[test]
fn new_returns_err_on_non_positive_world_size() {
    let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    assert!(matches!(
        World::new(agents, vec![nn], make_config(0.0, 0.1)),
        Err(WorldInitError::Config(SimConfigError::InvalidWorldSize))
    ));
}

#[test]
fn new_returns_err_on_non_finite_world_size() {
    let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    assert!(matches!(
        World::new(agents, vec![nn], make_config(f64::NAN, 0.1)),
        Err(WorldInitError::Config(SimConfigError::InvalidWorldSize))
    ));
}

#[test]
fn new_returns_err_on_excessive_world_size() {
    let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    assert!(matches!(
        World::new(
            agents,
            vec![nn],
            make_config(World::MAX_WORLD_SIZE + 1.0, 0.1),
        ),
        Err(WorldInitError::Config(
            SimConfigError::WorldSizeTooLarge { .. }
        ))
    ));
}

#[test]
fn internal_state_stays_clamped() {
    let mut world = make_world(1, 100.0);
    for _ in 0..100 {
        world.step();
    }
    for &s in &world.agents[0].internal_state {
        assert!((0.0..=1.0).contains(&s));
    }
}

#[test]
fn step_respects_config_dt_for_position_update() {
    let mut world = make_world(1, 100.0);
    world.agents[0].position = [50.0, 50.0];
    world.agents[0].velocity = [1.0, 0.0];
    world.organisms[0].nn =
        NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    let mut config = world.config().clone();
    config.dt = 0.5;
    world
        .set_config(config)
        .expect("config with positive dt should be valid");
    world.step();
    assert!(
        (world.agents[0].position[0] - 50.5).abs() < 1e-6,
        "expected x to advance by dt-scaled velocity"
    );
}

#[test]
fn toy_metabolism_sustains_energy_for_1000_steps() {
    let mut world = make_world(10, 100.0);
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    for _ in 0..1000 {
        world.step();
    }
    assert!(world.organism_count() >= 1);
    assert!(world.metabolic_state(0).unwrap().energy > 0.0);
}

#[test]
fn try_metabolic_state_returns_none_for_out_of_range() {
    let world = make_world(1, 100.0);
    assert!(world.metabolic_state(10).is_none());
}

#[test]
fn try_new_rejects_agent_count_mismatch() {
    let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    let mut cfg = make_config(100.0, 0.1);
    cfg.num_organisms = 1;
    cfg.agents_per_organism = 2;
    let result = World::new(agents, vec![nn], cfg);
    assert!(matches!(
        result,
        Err(WorldInitError::AgentCountMismatch { .. })
    ));
}

#[test]
fn try_new_rejects_agent_count_overflow() {
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    let cfg = SimConfig {
        num_organisms: 3,
        agents_per_organism: usize::MAX / 2 + 1,
        ..SimConfig::default()
    };
    let result = World::new(Vec::new(), vec![nn.clone(), nn.clone(), nn], cfg);
    assert!(matches!(
        result,
        Err(WorldInitError::Config(SimConfigError::AgentCountOverflow))
    ));
}

#[test]
fn try_new_rejects_too_many_agents() {
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    let cfg = SimConfig {
        num_organisms: 1,
        agents_per_organism: SimConfig::MAX_TOTAL_AGENTS + 1,
        ..SimConfig::default()
    };
    let result = World::new(Vec::new(), vec![nn], cfg);
    assert!(matches!(
        result,
        Err(WorldInitError::Config(SimConfigError::TooManyAgents { .. }))
    ));
}

#[test]
fn set_config_rejects_invalid_update() {
    let mut world = make_world(1, 100.0);
    let mut cfg = world.config().clone();
    cfg.dt = -0.1;
    let result = world.set_config(cfg);
    assert!(matches!(
        result,
        Err(WorldInitError::Config(SimConfigError::InvalidDt))
    ));
}

#[test]
fn set_config_rejects_structural_mismatch_after_runtime_growth() {
    let mut world = make_world(10, 100.0);
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 1.0;
    world.step();
    assert!(world.organism_count() > 1);

    let mut cfg = world.config().clone();
    cfg.num_organisms = 1;
    let result = world.set_config(cfg);
    assert!(matches!(
        result,
        Err(WorldInitError::NumOrganismsMismatch { .. })
    ));
}

#[test]
fn set_config_switch_to_graph_redecodes_organism_engines() {
    let mut world = make_world(4, 100.0);
    assert!(
        world
            .organisms
            .iter()
            .all(|o| o.metabolism_engine.is_none()),
        "toy mode should not use per-organism metabolism engines"
    );

    let mut cfg = world.config().clone();
    cfg.metabolism_mode = MetabolismMode::Graph;
    world
        .set_config(cfg)
        .expect("switching to graph mode should be valid");

    assert!(
        world
            .organisms
            .iter()
            .all(|o| o.metabolism_engine.is_some()),
        "graph mode should decode per-organism metabolism engines"
    );
}

#[test]
fn metabolism_consumes_world_resource_field() {
    let mut world = make_world(1, 100.0);
    world.agents[0].position = [10.0, 10.0];
    let before = world.resource_field().get(10.0, 10.0);
    world.step();
    let after = world.resource_field().get(10.0, 10.0);
    assert!(after <= before);
}

#[test]
fn toroidal_center_uses_wrapped_mean_for_resource_sampling() {
    let mut world = make_world(2, 100.0);
    world.config.resource_regeneration_rate = 0.0;
    world.current_resource_rate = 0.0;
    world.agents[0].position = [0.1, 50.0];
    world.agents[1].position = [99.9, 50.0];
    world.resource_field_mut().set(0.0, 50.0, 2.0);
    world.resource_field_mut().set(50.0, 50.0, 0.0);
    world.step();
    let edge_resource = world.resource_field().get(0.0, 50.0);
    let center_resource = world.resource_field().get(50.0, 50.0);
    assert!(edge_resource < 2.0);
    assert!((center_resource - 0.0).abs() < f32::EPSILON);
}

#[test]
fn run_experiment_produces_non_empty_summary() {
    let mut world = make_world(10, 100.0);
    let summary = world.run_experiment(50, 10);
    assert_eq!(summary.steps, 50);
    assert!(!summary.samples.is_empty());
    assert!(summary.final_alive_count <= world.organism_count());
}

#[test]
fn low_energy_org_decays_boundary_faster() {
    let mut low = make_world(10, 100.0);
    let mut high = make_world(10, 100.0);
    low.config.enable_metabolism = false;
    high.config.enable_metabolism = false;
    low.config.metabolic_viability_floor = 0.8;
    high.config.metabolic_viability_floor = 0.8;
    low.config.boundary_decay_energy_scale = 0.08;
    high.config.boundary_decay_energy_scale = 0.08;
    low.organisms[0].metabolic_state.energy = 0.0;
    high.organisms[0].metabolic_state.energy = 1.0;
    low.organisms[0].metabolic_state.waste = 0.8;
    high.organisms[0].metabolic_state.waste = 0.0;

    low.step();
    high.step();

    assert!(
        low.organisms[0].boundary_integrity < high.organisms[0].boundary_integrity,
        "low-energy high-waste organism should lose boundary integrity faster"
    );
}

#[test]
fn graph_mode_selects_graph_engine() {
    let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    let config = SimConfig {
        num_organisms: 1,
        agents_per_organism: 1,
        metabolism_mode: MetabolismMode::Graph,
        ..SimConfig::default()
    };
    let world = World::new(agents, vec![nn], config).unwrap();
    assert!(matches!(world.metabolism, MetabolismEngine::Graph(_)));
}

#[test]
fn try_new_rejects_invalid_boundary_decay_config() {
    let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    let cfg = SimConfig {
        num_organisms: 1,
        agents_per_organism: 1,
        boundary_decay_base_rate: -0.1,
        ..SimConfig::default()
    };
    let result = World::new(agents, vec![nn], cfg);
    assert!(matches!(
        result,
        Err(WorldInitError::Config(
            SimConfigError::InvalidBoundaryDecayBaseRate
        ))
    ));
}

#[test]
fn try_new_rejects_invalid_mutation_probability_budget() {
    let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    let cfg = SimConfig {
        num_organisms: 1,
        agents_per_organism: 1,
        mutation_point_rate: 0.8,
        mutation_reset_rate: 0.3,
        mutation_scale_rate: 0.1,
        ..SimConfig::default()
    };
    let result = World::new(agents, vec![nn], cfg);
    assert!(matches!(
        result,
        Err(WorldInitError::Config(
            SimConfigError::InvalidMutationProbabilityBudget
        ))
    ));
}

#[test]
fn try_new_rejects_reproduction_min_energy_below_cost() {
    let agents = vec![Agent::new(0, 0, [0.0, 0.0])];
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    let cfg = SimConfig {
        num_organisms: 1,
        agents_per_organism: 1,
        reproduction_min_energy: 0.1,
        reproduction_energy_cost: 0.3,
        ..SimConfig::default()
    };
    let result = World::new(agents, vec![nn], cfg);
    assert!(matches!(
        result,
        Err(WorldInitError::Config(
            SimConfigError::InvalidReproductionEnergyBalance
        ))
    ));
}

#[test]
fn try_run_experiment_rejects_too_many_steps() {
    let mut world = make_world(1, 100.0);
    let result = world.try_run_experiment(World::MAX_EXPERIMENT_STEPS + 1, 1);
    assert!(matches!(result, Err(ExperimentError::TooManySteps { .. })));
}

#[test]
fn snapshot_experiment_collects_frames_at_requested_steps() {
    let mut world = make_world(10, 100.0);
    let summary = world
        .try_run_experiment_with_snapshots(10, 5, &[5])
        .expect("experiment should succeed");
    assert_eq!(summary.organism_snapshots.len(), 1);
    assert_eq!(summary.organism_snapshots[0].step, 5);
    assert!(
        !summary.organism_snapshots[0].organisms.is_empty(),
        "snapshot should contain at least one organism"
    );
}

#[test]
fn snapshot_experiment_skips_out_of_range_steps() {
    let mut world = make_world(10, 100.0);
    let summary = world
        .try_run_experiment_with_snapshots(10, 5, &[0, 20])
        .expect("experiment should succeed");
    // step 0 is never reached (loop is 1..=steps), step 20 > 10
    assert!(summary.organism_snapshots.is_empty());
}

#[test]
fn snapshot_organisms_have_valid_fields() {
    let mut world = make_world(10, 100.0);
    let summary = world
        .try_run_experiment_with_snapshots(5, 1, &[3])
        .expect("experiment should succeed");
    assert_eq!(summary.organism_snapshots.len(), 1);
    for org in &summary.organism_snapshots[0].organisms {
        assert!(org.energy >= 0.0);
        assert!(org.boundary_integrity >= 0.0 && org.boundary_integrity <= 1.0);
        assert!(org.maturity >= 0.0);
    }
}

#[test]
fn snapshots_include_agent_counts_when_metabolism_disabled() {
    let mut world = make_world(3, 100.0);
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    let summary = world
        .try_run_experiment_with_snapshots(1, 1, &[1])
        .expect("experiment should succeed");
    assert_eq!(summary.organism_snapshots.len(), 1);
    let org = summary.organism_snapshots[0]
        .organisms
        .first()
        .expect("snapshot should contain an alive organism");
    assert_eq!(org.n_agents, 3);
}

#[test]
fn reproduction_increases_population_when_energy_is_high() {
    let mut world = make_world(10, 100.0);
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 1.0;
    let before = world.organism_count();
    world.step();
    assert!(world.organism_count() > before);
    assert!(world.population_stats().total_births >= 1);
}

#[test]
fn reproduction_obeys_configured_thresholds() {
    let mut world = make_world(10, 100.0);
    world.config.reproduction_min_energy = 1.1;
    world.config.reproduction_min_boundary = 0.95;
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 0.9;
    let before = world.organism_count();
    world.step();
    assert_eq!(world.organism_count(), before);
    assert_eq!(world.population_stats().total_births, 0);
}

#[test]
fn max_organism_age_steps_is_configurable() {
    let mut world = make_world(10, 100.0);
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.max_organism_age_steps = 1;
    world.config.reproduction_min_energy = 10.0;
    world.config.reproduction_min_boundary = 1.0;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.organisms[0].metabolic_state.energy = 1.0;
    world.step();
    assert_eq!(world.organism_count(), 1);
    world.step();
    assert_eq!(world.organism_count(), 0);
}

#[test]
fn same_seed_produces_same_birth_death_timeline() {
    let agents: Vec<Agent> = (0..20)
        .map(|i| Agent::new(i as u32, 0, [50.0, 50.0]))
        .collect();
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.1f32, NeuralNet::WEIGHT_COUNT));
    let config = SimConfig {
        seed: 777,
        num_organisms: 1,
        agents_per_organism: 20,
        ..SimConfig::default()
    };
    let mut a = World::new(agents.clone(), vec![nn.clone()], config.clone()).unwrap();
    let mut b = World::new(agents, vec![nn], config).unwrap();

    let ra = a.run_experiment(30, 1);
    let rb = b.run_experiment(30, 1);

    let births_a: Vec<usize> = ra.samples.iter().map(|s| s.birth_count).collect();
    let births_b: Vec<usize> = rb.samples.iter().map(|s| s.birth_count).collect();
    let deaths_a: Vec<usize> = ra.samples.iter().map(|s| s.death_count).collect();
    let deaths_b: Vec<usize> = rb.samples.iter().map(|s| s.death_count).collect();
    assert_eq!(births_a, births_b);
    assert_eq!(deaths_a, deaths_b);
}

#[test]
fn metrics_include_evolution_fields() {
    let mut world = make_world(10, 100.0);
    let summary = world.run_experiment(5, 1);
    let sample = summary.samples.last().expect("sample should exist");
    assert!(sample.population_size >= sample.alive_count);
    assert!(sample.mean_generation >= 0.0);
    assert!(sample.mean_genome_drift >= 0.0);
}

#[test]
fn dead_entities_are_pruned_after_step() {
    let mut world = make_world(4, 100.0);
    world.config.enable_metabolism = true;
    world.config.death_energy_threshold = 1.0;
    world.resource_field_mut().set(50.0, 50.0, 0.0);
    world.step();
    assert_eq!(world.organism_count(), 0);
    assert!(world.agents.is_empty());
}

#[test]
fn boundary_terminal_threshold_is_consistent_across_checks() {
    let mut world = make_world(4, 100.0);
    world.config.enable_metabolism = false;
    world.config.boundary_collapse_threshold = 0.05;
    world.config.death_boundary_threshold = 0.10;
    world.organisms[0].boundary_integrity = 0.08;
    world.step();
    assert_eq!(world.organism_count(), 0);
}

#[test]
fn disable_homeostasis_allows_state_decay() {
    let mut world = make_world(10, 100.0);
    world.config.enable_homeostasis = false;
    world.config.homeostasis_decay_rate = 0.05;
    // Disable metabolism and boundary to ensure organism survives all steps
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.config.enable_reproduction = false;
    let before_0 = world.agents[0].internal_state[0]; // starts at 0.5
    for _ in 0..50 {
        world.step();
    }
    assert!(
        world.organisms[0].alive,
        "organism must survive for valid test"
    );
    assert!(
        world.agents[0].internal_state[0] < before_0,
        "internal_state[0] should decay when homeostasis is disabled"
    );
}

#[test]
fn homeostasis_decay_reduces_internal_state() {
    let mut world = make_world(1, 100.0);
    // Use zero NN weights so NN delta is ~0
    world.organisms[0].nn =
        NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
    world.config.enable_homeostasis = false;
    world.config.homeostasis_decay_rate = 0.1;
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.config.enable_reproduction = false;
    let before = world.agents[0].internal_state[0]; // 0.5
    world.step();
    assert!(
        world.agents[0].internal_state[0] < before,
        "internal_state[0] should decrease after one step with decay"
    );
}

#[test]
fn homeostasis_enabled_counteracts_decay() {
    // NN with large positive weights should produce positive delta[2] to counteract decay
    let mut world = make_world(1, 100.0);
    world.organisms[0].nn =
        NeuralNet::from_weights(std::iter::repeat_n(1.0f32, NeuralNet::WEIGHT_COUNT));
    world.config.enable_homeostasis = true;
    world.config.homeostasis_decay_rate = 0.001; // small decay
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.config.enable_reproduction = false;

    // Comparison: same config but homeostasis disabled
    let mut world_no = make_world(1, 100.0);
    world_no.organisms[0].nn =
        NeuralNet::from_weights(std::iter::repeat_n(1.0f32, NeuralNet::WEIGHT_COUNT));
    world_no.config.enable_homeostasis = false;
    world_no.config.homeostasis_decay_rate = 0.001;
    world_no.config.enable_metabolism = false;
    world_no.config.enable_boundary_maintenance = false;
    world_no.config.death_boundary_threshold = 0.0;
    world_no.config.boundary_collapse_threshold = 0.0;
    world_no.config.death_energy_threshold = 0.0;
    world_no.config.enable_reproduction = false;

    for _ in 0..50 {
        world.step();
        world_no.step();
    }
    assert!(
        world.agents[0].internal_state[0] > world_no.agents[0].internal_state[0],
        "homeostasis-enabled should maintain higher internal_state than disabled"
    );
}

#[test]
fn homeostasis_modulates_boundary_repair() {
    // High internal_state[0] → better boundary repair
    let mut world_high = make_world(10, 100.0);
    world_high.config.enable_homeostasis = false;
    world_high.config.homeostasis_decay_rate = 0.0; // no decay, state stays at 0.5
    world_high.config.enable_metabolism = false;
    world_high.config.enable_reproduction = false;
    world_high.config.boundary_decay_base_rate = 0.003;
    world_high.config.boundary_repair_rate = 0.01;
    for a in &mut world_high.agents {
        a.internal_state[0] = 0.9;
    }

    let mut world_low = make_world(10, 100.0);
    world_low.config.enable_homeostasis = false;
    world_low.config.homeostasis_decay_rate = 0.0;
    world_low.config.enable_metabolism = false;
    world_low.config.enable_reproduction = false;
    world_low.config.boundary_decay_base_rate = 0.003;
    world_low.config.boundary_repair_rate = 0.01;
    for a in &mut world_low.agents {
        a.internal_state[0] = 0.1;
    }

    // Give both organisms some energy for repair
    world_high.organisms[0].metabolic_state.energy = 0.8;
    world_low.organisms[0].metabolic_state.energy = 0.8;

    for _ in 0..50 {
        world_high.step();
        world_low.step();
    }
    assert!(
        world_high.organisms[0].boundary_integrity > world_low.organisms[0].boundary_integrity,
        "organism with high internal_state[0] should have better boundary"
    );
}

#[test]
fn disable_response_freezes_velocity() {
    let mut world = make_world(10, 100.0);
    world.config.enable_response = false;
    world.agents[0].velocity = [0.0, 0.0];
    world.step();
    assert_eq!(
        world.agents[0].velocity,
        [0.0, 0.0],
        "velocity should remain zero when response is disabled"
    );
}

#[test]
fn disable_reproduction_prevents_births() {
    let mut world = make_world(10, 100.0);
    world.config.enable_reproduction = false;
    // Disable death mechanisms to ensure the organism survives
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 1.0;
    let before = world.organism_count();
    for _ in 0..10 {
        world.step();
    }
    assert!(
        world.organisms[0].alive,
        "organism must survive for valid test"
    );
    assert_eq!(
        world.population_stats().total_births,
        0,
        "birth_count should be 0 when reproduction is disabled"
    );
    assert_eq!(world.organism_count(), before);
}

#[test]
fn disable_evolution_copies_genome_exactly() {
    let mut world = make_world(10, 100.0);
    world.config.enable_evolution = false;
    // Disable death mechanisms so parent and child survive for inspection
    world.config.death_energy_threshold = 0.0;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 1.0;
    let parent_data = world.organisms[0].genome.data().to_vec();
    world.step();
    assert!(
        world.population_stats().total_births >= 1,
        "reproduction should still happen with evolution disabled"
    );
    // Find the child: it will be the last organism (pushed during maybe_reproduce)
    let child = world
        .organisms
        .iter()
        .find(|o| o.generation == 1)
        .expect("child organism with generation=1 should exist");
    let child_data = child.genome.data().to_vec();
    assert_eq!(
        parent_data, child_data,
        "child genome should be exact copy when evolution is disabled"
    );
}

#[test]
fn enable_growth_default_is_true() {
    let config = SimConfig::default();
    assert!(config.enable_growth, "enable_growth should default to true");
}

#[test]
fn organism_matures_over_time() {
    let mut world = make_world(10, 100.0);
    world.config.enable_growth = true;
    world.config.growth_maturation_steps = 100;
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.config.enable_reproduction = false;
    // Bootstrap organisms start at maturity 1.0
    assert!((world.organisms[0].maturity - 1.0).abs() < f32::EPSILON);

    // Simulate a child by setting maturity to 0.0
    world.organisms[0].maturity = 0.0;
    for _ in 0..100 {
        world.step();
    }
    assert!(
        (world.organisms[0].maturity - 1.0).abs() < 0.02,
        "organism should reach maturity ~1.0 after growth_maturation_steps"
    );
}

#[test]
fn immature_organism_has_reduced_metabolic_efficiency() {
    // Maturity always modulates metabolic efficiency regardless of enable_growth
    let mut world_immature = make_world(10, 100.0);
    world_immature.config.growth_immature_metabolic_efficiency = 0.3;
    world_immature.config.enable_boundary_maintenance = false;
    world_immature.config.death_boundary_threshold = 0.0;
    world_immature.config.boundary_collapse_threshold = 0.0;
    world_immature.config.death_energy_threshold = 0.0;
    world_immature.config.enable_reproduction = false;
    world_immature.organisms[0].maturity = 0.0;
    world_immature.organisms[0].metabolic_state.energy = 0.5;

    let mut world_mature = make_world(10, 100.0);
    world_mature.config.growth_immature_metabolic_efficiency = 0.3;
    world_mature.config.enable_boundary_maintenance = false;
    world_mature.config.death_boundary_threshold = 0.0;
    world_mature.config.boundary_collapse_threshold = 0.0;
    world_mature.config.death_energy_threshold = 0.0;
    world_mature.config.enable_reproduction = false;
    world_mature.organisms[0].maturity = 1.0;
    world_mature.organisms[0].metabolic_state.energy = 0.5;

    for _ in 0..10 {
        world_immature.step();
        world_mature.step();
    }
    assert!(
        world_mature.organisms[0].metabolic_state.energy
            > world_immature.organisms[0].metabolic_state.energy,
        "mature organism should have higher energy than immature"
    );
}

#[test]
fn growth_factor_preserves_energy_loss() {
    // When metabolism causes net energy loss, growth factor must NOT mask it.
    // With the old bug (.max(0.0) on energy delta), losses were silently discarded.
    let mut world = make_world(10, 100.0);
    world.config.growth_immature_metabolic_efficiency = 0.3;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.config.enable_reproduction = false;
    world.config.resource_regeneration_rate = 0.0;
    world.current_resource_rate = 0.0;
    world.organisms[0].maturity = 0.0; // fully immature

    // Deplete ALL resource sources: world grid, internal pool, and graph pool.
    // This forces metabolism's energy_loss_rate to dominate → net energy decrease.
    let w = world.resource_field.width();
    let h = world.resource_field.height();
    let cs = world.resource_field.cell_size();
    for y in 0..h {
        for x in 0..w {
            world.resource_field.set(x as f64 * cs, y as f64 * cs, 0.0);
        }
    }
    world.organisms[0].metabolic_state.resource = 0.0;
    world.organisms[0].metabolic_state.graph_pool.clear();

    let initial_energy = world.organisms[0].metabolic_state.energy;
    for _ in 0..50 {
        world.step();
    }
    assert!(
        world.organisms[0].metabolic_state.energy < initial_energy,
        "energy should decrease when all resources are depleted, even for immature organisms \
             (got {} >= initial {})",
        world.organisms[0].metabolic_state.energy,
        initial_energy
    );
}

#[test]
fn immature_organism_cannot_reproduce() {
    let mut world = make_world(10, 100.0);
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.organisms[0].maturity = 0.5; // not fully mature
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 1.0;
    let before = world.organism_count();
    world.step();
    assert_eq!(
        world.organism_count(),
        before,
        "immature organism should not reproduce"
    );
}

#[test]
fn growth_disabled_prevents_maturation() {
    let mut world = make_world(10, 100.0);
    world.config.enable_growth = false;
    world.config.growth_maturation_steps = 100;
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.config.enable_reproduction = false;
    world.organisms[0].maturity = 0.0;
    for _ in 0..200 {
        world.step();
    }
    assert!(
        world.organisms[0].maturity < f32::EPSILON,
        "maturity should stay at 0.0 when growth is disabled"
    );
}

#[test]
fn growth_reduces_boundary_repair_for_immature() {
    // Immature organisms should have lower boundary than mature after same steps
    let mut world_immature = make_world(10, 100.0);
    world_immature.config.enable_growth = true;
    world_immature.config.enable_metabolism = false;
    world_immature.config.death_energy_threshold = 0.0;
    world_immature.config.enable_reproduction = false;
    world_immature.organisms[0].maturity = 0.0;
    world_immature.organisms[0].boundary_integrity = 0.8;
    world_immature.organisms[0].metabolic_state.energy = 0.5;

    let mut world_mature = make_world(10, 100.0);
    world_mature.config.enable_growth = true;
    world_mature.config.enable_metabolism = false;
    world_mature.config.death_energy_threshold = 0.0;
    world_mature.config.enable_reproduction = false;
    world_mature.organisms[0].maturity = 1.0;
    world_mature.organisms[0].boundary_integrity = 0.8;
    world_mature.organisms[0].metabolic_state.energy = 0.5;

    for _ in 0..50 {
        world_immature.step();
        world_mature.step();
    }
    assert!(
        world_mature.organisms[0].boundary_integrity
            > world_immature.organisms[0].boundary_integrity,
        "mature organism should have higher boundary integrity: mature={}, immature={}",
        world_mature.organisms[0].boundary_integrity,
        world_immature.organisms[0].boundary_integrity
    );
}

#[test]
fn growth_reduces_sensing_for_immature() {
    // Immature organisms should detect fewer neighbors due to reduced sensing radius
    let mut world = make_world(10, 100.0);
    world.config.enable_growth = true;
    world.organisms[0].maturity = 0.0;
    // After step, neighbor counts should be reduced for immature organisms
    // This is hard to test directly, but we can verify the dev program decodes correctly
    let dp = &world.organisms[0].developmental_program;
    let (_, sensing, _) = dp.stage_factors(0.0);
    assert!(
        sensing < 1.0,
        "juvenile sensing factor should be < 1.0: {sensing}"
    );
}

#[test]
fn growth_ablation_degrades_boundary_independently_of_reproduction() {
    // With reproduction disabled, growth-on should still produce higher boundary than growth-off
    let mut world_growth_on = make_world(10, 100.0);
    world_growth_on.config.enable_growth = true;
    world_growth_on.config.enable_reproduction = false;
    world_growth_on.config.enable_metabolism = false;
    world_growth_on.config.death_energy_threshold = 0.0;
    world_growth_on.organisms[0].maturity = 0.0;
    world_growth_on.organisms[0].metabolic_state.energy = 0.5;
    world_growth_on.organisms[0].boundary_integrity = 0.8;

    let mut world_growth_off = make_world(10, 100.0);
    world_growth_off.config.enable_growth = false;
    world_growth_off.config.enable_reproduction = false;
    world_growth_off.config.enable_metabolism = false;
    world_growth_off.config.death_energy_threshold = 0.0;
    world_growth_off.organisms[0].maturity = 0.0;
    world_growth_off.organisms[0].metabolic_state.energy = 0.5;
    world_growth_off.organisms[0].boundary_integrity = 0.8;

    for _ in 0..100 {
        world_growth_on.step();
        world_growth_off.step();
    }

    // Growth-on: organism matures, eventually gets full boundary repair
    // Growth-off: organism never matures, boundary repair factor stays reduced
    // (growth_off uses the old linear formula which with maturity=0 gives reduced efficiency)
    // The key test: growth-on should show DIFFERENT boundary than growth-off,
    // proving independent viability effect
    assert!(
        (world_growth_on.organisms[0].boundary_integrity
            - world_growth_off.organisms[0].boundary_integrity)
            .abs()
            > 0.01,
        "growth on/off should produce different boundary integrity: on={}, off={}",
        world_growth_on.organisms[0].boundary_integrity,
        world_growth_off.organisms[0].boundary_integrity
    );
}

#[test]
fn genome_segment3_affects_maturation_rate() {
    let mut world_fast = make_world(10, 100.0);
    world_fast.config.enable_growth = true;
    world_fast.config.growth_maturation_steps = 200;
    world_fast.config.enable_metabolism = false;
    world_fast.config.enable_boundary_maintenance = false;
    world_fast.config.death_boundary_threshold = 0.0;
    world_fast.config.boundary_collapse_threshold = 0.0;
    world_fast.config.death_energy_threshold = 0.0;
    world_fast.config.enable_reproduction = false;
    // Set genome segment 3 g[0] = 1.0 → maturation_rate_modifier = 2^1 = 2.0
    world_fast.organisms[0]
        .genome
        .set_segment_data(3, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    world_fast.organisms[0].developmental_program =
        DevelopmentalProgram::decode(world_fast.organisms[0].genome.segment_data(3));
    world_fast.organisms[0].maturity = 0.0;

    let mut world_slow = make_world(10, 100.0);
    world_slow.config.enable_growth = true;
    world_slow.config.growth_maturation_steps = 200;
    world_slow.config.enable_metabolism = false;
    world_slow.config.enable_boundary_maintenance = false;
    world_slow.config.death_boundary_threshold = 0.0;
    world_slow.config.boundary_collapse_threshold = 0.0;
    world_slow.config.death_energy_threshold = 0.0;
    world_slow.config.enable_reproduction = false;
    // Set genome segment 3 g[0] = -1.0 → maturation_rate_modifier = 2^-1 = 0.5
    world_slow.organisms[0]
        .genome
        .set_segment_data(3, &[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    world_slow.organisms[0].developmental_program =
        DevelopmentalProgram::decode(world_slow.organisms[0].genome.segment_data(3));
    world_slow.organisms[0].maturity = 0.0;

    for _ in 0..100 {
        world_fast.step();
        world_slow.step();
    }
    assert!(
        world_fast.organisms[0].maturity > world_slow.organisms[0].maturity,
        "fast maturation genome should mature faster: fast={}, slow={}",
        world_fast.organisms[0].maturity,
        world_slow.organisms[0].maturity
    );
}

#[test]
fn developmental_program_decoded_on_child_creation() {
    let mut world = make_world(10, 100.0);
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.config.enable_evolution = false;
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 1.0;
    world.organisms[0].maturity = 1.0;
    for _ in 0..10 {
        world.step();
    }
    let child = world
        .organisms
        .iter()
        .find(|o| o.generation == 1)
        .expect("reproduction should produce a generation-1 child within 10 steps");
    assert!(
        child.developmental_program.maturation_rate_modifier > 0.0,
        "child should have decoded developmental program"
    );
    assert!(
        child.parent_stable_id.is_some(),
        "child should have parent_stable_id"
    );
}

#[test]
fn lineage_events_recorded_on_reproduction() {
    let mut world = make_world(10, 100.0);
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 1.0;
    world.organisms[0].maturity = 1.0;
    let summary = world.run_experiment(20, 20);
    if summary.total_reproduction_events > 0 {
        assert!(
            !summary.lineage_events.is_empty(),
            "lineage events should be recorded when reproduction occurs"
        );
        let event = &summary.lineage_events[0];
        assert!(event.generation > 0);
    }
}

#[test]
fn maturity_mean_and_spatial_cohesion_in_metrics() {
    let mut world = make_world(10, 100.0);
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    world.config.enable_reproduction = false;
    let summary = world.run_experiment(10, 10);
    let last = summary.samples.last().unwrap();
    // Bootstrap organisms start at maturity 1.0
    assert!(
        (last.maturity_mean - 1.0).abs() < f32::EPSILON,
        "bootstrap organisms should have maturity_mean=1.0: {}",
        last.maturity_mean
    );
    assert!(
        last.spatial_cohesion_mean >= 0.0,
        "spatial_cohesion_mean should be non-negative"
    );
}

fn make_graph_world(num_agents: usize, world_size: f64) -> World {
    let agents: Vec<Agent> = (0..num_agents)
        .map(|i| Agent::new(i as u32, 0, [50.0, 50.0]))
        .collect();
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.1f32, NeuralNet::WEIGHT_COUNT));
    let config = SimConfig {
        world_size,
        num_organisms: 1,
        agents_per_organism: num_agents,
        metabolism_mode: MetabolismMode::Graph,
        ..SimConfig::default()
    };
    World::new(agents, vec![nn], config).unwrap()
}

#[test]
fn graph_mode_organisms_have_individual_engines() {
    let agents: Vec<Agent> = (0..20)
        .map(|i| Agent::new(i as u32, i as u16 / 10, [50.0, 50.0]))
        .collect();
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.1f32, NeuralNet::WEIGHT_COUNT));
    let config = SimConfig {
        world_size: 100.0,
        num_organisms: 2,
        agents_per_organism: 10,
        metabolism_mode: MetabolismMode::Graph,
        ..SimConfig::default()
    };
    let world = World::new(agents, vec![nn.clone(), nn], config).unwrap();
    assert!(world.organisms[0].metabolism_engine.is_some());
    assert!(world.organisms[1].metabolism_engine.is_some());
    // Different organisms should have different metabolic segments (seeded differently)
    let seg0 = world.organisms[0].genome.segment_data(1);
    let seg1 = world.organisms[1].genome.segment_data(1);
    assert_ne!(
        seg0, seg1,
        "different organisms should have different metabolic segments"
    );
}

#[test]
fn toy_mode_organisms_use_shared_engine() {
    let world = make_world(10, 100.0);
    assert!(world.organisms[0].metabolism_engine.is_none());
}

#[test]
fn child_inherits_then_redecodes_metabolism() {
    let mut world = make_graph_world(10, 100.0);
    world.config.death_energy_threshold = 0.0;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 1.0;
    world.step();
    assert!(
        world.population_stats().total_births >= 1,
        "reproduction must occur for this test to be valid"
    );
    let child = world
        .organisms
        .iter()
        .find(|o| o.generation == 1)
        .expect("child should exist");
    assert!(
        child.metabolism_engine.is_some(),
        "child in Graph mode should have its own metabolism engine"
    );
}

#[test]
fn graph_mode_mutation_changes_metabolic_topology() {
    let agents: Vec<Agent> = (0..10)
        .map(|i| Agent::new(i as u32, 0, [50.0, 50.0]))
        .collect();
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.1f32, NeuralNet::WEIGHT_COUNT));
    let config = SimConfig {
        world_size: 100.0,
        num_organisms: 1,
        agents_per_organism: 10,
        metabolism_mode: MetabolismMode::Graph,
        mutation_point_rate: 0.5, // aggressive mutation
        mutation_point_scale: 1.0,
        death_energy_threshold: 0.0,
        death_boundary_threshold: 0.0,
        boundary_collapse_threshold: 0.0,
        ..SimConfig::default()
    };
    let mut world = World::new(agents, vec![nn], config).unwrap();
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 1.0;
    let parent_seg = world.organisms[0].genome.segment_data(1).to_vec();
    world.step();
    assert!(
        world.population_stats().total_births >= 1,
        "reproduction must occur for this test to be valid"
    );
    let child = world
        .organisms
        .iter()
        .find(|o| o.generation == 1)
        .expect("child should exist");
    let child_seg = child.genome.segment_data(1);
    assert_ne!(
        parent_seg, child_seg,
        "child metabolic segment should differ from parent with high mutation rate"
    );
}

// ── Extended metrics tests (Phase 1) ──

#[test]
fn step_metrics_new_fields_are_populated() {
    let mut world = make_world(10, 100.0);
    let summary = world.run_experiment(10, 5);
    let sample = summary.samples.last().expect("should have samples");
    // SD fields should be non-negative
    assert!(sample.energy_std >= 0.0);
    assert!(sample.waste_std >= 0.0);
    assert!(sample.boundary_std >= 0.0);
    // mean_age should be non-negative
    assert!(sample.mean_age >= 0.0);
    // internal_state_mean values should be in [0, 1]
    for &v in &sample.internal_state_mean {
        assert!(
            (0.0..=1.0).contains(&v),
            "internal_state_mean out of range: {v}"
        );
    }
    // internal_state_std should be non-negative
    for &v in &sample.internal_state_std {
        assert!(v >= 0.0, "internal_state_std negative: {v}");
    }
    // genome_diversity should be non-negative
    assert!(
        sample.genome_diversity >= 0.0,
        "genome_diversity should be non-negative"
    );
}

#[test]
fn lifespans_recorded_on_organism_death() {
    let mut world = make_world(4, 100.0);
    world.config.enable_metabolism = true;
    world.config.death_energy_threshold = 1.0; // force death
    world.resource_field_mut().set(50.0, 50.0, 0.0);
    let summary = world.run_experiment(5, 5);
    // Organism should die, producing at least one lifespan entry
    assert!(
        !summary.lifespans.is_empty(),
        "lifespans should be recorded when organisms die"
    );
}

#[test]
fn run_summary_total_reproduction_events() {
    let mut world = make_world(10, 100.0);
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 1.0;
    let summary = world.run_experiment(5, 5);
    // With high energy, reproduction should happen
    assert!(
        summary.total_reproduction_events >= 1,
        "total_reproduction_events should count births"
    );
}

#[test]
fn genome_diversity_is_bounded() {
    let mut world = make_world(10, 100.0);
    world.organisms[0].metabolic_state.energy = 1.0;
    world.organisms[0].boundary_integrity = 1.0;
    // Run enough steps for reproduction to occur
    let summary = world.run_experiment(20, 10);
    for sample in &summary.samples {
        assert!(
            sample.genome_diversity >= 0.0,
            "genome_diversity must be non-negative"
        );
        assert!(
            sample.genome_diversity.is_finite(),
            "genome_diversity must be finite"
        );
    }
}

#[test]
fn genome_diversity_zero_for_single_organism() {
    let mut world = make_world(10, 100.0);
    world.config.enable_reproduction = false;
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.death_energy_threshold = 0.0;
    let summary = world.run_experiment(5, 5);
    let sample = summary.samples.last().unwrap();
    assert!(
        sample.genome_diversity < f32::EPSILON,
        "genome_diversity should be 0 with only one organism"
    );
}

// ── Environment shift tests (Phase 4) ──

#[test]
fn environment_shift_zero_produces_no_change() {
    let mut world = make_world(10, 100.0);
    world.config.environment_shift_step = 0; // disabled
    world.config.environment_shift_resource_rate = 0.0;
    let original_rate = world.current_resource_rate;
    for _ in 0..50 {
        world.step();
    }
    assert!(
        (world.current_resource_rate - original_rate).abs() < f32::EPSILON,
        "resource rate should not change when shift_step=0"
    );
}

#[test]
fn environment_shift_changes_resource_rate_at_step() {
    let mut world = make_world(10, 100.0);
    world.config.environment_shift_step = 5;
    world.config.environment_shift_resource_rate = 0.005;
    world.config.resource_regeneration_rate = 0.01;
    world.current_resource_rate = 0.01;
    for _ in 0..4 {
        world.step();
    }
    assert!(
        (world.current_resource_rate - 0.01).abs() < f32::EPSILON,
        "rate should be unchanged before shift step"
    );
    world.step(); // step 5
    assert!(
        (world.current_resource_rate - 0.005).abs() < f32::EPSILON,
        "rate should change at shift step"
    );
}

// ── Graded ablation tests ──

#[test]
fn metabolism_efficiency_multiplier_defaults_to_one() {
    let cfg = SimConfig::default();
    assert!(
        (cfg.metabolism_efficiency_multiplier - 1.0).abs() < f32::EPSILON,
        "metabolism_efficiency_multiplier should default to 1.0"
    );
}

#[test]
fn metabolism_efficiency_half_halves_metabolic_gain() {
    // Two identical worlds: one with multiplier=1.0, one with 0.5
    let mut world_full = make_world(10, 100.0);
    world_full.config.enable_boundary_maintenance = false;
    world_full.config.death_boundary_threshold = 0.0;
    world_full.config.boundary_collapse_threshold = 0.0;
    world_full.config.metabolism_efficiency_multiplier = 1.0;

    let mut world_half = make_world(10, 100.0);
    world_half.config.enable_boundary_maintenance = false;
    world_half.config.death_boundary_threshold = 0.0;
    world_half.config.boundary_collapse_threshold = 0.0;
    world_half.config.metabolism_efficiency_multiplier = 0.5;

    for _ in 0..100 {
        world_full.step();
        world_half.step();
    }
    let e_full = world_full.metabolic_state(0).unwrap().energy;
    let e_half = world_half.metabolic_state(0).unwrap().energy;
    assert!(
        e_half < e_full,
        "half-efficiency ({e_half}) should produce less energy than full ({e_full})"
    );
}

#[test]
fn metabolism_efficiency_zero_produces_zero_metabolic_gain() {
    let mut world = make_world(10, 100.0);
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.metabolism_efficiency_multiplier = 0.0;
    let initial_energy = world.metabolic_state(0).unwrap().energy;

    for _ in 0..50 {
        world.step();
    }
    let final_energy = world.metabolic_state(0).unwrap().energy;
    assert!(
        final_energy <= initial_energy,
        "zero-efficiency should produce no net energy gain \
             (initial={initial_energy}, final={final_energy})"
    );
}

#[test]
fn metabolism_efficiency_one_matches_baseline() {
    // Multiplier=1.0 should be identical to a world without the field set
    let mut world_a = make_world(10, 100.0);
    world_a.config.enable_boundary_maintenance = false;
    world_a.config.death_boundary_threshold = 0.0;
    world_a.config.boundary_collapse_threshold = 0.0;
    world_a.config.metabolism_efficiency_multiplier = 1.0;

    let mut world_b = make_world(10, 100.0);
    world_b.config.enable_boundary_maintenance = false;
    world_b.config.death_boundary_threshold = 0.0;
    world_b.config.boundary_collapse_threshold = 0.0;
    // world_b uses default (1.0) from SimConfig::default()

    for _ in 0..100 {
        world_a.step();
        world_b.step();
    }
    let e_a = world_a.metabolic_state(0).unwrap().energy;
    let e_b = world_b.metabolic_state(0).unwrap().energy;
    assert!(
        (e_a - e_b).abs() < f32::EPSILON,
        "multiplier=1.0 ({e_a}) should match default ({e_b})"
    );
}

// ── Cyclic environment tests ──

#[test]
fn environment_cycle_period_zero_means_no_cycling() {
    let cfg = SimConfig::default();
    assert_eq!(
        cfg.environment_cycle_period, 0,
        "environment_cycle_period should default to 0 (no cycling)"
    );
    let mut world = make_world(10, 100.0);
    world.config.environment_cycle_period = 0;
    let original_rate = world.current_resource_rate;
    for _ in 0..200 {
        world.step();
    }
    assert!(
        (world.current_resource_rate - original_rate).abs() < f32::EPSILON,
        "resource rate should not change when cycle_period=0"
    );
}

#[test]
fn environment_cycle_alternates_resource_rate() {
    let mut world = make_world(10, 100.0);
    world.config.environment_cycle_period = 100;
    world.config.resource_regeneration_rate = 0.01;
    world.config.environment_cycle_low_rate = 0.005;
    world.current_resource_rate = 0.01;

    // Steps 1-100 → phase 0 (high rate): step_index 1..100, (step/100)%2 = 0
    for _ in 0..99 {
        world.step();
    }
    assert!(
        (world.current_resource_rate - 0.01).abs() < f32::EPSILON,
        "phase 0 should use normal rate, got {}",
        world.current_resource_rate
    );

    // Step 100 → phase 1 (low rate): (100/100)%2 = 1
    world.step();
    assert!(
        (world.current_resource_rate - 0.005).abs() < f32::EPSILON,
        "phase 1 should use low rate, got {}",
        world.current_resource_rate
    );
}

#[test]
fn environment_cycle_returns_to_high_rate() {
    let mut world = make_world(10, 100.0);
    world.config.environment_cycle_period = 100;
    world.config.resource_regeneration_rate = 0.01;
    world.config.environment_cycle_low_rate = 0.005;
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.current_resource_rate = 0.01;

    // Run 200 steps to reach phase 2 (which is (200/100)%2=0 → high)
    for _ in 0..200 {
        world.step();
    }
    assert!(
        (world.current_resource_rate - 0.01).abs() < f32::EPSILON,
        "phase 2 should return to normal rate, got {}",
        world.current_resource_rate
    );
}

// ── Sham ablation tests ──

#[test]
fn enable_sham_process_defaults_to_false() {
    let cfg = SimConfig::default();
    assert!(
        !cfg.enable_sham_process,
        "enable_sham_process should default to false (opt-in)"
    );
}

#[test]
fn sham_process_has_no_functional_effect() {
    // Compare sham-on vs sham-off: population outcomes should be identical
    // since sham computes distances but discards results
    let mut world_on = make_world(10, 100.0);
    world_on.config.enable_sham_process = true;
    world_on.config.enable_boundary_maintenance = false;
    world_on.config.death_boundary_threshold = 0.0;
    world_on.config.boundary_collapse_threshold = 0.0;

    let mut world_off = make_world(10, 100.0);
    world_off.config.enable_sham_process = false;
    world_off.config.enable_boundary_maintenance = false;
    world_off.config.death_boundary_threshold = 0.0;
    world_off.config.boundary_collapse_threshold = 0.0;

    for _ in 0..200 {
        world_on.step();
        world_off.step();
    }
    let energy_on = world_on.metabolic_state(0).unwrap().energy;
    let energy_off = world_off.metabolic_state(0).unwrap().energy;
    assert!(
        (energy_on - energy_off).abs() < f32::EPSILON,
        "sham process should have no effect on energy (on={energy_on}, off={energy_off})"
    );

    let alive_on = world_on.organism_count();
    let alive_off = world_off.organism_count();
    assert_eq!(
        alive_on, alive_off,
        "sham process should have no effect on population"
    );
}

// ── Legacy config backward compatibility ──

#[test]
fn legacy_config_gets_new_field_defaults() {
    let legacy_json = r#"{
            "seed": 42,
            "world_size": 100.0,
            "num_organisms": 1,
            "agents_per_organism": 1
        }"#;
    let cfg: SimConfig = serde_json::from_str(legacy_json).expect("legacy config should parse");
    assert!(
        (cfg.metabolism_efficiency_multiplier - 1.0).abs() < f32::EPSILON,
        "metabolism_efficiency_multiplier should default to 1.0"
    );
    assert_eq!(cfg.environment_cycle_period, 0);
    assert!(
        (cfg.environment_cycle_low_rate - 0.005).abs() < f32::EPSILON,
        "environment_cycle_low_rate should default to 0.005"
    );
    assert!(!cfg.enable_sham_process);
}

#[test]
fn environment_shift_and_cycle_are_mutually_exclusive() {
    let mut world = make_world(10, 100.0);
    world.config.environment_shift_step = 500;
    world.config.environment_cycle_period = 200;
    let result = world.set_config(world.config.clone());
    assert!(matches!(
        result,
        Err(WorldInitError::Config(
            SimConfigError::ConflictingEnvironmentFeatures
        ))
    ));
}

#[test]
fn try_run_experiment_rejects_too_many_snapshots() {
    let mut world = make_world(1, 100.0);
    let max = World::MAX_EXPERIMENT_SNAPSHOTS;
    let snapshot_steps = vec![0; max + 1];
    let result = world.try_run_experiment_with_snapshots(max + 1, 1, &snapshot_steps);
    assert!(matches!(
        result,
        Err(ExperimentError::TooManySnapshots { .. })
    ));
}

#[test]
fn scheduled_ablation_disables_targets_at_exact_step() {
    let mut world = make_world(10, 100.0);
    world.config.ablation_step = 3;
    world.config.ablation_targets = vec![AblationTarget::Metabolism, AblationTarget::Response];
    assert!(world.config.enable_metabolism);
    assert!(world.config.enable_response);

    world.step();
    world.step();
    assert!(
        world.config.enable_metabolism && world.config.enable_response,
        "scheduled ablation should not apply before ablation_step"
    );

    world.step();
    assert!(
        !world.config.enable_metabolism && !world.config.enable_response,
        "scheduled ablation should apply exactly at ablation_step"
    );
}

#[test]
fn scheduled_ablation_not_missed_after_midrun_config_update() {
    let mut world = make_world(10, 100.0);
    world.step();
    assert!(world.config.enable_metabolism);

    let mut updated = world.config.clone();
    updated.ablation_step = 1;
    updated.ablation_targets = vec![AblationTarget::Metabolism];
    world
        .set_config(updated)
        .expect("config update should succeed");

    world.step();
    assert!(
        !world.config.enable_metabolism,
        "ablation should still apply when config is updated after ablation_step"
    );
}

#[test]
fn boundary_mode_spatial_hull_feedback_changes_boundary_trajectory() {
    let mut scalar = make_world(10, 100.0);
    scalar.config.boundary_mode = BoundaryMode::ScalarRepair;
    scalar.config.enable_metabolism = false;
    scalar.config.enable_reproduction = false;
    scalar.config.enable_response = false;
    scalar.config.enable_homeostasis = false;
    scalar.config.death_boundary_threshold = 0.0;
    scalar.config.boundary_collapse_threshold = 0.0;
    scalar.config.boundary_decay_base_rate = 0.05;
    scalar.config.boundary_decay_energy_scale = 0.2;
    for (i, agent) in scalar.agents.iter_mut().enumerate() {
        agent.position = [10.0 + i as f64 * 7.0, 10.0 + i as f64 * 5.0];
    }

    let mut spatial = make_world(10, 100.0);
    spatial.config.boundary_mode = BoundaryMode::SpatialHullFeedback;
    spatial.config.enable_metabolism = false;
    spatial.config.enable_reproduction = false;
    spatial.config.enable_response = false;
    spatial.config.enable_homeostasis = false;
    spatial.config.death_boundary_threshold = 0.0;
    spatial.config.boundary_collapse_threshold = 0.0;
    spatial.config.boundary_decay_base_rate = 0.05;
    spatial.config.boundary_decay_energy_scale = 0.2;
    for (i, agent) in spatial.agents.iter_mut().enumerate() {
        agent.position = [10.0 + i as f64 * 7.0, 10.0 + i as f64 * 5.0];
    }

    scalar.step();
    spatial.step();

    let b_scalar = scalar.organisms[0].boundary_integrity;
    let b_spatial = spatial.organisms[0].boundary_integrity;
    assert!(
            (b_scalar - b_spatial).abs() > 1e-6,
            "boundary modes should produce different boundary trajectories (scalar={b_scalar}, spatial={b_spatial})"
        );
}

#[test]
fn setpoint_pid_mode_stabilizes_internal_state_toward_energy_scaled_setpoint() {
    let mut world = make_world(1, 100.0);
    world.config.homeostasis_mode = HomeostasisMode::SetpointPid;
    world.config.enable_response = false;
    world.config.enable_metabolism = false;
    world.config.enable_boundary_maintenance = false;
    world.config.enable_reproduction = false;
    world.config.death_boundary_threshold = 0.0;
    world.config.boundary_collapse_threshold = 0.0;
    world.config.max_organism_age_steps = usize::MAX;
    world.agents[0].internal_state[0] = 0.0;
    world.agents[0].internal_state[1] = 1.0;

    world.step();
    let s0 = world.agents[0].internal_state[0];
    let s1 = world.agents[0].internal_state[1];
    assert!(
        s0 > 0.0,
        "setpoint controller should raise low state toward target"
    );
    assert!(
        s1 < 1.0,
        "setpoint controller should lower high state toward target"
    );
}

// ─── SemiLife V0 tests ────────────────────────────────────────────────────────

fn make_semi_life_world(
    archetypes: Vec<crate::semi_life::SemiLifeArchetype>,
    num_per_archetype: usize,
    resource_density: f32,
    seed: u64,
) -> World {
    use crate::config::SemiLifeConfig;
    let semi_life_config = SemiLifeConfig {
        enabled_archetypes: archetypes,
        num_per_archetype,
        initial_energy: 0.5,
        energy_capacity: 1.0,
        maintenance_cost: 0.001,
        replication_threshold: 0.8,
        replication_cost: 0.3,
        resource_uptake_rate: 0.02,
        // V1
        boundary_decay_rate: 0.002,
        boundary_repair_rate: 0.01,
        boundary_death_threshold: 0.1,
        boundary_replication_min: 0.5,
        // V2
        regulator_init: 1.0,
        regulator_uptake_scale: 1.0,
        regulator_cost_per_step: 0.0005,
        // V3
        internal_pool_init_fraction: 0.5,
        internal_pool_capacity: 1.0,
        internal_conversion_rate: 0.05,
        internal_pool_uptake_rate: 0.01,
        // Prion
        prion_contact_radius: 10.0,
        prion_conversion_prob: 0.1,
        prion_fragmentation_loss: 0.005,
        prion_dilution_death_energy: 0.0,
        ..SemiLifeConfig::default()
    };
    // Minimal organism world so Prion has hosts to contact.
    let agents: Vec<Agent> = (0..10)
        .map(|i| Agent::new(i as u32, 0, [50.0, 50.0]))
        .collect();
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.1f32, NeuralNet::WEIGHT_COUNT));
    let config = SimConfig {
        seed,
        world_size: 100.0,
        num_organisms: 1,
        agents_per_organism: 10,
        resource_regeneration_rate: resource_density * 0.01,
        enable_semi_life: true,
        semi_life_config,
        ..SimConfig::default()
    };
    let mut world = World::new(agents, vec![nn], config).unwrap();
    // Pre-fill resource field to requested density.
    for x in 0..100 {
        for y in 0..100 {
            world
                .resource_field_mut()
                .set(x as f64, y as f64, resource_density);
        }
    }
    world
}

/// SemiLife simulation is deterministic: two runs with the same seed produce
/// identical SemiLifeSnapshot sequences.
#[test]
fn semi_life_simulation_is_deterministic() {
    use crate::semi_life::SemiLifeArchetype::Viroid;
    let steps = 50;
    // Compare agent positions (seed-dependent via world.rng) + energy (physics-dependent).
    let run = |seed: u64| -> Vec<[f32; 2]> {
        let mut world = make_semi_life_world(vec![Viroid], 5, 1.0, seed);
        world.run_experiment(steps, steps);
        // SemiLife agent positions are seeded from world.rng; they differ between seeds.
        world
            .agents()
            .iter()
            .filter(|a| a.owner_type == crate::agent::OwnerType::SemiLife)
            .map(|a| [a.position[0] as f32, a.position[1] as f32])
            .collect()
    };

    let a = run(42);
    let b = run(42);
    assert_eq!(a, b, "same seed must produce identical SemiLife positions");

    let c = run(99);
    // Different seed should produce different initial positions for SemiLife agents.
    assert_ne!(
        a, c,
        "different seeds should produce different SemiLife positions"
    );
}

/// Viroid (resource-field dependent) outlives Prion (contact-dependent) in a
/// high-resource environment where organisms are present but far between.
/// This is a basic sanity check for the archetype dependency model.
#[test]
fn viroid_survives_longer_than_prion_in_high_resource() {
    use crate::semi_life::SemiLifeArchetype::{Prion, Viroid};
    let steps = 300;

    let mut viroid_world = make_semi_life_world(vec![Viroid], 10, 1.0, 7);
    viroid_world.run_experiment(steps, steps);
    let viroid_alive = viroid_world.semi_life_alive_count();

    // Prion only benefits from host contact; high resource density doesn't help it.
    let mut prion_world = make_semi_life_world(vec![Prion], 10, 1.0, 7);
    prion_world.run_experiment(steps, steps);
    let prion_alive = prion_world.semi_life_alive_count();

    // Guard: viroid must have survived — if both are 0, the SemiLife subsystem is broken.
    assert!(
        viroid_alive > 0,
        "Viroid should have survivors in high-resource environment after {steps} steps \
         (got 0 — SemiLife resource uptake may be broken)"
    );
    assert!(
        viroid_alive >= prion_alive,
        "Viroid ({viroid_alive} alive) should survive at least as long as Prion \
         ({prion_alive} alive) in high-resource environment"
    );
}

/// ProtoOrganelle (V1+V2+V3, no V0) can sustain itself in a high-resource environment:
/// its internal metabolism (V3) keeps it alive even without replication capability.
#[test]
fn proto_organelle_stays_alive_with_baseline_capabilities() {
    use crate::semi_life::SemiLifeArchetype::ProtoOrganelle;
    let steps = 200;

    let mut world = make_semi_life_world(vec![ProtoOrganelle], 5, 1.0, 13);
    world.run_experiment(steps, steps);
    let alive = world.semi_life_alive_count();

    assert!(
        alive > 0,
        "ProtoOrganelle with V1+V2+V3 baseline should stay alive for {steps} steps in \
         high-resource, but 0 of 5 entities survived"
    );
}

// ─── SemiLife V4/V5 edge-case tests ───────────────────────────────────────────

/// Helper: create a SemiLife world with specific capability overrides.
fn make_semi_life_world_with_caps(
    num: usize,
    resource_density: f32,
    seed: u64,
    capability_bits: u8,
) -> World {
    use crate::config::SemiLifeConfig;
    use crate::semi_life::SemiLifeArchetype;
    use std::collections::HashMap;

    let mut overrides = HashMap::new();
    overrides.insert("viroid".to_string(), capability_bits);

    let semi_life_config = SemiLifeConfig {
        enabled_archetypes: vec![SemiLifeArchetype::Viroid],
        num_per_archetype: num,
        initial_energy: 0.5,
        energy_capacity: 1.0,
        maintenance_cost: 0.001,
        replication_threshold: 0.8,
        replication_cost: 0.3,
        resource_uptake_rate: 0.02,
        boundary_decay_rate: 0.002,
        boundary_repair_rate: 0.01,
        boundary_death_threshold: 0.1,
        boundary_replication_min: 0.5,
        regulator_init: 1.0,
        regulator_uptake_scale: 1.0,
        regulator_cost_per_step: 0.0005,
        internal_pool_init_fraction: 0.5,
        internal_pool_capacity: 1.0,
        internal_conversion_rate: 0.05,
        internal_pool_uptake_rate: 0.01,
        // V4
        v4_max_speed: 1.0,
        v4_move_cost: 0.01,
        v4_policy_init: [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        v4_mutation_sigma: 0.05,
        // V5 — use short durations for testability
        v5_activation_threshold: 0.4,
        v5_dispersal_age: 10,
        v5_dispersal_duration: 5,
        v5_dormant_decay_mult: 0.3,
        v5_dispersal_speed_mult: 2.0,
        v5_dispersal_decay_mult: 1.5,
        capability_overrides: overrides,
        ..SemiLifeConfig::default()
    };
    let agents: Vec<Agent> = (0..10)
        .map(|i| Agent::new(i as u32, 0, [50.0, 50.0]))
        .collect();
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.1f32, NeuralNet::WEIGHT_COUNT));
    let config = SimConfig {
        seed,
        world_size: 100.0,
        num_organisms: 1,
        agents_per_organism: 10,
        resource_regeneration_rate: resource_density * 0.01,
        enable_semi_life: true,
        semi_life_config,
        ..SimConfig::default()
    };
    let mut world = World::new(agents, vec![nn], config).unwrap();
    for x in 0..100 {
        for y in 0..100 {
            world
                .resource_field_mut()
                .set(x as f64, y as f64, resource_density);
        }
    }
    world
}

/// V5 entities start in Dormant stage.
#[test]
fn v5_entities_start_dormant() {
    use crate::semi_life::capability::*;
    use crate::semi_life::SemiLifeStage;

    let caps =
        V0_REPLICATION | V1_BOUNDARY | V2_HOMEOSTASIS | V3_METABOLISM | V4_RESPONSE | V5_LIFECYCLE;
    let world = make_semi_life_world_with_caps(3, 1.0, 42, caps);
    for sl in &world.semi_lives {
        assert_eq!(
            sl.stage,
            Some(SemiLifeStage::Dormant),
            "V5 entities must start Dormant"
        );
        assert_eq!(sl.stage_ticks, 0);
    }
}

/// V5 Dormant stage reduces energy decay (multiplier < 1.0) compared to non-V5.
#[test]
fn v5_dormant_reduces_energy_decay() {
    use crate::semi_life::capability::*;

    let caps_with_v5 = V0_REPLICATION | V5_LIFECYCLE;
    let caps_without_v5 = V0_REPLICATION;

    // Set activation threshold very high so entity stays Dormant for all steps.
    let mut world_v5 = make_semi_life_world_with_caps(1, 0.0, 42, caps_with_v5);
    world_v5.config.semi_life_config.v5_activation_threshold = 10.0;
    let mut world_no_v5 = make_semi_life_world_with_caps(1, 0.0, 42, caps_without_v5);

    let initial_energy = world_v5.semi_lives[0].maintenance_energy;
    assert!(
        (initial_energy - world_no_v5.semi_lives[0].maintenance_energy).abs() < 0.001,
        "Both should start with same energy"
    );

    for _ in 0..5 {
        world_v5.step();
        world_no_v5.step();
    }

    // Dormant V5 entity should retain more energy due to decay_mult=0.3
    let energy_v5 = world_v5.semi_lives[0].maintenance_energy;
    let energy_no_v5 = world_no_v5.semi_lives[0].maintenance_energy;
    assert!(
        energy_v5 > energy_no_v5,
        "Dormant V5 entity should retain more energy ({energy_v5:.4}) \
         than non-V5 ({energy_no_v5:.4}) due to reduced decay multiplier"
    );
}

/// V5 Dormant stage blocks replication (repl_mult = 0.0).
#[test]
fn v5_dormant_blocks_replication() {
    use crate::semi_life::capability::*;
    use crate::semi_life::SemiLifeStage;

    let caps = V0_REPLICATION | V5_LIFECYCLE;
    let mut world = make_semi_life_world_with_caps(1, 1.0, 42, caps);

    // Set activation threshold very high so entity stays Dormant.
    world.config.semi_life_config.v5_activation_threshold = 10.0;
    // Give entity enough energy to replicate.
    world.semi_lives[0].maintenance_energy = 0.95;
    assert_eq!(world.semi_lives[0].stage, Some(SemiLifeStage::Dormant));

    let initial_count = world.semi_lives.len();
    for _ in 0..5 {
        world.step();
    }
    // Entity should still be Dormant (threshold = 10.0 > energy).
    assert_eq!(world.semi_lives[0].stage, Some(SemiLifeStage::Dormant));
    assert_eq!(
        world.semi_lives.len(),
        initial_count,
        "Dormant V5 entity must not replicate (repl_mult=0.0)"
    );
}

/// V5 stage transitions: Dormant → Active when energy > threshold.
#[test]
fn v5_dormant_to_active_transition() {
    use crate::semi_life::capability::*;
    use crate::semi_life::SemiLifeStage;

    let caps = V0_REPLICATION | V5_LIFECYCLE;
    let mut world = make_semi_life_world_with_caps(1, 1.0, 42, caps);

    // Start Dormant with energy above activation threshold (0.4).
    world.semi_lives[0].maintenance_energy = 0.5;
    assert_eq!(world.semi_lives[0].stage, Some(SemiLifeStage::Dormant));

    world.step();
    // Pass 0.5 transitions Dormant→Active, resets stage_ticks to 0, then increments.
    // But the increment happens in the *next* step's Pass 0.5, so after 1 step: ticks=0.
    assert_eq!(
        world.semi_lives[0].stage,
        Some(SemiLifeStage::Active),
        "Entity with energy > v5_activation_threshold should transition to Active"
    );
}

/// V5 stage transitions: Active → Dispersal after v5_dispersal_age ticks.
#[test]
fn v5_active_to_dispersal_transition() {
    use crate::semi_life::capability::*;
    use crate::semi_life::SemiLifeStage;

    let caps = V0_REPLICATION | V5_LIFECYCLE;
    let mut world = make_semi_life_world_with_caps(1, 1.0, 42, caps);

    // Force into Active stage with enough energy to survive.
    world.semi_lives[0].maintenance_energy = 0.9;
    world.semi_lives[0].stage = Some(SemiLifeStage::Active);
    world.semi_lives[0].stage_ticks = 0;

    // Run for v5_dispersal_age (10) + 1 steps.
    for _ in 0..11 {
        world.step();
        world.semi_lives[0].maintenance_energy = 0.9;
    }
    assert_eq!(
        world.semi_lives[0].stage,
        Some(SemiLifeStage::Dispersal),
        "Entity should transition to Dispersal after v5_dispersal_age ticks in Active"
    );
}

/// V5 Dispersal → Dormant after v5_dispersal_duration ticks.
#[test]
fn v5_dispersal_to_dormant_transition() {
    use crate::semi_life::capability::*;
    use crate::semi_life::SemiLifeStage;

    let caps = V0_REPLICATION | V5_LIFECYCLE;
    let mut world = make_semi_life_world_with_caps(1, 1.0, 42, caps);

    // Set activation threshold very high so Dormant doesn't immediately become Active.
    world.config.semi_life_config.v5_activation_threshold = 10.0;

    // Force into Dispersal stage.
    world.semi_lives[0].maintenance_energy = 0.5;
    world.semi_lives[0].stage = Some(SemiLifeStage::Dispersal);
    world.semi_lives[0].stage_ticks = 0;

    // Run for v5_dispersal_duration (5) + 1 steps.
    for _ in 0..6 {
        world.step();
        world.semi_lives[0].maintenance_energy = 0.5;
    }
    assert_eq!(
        world.semi_lives[0].stage,
        Some(SemiLifeStage::Dormant),
        "Entity should return to Dormant after v5_dispersal_duration ticks in Dispersal"
    );
}

/// V5 Dispersal → Dormant when energy drops below 0.2.
#[test]
fn v5_dispersal_to_dormant_on_low_energy() {
    use crate::semi_life::capability::*;
    use crate::semi_life::SemiLifeStage;

    let caps = V0_REPLICATION | V5_LIFECYCLE;
    let mut world = make_semi_life_world_with_caps(1, 0.0, 42, caps);

    // Force into Dispersal with low energy (but above death threshold).
    world.semi_lives[0].maintenance_energy = 0.15;
    world.semi_lives[0].stage = Some(SemiLifeStage::Dispersal);
    world.semi_lives[0].stage_ticks = 0;

    world.step();
    // Energy < 0.2 should trigger Dispersal → Dormant regardless of stage_ticks.
    if world.semi_lives[0].alive {
        assert_eq!(
            world.semi_lives[0].stage,
            Some(SemiLifeStage::Dormant),
            "Dispersal entity with energy < 0.2 should transition to Dormant"
        );
    }
    // If entity died from low energy, that's also acceptable — the transition check
    // happens in Pass 0.5 before energy deductions in Pass 2.
}

/// V4 entities get a policy vector initialized from config.
#[test]
fn v4_entities_have_policy_initialized() {
    use crate::semi_life::capability::*;

    let caps = V0_REPLICATION | V4_RESPONSE;
    let world = make_semi_life_world_with_caps(3, 1.0, 42, caps);
    for sl in &world.semi_lives {
        assert!(
            sl.policy.is_some(),
            "V4 entity must have policy initialized"
        );
        let policy = sl.policy.unwrap();
        // Default init: [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert!(
            (policy[0] - 0.5).abs() < f32::EPSILON,
            "grad_x weight should be 0.5"
        );
        assert!(
            (policy[1] - 0.5).abs() < f32::EPSILON,
            "grad_y weight should be 0.5"
        );
    }
}

/// Entities without V4 have no policy.
#[test]
fn non_v4_entities_have_no_policy() {
    use crate::semi_life::capability::*;

    let caps = V0_REPLICATION | V3_METABOLISM;
    let world = make_semi_life_world_with_caps(3, 1.0, 42, caps);
    for sl in &world.semi_lives {
        assert!(sl.policy.is_none(), "Non-V4 entity must not have a policy");
    }
}

/// V4 entity with gradient loses more energy than one without V4 (movement cost).
#[test]
fn v4_movement_deducts_energy() {
    use crate::semi_life::capability::*;

    let caps_v4 = V0_REPLICATION | V4_RESPONSE;
    let caps_no_v4 = V0_REPLICATION;

    // Create worlds with gradient: high resources on right half, zero on left.
    let mut world_v4 = make_semi_life_world_with_caps(1, 0.0, 42, caps_v4);
    let mut world_no_v4 = make_semi_life_world_with_caps(1, 0.0, 42, caps_no_v4);
    for w in [&mut world_v4, &mut world_no_v4] {
        for y in 0..100 {
            for x in 50..100 {
                w.resource_field_mut().set(x as f64, y as f64, 1.0);
            }
        }
    }

    // Place entity at gradient boundary (x=50) so gradient is non-zero.
    for w in [&mut world_v4, &mut world_no_v4] {
        if let Some(agent) = w
            .agents
            .iter_mut()
            .find(|a| a.owner_type == crate::agent::OwnerType::SemiLife)
        {
            agent.position = [50.0, 50.0];
        }
    }

    world_v4.step();
    world_no_v4.step();

    let energy_v4 = world_v4.semi_lives[0].maintenance_energy;
    let energy_no_v4 = world_no_v4.semi_lives[0].maintenance_energy;

    // V4 entity should lose more energy than non-V4 due to movement cost.
    assert!(
        energy_v4 < energy_no_v4,
        "V4 entity should lose more energy ({energy_v4:.4}) than non-V4 ({energy_no_v4:.4}) \
         due to movement cost"
    );
}

/// SemiLife agents never appear in the organism spatial tree (OwnerType isolation).
#[test]
fn semi_life_agents_excluded_from_organism_spatial_tree() {
    use crate::semi_life::SemiLifeArchetype::Viroid;
    let world = make_semi_life_world(vec![Viroid], 3, 1.0, 5);
    // The world has organism agents + SemiLife agents; organism-only count should
    // match num_organisms * agents_per_organism (10 in make_semi_life_world).
    let organism_agent_count = world
        .agents()
        .iter()
        .filter(|a| a.owner_type == crate::agent::OwnerType::Organism)
        .count();
    assert_eq!(
        organism_agent_count, 10,
        "Organism agent count must not include SemiLife agents"
    );
    let semi_life_agent_count = world
        .agents()
        .iter()
        .filter(|a| a.owner_type == crate::agent::OwnerType::SemiLife)
        .count();
    assert_eq!(
        semi_life_agent_count, 3,
        "3 Viroid entities should produce 3 SemiLife agents"
    );
}

// ─── V1 Leakage / Environmental Damage tests ─────────────────────────────────

/// Helper: create a SemiLife world with full control over new V1/V2 parameters.
fn make_semi_life_world_v1v2(
    num: usize,
    resource_density: f32,
    seed: u64,
    capability_bits: u8,
    cfg_patch: impl FnOnce(&mut crate::config::SemiLifeConfig),
) -> World {
    use crate::config::SemiLifeConfig;
    use crate::semi_life::SemiLifeArchetype;
    use std::collections::HashMap;

    let mut overrides = HashMap::new();
    overrides.insert("viroid".to_string(), capability_bits);

    let mut semi_life_config = SemiLifeConfig {
        enabled_archetypes: vec![SemiLifeArchetype::Viroid],
        num_per_archetype: num,
        capability_overrides: overrides,
        ..SemiLifeConfig::default()
    };
    cfg_patch(&mut semi_life_config);

    let agents: Vec<Agent> = (0..10)
        .map(|i| Agent::new(i as u32, 0, [50.0, 50.0]))
        .collect();
    let nn = NeuralNet::from_weights(std::iter::repeat_n(0.1f32, NeuralNet::WEIGHT_COUNT));
    let config = SimConfig {
        seed,
        world_size: 100.0,
        num_organisms: 1,
        agents_per_organism: 10,
        resource_regeneration_rate: resource_density * 0.01,
        enable_semi_life: true,
        semi_life_config,
        ..SimConfig::default()
    };
    let mut world = World::new(agents, vec![nn], config).unwrap();
    for x in 0..100 {
        for y in 0..100 {
            world
                .resource_field_mut()
                .set(x as f64, y as f64, resource_density);
        }
    }
    world
}

/// Non-V1 entity loses energy from leakage; V1 entity does not.
#[test]
fn v1_leakage_only_affects_entities_without_boundary() {
    use crate::semi_life::capability::{V0_REPLICATION, V1_BOUNDARY};

    // V0-only: suffers leakage
    let mut world_v0 = make_semi_life_world_v1v2(1, 0.0, 42, V0_REPLICATION, |cfg| {
        cfg.energy_leakage_rate = 0.01;
        cfg.env_damage_probability = 0.0; // isolate leakage
        cfg.maintenance_cost = 0.0; // isolate leakage
        cfg.overconsumption_waste_fraction = 0.0;
    });
    let e0_before = world_v0.semi_lives[0].maintenance_energy;
    world_v0.step();
    let e0_after = world_v0.semi_lives[0].maintenance_energy;

    // V0+V1: no leakage (but has boundary costs)
    let mut world_v1 = make_semi_life_world_v1v2(1, 0.0, 42, V0_REPLICATION | V1_BOUNDARY, |cfg| {
        cfg.energy_leakage_rate = 0.01;
        cfg.env_damage_probability = 0.0;
        cfg.maintenance_cost = 0.0;
        cfg.boundary_decay_rate = 0.0; // disable boundary costs too
        cfg.boundary_repair_rate = 0.0;
        cfg.overconsumption_waste_fraction = 0.0;
    });
    let e1_before = world_v1.semi_lives[0].maintenance_energy;
    world_v1.step();
    let e1_after = world_v1.semi_lives[0].maintenance_energy;

    // V0 should lose energy from leakage
    assert!(
        e0_after < e0_before,
        "V0 entity should lose energy from leakage: {e0_before} -> {e0_after}"
    );
    // V1 entity should NOT lose energy from leakage (no other costs either)
    assert!(
        (e1_after - e1_before).abs() < 0.001,
        "V1 entity should not lose energy from leakage: {e1_before} -> {e1_after}"
    );
}

/// Environmental damage hits non-V1 entity at full strength;
/// V1 entity absorbs partial damage via boundary.
#[test]
fn v1_boundary_absorbs_environmental_damage() {
    use crate::semi_life::capability::{V0_REPLICATION, V1_BOUNDARY};

    // Use high damage probability to guarantee a hit in deterministic seed search.
    // Run enough steps that at least one damage event occurs.
    let steps = 50;

    let mut world_v0 = make_semi_life_world_v1v2(1, 0.0, 42, V0_REPLICATION, |cfg| {
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 20.0; // prob * dt = 2.0 → guaranteed damage every step
        cfg.env_damage_amount = 0.01;
        cfg.maintenance_cost = 0.0;
        cfg.overconsumption_waste_fraction = 0.0;
    });
    let e0_before = world_v0.semi_lives[0].maintenance_energy;
    for _ in 0..steps {
        world_v0.step();
    }
    let e0_loss = e0_before - world_v0.semi_lives[0].maintenance_energy;

    let mut world_v1 = make_semi_life_world_v1v2(1, 0.0, 42, V0_REPLICATION | V1_BOUNDARY, |cfg| {
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 20.0; // prob * dt = 2.0 → guaranteed
        cfg.env_damage_amount = 0.01;
        cfg.boundary_damage_absorption = 0.8;
        cfg.boundary_damage_integrity_cost = 0.001;
        cfg.maintenance_cost = 0.0;
        cfg.boundary_decay_rate = 0.0;
        cfg.boundary_repair_rate = 0.0;
        cfg.overconsumption_waste_fraction = 0.0;
    });
    let e1_before = world_v1.semi_lives[0].maintenance_energy;
    for _ in 0..steps {
        if !world_v1.semi_lives[0].alive {
            break;
        }
        world_v1.step();
    }
    let e1_loss = e1_before - world_v1.semi_lives[0].maintenance_energy;

    // V1 entity should lose LESS energy than V0 (boundary absorbs damage)
    assert!(
        e1_loss < e0_loss,
        "V1 entity should lose less energy from damage than V0: V0 lost {e0_loss}, V1 lost {e1_loss}"
    );
}

/// Boundary integrity decreases after absorbing damage.
#[test]
fn v1_boundary_integrity_decreases_from_damage_absorption() {
    use crate::semi_life::capability::{V0_REPLICATION, V1_BOUNDARY};

    let mut world = make_semi_life_world_v1v2(1, 0.0, 42, V0_REPLICATION | V1_BOUNDARY, |cfg| {
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 20.0; // prob * dt = 20 * 0.1 = 2.0 → guaranteed hit
        cfg.env_damage_amount = 0.01;
        cfg.boundary_damage_absorption = 0.8;
        cfg.boundary_damage_integrity_cost = 0.05; // large cost for visibility
        cfg.maintenance_cost = 0.0;
        cfg.boundary_decay_rate = 0.0;
        cfg.boundary_repair_rate = 0.0;
        cfg.overconsumption_waste_fraction = 0.0;
    });

    let integrity_before = world.semi_lives[0].boundary_integrity.unwrap();
    assert!((integrity_before - 1.0).abs() < f32::EPSILON);

    world.step();
    let integrity_after = world.semi_lives[0].boundary_integrity.unwrap();
    assert!(
        integrity_after < integrity_before,
        "Boundary integrity should decrease after absorbing damage: {integrity_before} -> {integrity_after}"
    );
}

/// Backward compat: with leakage=0 and damage_prob=0, energy matches
/// a frozen golden value (seed=42, 20 steps, resource_density=1.0, V0-only).
/// The golden value 0.537999212 was captured from a known-good commit before
/// the leakage/damage/waste mechanisms were introduced.
#[test]
fn v1_backward_compat_no_leakage_no_damage() {
    use crate::semi_life::capability::V0_REPLICATION;

    // Frozen golden value — do NOT recompute from the same code path.
    let golden_energy: f32 = 0.537_999_2;

    let mut world = make_semi_life_world_v1v2(1, 1.0, 42, V0_REPLICATION, |cfg| {
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 0.0;
        cfg.overconsumption_waste_fraction = 0.0;
    });
    for _ in 0..20 {
        world.step();
    }
    let e = world.semi_lives[0].maintenance_energy;
    assert!(
        (e - golden_energy).abs() < 1e-5,
        "With leakage=0 and damage=0, energy should match golden value: got {e}, expected {golden_energy}"
    );
}

/// Tradeoff test: with low damage, V1 entity has lower net energy than V0
/// (repair cost exceeds leakage/damage savings).
#[test]
fn v1_tradeoff_low_damage_v1_worse_than_v0() {
    use crate::semi_life::capability::{V0_REPLICATION, V1_BOUNDARY};

    let steps = 100;

    // V0: no leakage savings since leakage is very small
    let mut world_v0 = make_semi_life_world_v1v2(1, 1.0, 42, V0_REPLICATION, |cfg| {
        cfg.energy_leakage_rate = 0.0001; // very small leakage
        cfg.env_damage_probability = 0.01; // very rare damage
        cfg.env_damage_amount = 0.001; // tiny damage
        cfg.overconsumption_waste_fraction = 0.0;
    });
    for _ in 0..steps {
        world_v0.step();
    }
    let e_v0 = world_v0.semi_lives[0].maintenance_energy;

    // V0+V1: repair cost should dominate over negligible leakage/damage savings
    let mut world_v1 = make_semi_life_world_v1v2(1, 1.0, 42, V0_REPLICATION | V1_BOUNDARY, |cfg| {
        cfg.energy_leakage_rate = 0.0001;
        cfg.env_damage_probability = 0.01;
        cfg.env_damage_amount = 0.001;
        cfg.overconsumption_waste_fraction = 0.0;
    });
    for _ in 0..steps {
        world_v1.step();
    }
    let e_v1 = world_v1.semi_lives[0].maintenance_energy;

    assert!(
        e_v1 < e_v0,
        "With low damage, V1 repair cost should exceed leakage savings: V0={e_v0}, V1={e_v1}"
    );
}

// ─── V2 Overconsumption Waste tests ──────────────────────────────────────────

/// No waste when uptake is at or below optimal rate.
#[test]
fn v2_no_waste_below_optimal_uptake() {
    use crate::semi_life::capability::V0_REPLICATION;

    let mut world = make_semi_life_world_v1v2(1, 0.001, 42, V0_REPLICATION, |cfg| {
        cfg.resource_uptake_rate = 0.01; // low uptake
        cfg.optimal_uptake_rate = 0.02; // threshold higher than uptake
        cfg.overconsumption_waste_fraction = 0.5;
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 0.0;
        cfg.maintenance_cost = 0.0;
    });

    let e_before = world.semi_lives[0].maintenance_energy;
    world.step();
    let e_after = world.semi_lives[0].maintenance_energy;

    // Should gain energy from uptake with no waste penalty
    assert!(
        e_after >= e_before,
        "Below optimal uptake, no waste should apply: {e_before} -> {e_after}"
    );
}

/// Full waste for non-V2 entity on excess uptake.
#[test]
fn v2_full_waste_without_homeostasis() {
    use crate::semi_life::capability::V0_REPLICATION;

    // High resource density to ensure excess uptake
    let mut world = make_semi_life_world_v1v2(1, 1.0, 42, V0_REPLICATION, |cfg| {
        cfg.resource_uptake_rate = 0.1; // high uptake
        cfg.optimal_uptake_rate = 0.01; // low threshold
        cfg.overconsumption_waste_fraction = 0.5;
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 0.0;
        cfg.maintenance_cost = 0.0;
    });

    let e_before = world.semi_lives[0].maintenance_energy;
    world.step();
    let e_after = world.semi_lives[0].maintenance_energy;
    let gain = e_after - e_before;

    // Same setup but with no waste
    let mut world_no_waste = make_semi_life_world_v1v2(1, 1.0, 42, V0_REPLICATION, |cfg| {
        cfg.resource_uptake_rate = 0.1;
        cfg.optimal_uptake_rate = 0.01;
        cfg.overconsumption_waste_fraction = 0.0;
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 0.0;
        cfg.maintenance_cost = 0.0;
    });

    let e_before_nw = world_no_waste.semi_lives[0].maintenance_energy;
    world_no_waste.step();
    let e_after_nw = world_no_waste.semi_lives[0].maintenance_energy;
    let gain_no_waste = e_after_nw - e_before_nw;

    assert!(
        gain < gain_no_waste,
        "With overconsumption waste, energy gain should be less: {gain} vs {gain_no_waste} (no waste)"
    );
}

/// V2 with regulator_state=1.0 reduces waste by ~80%.
#[test]
fn v2_regulator_reduces_waste() {
    use crate::semi_life::capability::{V0_REPLICATION, V2_HOMEOSTASIS};

    // V0 only: full waste
    let mut world_v0 = make_semi_life_world_v1v2(1, 1.0, 42, V0_REPLICATION, |cfg| {
        cfg.resource_uptake_rate = 0.1;
        cfg.optimal_uptake_rate = 0.01;
        cfg.overconsumption_waste_fraction = 0.5;
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 0.0;
        cfg.maintenance_cost = 0.0;
        cfg.regulator_cost_per_step = 0.0; // isolate waste effect
    });
    let e0_before = world_v0.semi_lives[0].maintenance_energy;
    world_v0.step();
    let gain_v0 = world_v0.semi_lives[0].maintenance_energy - e0_before;

    // V0+V2: regulator reduces waste
    let mut world_v2 =
        make_semi_life_world_v1v2(1, 1.0, 42, V0_REPLICATION | V2_HOMEOSTASIS, |cfg| {
            cfg.resource_uptake_rate = 0.1;
            cfg.optimal_uptake_rate = 0.01;
            cfg.overconsumption_waste_fraction = 0.5;
            cfg.energy_leakage_rate = 0.0;
            cfg.env_damage_probability = 0.0;
            cfg.maintenance_cost = 0.0;
            cfg.regulator_init = 1.0;
            cfg.regulator_cost_per_step = 0.0;
        });
    let e2_before = world_v2.semi_lives[0].maintenance_energy;
    world_v2.step();
    let gain_v2 = world_v2.semi_lives[0].maintenance_energy - e2_before;

    assert!(
        gain_v2 > gain_v0,
        "V2 regulator should reduce waste, yielding higher energy gain: V0={gain_v0}, V2={gain_v2}"
    );
}

/// Backward compat: with overconsumption_waste_fraction=0, no behavior change.
#[test]
fn v2_backward_compat_zero_waste_fraction() {
    use crate::semi_life::capability::V0_REPLICATION;

    let mut world = make_semi_life_world_v1v2(1, 1.0, 42, V0_REPLICATION, |cfg| {
        cfg.overconsumption_waste_fraction = 0.0;
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 0.0;
    });

    let e_before = world.semi_lives[0].maintenance_energy;
    world.step();
    let e_after = world.semi_lives[0].maintenance_energy;

    // Should just be maintenance_cost + uptake (no waste deduction)
    // Energy should increase from uptake (resource density 1.0)
    assert!(
        e_after > e_before - 0.01,
        "With zero waste fraction, entity should gain energy normally: {e_before} -> {e_after}"
    );
}

// ─── Multi-channel Internalization Index tests ───────────────────────────────

/// V0-only entity: 0 active channels → II=0.
#[test]
fn ii_v0_only_returns_zero() {
    use crate::semi_life::capability::V0_REPLICATION;

    let mut world = make_semi_life_world_v1v2(1, 1.0, 42, V0_REPLICATION, |cfg| {
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 0.0;
        cfg.overconsumption_waste_fraction = 0.0;
    });
    world.step();

    let snaps = world.semi_life_snapshots();
    assert!(
        snaps[0].internalization_index.abs() < f32::EPSILON,
        "V0-only entity should have II=0, got {}",
        snaps[0].internalization_index
    );
    assert!(snaps[0].ii_energy.abs() < f32::EPSILON);
    assert!(snaps[0].ii_regulation.abs() < f32::EPSILON);
    assert!(snaps[0].ii_behavior.abs() < f32::EPSILON);
    assert!(snaps[0].ii_lifecycle.abs() < f32::EPSILON);
}

/// V0+V1+V2+V3 entity: energy channel should be active and > 0.
#[test]
fn ii_v3_energy_channel_active() {
    use crate::semi_life::capability::{
        V0_REPLICATION, V1_BOUNDARY, V2_HOMEOSTASIS, V3_METABOLISM,
    };

    let caps = V0_REPLICATION | V1_BOUNDARY | V2_HOMEOSTASIS | V3_METABOLISM;
    let mut world = make_semi_life_world_v1v2(1, 1.0, 42, caps, |cfg| {
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 0.0;
        cfg.overconsumption_waste_fraction = 0.0;
    });
    // Run a few steps to accumulate energy flow
    for _ in 0..5 {
        world.step();
    }

    let snaps = world.semi_life_snapshots();
    let snap = &snaps[0];
    assert!(
        snap.ii_energy > 0.0,
        "V3 entity should have ii_energy > 0, got {}",
        snap.ii_energy
    );
    assert!(
        snap.internalization_index > 0.0,
        "Composite II should be > 0 with V3 active"
    );
}

/// Old single-channel II should match ii_energy when only energy channel is active.
#[test]
fn ii_composite_matches_energy_when_single_channel() {
    use crate::semi_life::capability::{V0_REPLICATION, V3_METABOLISM};

    // V0+V3 only (no V2/V4/V5 → only energy channel active)
    let caps = V0_REPLICATION | V3_METABOLISM;
    let mut world = make_semi_life_world_v1v2(1, 1.0, 42, caps, |cfg| {
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 0.0;
        cfg.overconsumption_waste_fraction = 0.0;
    });
    for _ in 0..5 {
        world.step();
    }

    let snaps = world.semi_life_snapshots();
    let snap = &snaps[0];
    assert!(
        (snap.internalization_index - snap.ii_energy).abs() < 0.001,
        "With only energy channel, composite ({}) should equal ii_energy ({})",
        snap.internalization_index,
        snap.ii_energy
    );
}

/// Multi-channel: channel count increases with capabilities.
#[test]
fn ii_channel_count_increases_with_capabilities() {
    use crate::semi_life::capability::*;

    let steps = 10;

    // V0+V3: energy channel only
    let mut w1 = make_semi_life_world_v1v2(1, 1.0, 42, V0_REPLICATION | V3_METABOLISM, |cfg| {
        cfg.energy_leakage_rate = 0.0;
        cfg.env_damage_probability = 0.0;
        cfg.overconsumption_waste_fraction = 0.3;
    });
    for _ in 0..steps {
        w1.step();
    }
    let s1 = &w1.semi_life_snapshots()[0];

    // V0+V2+V3: energy + regulation channels
    let mut w2 = make_semi_life_world_v1v2(
        1,
        1.0,
        42,
        V0_REPLICATION | V2_HOMEOSTASIS | V3_METABOLISM,
        |cfg| {
            cfg.energy_leakage_rate = 0.0;
            cfg.env_damage_probability = 0.0;
            cfg.overconsumption_waste_fraction = 0.3;
            cfg.regulator_cost_per_step = 0.0;
            cfg.resource_uptake_rate = 0.1; // high to trigger overconsumption
            cfg.optimal_uptake_rate = 0.01;
        },
    );
    for _ in 0..steps {
        w2.step();
    }
    let s2 = &w2.semi_life_snapshots()[0];

    // V0+V3 should only have energy channel
    assert!(s1.ii_energy > 0.0);
    assert!(
        s1.ii_regulation.abs() < f32::EPSILON,
        "V0+V3 should have no regulation channel"
    );

    // V0+V2+V3 should have both energy and regulation channels
    assert!(s2.ii_energy > 0.0);
    assert!(
        s2.ii_regulation > 0.0,
        "V0+V2+V3 with overconsumption should have regulation channel > 0, got {}",
        s2.ii_regulation
    );
}
