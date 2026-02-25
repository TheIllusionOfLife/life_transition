# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Life Transition** is an artificial life (ALife) research project studying the **Virus → Semi-Life → Life** transition: can a virus-like replicator become life-like by internalizing the biological functions it originally outsourced?

**Target venue**: ALIFE 2026 Full Paper (8p), deadline ~April 1, 2026.

**Stance**: Weak ALife — the system is a functional model, not a claim of literal life.

## Document Structure

| Document | Role |
|----------|------|
| `docs/research/research-plan.md` | **Authoritative plan**: research thesis, capability ladder V0–V5, experimental program, measurement framework |
| `docs/archive/action-plan.md` | Archived: prior 7-criteria project schedule (historical reference only) |
| `docs/archive/digital-life-project-overview.md` | Archived: initial 7-criteria project overview |
| `docs/archive/unified-review.md` | Archived: peer review of prior project (Japanese) |

When documents conflict, `docs/research/research-plan.md` takes precedence.

## Architecture Decisions

- **Hybrid two-layer**: Swarm agents (10-50 per organism) form organism-level structures; organisms (10-50) inhabit a continuous 2D environment
- **Language**: Rust (core simulation) + Python (experiment management, analysis). Bound via PyO3/maturin
- **LaTeX**: Use `tectonic` for paper compilation (not pdflatex/latexmk)
- **Neural controllers**: Evolutionary NN (main). LLM (Ollama) only for ablation study
- **Compute**: Mac Mini M2 Pro. Target: >100 timesteps/sec for 2,500 agents
- **Metabolism**: Graph-based metabolic networks, genetically encoded and evolvable
- **Module names**: `life_transition` (Python), `life_transition_core` (Rust)

## Background Platform (Seven Biological Criteria)

The existing simulation implements all seven textbook criteria as interdependent processes. This serves as the **baseline world** into which virus-like entities are introduced:

1. **Cellular Organization** — Active boundary maintenance (swarm coordination)
2. **Metabolism** — Graph-based multi-step transformation network
3. **Homeostasis** — NN controller regulates internal state vector
4. **Growth/Development** — Seed → mature organism via developmental program
5. **Reproduction** — Organism-initiated division when metabolically ready
6. **Response to Stimuli** — Local sensory field + NN processing → behavior
7. **Evolution** — Heritable genomes, mutation/recombination, differential survival

## Core Experiment: Capability Ladder

**New entity type**: `Virus` — a replicator that starts with minimal internal function and gains capabilities one step at a time:

| Level | Added function |
|-------|---------------|
| V0 | Base replicator: reproduction only, minimal internal state |
| V1 | Boundary / cellular organization (capsid-like integrity) |
| V2 | Homeostasis (replication throttling, internal regulation) |
| V3 | Metabolism (internal resource conversion) |
| V4 | Response to stimuli (sensing + action selection) |
| V5 | Growth/development (staged lifecycle: dormant → active → dispersal) |

**Key principle**: each added function must be a dynamic process (resource-consuming every timestep) — matching the functional analogy condition from the prior paper.

## Signature Result: Phase Diagrams

For each virus archetype × capability level V0–V5 × environment harshness:
- **Survival phase diagram**: where the population persists
- **Recovery phase diagram**: rebound rate after shocks
- **Tradeoff plots**: replication rate vs persistence

**Virus archetypes** (3–5 families, same world, different parameterizations):
- Fast/fragile, Slow/persistent, Aggressive parasite, Stealth, High-mutation vs Low-mutation

## New Metric: Internalization Index

Fraction of energy/regulation obtained internally vs externally. This becomes the continuous "semi-life → life" axis.

## Statistical Design

Carry forward from prior paper:
- Mann-Whitney U, Holm-Bonferroni correction, Cliff's δ (effect size)
- Data separation: calibration seeds 0–99, final test seeds 100–199
- Negative controls: sham capability (compute but no state effect), proxy controls

## Functional Analogy Framework (Evaluation, Carried Forward)

A computational process is a functional analogy of a biological criterion iff:
- (a) It is a **dynamic process** requiring sustained resource consumption
- (b) Its removal causes **measurable degradation** of organism self-maintenance
- (c) It forms a **feedback loop** with at least one other criterion

Apply to each added V-level capability via ablation test.

## Pivot Strategy

| Trigger | Pivot |
|---------|-------|
| Virus mechanics unstable | Simplify to a single archetype; drop the family library |
| Phase transitions absent | Focus on tradeoff emergence instead; reframe as "cost-benefit" paper |
| V0–V5 integration infeasible by deadline | Report V0–V3 only; frame as "partial internalization ladder" |
| Full paper infeasible by Week 4 | Switch to Extended Abstract (2-4p) |
| Metabolic network unsustainable | Graph-based → ODE-based metabolism |

## Language Notes

Research documents are bilingual (Japanese + English). `docs/archive/unified-review.md` is primarily Japanese. When generating research content, match the language of the target document.
