# ALIFE Conference Research Plan: **Virus → Semi-Life → Life Transition** in Your “Digital Life” World

> This document is written to be **actionable for research + camera-ready for ALIFE**. It builds directly on your paper’s current framing: **functional analogy** (dynamic process + measurable degradation + feedback coupling) and the **criterion-ablation methodology** as a reusable evaluation protocol.  

---

## 0) Context: What you already have (baseline platform)

Your current paper establishes:

* A system that integrates the **seven textbook criteria** (cellular organization, metabolism, homeostasis, growth/development, reproduction, response, evolution) as **interdependent processes** in a single ALife world. 
* An operational definition of “functional analogy” requiring:

  1. **Dynamic process** (resource consumption every timestep),
  2. **Measurable degradation** upon ablation,
  3. **Feedback coupling** (at least one feedback loop with other criteria). 
* A strong empirical base: disabling any single criterion causes **statistically significant population decline**, with effect sizes reported (Cliff’s δ). 
* Additional credibility checks: pairwise ablations show sub-additive interactions; proxy controls show implementation matters; sham ablation no effect; graded ablation dose-response. 
* You explicitly position the work as **weak ALife** and methodological (not claiming “literally alive”). 

**This is a perfect launchpad** to study your new passion question:

> **Can a virus-like replicator become life-like by internalizing missing functions?**

---

## 1) New research thesis (what this ALIFE submission should be “about”)

### 1.1 Core claim (high impact + testable)

> A “virus-like” replicator in the same world can cross a **life-likeness transition** when it internalizes certain functions, and this transition can be demonstrated using your existing falsifiable framework (dynamic process / degradation / coupling) rather than philosophical argument. 

### 1.2 Why this is a “drastic” leap vs incremental improvement

Your current paper proves “all 7 criteria can be integrated and each matters” (methodology paper). 
This new work asks a more ambitious origin-style question:

* **Not**: “Can we implement criteria X?”
* **But**: “When does a **non-life replicator** become **life-like** under measurable conditions?”

That’s a story ALIFE reviewers remember.

---

## 2) Definitions for this project (tight and reviewer-proof)

### 2.1 “Virus-like” in your world (minimal operational definition)

A virus-like entity is a replicator that:

1. **Reproduces** and can evolve (optional at start), but
2. Has **little or no internal resource-processing** and/or regulation,
3. Relies on the world’s existing resource/interaction pathways so that its survival and copying are largely **outsourced**.

> Important: you do **not** need a “Host class”. “Outsourcing” can be implemented as “viral replication draws from external resource channels / interacts parasitically with other organisms.” The goal is *operational dependence*, not biology cosplay.

### 2.2 “Life-likeness transition” (what counts as success)

We define a transition as a qualitative change in at least **two** of:

1. **Persistence region expands** (survives harsher environments / perturbations),
2. **Recovery improves** after shocks (resilience),
3. **Tradeoffs emerge** (e.g., replication speed vs persistence),
4. **Coupling/closure strengthens** (more criteria become mutually dependent).

This aligns naturally with your framework emphasizing interdependence and measurable degradation. 

---

## 3) Experimental program (the “capability ladder”)

### 3.1 Virus family library (diversity is crucial)

Create **3–5 viral archetypes** (same world, different parameterizations), e.g.:

1. **Fast / fragile**: high replication rate, short survival outside favorable conditions
2. **Slow / persistent**: low replication, high durability
3. **Aggressive parasite**: harms nearby organisms strongly
4. **Stealth**: low detection/interaction footprint
5. **High-mutation** vs **low-mutation**

This prevents the result from being “one hand-tuned virus did X”.

### 3.2 Capability ladder: internalize missing functions (one-by-one)

Start at V0 (“virus-only”) then add functions:

* **V0: Base virus**

  * reproduction (and optionally mutation), minimal internal state.

* **V1: Boundary / cellular organization**

  * Adds a “capsid-like” integrity variable analogous to your boundary integrity / cohesion proxy logic. (Your current work already tracks boundary via scalar integrity + cohesion validation.)  

* **V2: Homeostasis**

  * Adds regulation (e.g., replication throttling when internal state is unstable), analogous to your active homeostatic regulation claim. 

* **V3: Metabolism**

  * Adds internal resource conversion (however minimal), echoing your emphasis that meaningful criteria should be dynamic, resource-consuming processes. 

* **V4: Response to stimuli**

  * Adds sensing + action selection (movement or interaction choices), which in your system is strongly necessary (large ablation effect). 

* **V5: Growth/development**

  * Adds staged lifecycle (dormant → active → dispersal), paralleling your staged developmental program. 

**Key principle:** each added function should be implemented as a **dynamic process**, not a static “buff”, matching your functional analogy condition #1. 

---

## 4) The signature result: Phase diagrams (this is what makes it “drastic”)

### 4.1 Environment harshness axes (choose 2–3)

Construct environment regimes that stress different capabilities:

* **Resource scarcity / patchiness** (metabolism + response matter)
* **Periodic shocks** (homeostasis + boundary matter)
* **Competition intensity / predation-like pressure** (response + boundary + growth)
* (Optional) **Mutation load** (evolution robustness)

Your current paper already explored cyclic stress and treats it as supplemental because pooled significance wasn’t reached; here, we’ll use shocks mainly to reveal qualitative transitions and tradeoffs. 

### 4.2 Plot: capability ladder × harshness → survival region

For each virus family × capability level V0..V5:

* Run **n seeds** (keep your held-out seed discipline) 
* Compute survival/persistence & resilience

Deliverables:

1. **Survival phase diagram**: where the population persists
2. **Recovery phase diagram**: how quickly it rebounds after shock
3. **Tradeoff plots**: replication rate vs persistence

This set of figures will likely be the “centerpiece” of the ALIFE submission.

---

## 5) Measurement: reuse your paper’s rigor, but avoid tautologies

Your current work already anticipates the “tautology” problem and includes proxy controls, sham ablations, graded ablations, and criterion-orthogonal measures (cohesion, lifespan).  

We should apply the same discipline here.

### 5.1 Primary outcomes (not structurally entailed)

* **Median lifespan** per entity (works well for “reproduction is tautological” criticism). 
* **Persistence under perturbation** (time-to-extinction, rebound rate)
* **Diversity / lineage persistence** (if you track lineages)

### 5.2 “Internalization index” (new metric you should add)

A quantitative measure of how dependent the virus is on external support:

* Fraction of required energy obtained internally vs externally
* Fraction of time “survival depends on contact with others”
* Amount of regulation performed internally

This becomes your “semi-life → life” continuous axis.

### 5.3 Causal tests: apply your functional analogy framework to viruses too

For each “added capability”, verify:

1. **Dynamic operation:** it consumes resources each step 
2. **Ablation hurts:** remove that capability from V(k) and show a significant drop
3. **Feedback coupling exists:** show at least one feedback loop with another capability 

This is your strongest “continuity” with the existing paper and turns the new work into an extension rather than a tangent.

---

## 6) Analysis plan (statistics + reviewer expectations)

### 6.1 Statistical structure (recommended)

* Same as your paper: Mann–Whitney U, Holm-Bonferroni (or BH if you prefer), report Cliff’s δ. 
* Use **dose–response** logic where possible (graded internalization), mirroring your graded ablation logic. 
* When comparing many models: pre-register “main comparisons” vs “exploratory.”

### 6.2 Negative controls (carry over your strongest credibility move)

* Sham capability (consumes compute but no state effect) should not change outcomes. 
* Proxy controls: alternative implementations for a capability should change ecology in interpretable ways (as you already demonstrated with metabolism). 

---

## 7) Expected contributions (how this becomes an ALIFE Full Paper)

### 7.1 Contribution statement (suggested)

1. A **virus-to-life capability ladder** within a single digital ecology
2. Evidence for a **life-likeness transition** (phase diagram + tradeoff emergence)
3. Extension of your **functional analogy / ablation framework** to borderline systems (semi-life) 
4. A reusable benchmark setup (“borderline life suite”) akin to how your current paper frames portability of the methodology 

### 7.2 How it fits your current paper’s scope statements

Your current conclusion emphasizes portability and future work on larger populations and perturbations. 
This project **directly** instantiates “systematic environmental perturbation studies” and adds an origin-of-life style investigation.

---

## 8) Practical implementation blueprint (high-level)

### 8.1 Minimal data model additions

* New entity type: `Virus` (or “Genome-only agent”) with state:

  * `integrity` (boundary)
  * `energy` (metabolism)
  * `regulator_state` (homeostasis)
  * `stage` (development)
  * `policy` (response)

### 8.2 Interaction rules (conceptual)

* V0 replicates opportunistically, but survival is fragile and depends on favorable external conditions.
* Each additional capability reduces dependence / increases resilience, but introduces **costs**, creating tradeoffs.

> Don’t make added functions “free upgrades”. Make them require resources each timestep (to preserve your “dynamic process” principle). 

---

## 9) Risks and how to preempt them (reviewer psychology)

### Risk A: “You just engineered tiny organisms.”

**Mitigation:** emphasize “internalization” + show tradeoffs + show phase transition.

### Risk B: “Virus definition is vague.”

**Mitigation:** operationalize “outsourcing/dependence” via the **Internalization Index** and ablation tests.

### Risk C: “Reproduction makes population metrics tautological.”

**Mitigation:** prioritize lifespan and resilience as primary DVs (you already use lifespan as orthogonal). 

### Risk D: “Evolution still weak.”

**Mitigation:** treat evolution as optional in the ladder: you can demonstrate a life-likeness transition even before strong adaptation, then include a smaller evolution section as secondary evidence (consistent with your current caveats). 

---

## 10) Draft outline for the ALIFE submission (suggested)

1. **Motivation:** borderline cases (viruses) + need for testable criteria
2. **Baseline platform:** your 7-criteria world + functional analogy + ablation methodology 
3. **Virus ladder:** V0..V5 definition + costs
4. **Experiments:** phase diagrams + perturbations + controls
5. **Results:** transition regions + tradeoffs + coupling evidence
6. **Discussion:** what “life-likeness” means operationally; limits; future work

---

# Questions to update this plan (please answer briefly; we’ll revise this doc)

## Q1) What does “resource/energy” mean in your current world?

* Is there a single energy budget per organism (like Polyworld-style), a spatial resource field, multiple resources, or something else?

This determines how to implement **viral outsourcing** and **metabolism internalization** cleanly.

## Q2) What is the easiest “viral action” in your codebase?

Pick one:

1. viruses are **their own agents** moving/acting in the world
2. viruses are **payloads that attach to existing organisms** (parasite/infection)
3. viruses are **environmental replicators** (replicate in resource zones)

Any is fine; we’ll tailor the ladder mechanics to the least invasive option.

## Q3) Can viruses kill hosts / reduce host fitness?

Yes/no. If yes, we can get the best tradeoff story: replication speed vs sustainability.

## Q4) Which transition do you personally want to demonstrate most?

Pick one:

1. **host-independent persistence** (even partial)
2. **resilience under shocks**
3. **emergent tradeoffs**
4. **coupling/closure increase**

Your answer determines what becomes Figure 1.

---

If you answer Q1–Q4, I’ll **rewrite this document into a “v1 research spec”** with:

* exact experiment matrix (runs × seeds × conditions),
* specific plots to generate,
* and “minimum publishable results” checkpoints.
