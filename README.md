# sheaf-coherence

**Sheaf-theoretic multi-agent knowledge consistency for Rust.**

This library models a fleet of agents as a sheaf over an open cover of a shared information space. Each agent maintains a *local section* — its view of the global state restricted to the variables it can observe. The library computes Čech sheaf cohomology groups to detect and classify contradictions across the fleet, and can repair them via sheaf gluing when possible.

## Why Sheaf Cohomology?

In a multi-agent system, each agent sees only part of the world. When two agents' views overlap, they must agree on the overlap — otherwise the fleet is *inconsistent*. Sheaf cohomology gives a rigorous algebraic framework for this:

- **H⁰ (global sections):** The space of globally consistent belief vectors. `dim H⁰ > 0` means a global consensus exists.
- **H¹ (obstructions):** The local-to-global obstruction space. `dim H¹ > 0` means agents *cannot* reconcile their views — there are fundamental contradictions that no local patching can resolve.

This is the difference between "agents disagree but can converge" (H¹ = 0) and "agents are trapped in irreconcilable contradictions" (H¹ ≠ 0).

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  OpenCover  │────▶│ LocalSection │────▶│ Čech Cochain  │
│ (who sees   │     │ (what they   │     │ Complex       │
│  what)      │     │  believe)    │     │ (C⁰→C¹→C²)   │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                            ┌─────────────────────┼──────────────────┐
                            ▼                     ▼                  ▼
                    ┌──────────────┐    ┌──────────────────┐  ┌────────────┐
                    │   H⁰: Global │    │ H¹: Obstructions │  │Persistence │
                    │   Sections   │    │ & Conflicts      │  │ Diagrams   │
                    └──────┬───────┘    └────────┬─────────┘  └────────────┘
                           │                     │
                           ▼                     ▼
                    ┌──────────────┐    ┌──────────────────┐
                    │   Gluing:    │    │ Classification:  │
                    │   Repair     │    │ Pairwise/Multi/  │
                    │   (H¹=0)     │    │ Hidden           │
                    └──────────────┘    └──────────────────┘
```

## Modules

| Module | Purpose |
|--------|---------|
| `cover` | Open covers over agent knowledge domains with intersection lattices |
| `section` | Local sections: per-agent belief vectors with restriction maps |
| `cochain` | Čech cochain complexes and Gaussian elimination (no external deps) |
| `obstruction` | Detects and classifies local-to-global obstruction classes |
| `gluing` | Consistency repair via sheaf gluing (collate compatible sections) |
| `persistence` | Persistent sheaf cohomology over varying resolution thresholds |

## Quick Start

```rust
use sheaf_coherence::*;

// Define what each agent can see
let cover = cover::OpenCover::new(vec![
    vec![0, 1, 2],  // Agent 0 sees variables 0, 1, 2
    vec![2, 3, 4],  // Agent 1 sees variables 2, 3, 4
]);

// Each agent has beliefs over its visible variables
let fam = section::SectionFamily::new(vec![
    section::LocalSection::new(0, vec![1.0, 2.0, 3.0]),
    section::LocalSection::new(1, vec![3.0, 4.0, 5.0]),
]);

// Check consistency
assert!(fam.is_consistent(&cover, 1e-8));

// Compute cohomology
let (h0, h1) = cochain::compute_cohomology(&cover, 1e-8);
println!("H⁰ dimension: {} (global sections)", h0.dimension);
println!("H¹ dimension: {} (obstructions)", h1.dimension);

// Glue into a global section
let result = gluing::glue(&cover, &fam, 1e-8);
match result {
    gluing::GluingResult::Success(global) => {
        println!("Global section: {:?}", global);
        // [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    gluing::GluingResult::Failed { h1_dimension } => {
        println!("Cannot glue! {} obstruction(s)", h1_dimension);
    }
}
```

### Detecting Contradictions

```rust
// Agent 0 thinks variable 1 = 2.0, Agent 1 thinks variable 1 = 9.0
let bad_fam = section::SectionFamily::new(vec![
    section::LocalSection::new(0, vec![1.0, 2.0]),
    section::LocalSection::new(1, vec![9.0, 3.0]),
]);

let obs = obstruction::ObstructionClass::detect(&cover, &bad_fam, 1e-8);
assert!(!obs.is_consistent());
assert_eq!(obs.conflicting_agents, vec![(0, 1)]);
```

### Persistent Cohomology

Track how cohomology evolves as the cover gets finer:

```rust
let stages = vec![
    (1.0, coarse_cover, coarse_sections),
    (2.0, medium_cover, medium_sections),
    (3.0, fine_cover, fine_sections),
];
let (snapshots, diagram) = persistence::persistent_cohomology(&stages, 1e-8);
for snap in &snapshots {
    println!("res={}: H⁰={}, H¹={}", snap.resolution, snap.h0_dimension, snap.h1_dimension);
}
```

## Core Types

```rust
struct OpenCover { sets: Vec<Vec<usize>> }
struct LocalSection { agent_id: usize, data: Vec<f64> }
struct RestrictionMap { from_set: usize, to_intersection: Vec<usize>, mapping: Vec<usize> }
struct CohomologyGroup { degree: usize, dimension: usize, generators: Vec<Vec<f64>> }
struct ObstructionClass { h1_dimension: usize, conflicting_agents: Vec<(usize, usize)> }
```

All public types derive `Serialize` and `Deserialize` via Serde.

## How It Works

### Čech Cohomology

Given an open cover $\{U_i\}$ of the state space, we form the Čech cochain complex:

$$0 \to C^0 \xrightarrow{d^0} C^1 \xrightarrow{d^1} C^2 \to \cdots$$

where:
- $C^0 = \prod_i \mathcal{F}(U_i)$ — sections on each open set
- $C^1 = \prod_{i<j} \mathcal{F}(U_i \cap U_j)$ — sections on pairwise overlaps
- $C^2 = \prod_{i<j<k} \mathcal{F}(U_i \cap U_j \cap U_k)$ — triple overlaps
- $d^0$ computes pairwise differences (restriction + subtraction)
- $d^1$ enforces the cocycle condition on triple overlaps

The cohomology groups are:
- $H^0 = \ker(d^0)$ — globally compatible sections
- $H^1 = \ker(d^1) / \text{im}(d^0)$ — obstructions to gluing

### Gaussian Elimination

All linear algebra (rank, nullspace, image) is computed via from-scratch Gaussian elimination with partial pivoting. No external math dependencies.

## Testing

```bash
cargo test    # 45 tests (5 unit + 40 integration)
```

Test coverage includes:
- Single agent (trivial cover) → H⁰ full, H¹ = 0
- Two agents with contradictory beliefs → detected
- Consistent fleet → global section exists
- Inconsistent fleet → nonzero H¹ or conflicts
- Restriction maps compose correctly
- Čech complex has correct dimensionality
- Serde roundtrips for all public types
- Persistence diagrams over multiple resolutions
- Gluing succeeds/fails based on H¹
- Cover operations (intersection, triple, universe, validity)

## License

MIT

---

## Fleet Integration

Part of the **SuperInstance research program**. This repo connects to the
200-repo MIDI fleet through the shared mathematical framework.

**Direct counterpart:** [$fleet_link](https://github.com/SuperInstance/$fleet_link)

**Bridge:** $bridge_desc
