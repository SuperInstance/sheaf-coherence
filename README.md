# sheaf-coherence

[![crates.io](https://img.shields.io/crates/v/sheaf-coherence.svg)](https://crates.io/crates/sheaf-coherence)
[![docs.rs](https://docs.rs/sheaf-coherence/badge.svg)](https://docs.rs/sheaf-coherence)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The Problem

A group of agents needs to agree. But agreement isn't binary — agents can agree on some things and disagree on others, and the disagreements have structure. Agent A and B agree on X but disagree on Y. Agent B and C agree on Y but disagree on X. Who's "right"?

Standard consensus algorithms (majority vote, averaging) collapse this structure. They give you an answer but not the *topology* of agreement. You can't tell if a disagreement is a local miscommunication (fixable) or a fundamental value conflict (irreconcilable).

## The Idea: Cellular Sheaves

A **cellular sheaf** assigns data to every node and edge of a graph, plus **restriction maps** that say how node-data should relate across edges. If agent i and agent j are connected, the restriction maps fᵢⱼ and fⱼᵢ specify what "agreement" means between them.

The **sheaf Laplacian** L_𝓕 generalizes the graph Laplacian. Where the ordinary Laplacian measures how different a node's value is from its neighbors', the sheaf Laplacian measures how much the restriction maps are violated:

```
(L_𝓕 x)ᵢ = Σⱼ (fᵢⱼᵀ fᵢⱼ xᵢ - fᵢⱼᵀ fⱼᵢ xⱼ)
```

When L_𝓕 x = 0, every restriction map is satisfied — the agents are in perfect coherence. The eigenvalues of L_𝓕 tell you how far from coherence the system is, and the eigenvectors tell you *which* disagreements are the most costly.

## How To Use It

### Define a sheaf on a graph

```rust
use sheaf_coherence::{WeightedGraph, CellularSheaf, RestrictionMap};

let graph = WeightedGraph::from_edges(4, &[
    (0, 1), (1, 2), (2, 3), (0, 3),
]);

// Each agent has a 2D belief vector
// Restriction maps: what "agreement" means between connected agents
let sheaf = CellularSheaf::new(graph, 2)
    .restriction(0, 1, RestrictionMap::identity())  // agents 0,1 must agree exactly
    .restriction(1, 2, RestrictionMap::identity())
    .restriction(2, 3, RestrictionMap::projection(0)) // agents 2,3 agree on dimension 0 only
    .restriction(0, 3, RestrictionMap::rotation(0.1)); // agents 0,3 agree up to small rotation
```

### Assign beliefs and measure coherence

```rust
let beliefs = vec![
    vec![1.0, 0.0],  // agent 0
    vec![1.0, 0.0],  // agent 1 (agrees with 0)
    vec![1.0, 1.0],  // agent 2 (agrees with 1 on dim 0, disagrees on dim 1)
    vec![0.9, 0.5],  // agent 3
];

let laplacian = sheaf.laplacian();
let coherence_energy = sheaf.coherence_energy(&beliefs);
println!("Coherence energy: {:.4} (0 = perfect coherence)", coherence_energy);
```

### Find the closest coherent state

The kernel of L_𝓕 is the space of perfectly coherent belief assignments. To find the closest coherent state to your agents' current beliefs:

```rust
let coherent = sheaf.project_to_coherent(&beliefs);
// This is the set of beliefs that satisfies all restriction maps
// and is closest (in L²) to the original beliefs
```

### Spectral analysis

The eigenvalues of the sheaf Laplacian tell you the "cost surface" of disagreement:

```rust
use sheaf_coherence::spectral::SheafSpectrum;

let spectrum = SheafSpectrum::compute(&sheaf);
println!("λ₁ = {:.4} (0 = non-trivial coherent state exists)", spectrum.smallest());
println!("λ₂ = {:.4} (gap = how stable coherence is)", spectrum.gap());
```

- λ₁ = 0: the sheaf admits non-trivial global sections (agents can all agree without being trivial)
- Small spectral gap: coherence is fragile — small perturbations break agreement
- Large spectral gap: coherence is robust

## Key Types

| Type | What it represents |
|---|---|
| `CellularSheaf` | Data + restriction maps on a graph |
| `RestrictionMap` | How agreement is defined between two agents |
| `SheafLaplacian` | Generalized Laplacian measuring coherence violation |
| `CoherenceReport` | Energy, violation per edge, global coherence score |

## Module Map

| Module | What it does |
|---|---|
| `graph` | `WeightedGraph` — the underlying communication topology |
| `sheaf` | `CellularSheaf` — data + restriction maps |
| `restriction` | `RestrictionMap` — identity, projection, rotation, custom |
| `laplacian` | `SheafLaplacian` — the generalized Laplacian |
| `coherence` | Coherence energy, projection to coherent states |
| `spectral` | Eigenvalue analysis of the sheaf Laplacian |
| `error` | `SheafError` |

## When To Use This

- **Multi-agent consensus** with structured disagreement (not just "agree/disagree")
- **Belief alignment**: detect which beliefs are fixable vs fundamental conflicts
- **Communication topology design**: which edges (communication channels) matter most for coherence?
- **Robustness analysis**: if an agent's beliefs shift slightly, does coherence collapse?

## Design Decisions

- **Why not just average beliefs?** Averaging ignores the *structure* of agreement. Two agents can agree on "climate change is real" while disagreeing on "we should act now." The sheaf captures this — restriction maps specify which dimensions of belief must match.
- **Why cellular sheaves, not persistent sheaves?** Cellular sheaves operate on a fixed graph (the communication topology). Persistent sheaves would track how coherence changes as the topology evolves — that's a separate crate.
- **Restriction maps as the core abstraction**: The restriction map is where you encode domain knowledge. Identity = agents must agree exactly. Projection = agents agree on a subspace. Rotation = agents agree up to a known transformation. Custom = you define what agreement means.

## Links

- [Documentation](https://docs.rs/sheaf-coherence)
- [Repository](https://github.com/SuperInstance/sheaf-coherence)
- [crates.io](https://crates.io/crates/sheaf-coherence)
- Hansen & Ghrist (2019) — *Opinion Dynamics on Discourse Sheaves*

## License

MIT
