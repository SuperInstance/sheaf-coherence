# sheaf-coherence

[![crates.io](https://img.shields.io/crates/v/sheaf-coherence.svg)](https://crates.io/crates/sheaf-coherence)
[![docs.rs](https://docs.rs/sheaf-coherence/badge.svg)](https://docs.rs/sheaf-coherence)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Cellular sheaf coherence for multi-agent belief alignment.**

A **cellular sheaf** assigns vector spaces (stalks) to nodes and linear maps
(restriction maps) to edges of a graph. The **sheaf Laplacian** `L_F` measures
how much a section (belief assignment) disagrees across edges:

- `L_F x = 0` ⟹ **global section** (perfect agreement among all agents)
- `||L_F x|| / ||x||` ⟹ **disagreement level**
- `1 - ||L_F x|| / ||x||` ⟹ **alignment score** (0 = chaos, 1 = consensus)

Use this crate to model multi-agent belief systems, measure coherence, find
global sections (consensus states), and detect which agent pairs disagree most.

## Features

- **Cellular sheaf construction** — `CellularSheaf` with presets (`constant`,
  `path`, `cycle`, `complete`) and a builder API for custom graph topologies
  and heterogeneous stalk dimensions
- **Sheaf Laplacian** — `SheafLaplacian` constructs the block matrix `L_F`,
  computes quadratic forms, applies to belief vectors, and finds eigenvalues
  via power iteration
- **Coherence measurement** — `CoherenceMeasure` provides alignment score,
  per-edge disagreement, average/max disagreement, dominant disagreement mode,
  and alignment threshold checks
- **Global sections** — `GlobalSection` finds consensus states (exact or
  approximate) via gradient descent on the sheaf energy functional
- **Agent-based interface** — `AgentSheaf` and `AgentBelief` provide a
  high-level API where each agent has a name, belief vector, and confidence
  score (0–1); construct `complete`, `path`, or custom-edge sheaves
- **Serialization** — full serde support for sheaf structures

## Quick Start

```rust
use sheaf_coherence::{CellularSheaf, CoherenceMeasure};

// Build a complete sheaf on 3 agents with 2D belief spaces
let sheaf = CellularSheaf::complete(3, 2).unwrap();

// All agents agree perfectly → alignment = 1.0
let beliefs = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
let coherence = CoherenceMeasure::from_flat(&sheaf, &beliefs, 100, 1e-10).unwrap();
println!("Alignment: {:.3}", coherence.alignment);
assert!(coherence.is_aligned(0.99));
```

## Agent-Based API

```rust
use sheaf_coherence::{AgentBelief, AgentSheaf};

let agents = vec![
    AgentBelief::new("alice", vec![1.0, 0.0], 0.9),
    AgentBelief::new("bob",   vec![1.0, 0.0], 0.8),
    AgentBelief::new("carol", vec![0.0, 1.0], 0.7),  // disagrees
];

let sheaf = AgentSheaf::complete(agents).unwrap();
let coh = sheaf.coherence(100, 1e-10).unwrap();
println!("Team alignment: {:.1}%", coh.alignment * 100.0);
println!("Max disagreement edge: {:.3}", coh.max_disagreement());
```

## Module Overview

| Module | Description |
|---|---|
| `sheaf` | `CellularSheaf`, `SheafBuilder` — sheaf construction |
| `laplacian` | `SheafLaplacian` — block matrix, quadratic forms, eigenvalues |
| `coherence` | `CoherenceMeasure` — alignment, disagreement metrics |
| `section` | `GlobalSection` — consensus state computation |
| `agent` | `AgentBelief`, `AgentSheaf` — agent-based interface |
| `error` | `SheafError` — error types |

## Links

- [Documentation](https://docs.rs/sheaf-coherence)
- [Repository](https://github.com/nightshift-crates/sheaf-coherence)
- [Crates.io](https://crates.io/crates/sheaf-coherence)

## License

MIT
