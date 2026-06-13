//! # sheaf-coherence
//!
//! Sheaf-theoretic multi-agent knowledge consistency.
//!
//! Each agent maintains a local section over an open cover of a shared
//! information space. The library computes sheaf cohomology groups H⁰
//! (global consistency) and H¹ (local-to-global obstructions) to detect
//! and repair contradictions across a fleet of agents.

pub mod cover;
pub mod section;
pub mod cochain;
pub mod obstruction;
pub mod gluing;
pub mod persistence;

pub use cover::{OpenCover, RestrictionMap};
pub use section::LocalSection;
pub use cochain::CohomologyGroup;
pub use obstruction::ObstructionClass;
