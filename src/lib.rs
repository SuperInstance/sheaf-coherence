//! # sheaf-coherence
//!
//! Sheaf-theoretic multi-agent knowledge consistency.
//!
//! Each agent maintains a local section over an open cover of a shared
//! information space. The library computes sheaf cohomology groups H⁰
//! (global consistency) and H¹ (local-to-global obstructions) to detect
//! and repair contradictions across a fleet of agents.

// Pre-existing numeric code triggers several clippy pedantic lints.
#![allow(
    unused_variables,
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::type_complexity,
    clippy::ptr_arg
)]

pub mod cochain;
pub mod cover;
pub mod gluing;
pub mod obstruction;
pub mod persistence;
pub mod section;

pub use cochain::CohomologyGroup;
pub use cover::{OpenCover, RestrictionMap};
pub use obstruction::ObstructionClass;
pub use section::LocalSection;
