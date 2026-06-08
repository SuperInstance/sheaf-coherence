//! Detection and classification of local-to-global obstruction classes.
//!
//! A nonzero H¹ dimension means the fleet cannot reconcile its local views
//! into a single global section. This module identifies which agent pairs
//! contribute to the obstruction.

use crate::cochain::compute_cohomology;
use crate::cover::OpenCover;
use crate::section::SectionFamily;
use serde::{Deserialize, Serialize};

/// An obstruction class describing why local sections cannot be glued globally.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObstructionClass {
    /// Dimension of H¹ (number of independent obstructions).
    pub h1_dimension: usize,
    /// Agent pairs whose beliefs contradict on their overlap.
    pub conflicting_agents: Vec<(usize, usize)>,
}

impl ObstructionClass {
    /// Detect obstructions from a section family and cover.
    pub fn detect(cover: &OpenCover, sections: &SectionFamily, tol: f64) -> Self {
        let (_h0, h1) = compute_cohomology(cover, tol);

        // Find which agent pairs actually disagree
        let diffs = sections.pairwise_differences(cover);
        let conflicting: Vec<(usize, usize)> = diffs
            .iter()
            .filter(|(_, _, d)| d.iter().any(|x| x.abs() > tol))
            .map(|(i, j, _)| (*i, *j))
            .collect();

        Self {
            h1_dimension: h1.dimension,
            conflicting_agents: conflicting,
        }
    }

    /// Whether the fleet is globally consistent (H¹ = 0, no conflicts).
    pub fn is_consistent(&self) -> bool {
        self.h1_dimension == 0 && self.conflicting_agents.is_empty()
    }

    /// Classify the type of obstruction.
    pub fn classify(&self) -> ObstructionType {
        if self.h1_dimension == 0 {
            ObstructionType::None
        } else if self.conflicting_agents.len() == 1 {
            ObstructionType::Pairwise(self.conflicting_agents[0])
        } else if self.conflicting_agents.len() > 1 {
            ObstructionType::MultiAgent(self.conflicting_agents.clone())
        } else {
            ObstructionType::Hidden(self.h1_dimension)
        }
    }
}

/// Classification of the obstruction type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ObstructionType {
    /// No obstruction — fleet is consistent.
    None,
    /// A single pair of agents contradicts.
    Pairwise((usize, usize)),
    /// Multiple pairs contradict.
    MultiAgent(Vec<(usize, usize)>),
    /// Obstruction exists but no pairwise conflict detected (higher-order).
    Hidden(usize),
}
