//! Local sections: per-agent belief vectors with restriction maps.
//!
//! A local section is an agent's view of the shared state — a vector of
//! values for the variables in its open set. Consistency demands that
//! whenever two agents' views overlap, they agree on the overlap.

use crate::cover::{OpenCover, RestrictionMap};
use serde::{Deserialize, Serialize};

/// A local section: one agent's belief vector over its open set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalSection {
    /// Which agent / open set this section belongs to.
    pub agent_id: usize,
    /// The belief values, one per variable in `cover.sets[agent_id]`.
    pub data: Vec<f64>,
}

impl LocalSection {
    /// Create a new local section.
    pub fn new(agent_id: usize, data: Vec<f64>) -> Self {
        Self { agent_id, data }
    }

    /// Restrict this section to an intersection with another set.
    ///
    /// Returns the sub-vector over the intersection indices.
    pub fn restrict(&self, cover: &OpenCover, intersection: &[usize]) -> Vec<f64> {
        let rmap = RestrictionMap::from_cover(cover, self.agent_id, intersection);
        rmap.apply(&self.data)
    }

    /// Check if this section agrees with another over their intersection.
    ///
    /// Two sections agree if their restricted values match within `tol`.
    pub fn agrees_with(&self, other: &LocalSection, cover: &OpenCover, tol: f64) -> bool {
        let inter = cover.intersection(self.agent_id, other.agent_id);
        if inter.is_empty() {
            return true; // No overlap → vacuously consistent
        }
        let a = self.restrict(cover, &inter);
        let b = other.restrict(cover, &inter);
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    /// Dimension (number of variables in this section's domain).
    pub fn dim(&self) -> usize {
        self.data.len()
    }
}

/// A collection of local sections forming a presheaf.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SectionFamily {
    pub sections: Vec<LocalSection>,
}

impl SectionFamily {
    pub fn new(sections: Vec<LocalSection>) -> Self {
        Self { sections }
    }

    /// Get section by agent_id.
    pub fn get(&self, agent_id: usize) -> Option<&LocalSection> {
        self.sections.iter().find(|s| s.agent_id == agent_id)
    }

    /// Check pairwise consistency of all sections against a cover.
    pub fn is_consistent(&self, cover: &OpenCover, tol: f64) -> bool {
        for i in 0..self.sections.len() {
            for j in (i + 1)..self.sections.len() {
                if !self.sections[i].agrees_with(&self.sections[j], cover, tol) {
                    return false;
                }
            }
        }
        true
    }

    /// Compute pairwise differences on all non-empty intersections.
    ///
    /// Returns `diffs[k]` = (i, j, difference_vector) for the k-th
    /// non-empty intersection between sets i and j, where
    /// difference_vector = restrict(section_i) - restrict(section_j).
    pub fn pairwise_differences(&self, cover: &OpenCover) -> Vec<(usize, usize, Vec<f64>)> {
        let intersections = cover.nonempty_intersections();
        let mut diffs = Vec::new();
        for (i, j, inter) in &intersections {
            let si = self.get(*i);
            let sj = self.get(*j);
            if let (Some(si), Some(sj)) = (si, sj) {
                let ri = si.restrict(cover, inter);
                let rj = sj.restrict(cover, inter);
                let diff: Vec<f64> = ri.iter().zip(rj.iter()).map(|(a, b)| a - b).collect();
                diffs.push((*i, *j, diff));
            }
        }
        diffs
    }

    /// Number of sections.
    pub fn len(&self) -> usize {
        self.sections.len()
    }

    /// Whether the family is empty.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }
}
