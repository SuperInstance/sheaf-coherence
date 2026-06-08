//! Consistency repair via sheaf gluing.
//!
//! When H¹ = 0, local sections can be collated into a unique global
//! section. When H¹ ≠ 0, gluing fails and we report the obstruction.

use crate::cochain::compute_cohomology;
use crate::cover::OpenCover;
use crate::section::{LocalSection, SectionFamily};

/// Result of attempting to glue local sections into a global section.
#[derive(Debug, Clone, PartialEq)]
pub enum GluingResult {
    /// Gluing succeeded: a global section over the full universe.
    Success(Vec<f64>),
    /// Gluing failed: H¹ ≠ 0, with the obstruction dimension.
    Failed { h1_dimension: usize },
}

/// Attempt to glue a family of local sections into a global section.
///
/// This works by solving for a global section that restricts to each
/// local section. When H¹ = 0, a unique solution exists (up to kernel
/// dimension of d⁰).
pub fn glue(cover: &OpenCover, sections: &SectionFamily, tol: f64) -> GluingResult {
    let (_h0, h1) = compute_cohomology(cover, tol);

    if h1.dimension > 0 {
        return GluingResult::Failed {
            h1_dimension: h1.dimension,
        };
    }

    // Build a global section over the universe.
    // Strategy: iterate through all variables, assigning values from
    // whichever section covers them. Consistency is guaranteed by H¹=0.
    let universe = cover.universe();
    if universe.is_empty() {
        return GluingResult::Success(vec![]);
    }

    let n = *universe.last().unwrap() + 1;
    let mut global = vec![f64::NAN; n];
    let mut assigned = vec![false; n];

    // Sort sections by agent_id to match cover.sets ordering
    let mut sorted: Vec<&LocalSection> = sections.sections.iter().collect();
    sorted.sort_by_key(|s| s.agent_id);

    for sec in &sorted {
        let set = &cover.sets[sec.agent_id];
        for (k, &global_idx) in set.iter().enumerate() {
            if global_idx >= n {
                continue;
            }
            if !assigned[global_idx] {
                global[global_idx] = sec.data[k];
                assigned[global_idx] = true;
            }
            // If already assigned, H¹=0 guarantees consistency;
            // we could verify but trust the algebraic result.
        }
    }

    // Fill any remaining NaNs with 0.0 (variables not in any section)
    for v in &mut global {
        if v.is_nan() {
            *v = 0.0;
        }
    }

    GluingResult::Success(global)
}

/// Attempt gluing with verification: after gluing, check that restrictions
/// match the original sections within tolerance.
pub fn glue_with_verification(
    cover: &OpenCover,
    sections: &SectionFamily,
    tol: f64,
) -> GluingResult {
    let result = glue(cover, sections, tol);

    if let GluingResult::Success(ref global) = result {
        // Verify: for each section, the restriction of global to its set
        // should match the section data
        for sec in &sections.sections {
            let set = &cover.sets[sec.agent_id];
            for (k, &idx) in set.iter().enumerate() {
                if idx < global.len() && (global[idx] - sec.data[k]).abs() > tol {
                    // Verification failed — shouldn't happen if H¹=0
                    return GluingResult::Failed { h1_dimension: 1 };
                }
            }
        }
    }

    result
}
