//! Persistent sheaf cohomology over varying resolution thresholds.
//!
//! As the cover gets finer (more sets, smaller overlaps), cohomology
//! groups can change. This module tracks H⁰ and H¹ across a sequence
//! of covers at different resolutions, producing a persistence diagram.

use serde::{Deserialize, Serialize};
use crate::cover::OpenCover;
use crate::section::SectionFamily;
use crate::cochain::compute_cohomology;

/// A point in the persistence diagram: (birth_resolution, death_resolution, degree).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PersistencePoint {
    /// Resolution at which this feature first appears.
    pub birth: f64,
    /// Resolution at which this feature dies (f64::INFINITY if still alive).
    pub death: f64,
    /// Cohomology degree (0 or 1).
    pub degree: usize,
}

/// A persistence diagram capturing the lifetime of cohomological features.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PersistenceDiagram {
    pub points: Vec<PersistencePoint>,
}

/// Snapshot of cohomology at a specific resolution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CohomologySnapshot {
    pub resolution: f64,
    pub h0_dimension: usize,
    pub h1_dimension: usize,
}

/// Compute persistent cohomology across a sequence of (resolution, cover, sections) tuples.
///
/// The `stages` slice must be sorted by increasing resolution.
pub fn persistent_cohomology(
    stages: &[(f64, OpenCover, SectionFamily)],
    tol: f64,
) -> (Vec<CohomologySnapshot>, PersistenceDiagram) {
    let mut snapshots = Vec::new();
    let mut prev_h0 = None;
    let mut prev_h1 = None;
    let mut h0_births: Vec<f64> = Vec::new();
    let mut h1_births: Vec<f64> = Vec::new();
    let mut points = Vec::new();

    for (resolution, cover, _sections) in stages {
        let (h0, h1) = compute_cohomology(cover, tol);

        snapshots.push(CohomologySnapshot {
            resolution: *resolution,
            h0_dimension: h0.dimension,
            h1_dimension: h1.dimension,
        });

        // Track H⁰ features
        if let Some(prev) = prev_h0 {
            if h0.dimension > prev {
                // New H⁰ features born
                let n_new = h0.dimension - prev;
                for _ in 0..n_new {
                    h0_births.push(*resolution);
                }
            } else if h0.dimension < prev {
                // Some H⁰ features died
                let n_died = prev - h0.dimension;
                for _ in 0..n_died {
                    if let Some(birth) = h0_births.pop() {
                        points.push(PersistencePoint {
                            birth,
                            death: *resolution,
                            degree: 0,
                        });
                    }
                }
            }
        } else {
            // First stage — all features are born
            for _ in 0..h0.dimension {
                h0_births.push(*resolution);
            }
        }

        // Track H¹ features
        if let Some(prev) = prev_h1 {
            if h1.dimension > prev {
                let n_new = h1.dimension - prev;
                for _ in 0..n_new {
                    h1_births.push(*resolution);
                }
            } else if h1.dimension < prev {
                let n_died = prev - h1.dimension;
                for _ in 0..n_died {
                    if let Some(birth) = h1_births.pop() {
                        points.push(PersistencePoint {
                            birth,
                            death: *resolution,
                            degree: 1,
                        });
                    }
                }
            }
        } else {
            for _ in 0..h1.dimension {
                h1_births.push(*resolution);
            }
        }

        prev_h0 = Some(h0.dimension);
        prev_h1 = Some(h1.dimension);
    }

    // Features still alive at the end
    if let Some(&last_res) = stages.last().map(|(r, _, _)| r) {
        for birth in h0_births {
            points.push(PersistencePoint {
                birth,
                death: f64::INFINITY,
                degree: 0,
            });
        }
        for birth in h1_births {
            points.push(PersistencePoint {
                birth,
                death: f64::INFINITY,
                degree: 1,
            });
        }
    }

    (snapshots, PersistenceDiagram { points })
}

/// Refine a cover by splitting each set into smaller pieces.
///
/// `resolution` controls how many pieces each set is split into.
/// At resolution 1, the cover is unchanged. Higher values produce finer covers.
pub fn refine_cover(cover: &OpenCover, resolution: usize) -> OpenCover {
    if resolution <= 1 {
        return cover.clone();
    }
    let mut new_sets = Vec::new();
    for set in &cover.sets {
        if set.len() <= resolution {
            new_sets.push(set.clone());
        } else {
            let chunk_size = (set.len() + resolution - 1) / resolution;
            for chunk in set.chunks(chunk_size) {
                new_sets.push(chunk.to_vec());
            }
        }
    }
    OpenCover::new(new_sets)
}

/// Build a sequence of refined covers at different resolutions.
pub fn refinement_sequence(cover: &OpenCover, max_resolution: usize) -> Vec<OpenCover> {
    (1..=max_resolution)
        .map(|r| refine_cover(cover, r))
        .collect()
}
