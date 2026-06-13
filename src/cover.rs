//! Open covers over agent knowledge domains with intersection lattices.
//!
//! An `OpenCover` partitions the global information space into subsets
//! ("open sets"), each representing what one agent can see. Intersections
//! capture overlapping knowledge between agents.

use serde::{Deserialize, Serialize};

/// An open cover of the global state space.
///
/// Each entry in `sets` is a sorted vector of state-variable indices
/// that one agent can observe. The collection must cover every index
/// at least once.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenCover {
    /// Open sets: `sets[i]` = sorted list of variable indices visible to agent i.
    pub sets: Vec<Vec<usize>>,
}

impl OpenCover {
    /// Create a new open cover from a list of index sets.
    pub fn new(sets: Vec<Vec<usize>>) -> Self {
        let mut sets: Vec<Vec<usize>> = sets
            .into_iter()
            .map(|mut s| {
                s.sort_unstable();
                s.dedup();
                s
            })
            .collect();
        // Ensure at least one set exists
        if sets.is_empty() {
            sets.push(vec![]);
        }
        Self { sets }
    }

    /// Trivial cover: a single set containing all indices `0..n`.
    pub fn trivial(n: usize) -> Self {
        Self {
            sets: vec![(0..n).collect()],
        }
    }

    /// Number of open sets (agents).
    pub fn num_sets(&self) -> usize {
        self.sets.len()
    }

    /// Total number of state variables covered.
    pub fn universe_size(&self) -> usize {
        let mut all: Vec<usize> = self.sets.iter().flatten().copied().collect();
        all.sort_unstable();
        all.dedup();
        all.len()
    }

    /// Universe: sorted, deduplicated list of all indices.
    pub fn universe(&self) -> Vec<usize> {
        let mut all: Vec<usize> = self.sets.iter().flatten().copied().collect();
        all.sort_unstable();
        all.dedup();
        all
    }

    /// Pairwise intersection of sets i and j.
    pub fn intersection(&self, i: usize, j: usize) -> Vec<usize> {
        let a = &self.sets[i];
        let b = &self.sets[j];
        let mut result = Vec::new();
        let (mut ai, mut bi) = (0, 0);
        while ai < a.len() && bi < b.len() {
            if a[ai] == b[bi] {
                result.push(a[ai]);
                ai += 1;
                bi += 1;
            } else if a[ai] < b[bi] {
                ai += 1;
            } else {
                bi += 1;
            }
        }
        result
    }

    /// All pairwise intersections (i < j).
    pub fn pairwise_intersections(&self) -> Vec<(usize, usize, Vec<usize>)> {
        let n = self.num_sets();
        let mut out = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let inter = self.intersection(i, j);
                out.push((i, j, inter));
            }
        }
        out
    }

    /// Non-empty pairwise intersections only.
    pub fn nonempty_intersections(&self) -> Vec<(usize, usize, Vec<usize>)> {
        self.pairwise_intersections()
            .into_iter()
            .filter(|(_, _, v)| !v.is_empty())
            .collect()
    }

    /// Triple intersection of sets i, j, k.
    pub fn triple_intersection(&self, i: usize, j: usize, k: usize) -> Vec<usize> {
        let ab = self.intersection(i, j);
        let a = &ab;
        let b = &self.sets[k];
        let mut result = Vec::new();
        let (mut ai, mut bi) = (0, 0);
        while ai < a.len() && bi < b.len() {
            if a[ai] == b[bi] {
                result.push(a[ai]);
                ai += 1;
                bi += 1;
            } else if a[ai] < b[bi] {
                ai += 1;
            } else {
                bi += 1;
            }
        }
        result
    }

    /// All triple intersections (i < j < k) that are non-empty.
    pub fn triple_intersections(&self) -> Vec<(usize, usize, usize, Vec<usize>)> {
        let n = self.num_sets();
        let mut out = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let inter = self.triple_intersection(i, j, k);
                    if !inter.is_empty() {
                        out.push((i, j, k, inter));
                    }
                }
            }
        }
        out
    }

    /// Check if the cover is a valid cover (every index appears in at least one set).
    pub fn is_valid_cover(&self) -> bool {
        let universe = self.universe();
        if universe.is_empty() && self.sets.iter().all(|s| s.is_empty()) {
            return true;
        }
        // Check no gaps: universe should be 0..max+1
        if universe.is_empty() {
            return true;
        }
        let max = *universe.last().unwrap();
        universe.len() == max + 1
    }
}

/// A restriction map from one open set to a subset (intersection).
///
/// Describes which indices of the parent section map to which indices
/// of the sub-section. `mapping[k]` means: element k of the intersection
/// comes from element `mapping[k]` of the parent set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RestrictionMap {
    /// Index of the source open set.
    pub from_set: usize,
    /// The target intersection (sorted indices in global space).
    pub to_intersection: Vec<usize>,
    /// mapping[k] = position in from_set's data vector for the k-th intersection element.
    pub mapping: Vec<usize>,
}

impl RestrictionMap {
    /// Build a restriction map from set `from_idx` to the given intersection.
    pub fn from_cover(cover: &OpenCover, from_idx: usize, intersection: &[usize]) -> Self {
        let source = &cover.sets[from_idx];
        let mapping: Vec<usize> = intersection
            .iter()
            .map(|&idx| {
                source
                    .iter()
                    .position(|&x| x == idx)
                    .expect("intersection element must be in source set")
            })
            .collect();
        Self {
            from_set: from_idx,
            to_intersection: intersection.to_vec(),
            mapping,
        }
    }

    /// Apply this restriction map to a source vector, producing the sub-vector.
    pub fn apply(&self, data: &[f64]) -> Vec<f64> {
        self.mapping.iter().map(|&i| data[i]).collect()
    }
}
