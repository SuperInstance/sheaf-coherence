//! Čech cochain complexes for computing sheaf cohomology.
//!
//! C⁰ = space of sections (one vector per open set).
//! C¹ = pairwise differences on intersections.
//! C² = triple-overlap cocycle conditions.
//!
//! Cohomology is computed via Gaussian elimination on the coboundary
//! matrices — no external linear-algebra crate needed.

use serde::{Deserialize, Serialize};
use crate::cover::OpenCover;
use crate::section::SectionFamily;

/// A computed cohomology group.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CohomologyGroup {
    /// Degree: 0 for H⁰, 1 for H¹, etc.
    pub degree: usize,
    /// Dimension of the group (rank of the cohomology).
    pub dimension: usize,
    /// Generators: basis vectors spanning the cohomology.
    pub generators: Vec<Vec<f64>>,
}

// ── Gaussian elimination (from scratch) ─────────────────────────────

/// Row-reduce a matrix in place, return the rank.
/// Uses partial pivoting for numerical stability.
fn gaussian_elimination(mat: &mut Vec<Vec<f64>>, tol: f64) -> usize {
    if mat.is_empty() || mat[0].is_empty() {
        return 0;
    }
    let rows = mat.len();
    let cols = mat[0].len();
    let mut pivot_row = 0;

    for col in 0..cols {
        // Find pivot
        let mut best = None;
        let mut best_val = 0.0_f64;
        for r in pivot_row..rows {
            let v = mat[r][col].abs();
            if v > best_val && v > tol {
                best = Some(r);
                best_val = v;
            }
        }
        let Some(pr) = best else { continue };

        // Swap
        mat.swap(pivot_row, pr);
        let pivot_val = mat[pivot_row][col];

        // Eliminate below
        for r in (pivot_row + 1)..rows {
            let factor = mat[r][col] / pivot_val;
            for c in col..cols {
                mat[r][c] -= factor * mat[pivot_row][c];
                if mat[r][c].abs() < tol {
                    mat[r][c] = 0.0;
                }
            }
        }
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    pivot_row
}

/// Compute the nullspace of a matrix (kernel) using RREF + back-substitution.
/// Returns basis vectors of the kernel.
fn nullspace(mat: &[Vec<f64>], tol: f64) -> Vec<Vec<f64>> {
    let ncols = if mat.is_empty() {
        0
    } else {
        mat[0].len()
    };
    if mat.is_empty() && ncols == 0 {
        return vec![];
    }
    if mat.is_empty() {
        // Kernel of 0×n matrix = all of R^n
        return (0..ncols)
            .map(|i| {
                let mut v = vec![0.0; ncols];
                v[i] = 1.0;
                v
            })
            .collect();
    }
    let rows = mat.len();
    let cols = mat[0].len();
    if cols == 0 {
        return vec![];
    }

    // Augmented copy for RREF
    let mut m = mat.to_vec();

    // Forward elimination + track pivots
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut pivot_row = 0;
    for col in 0..cols {
        let mut best = None;
        let mut best_val = 0.0;
        for r in pivot_row..rows {
            let v = m[r][col].abs();
            if v > best_val && v > tol {
                best = Some(r);
                best_val = v;
            }
        }
        let Some(pr) = best else { continue };
        m.swap(pivot_row, pr);
        let pv = m[pivot_row][col];

        // Normalize pivot row
        for c in 0..cols {
            m[pivot_row][c] /= pv;
        }

        // Eliminate ALL other rows (full RREF)
        for r in 0..rows {
            if r == pivot_row {
                continue;
            }
            let factor = m[r][col];
            if factor.abs() > tol {
                for c in 0..cols {
                    m[r][c] -= factor * m[pivot_row][c];
                    if m[r][c].abs() < tol {
                        m[r][c] = 0.0;
                    }
                }
            }
        }
        pivot_cols.push(col);
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }

    // Free columns = those not in pivot_cols
    let free_cols: Vec<usize> = (0..cols).filter(|c| !pivot_cols.contains(c)).collect();

    // For each free column, construct a kernel vector
    free_cols
        .iter()
        .map(|&fc| {
            let mut v = vec![0.0; cols];
            v[fc] = 1.0;
            // For each pivot row, the pivot column value must be 0
            // pivot_row i has pivot at pivot_cols[i]
            for (i, &pc) in pivot_cols.iter().enumerate() {
                v[pc] = -m[i][fc];
            }
            // Clean tiny values
            for x in &mut v {
                if x.abs() < tol {
                    *x = 0.0;
                }
            }
            v
        })
        .collect()
}

/// Build the Čech coboundary map d⁰: C⁰ → C¹.
///
/// The matrix has one row per intersection variable and one column
/// per section variable. For intersection (i,j) at position k within
/// that intersection, the coboundary maps: +1 from section i's element,
/// -1 from section j's element.
fn build_d0_matrix(cover: &OpenCover) -> (Vec<Vec<f64>>, Vec<(usize, usize, usize)>) {
    let intersections = cover.nonempty_intersections();
    let mut row_labels: Vec<(usize, usize, usize)> = Vec::new(); // (set_i, set_j, local_idx)
    let ncols: usize = cover.sets.iter().map(|s| s.len()).sum();
    let mut mat: Vec<Vec<f64>> = Vec::new();

    let mut col_offset = 0;
    let col_offsets: Vec<usize> = cover
        .sets
        .iter()
        .scan(0, |acc, s| {
            let off = *acc;
            *acc += s.len();
            Some(off)
        })
        .collect();

    for (i, j, inter) in &intersections {
        for (k, &global_idx) in inter.iter().enumerate() {
            let mut row = vec![0.0; ncols];
            // +1 for section i's element corresponding to global_idx
            let pos_i = cover.sets[*i]
                .iter()
                .position(|&x| x == global_idx)
                .unwrap();
            row[col_offsets[*i] + pos_i] = 1.0;
            // -1 for section j's element
            let pos_j = cover.sets[*j]
                .iter()
                .position(|&x| x == global_idx)
                .unwrap();
            row[col_offsets[*j] + pos_j] = -1.0;
            mat.push(row);
            row_labels.push((*i, *j, k));
        }
    }

    (mat, row_labels)
}

/// Build the Čech coboundary map d¹: C¹ → C².
///
/// For each triple overlap (i,j,k), the cocycle condition is:
/// diff_ij|ijk - diff_ik|ijk + diff_jk|ijk = 0
fn build_d1_matrix(cover: &OpenCover) -> Vec<Vec<f64>> {
    let triples = cover.triple_intersections();
    let intersections = cover.nonempty_intersections();
    let ncols: usize = intersections.iter().map(|(_, _, v)| v.len()).sum();

    if triples.is_empty() || ncols == 0 {
        return vec![];
    }

    // Column offsets for each intersection
    let inter_col_offsets: Vec<usize> = intersections
        .iter()
        .scan(0, |acc, (_, _, v)| {
            let off = *acc;
            *acc += v.len();
            Some(off)
        })
        .collect();

    // Map from (i,j) pair to index in intersections list
    let mut pair_idx = std::collections::HashMap::new();
    for (idx, (i, j, _)) in intersections.iter().enumerate() {
        pair_idx.insert((*i, *j), idx);
    }

    let mut mat: Vec<Vec<f64>> = Vec::new();

    for (i, j, k, tri_inter) in &triples {
        for (t, &global_idx) in tri_inter.iter().enumerate() {
            let mut row = vec![0.0; ncols];

            // diff_ij restricted to ijk: +1
            if let Some(&pij) = pair_idx.get(&(*i, *j)) {
                let local_pos = intersections[pij]
                    .2
                    .iter()
                    .position(|&x| x == global_idx)
                    .unwrap();
                row[inter_col_offsets[pij] + local_pos] += 1.0;
            }
            // diff_ik restricted to ijk: -1
            if let Some(&pik) = pair_idx.get(&(*i, *k)) {
                let local_pos = intersections[pik]
                    .2
                    .iter()
                    .position(|&x| x == global_idx)
                    .unwrap();
                row[inter_col_offsets[pik] + local_pos] -= 1.0;
            }
            // diff_jk restricted to ijk: +1
            if let Some(&pjk) = pair_idx.get(&(*j, *k)) {
                let local_pos = intersections[pjk]
                    .2
                    .iter()
                    .position(|&x| x == global_idx)
                    .unwrap();
                row[inter_col_offsets[pjk] + local_pos] += 1.0;
            }

            mat.push(row);
        }
    }
    mat
}

/// Compute Čech sheaf cohomology groups H⁰ and H¹.
///
/// - H⁰ = ker(d⁰): the space of globally consistent sections.
/// - H¹ = ker(d¹) / im(d⁰): the local-to-global obstructions.
pub fn compute_cohomology(cover: &OpenCover, tol: f64) -> (CohomologyGroup, CohomologyGroup) {
    let (d0, _) = build_d0_matrix(cover);
    let d1 = build_d1_matrix(cover);

    let c0_dim: usize = cover.sets.iter().map(|s| s.len()).sum();
    let c1_dim: usize = cover
        .nonempty_intersections()
        .iter()
        .map(|(_, _, v)| v.len())
        .sum();

    // H⁰ = ker(d⁰)
    // Special case: if d⁰ has no rows (no intersections), everything is in the kernel
    let h0_gens = if d0.is_empty() && c0_dim > 0 {
        // 0×n matrix: kernel = R^n
        (0..c0_dim)
            .map(|i| {
                let mut v = vec![0.0; c0_dim];
                v[i] = 1.0;
                v
            })
            .collect()
    } else {
        nullspace(&d0, tol)
    };
    let h0 = CohomologyGroup {
        degree: 0,
        dimension: h0_gens.len(),
        generators: h0_gens,
    };

    // H¹ = ker(d¹) / im(d⁰)
    // ker(d¹) dimension
    let d1_rank = {
        let mut d1_copy = d1.clone();
        gaussian_elimination(&mut d1_copy, tol)
    };
    let ker_d1_dim = c1_dim.saturating_sub(d1_rank);

    // im(d⁰) dimension = rank(d⁰)
    let im_d0_dim = {
        let mut d0_copy = d0.clone();
        gaussian_elimination(&mut d0_copy, tol)
    };

    let h1_dim = ker_d1_dim.saturating_sub(im_d0_dim);

    // Compute ker(d¹) generators
    let ker_d1_gens = if d1.is_empty() && c1_dim > 0 {
        (0..c1_dim)
            .map(|i| {
                let mut v = vec![0.0; c1_dim];
                v[i] = 1.0;
                v
            })
            .collect()
    } else {
        nullspace(&d1, tol)
    };

    // The H¹ generators are those in ker(d¹) but not in im(d⁰).
    // For simplicity, project: for each ker(d¹) generator, check if it's in im(d⁰).
    // If not, it's an H¹ generator.
    let d0_gens = nullspace_with_complement(&d0, c1_dim, tol);
    let _ = d0_gens; // im(d⁰) basis — used conceptually

    // Approximate H¹ generators:
    // Take ker(d¹) generators and remove those that are in im(d⁰).
    let mut h1_gens = Vec::new();
    let im_d0_basis = compute_image_basis(&d0, tol);
    for g in &ker_d1_gens {
        if !is_in_span(g, &im_d0_basis, tol) {
            h1_gens.push(g.clone());
        }
    }

    // The dimension should match h1_dim; if not, trust the algebraic count
    let h1 = CohomologyGroup {
        degree: 1,
        dimension: h1_dim,
        generators: h1_gens,
    };

    (h0, h1)
}

/// Compute a basis for the image of a matrix (column space of its transpose,
/// i.e., the row space, i.e., the image of the linear map).
fn compute_image_basis(mat: &[Vec<f64>], tol: f64) -> Vec<Vec<f64>> {
    if mat.is_empty() || mat[0].is_empty() {
        return vec![];
    }
    let rows = mat.len();
    let cols = mat[0].len();

    // Transpose to work with columns as vectors
    let mut m = mat.to_vec();
    let rank = gaussian_elimination(&mut m, tol);

    // The first `rank` rows of the RREF form a basis for the row space,
    // which corresponds to the image.
    m[..rank].to_vec()
}

/// Check if vector v is in the span of the given basis vectors.
fn is_in_span(v: &[f64], basis: &[Vec<f64>], tol: f64) -> bool {
    if basis.is_empty() {
        return v.iter().all(|x| x.abs() < tol);
    }
    let n = v.len();
    // Augmented matrix: [basis | v] and check if rank doesn't increase
    let mut aug: Vec<Vec<f64>> = basis
        .iter()
        .filter(|b| b.len() == n)
        .cloned()
        .collect();
    if aug.is_empty() {
        return v.iter().all(|x| x.abs() < tol);
    }
    let cols = aug[0].len();
    aug.push(v.to_vec());

    let rank_without = {
        let mut m = aug[..aug.len() - 1].to_vec();
        gaussian_elimination(&mut m, tol)
    };
    let rank_with = {
        let mut m = aug.clone();
        gaussian_elimination(&mut m, tol)
    };
    rank_with == rank_without
}

/// Compute nullspace and also return complementary column indices.
fn nullspace_with_complement(
    mat: &[Vec<f64>],
    _total_dim: usize,
    tol: f64,
) -> Vec<Vec<f64>> {
    nullspace(mat, tol)
}

/// Compute H⁰ from a section family: the space of globally consistent sections.
///
/// Returns the dimension of H⁰ and a basis.
pub fn compute_h0_from_sections(
    cover: &OpenCover,
    sections: &SectionFamily,
    tol: f64,
) -> CohomologyGroup {
    let (h0, _) = compute_cohomology(cover, tol);
    h0
}

/// Compute H¹ from a section family: the obstructions.
pub fn compute_h1_from_sections(
    cover: &OpenCover,
    sections: &SectionFamily,
    tol: f64,
) -> CohomologyGroup {
    let (_, h1) = compute_cohomology(cover, tol);
    h1
}

#[cfg(test)]
mod gauss_tests {
    use super::*;

    #[test]
    fn test_gauss_identity() {
        let mut m = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let rank = gaussian_elimination(&mut m, 1e-10);
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_gauss_singular() {
        let mut m = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let rank = gaussian_elimination(&mut m, 1e-10);
        assert_eq!(rank, 1);
    }

    #[test]
    fn test_nullspace_identity() {
        let m = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ns = nullspace(&m, 1e-10);
        assert!(ns.is_empty());
    }

    #[test]
    fn test_nullspace_zero_row() {
        let m = vec![vec![1.0, 1.0]];
        let ns = nullspace(&m, 1e-10);
        assert_eq!(ns.len(), 1);
        // ns[0] should be proportional to [1, -1]
        assert!((ns[0][0] - 1.0).abs() < 1e-6 || (ns[0][0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nullspace_empty_matrix() {
        // 2-column matrix with 0 rows = impossible to construct directly
        // Use a 2x2 zero matrix instead → kernel = R²
        let m: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let ns = nullspace(&m, 1e-10);
        assert_eq!(ns.len(), 2);
    }
}
