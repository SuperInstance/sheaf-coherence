use crate::error::SheafError;
use crate::sheaf::CellularSheaf;

/// The sheaf Laplacian matrix.
///
/// For a sheaf `F` on graph `G` with incidence matrix `B`,
/// the sheaf Laplacian is `L_F = B^T diag(F_e^T F_e) B`
/// where `F_e` is the restriction map for edge `e`.
///
/// In expanded form, for each edge `(i,j)` with restriction map `F_{ij}`:
///   L_F[i,i] += F_{ij}^T F_{ij}
///   L_F[j,j] += F_{ij}^T F_{ij}
///   L_F[i,j] -= F_{ij}^T F_{ij}
///   L_F[j,i] -= F_{ij}^T F_{ij}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SheafLaplacian {
    /// The sheaf Laplacian as a dense matrix of size (total_dim × total_dim).
    pub matrix: Vec<Vec<f64>>,
    /// Total dimension (sum of all stalk dimensions).
    pub n: usize,
}

impl SheafLaplacian {
    /// Build the sheaf Laplacian from a cellular sheaf.
    pub fn from_sheaf(sheaf: &CellularSheaf) -> Result<Self, SheafError> {
        sheaf.validate()?;
        let n = sheaf.total_dim();
        let mut matrix = vec![vec![0.0; n]; n];

        // Build offset map: node i starts at row/col offsets[i]
        let mut offsets = Vec::with_capacity(sheaf.node_count());
        let mut off = 0;
        for &d in &sheaf.stalk_dims {
            offsets.push(off);
            off += d;
        }

        for (src, tgt, f_map) in &sheaf.restriction_maps {
            let i = *src;
            let j = *tgt;
            // F_{ij}^T F_{ij} (matrices are stored as row-major vec<vec<f64>>)
            let ft_f = mat_mul_transpose(f_map);

            // Add to diagonal blocks
            add_to_block(&mut matrix, offsets[i], offsets[i], &ft_f);
            add_to_block(&mut matrix, offsets[j], offsets[j], &ft_f);

            // Subtract from off-diagonal blocks
            sub_from_block(&mut matrix, offsets[i], offsets[j], &ft_f);
            sub_from_block(&mut matrix, offsets[j], offsets[i], &ft_f);
        }

        Ok(Self { matrix, n })
    }

    /// Apply the Laplacian to a flattened vector (length = total_dim).
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        mat_vec(&self.matrix, x)
    }

    /// Compute the quadratic form x^T L_F x (measures total disagreement).
    pub fn quadratic_form(&self, x: &[f64]) -> f64 {
        let lx = self.apply(x);
        dot(x, &lx)
    }

    /// Compute the norm of L_F x (residual).
    pub fn residual_norm(&self, x: &[f64]) -> f64 {
        let lx = self.apply(x);
        norm(&lx)
    }

    /// Power iteration to estimate the smallest eigenvalue.
    /// Returns (eigenvalue, eigenvector) for the smallest eigenvalue.
    pub fn smallest_eigenvalue(&self, max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
        // Inverse power iteration with shift: we approximate by computing
        // the eigenvalue decomposition via repeated application.
        // For small matrices, we use a direct approach.
        let n = self.n;
        if n == 0 {
            return (0.0, vec![]);
        }

        // Use power iteration on (L_F - shift*I) to find largest eigenvalue,
        // then shift back. Start with finding the largest eigenvalue first.
        let largest = self.power_iteration(max_iter, tol);
        let shift = largest.0;

        // Now do inverse iteration: (L_F - shift*I)^-1 x
        // We approximate by doing shifted power iteration
        let mut v: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sqrt()).collect();
        let v_norm = norm(&v);
        for x in &mut v { *x /= v_norm; }
        for _ in 0..max_iter {
            // Apply (L - shift*I)
            let mut lv = self.apply(&v);
            for (lvi, vi) in lv.iter_mut().zip(&v) {
                *lvi -= shift * vi;
            }
            // We want the smallest magnitude, so we negate
            for val in lv.iter_mut() {
                *val = -*val;
            }
            // Add shift back to make it positive semi-definite
            let lv_norm = norm(&lv);
            if lv_norm < tol {
                break;
            }
            for (vi, lvi) in v.iter_mut().zip(&lv) {
                *vi = lvi / lv_norm;
            }
        }

        // Compute Rayleigh quotient
        let lx = self.apply(&v);
        let eigenvalue = dot(&v, &lx) / dot(&v, &v).max(1e-15);

        (eigenvalue, v)
    }

    /// Standard power iteration to find the largest eigenvalue and eigenvector.
    pub fn power_iteration(&self, max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
        let n = self.n;
        if n == 0 {
            return (0.0, vec![]);
        }
        // Use a non-uniform seed to avoid landing in the null space
        let mut v: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sqrt()).collect();
        let v_norm = norm(&v);
        for x in &mut v { *x /= v_norm; }
        let mut eigenvalue = 0.0;

        for _ in 0..max_iter {
            let lv = self.apply(&v);
            let new_eigenvalue = dot(&v, &lv);
            let lv_norm = norm(&lv);
            if lv_norm < tol {
                break;
            }
            for i in 0..n {
                v[i] = lv[i] / lv_norm;
            }
            if (new_eigenvalue - eigenvalue).abs() < tol {
                eigenvalue = new_eigenvalue;
                break;
            }
            eigenvalue = new_eigenvalue;
        }

        (eigenvalue, v)
    }

    /// Compute all eigenvalues using QR-like iteration (for small matrices).
    /// Returns eigenvalues sorted in ascending order.
    pub fn eigenvalues(&self, max_iter: usize) -> Vec<f64> {
        let n = self.n;
        if n == 0 {
            return vec![];
        }
        // Simple approach: compute characteristic polynomial roots not practical.
        // Use iterative deflation with power iteration.
        let mut eigenvalues = Vec::new();
        let mut deflated = self.matrix.clone();

        for _ in 0..n {
            let lap = SheafLaplacian { matrix: deflated.clone(), n };
            let (ev, v) = lap.power_iteration(max_iter, 1e-10);
            eigenvalues.push(ev);
            // Deflate: remove component along v
            let vv = dot(&v, &v);
            if vv > 1e-15 {
                for i in 0..n {
                    for j in 0..n {
                        deflated[i][j] -= ev * v[i] * v[j] / vv;
                    }
                }
            }
        }

        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        eigenvalues
    }
}

/// Compute F^T * F for a matrix F stored as vec of rows.
fn mat_mul_transpose(f: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = f.len();
    let cols = if rows == 0 { 0 } else { f[0].len() };
    // Result is cols × cols
    let mut result = vec![vec![0.0; cols]; cols];
    for i in 0..cols {
        for j in 0..cols {
            let sum: f64 = f.iter().map(|row| row[i] * row[j]).sum();
            result[i][j] = sum;
        }
    }
    result
}

fn add_to_block(mat: &mut [Vec<f64>], r: usize, c: usize, block: &[Vec<f64>]) {
    for (di, row) in block.iter().enumerate() {
        for (dj, &val) in row.iter().enumerate() {
            mat[r + di][c + dj] += val;
        }
    }
}

fn sub_from_block(mat: &mut [Vec<f64>], r: usize, c: usize, block: &[Vec<f64>]) {
    for (di, row) in block.iter().enumerate() {
        for (dj, &val) in row.iter().enumerate() {
            mat[r + di][c + dj] -= val;
        }
    }
}

fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}
