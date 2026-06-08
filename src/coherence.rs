use crate::error::SheafError;
use crate::laplacian::SheafLaplacian;
use crate::sheaf::CellularSheaf;

/// Measures how well a belief assignment aligns across the sheaf.
///
/// Alignment = 1 - ||L_F x|| / ||x|| (higher = more coherent).
/// Per-edge disagreement = ||F_{ij} x_i - F_{ij} x_j|| where F_{ij} is the restriction map.
/// Dominant mode = eigenvector of the smallest nonzero eigenvalue (most coherent non-trivial pattern).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoherenceMeasure {
    /// Overall alignment score: 1 - ||L_F x|| / ||x||.
    pub alignment: f64,
    /// Per-edge disagreement values.
    pub disagreement: Vec<f64>,
    /// Eigenvector of smallest nonzero eigenvalue.
    pub dominant_mode: Vec<f64>,
}

impl CoherenceMeasure {
    /// Compute coherence measures for a flat belief vector.
    pub fn from_flat(sheaf: &CellularSheaf, x: &[f64], max_iter: usize, tol: f64) -> Result<Self, SheafError> {
        sheaf.validate()?;
        let total_dim: usize = sheaf.stalk_dims.iter().sum();
        if x.len() != total_dim {
            return Err(SheafError::BeliefDimensionMismatch {
                agent: "global".into(),
                expected: total_dim,
                got: x.len(),
            });
        }

        let lap = SheafLaplacian::from_sheaf(sheaf)?;

        // Alignment
        let x_norm = norm(x);
        let lx_norm = lap.residual_norm(x);
        let alignment = if x_norm > 1e-15 {
            1.0 - lx_norm / x_norm
        } else {
            1.0
        };

        // Per-edge disagreement
        let mut offsets = Vec::with_capacity(sheaf.node_count());
        let mut off = 0;
        for &d in &sheaf.stalk_dims {
            offsets.push(off);
            off += d;
        }

        let mut disagreement = Vec::new();
        for (src, tgt, f_map) in &sheaf.restriction_maps {
            let x_src = &x[offsets[*src]..offsets[*src] + sheaf.stalk_dims[*src]];
            let x_tgt = &x[offsets[*tgt]..offsets[*tgt] + sheaf.stalk_dims[*tgt]];
            // disagreement = ||F x_src - F x_tgt|| (for undirected, both directions)
            let f_src = mat_vec(f_map, x_src);
            let f_tgt = mat_vec(f_map, x_tgt);
            let diff: Vec<f64> = f_src.iter().zip(&f_tgt).map(|(a, b)| a - b).collect();
            disagreement.push(norm(&diff));
        }

        // Dominant mode: eigenvector of smallest eigenvalue
        let (_, dominant_mode) = lap.smallest_eigenvalue(max_iter, tol);

        Ok(Self {
            alignment: alignment.clamp(0.0, 1.0),
            disagreement,
            dominant_mode,
        })
    }

    /// Compute coherence from per-node values.
    pub fn from_values(sheaf: &CellularSheaf, values: &[Vec<f64>], max_iter: usize, tol: f64) -> Result<Self, SheafError> {
        let flat: Vec<f64> = values.iter().flatten().copied().collect();
        Self::from_flat(sheaf, &flat, max_iter, tol)
    }

    /// True if alignment is above a threshold.
    pub fn is_aligned(&self, threshold: f64) -> bool {
        self.alignment >= threshold
    }

    /// Average per-edge disagreement.
    pub fn avg_disagreement(&self) -> f64 {
        if self.disagreement.is_empty() {
            return 0.0;
        }
        self.disagreement.iter().sum::<f64>() / self.disagreement.len() as f64
    }

    /// Maximum per-edge disagreement.
    pub fn max_disagreement(&self) -> f64 {
        self.disagreement.iter().cloned().fold(0.0_f64, f64::max)
    }
}

fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
