use crate::error::SheafError;
use crate::laplacian::SheafLaplacian;
use crate::sheaf::CellularSheaf;

/// A (possibly approximate) global section of a cellular sheaf.
///
/// A global section assigns a vector to each stalk such that
/// restriction maps are satisfied on every edge. Equivalently,
/// it lies in the kernel of the sheaf Laplacian.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GlobalSection {
    /// Stalk values at each node.
    pub values: Vec<Vec<f64>>,
    /// True if this is an exact global section (residual ≈ 0).
    pub is_exact: bool,
    /// The residual ||L_F x||.
    pub residual: f64,
}

impl GlobalSection {
    /// Build a section from per-node values and check exactness.
    pub fn new(sheaf: &CellularSheaf, values: Vec<Vec<f64>>, tol: f64) -> Result<Self, SheafError> {
        sheaf.validate()?;
        if values.len() != sheaf.node_count() {
            return Err(SheafError::InvalidNode(values.len()));
        }
        for (i, v) in values.iter().enumerate() {
            if v.len() != sheaf.stalk_dims[i] {
                return Err(SheafError::BeliefDimensionMismatch {
                    agent: format!("node_{i}"),
                    expected: sheaf.stalk_dims[i],
                    got: v.len(),
                });
            }
        }

        let lap = SheafLaplacian::from_sheaf(sheaf)?;
        let flat = values.iter().flatten().copied().collect::<Vec<_>>();
        let residual = lap.residual_norm(&flat);
        let is_exact = residual < tol;

        Ok(Self {
            values,
            is_exact,
            residual,
        })
    }

    /// Try to find a nontrivial global section by solving L_F x = 0
    /// using inverse iteration toward the smallest eigenvalue.
    pub fn find(sheaf: &CellularSheaf, max_iter: usize, tol: f64) -> Result<Self, SheafError> {
        sheaf.validate()?;
        let lap = SheafLaplacian::from_sheaf(sheaf)?;

        let (eigenvalue, flat) = lap.smallest_eigenvalue(max_iter, tol);

        // Split flat vector back into per-stalk values
        let mut values = Vec::new();
        let mut offset = 0;
        for &d in &sheaf.stalk_dims {
            values.push(flat[offset..offset + d].to_vec());
            offset += d;
        }

        let residual = lap.residual_norm(&flat);
        let is_exact = eigenvalue.abs() < tol;

        Ok(Self {
            values,
            is_exact,
            residual,
        })
    }

    /// Flatten per-node values into a single vector.
    pub fn flatten(&self) -> Vec<f64> {
        self.values.iter().flatten().copied().collect()
    }
}
