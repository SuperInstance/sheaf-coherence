use crate::error::SheafError;

/// A cellular sheaf over a graph.
///
/// Assigns a vector space (stalk) of dimension `stalk_dims[i]` to each node `i`,
/// and a linear restriction map to each directed edge `(i, j)`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CellularSheaf {
    /// Adjacency list: `graph[i]` contains neighbors of node `i`.
    pub graph: Vec<Vec<usize>>,
    /// Dimension of the stalk (vector space) at each node.
    pub stalk_dims: Vec<usize>,
    /// Restriction maps as `(source, target, matrix)` triples.
    /// The matrix maps from stalk[source] to stalk[target].
    pub restriction_maps: Vec<(usize, usize, Vec<Vec<f64>>)>,
}

impl CellularSheaf {
    /// Build a constant sheaf: every stalk has the same dimension,
    /// and every restriction map is the identity matrix.
    pub fn constant(n: usize, stalk_dim: usize) -> Result<Self, SheafError> {
        if n == 0 {
            return Err(SheafError::EmptySheaf);
        }
        let graph = vec![vec![]; n];
        let stalk_dims = vec![stalk_dim; n];
        Ok(Self {
            graph,
            stalk_dims,
            restriction_maps: vec![],
        })
    }

    /// Build a sheaf on a path graph `0 — 1 — 2 — … — (n-1)` with constant stalk dimension.
    pub fn path(n: usize, stalk_dim: usize) -> Result<Self, SheafError> {
        if n == 0 {
            return Err(SheafError::EmptySheaf);
        }
        let mut graph = vec![vec![]; n];
        let mut restriction_maps = Vec::new();
        let identity = identity_matrix(stalk_dim);
        for i in 0..n.saturating_sub(1) {
            graph[i].push(i + 1);
            graph[i + 1].push(i);
            restriction_maps.push((i, i + 1, identity.clone()));
        }
        Ok(Self {
            graph,
            stalk_dims: vec![stalk_dim; n],
            restriction_maps,
        })
    }

    /// Build a sheaf on a cycle graph with constant stalk dimension.
    pub fn cycle(n: usize, stalk_dim: usize) -> Result<Self, SheafError> {
        if n < 3 {
            return Err(SheafError::EmptySheaf);
        }
        let mut sheaf = Self::path(n, stalk_dim)?;
        // close the cycle: edge (n-1, 0)
        let identity = identity_matrix(stalk_dim);
        sheaf.graph[n - 1].push(0);
        sheaf.graph[0].push(n - 1);
        sheaf.restriction_maps.push((n - 1, 0, identity));
        Ok(sheaf)
    }

    /// Build a sheaf on a complete graph with constant stalk dimension.
    pub fn complete(n: usize, stalk_dim: usize) -> Result<Self, SheafError> {
        if n == 0 {
            return Err(SheafError::EmptySheaf);
        }
        let mut graph = vec![vec![]; n];
        let mut restriction_maps = Vec::new();
        let identity = identity_matrix(stalk_dim);
        for i in 0..n {
            for j in (i + 1)..n {
                graph[i].push(j);
                graph[j].push(i);
                restriction_maps.push((i, j, identity.clone()));
            }
        }
        Ok(Self {
            graph,
            stalk_dims: vec![stalk_dim; n],
            restriction_maps,
        })
    }

    /// Build a sheaf with a custom adjacency list, stalk dims, and restriction maps.
    pub fn builder() -> SheafBuilder {
        SheafBuilder::default()
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.stalk_dims.len()
    }

    /// Total dimension of the global section space (sum of stalk dims).
    pub fn total_dim(&self) -> usize {
        self.stalk_dims.iter().sum()
    }

    /// Validate internal consistency.
    pub fn validate(&self) -> Result<(), SheafError> {
        if self.stalk_dims.is_empty() {
            return Err(SheafError::EmptySheaf);
        }
        for (i, j, mat) in &self.restriction_maps {
            let max = self.stalk_dims.len();
            if *i >= max || *j >= max {
                return Err(SheafError::InvalidEdge(*i, *j));
            }
            let expected_rows = self.stalk_dims[*j];
            let expected_cols = self.stalk_dims[*i];
            let got_rows = mat.len();
            let got_cols = mat.first().map_or(0, |r| r.len());
            if got_rows != expected_rows || got_cols != expected_cols {
                return Err(SheafError::DimensionMismatch {
                    edge: (*i, *j),
                    expected_rows,
                    expected_cols,
                    got_rows,
                    got_cols,
                });
            }
        }
        Ok(())
    }

    /// Get restriction map from node `i` to node `j`, if it exists.
    pub fn get_restriction_map(&self, i: usize, j: usize) -> Option<&Vec<Vec<f64>>> {
        self.restriction_maps
            .iter()
            .find(|(src, tgt, _)| (*src, *tgt) == (i, j) || (*src, *tgt) == (j, i))
            .map(|(_, _, mat)| mat)
    }
}

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

/// Builder for constructing a `CellularSheaf` incrementally.
#[derive(Debug, Default)]
pub struct SheafBuilder {
    graph: Vec<Vec<usize>>,
    stalk_dims: Vec<usize>,
    restriction_maps: Vec<(usize, usize, Vec<Vec<f64>>)>,
}

impl SheafBuilder {
    pub fn add_node(mut self, stalk_dim: usize) -> Self {
        self.graph.push(vec![]);
        self.stalk_dims.push(stalk_dim);
        self
    }

    pub fn add_edge(mut self, i: usize, j: usize, map: Vec<Vec<f64>>) -> Self {
        if i < self.graph.len() && j < self.graph.len() {
            self.graph[i].push(j);
            self.graph[j].push(i);
            self.restriction_maps.push((i, j, map));
        }
        self
    }

    pub fn build(self) -> Result<CellularSheaf, SheafError> {
        if self.stalk_dims.is_empty() {
            return Err(SheafError::EmptySheaf);
        }
        let sheaf = CellularSheaf {
            graph: self.graph,
            stalk_dims: self.stalk_dims,
            restriction_maps: self.restriction_maps,
        };
        sheaf.validate()?;
        Ok(sheaf)
    }
}
