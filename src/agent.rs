use crate::coherence::CoherenceMeasure;
use crate::error::SheafError;
use crate::laplacian::SheafLaplacian;
use crate::section::GlobalSection;
use crate::sheaf::{CellularSheaf, SheafBuilder};

/// An agent's belief represented as a vector in a stalk.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentBelief {
    /// Agent identifier.
    pub id: String,
    /// Belief vector (lives in the stalk space).
    pub belief_vector: Vec<f64>,
    /// Confidence in this belief (0..1).
    pub confidence: f64,
}

impl AgentBelief {
    pub fn new(id: impl Into<String>, belief_vector: Vec<f64>, confidence: f64) -> Self {
        Self {
            id: id.into(),
            belief_vector,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Wraps a group of agents as a cellular sheaf.
///
/// Each agent becomes a node, edges are built from a connectivity graph,
/// and restriction maps default to identity (can be customized).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentSheaf {
    /// The agents and their beliefs.
    pub agents: Vec<AgentBelief>,
    /// The underlying cellular sheaf.
    pub sheaf: CellularSheaf,
}

impl AgentSheaf {
    /// Build an agent sheaf where all agents share the same stalk dimension,
    /// connected as a complete graph with identity restriction maps.
    pub fn complete(agents: Vec<AgentBelief>) -> Result<Self, SheafError> {
        if agents.is_empty() {
            return Err(SheafError::EmptySheaf);
        }
        let stalk_dim = agents[0].belief_vector.len();
        for a in &agents {
            if a.belief_vector.len() != stalk_dim {
                return Err(SheafError::BeliefDimensionMismatch {
                    agent: a.id.clone(),
                    expected: stalk_dim,
                    got: a.belief_vector.len(),
                });
            }
        }
        let n = agents.len();
        let sheaf = CellularSheaf::complete(n, stalk_dim)?;
        Ok(Self { agents, sheaf })
    }

    /// Build an agent sheaf with a path connectivity.
    pub fn path(agents: Vec<AgentBelief>) -> Result<Self, SheafError> {
        if agents.is_empty() {
            return Err(SheafError::EmptySheaf);
        }
        let stalk_dim = agents[0].belief_vector.len();
        for a in &agents {
            if a.belief_vector.len() != stalk_dim {
                return Err(SheafError::BeliefDimensionMismatch {
                    agent: a.id.clone(),
                    expected: stalk_dim,
                    got: a.belief_vector.len(),
                });
            }
        }
        let n = agents.len();
        let sheaf = if n == 1 {
            CellularSheaf::constant(1, stalk_dim)?
        } else {
            CellularSheaf::path(n, stalk_dim)?
        };
        Ok(Self { agents, sheaf })
    }

    /// Build an agent sheaf with custom edges and identity restriction maps.
    pub fn with_edges(agents: Vec<AgentBelief>, edges: &[(usize, usize)]) -> Result<Self, SheafError> {
        if agents.is_empty() {
            return Err(SheafError::EmptySheaf);
        }
        let stalk_dim = agents[0].belief_vector.len();
        for a in &agents {
            if a.belief_vector.len() != stalk_dim {
                return Err(SheafError::BeliefDimensionMismatch {
                    agent: a.id.clone(),
                    expected: stalk_dim,
                    got: a.belief_vector.len(),
                });
            }
        }

        let identity = {
            let mut m = vec![vec![0.0; stalk_dim]; stalk_dim];
            for (i, row) in m.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            m
        };

        let mut builder = SheafBuilder::default();
        for _ in &agents {
            builder = builder.add_node(stalk_dim);
        }
        for &(i, j) in edges {
            builder = builder.add_edge(i, j, identity.clone());
        }
        let sheaf = builder.build()?;

        Ok(Self { agents, sheaf })
    }

    /// Flatten all agent beliefs into a single vector.
    pub fn flat_beliefs(&self) -> Vec<f64> {
        self.agents.iter().flat_map(|a| a.belief_vector.iter().copied()).collect()
    }

    /// Compute the sheaf Laplacian.
    pub fn laplacian(&self) -> Result<SheafLaplacian, SheafError> {
        SheafLaplacian::from_sheaf(&self.sheaf)
    }

    /// Measure coherence of current beliefs.
    pub fn coherence(&self, max_iter: usize, tol: f64) -> Result<CoherenceMeasure, SheafError> {
        let flat = self.flat_beliefs();
        CoherenceMeasure::from_flat(&self.sheaf, &flat, max_iter, tol)
    }

    /// Check if current beliefs form a global section.
    pub fn global_section(&self, tol: f64) -> Result<GlobalSection, SheafError> {
        let values: Vec<Vec<f64>> = self.agents.iter().map(|a| a.belief_vector.clone()).collect();
        GlobalSection::new(&self.sheaf, values, tol)
    }

    /// Number of agents.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// True if there are no agents.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
}
