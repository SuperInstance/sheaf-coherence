use std::fmt;

/// Errors that can arise when building or using cellular sheaves.
#[derive(Debug)]
pub enum SheafError {
    /// A restriction map dimension does not match its target stalk.
    DimensionMismatch {
        edge: (usize, usize),
        expected_rows: usize,
        expected_cols: usize,
        got_rows: usize,
        got_cols: usize,
    },
    /// A node index is out of range.
    InvalidNode(usize),
    /// An edge references an unknown node.
    InvalidEdge(usize, usize),
    /// Belief vector length does not match stalk dimension.
    BeliefDimensionMismatch {
        agent: String,
        expected: usize,
        got: usize,
    },
    /// Empty sheaf (no nodes).
    EmptySheaf,
    /// Linear algebra failure.
    SingularMatrix,
}

impl fmt::Display for SheafError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { edge, expected_rows, expected_cols, got_rows, got_cols } => {
                write!(f, "restriction map on edge {:?}: expected {}x{}, got {}x{}", edge, expected_rows, expected_cols, got_rows, got_cols)
            }
            Self::InvalidNode(i) => write!(f, "invalid node index {i}"),
            Self::InvalidEdge(i, j) => write!(f, "invalid edge ({i}, {j})"),
            Self::BeliefDimensionMismatch { agent, expected, got } => {
                write!(f, "agent '{agent}': belief vector length {got}, expected {expected}")
            }
            Self::EmptySheaf => write!(f, "sheaf has no nodes"),
            Self::SingularMatrix => write!(f, "singular matrix encountered"),
        }
    }
}

impl std::error::Error for SheafError {}
