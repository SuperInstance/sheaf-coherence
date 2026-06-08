//! # sheaf-coherence
//!
//! Cellular sheaf coherence for multi-agent belief alignment.
//!
//! A **cellular sheaf** assigns vector spaces (stalks) to nodes and linear maps
//! (restriction maps) to edges of a graph. The **sheaf Laplacian** `L_F` measures
//! how much a section (belief assignment) disagrees across edges:
//!
//! - `L_F x = 0` ⟹ **global section** (perfect agreement)
//! - `||L_F x|| / ||x||` ⟹ **disagreement level**
//! - `1 - ||L_F x|| / ||x||` ⟹ **alignment score**
//!
//! # Quick Start
//!
//! ```
//! use sheaf_coherence::{CellularSheaf, SheafLaplacian, CoherenceMeasure, AgentSheaf, AgentBelief};
//!
//! // Build a complete sheaf on 3 nodes with 2D stalks
//! let sheaf = CellularSheaf::complete(3, 2).unwrap();
//! let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
//!
//! // Perfect agreement → alignment = 1.0
//! let beliefs = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
//! let coherence = CoherenceMeasure::from_flat(&sheaf, &beliefs, 100, 1e-10).unwrap();
//! assert!(coherence.alignment > 0.99);
//!
//! // Agent-based interface
//! let agents = vec![
//!     AgentBelief::new("alice", vec![1.0, 0.0], 0.9),
//!     AgentBelief::new("bob",   vec![1.0, 0.0], 0.8),
//!     AgentBelief::new("carol", vec![0.0, 1.0], 0.7),
//! ];
//! let asheaf = AgentSheaf::complete(agents).unwrap();
//! let coh = asheaf.coherence(100, 1e-10).unwrap();
//! ```

pub mod agent;
pub mod coherence;
pub mod error;
pub mod laplacian;
pub mod section;
pub mod sheaf;

pub use agent::{AgentBelief, AgentSheaf};
pub use coherence::CoherenceMeasure;
pub use error::SheafError;
pub use laplacian::SheafLaplacian;
pub use section::GlobalSection;
pub use sheaf::{CellularSheaf, SheafBuilder};

#[cfg(test)]
mod tests {
    use super::*;

    // ── CellularSheaf construction ──────────────────────────────

    #[test]
    fn test_constant_sheaf() {
        let s = CellularSheaf::constant(4, 3).unwrap();
        assert_eq!(s.node_count(), 4);
        assert_eq!(s.total_dim(), 12);
        assert!(s.restriction_maps.is_empty());
    }

    #[test]
    fn test_path_sheaf() {
        let s = CellularSheaf::path(3, 2).unwrap();
        assert_eq!(s.node_count(), 3);
        assert_eq!(s.restriction_maps.len(), 2); // edges: 0-1, 1-2
    }

    #[test]
    fn test_cycle_sheaf() {
        let s = CellularSheaf::cycle(4, 2).unwrap();
        assert_eq!(s.restriction_maps.len(), 4); // 3 path edges + 1 cycle-closing
    }

    #[test]
    fn test_complete_sheaf() {
        let s = CellularSheaf::complete(4, 2).unwrap();
        // Complete graph K4 has 6 edges
        assert_eq!(s.restriction_maps.len(), 6);
    }

    #[test]
    fn test_empty_sheaf_rejected() {
        assert!(CellularSheaf::constant(0, 2).is_err());
    }

    #[test]
    fn test_cycle_too_small() {
        assert!(CellularSheaf::cycle(2, 1).is_err());
    }

    #[test]
    fn test_builder_custom_sheaf() {
        let s = CellularSheaf::builder()
            .add_node(2)
            .add_node(2)
            .add_node(3)
            .add_edge(0, 1, vec![vec![1.0, 0.0], vec![0.0, 1.0]])
            .build()
            .unwrap();
        assert_eq!(s.node_count(), 3);
        assert_eq!(s.total_dim(), 7);
        assert_eq!(s.restriction_maps.len(), 1);
    }

    #[test]
    fn test_dimension_mismatch_detected() {
        // Edge (0,1): map is 2x2 but stalk[1]=3
        let res = CellularSheaf::builder()
            .add_node(2)
            .add_node(3)
            .add_edge(0, 1, vec![vec![1.0, 0.0], vec![0.0, 1.0]])
            .build();
        assert!(matches!(res, Err(SheafError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_get_restriction_map() {
        let s = CellularSheaf::path(3, 2).unwrap();
        let m = s.get_restriction_map(0, 1);
        assert!(m.is_some());
        let m = s.get_restriction_map(1, 0);
        assert!(m.is_some()); // undirected lookup
        let m = s.get_restriction_map(0, 2);
        assert!(m.is_none());
    }

    // ── SheafLaplacian ──────────────────────────────────────────

    #[test]
    fn test_laplacian_single_node() {
        let s = CellularSheaf::constant(1, 2).unwrap();
        let lap = SheafLaplacian::from_sheaf(&s).unwrap();
        assert_eq!(lap.n, 2);
        // Zero matrix (no edges)
        assert_eq!(lap.matrix, vec![vec![0.0; 2]; 2]);
    }

    #[test]
    fn test_laplacian_path_2_nodes() {
        let s = CellularSheaf::path(2, 1).unwrap();
        let lap = SheafLaplacian::from_sheaf(&s).unwrap();
        // Identity restriction map on 1D stalks: F^T F = [[1]]
        // L = [[1, -1], [-1, 1]]
        assert_eq!(lap.matrix, vec![vec![1.0, -1.0], vec![-1.0, 1.0]]);
    }

    #[test]
    fn test_laplacian_complete_3_nodes_1d() {
        let s = CellularSheaf::complete(3, 1).unwrap();
        let lap = SheafLaplacian::from_sheaf(&s).unwrap();
        // 3 edges, each contributes [[1]] to diag and -[[1]] to off-diag
        // Each node has degree 2: diag = [2, 2, 2], off-diag = -1
        assert!((lap.matrix[0][0] - 2.0).abs() < 1e-10);
        assert!((lap.matrix[0][1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_quadratic_form_constant_section() {
        // Constant section on path should have zero quadratic form
        let s = CellularSheaf::path(3, 2).unwrap();
        let lap = SheafLaplacian::from_sheaf(&s).unwrap();
        let x = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // same value at each node
        let q = lap.quadratic_form(&x);
        assert!(q.abs() < 1e-10, "constant section should have zero energy, got {q}");
    }

    #[test]
    fn test_laplacian_apply_zero_for_global_section() {
        let s = CellularSheaf::complete(3, 2).unwrap();
        let lap = SheafLaplacian::from_sheaf(&s).unwrap();
        // All nodes same value → global section → L_F x = 0
        let x = vec![2.0, 3.0, 2.0, 3.0, 2.0, 3.0];
        let lx = lap.apply(&x);
        for v in &lx {
            assert!(v.abs() < 1e-10, "L_F x should be zero for global section");
        }
    }

    #[test]
    fn test_laplacian_eigenvalues() {
        let s = CellularSheaf::path(2, 1).unwrap();
        let lap = SheafLaplacian::from_sheaf(&s).unwrap();
        // [[1,-1],[-1,1]] has eigenvalues 0 and 2
        let (largest, _) = lap.power_iteration(500, 1e-10);
        assert!((largest - 2.0).abs() < 0.05, "largest eigenvalue should be ~2, got {largest}");
        // Smallest should be near 0
        let (smallest, _) = lap.smallest_eigenvalue(500, 1e-10);
        assert!(smallest.abs() < 0.05, "smallest eigenvalue should be ~0, got {smallest}");
    }

    // ── GlobalSection ───────────────────────────────────────────

    #[test]
    fn test_global_section_exact() {
        let s = CellularSheaf::complete(3, 1).unwrap();
        let values = vec![vec![1.0], vec![1.0], vec![1.0]];
        let gs = GlobalSection::new(&s, values, 1e-8).unwrap();
        assert!(gs.is_exact);
        assert!(gs.residual < 1e-8);
    }

    #[test]
    fn test_global_section_not_exact() {
        let s = CellularSheaf::complete(3, 1).unwrap();
        let values = vec![vec![1.0], vec![0.0], vec![0.0]];
        let gs = GlobalSection::new(&s, values, 1e-8).unwrap();
        assert!(!gs.is_exact);
        assert!(gs.residual > 0.1);
    }

    #[test]
    fn test_find_global_section() {
        let s = CellularSheaf::complete(3, 1).unwrap();
        let gs = GlobalSection::find(&s, 200, 1e-10).unwrap();
        assert!(gs.is_exact); // constant sheaf on connected graph has global section
    }

    #[test]
    fn test_section_flatten() {
        let s = CellularSheaf::path(2, 2).unwrap();
        let values = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let gs = GlobalSection::new(&s, values, 1e-8).unwrap();
        assert_eq!(gs.flatten(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ── CoherenceMeasure ────────────────────────────────────────

    #[test]
    fn test_perfect_coherence() {
        let s = CellularSheaf::complete(3, 2).unwrap();
        let x = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let c = CoherenceMeasure::from_flat(&s, &x, 100, 1e-10).unwrap();
        assert!(c.alignment > 0.99, "alignment = {}", c.alignment);
        for d in &c.disagreement {
            assert!(d < &0.01, "per-edge disagreement should be ~0");
        }
    }

    #[test]
    fn test_zero_coherence() {
        let s = CellularSheaf::path(2, 1).unwrap();
        let x = vec![1.0, -1.0]; // opposite beliefs
        let c = CoherenceMeasure::from_flat(&s, &x, 100, 1e-10).unwrap();
        assert!(c.alignment < 0.01, "alignment should be near 0, got {}", c.alignment);
    }

    #[test]
    fn test_partial_coherence() {
        let s = CellularSheaf::complete(3, 2).unwrap();
        // Two agents agree, one partially disagrees
        let x = vec![1.0, 0.0, 1.0, 0.0, 0.8, 0.6];
        let c = CoherenceMeasure::from_flat(&s, &x, 100, 1e-10).unwrap();
        // Not perfectly aligned (third agent differs), not maximally misaligned
        assert!(c.alignment > 0.0 && c.alignment < 1.0, "alignment = {}", c.alignment);
    }

    #[test]
    fn test_disagreement_per_edge() {
        let s = CellularSheaf::path(3, 1).unwrap();
        let x = vec![0.0, 1.0, 2.0];
        let c = CoherenceMeasure::from_flat(&s, &x, 100, 1e-10).unwrap();
        assert_eq!(c.disagreement.len(), 2); // 2 edges in path of 3
    }

    #[test]
    fn test_avg_and_max_disagreement() {
        let s = CellularSheaf::path(3, 1).unwrap();
        let x = vec![0.0, 1.0, 2.0];
        let c = CoherenceMeasure::from_flat(&s, &x, 100, 1e-10).unwrap();
        assert!(c.max_disagreement() >= c.avg_disagreement());
    }

    #[test]
    fn test_is_aligned() {
        let s = CellularSheaf::complete(3, 1).unwrap();
        let x = vec![1.0, 1.0, 1.0];
        let c = CoherenceMeasure::from_flat(&s, &x, 100, 1e-10).unwrap();
        assert!(c.is_aligned(0.95));
    }

    #[test]
    fn test_dominant_mode_nonempty() {
        let s = CellularSheaf::complete(3, 2).unwrap();
        let x = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let c = CoherenceMeasure::from_flat(&s, &x, 100, 1e-10).unwrap();
        assert_eq!(c.dominant_mode.len(), 6);
    }

    // ── AgentSheaf ──────────────────────────────────────────────

    #[test]
    fn test_agent_sheaf_complete() {
        let agents = vec![
            AgentBelief::new("a", vec![1.0, 0.0], 0.9),
            AgentBelief::new("b", vec![1.0, 0.0], 0.8),
        ];
        let asheaf = AgentSheaf::complete(agents).unwrap();
        assert_eq!(asheaf.len(), 2);
        assert_eq!(asheaf.sheaf.restriction_maps.len(), 1);
    }

    #[test]
    fn test_agent_sheaf_path() {
        let agents = vec![
            AgentBelief::new("a", vec![1.0], 1.0),
            AgentBelief::new("b", vec![1.0], 1.0),
            AgentBelief::new("c", vec![1.0], 1.0),
        ];
        let asheaf = AgentSheaf::path(agents).unwrap();
        assert_eq!(asheaf.sheaf.restriction_maps.len(), 2);
    }

    #[test]
    fn test_agent_sheaf_custom_edges() {
        let agents = vec![
            AgentBelief::new("a", vec![1.0], 1.0),
            AgentBelief::new("b", vec![1.0], 1.0),
            AgentBelief::new("c", vec![1.0], 1.0),
        ];
        let asheaf = AgentSheaf::with_edges(agents, &[(0, 1), (1, 2)]).unwrap();
        assert_eq!(asheaf.sheaf.restriction_maps.len(), 2);
    }

    #[test]
    fn test_agent_coherence_perfect() {
        let agents = vec![
            AgentBelief::new("a", vec![5.0, 3.0], 1.0),
            AgentBelief::new("b", vec![5.0, 3.0], 1.0),
            AgentBelief::new("c", vec![5.0, 3.0], 1.0),
        ];
        let asheaf = AgentSheaf::complete(agents).unwrap();
        let coh = asheaf.coherence(200, 1e-10).unwrap();
        assert!(coh.alignment > 0.99);
    }

    #[test]
    fn test_agent_global_section() {
        let agents = vec![
            AgentBelief::new("a", vec![1.0, 0.0], 1.0),
            AgentBelief::new("b", vec![1.0, 0.0], 1.0),
        ];
        let asheaf = AgentSheaf::complete(agents).unwrap();
        let gs = asheaf.global_section(1e-8).unwrap();
        assert!(gs.is_exact);
    }

    #[test]
    fn test_agent_dimension_mismatch() {
        let agents = vec![
            AgentBelief::new("a", vec![1.0, 0.0], 1.0),
            AgentBelief::new("b", vec![1.0], 1.0), // wrong dim
        ];
        assert!(AgentSheaf::complete(agents).is_err());
    }

    #[test]
    fn test_agent_flat_beliefs() {
        let agents = vec![
            AgentBelief::new("a", vec![1.0, 2.0], 1.0),
            AgentBelief::new("b", vec![3.0, 4.0], 1.0),
        ];
        let asheaf = AgentSheaf::complete(agents).unwrap();
        assert_eq!(asheaf.flat_beliefs(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_agent_confidence_clamped() {
        let a = AgentBelief::new("x", vec![1.0], 1.5);
        assert!((a.confidence - 1.0).abs() < 1e-10);
        let a = AgentBelief::new("x", vec![1.0], -0.5);
        assert!(a.confidence.abs() < 1e-10);
    }

    #[test]
    fn test_agent_empty_rejected() {
        let agents: Vec<AgentBelief> = vec![];
        assert!(AgentSheaf::complete(agents).is_err());
    }

    // ── Serialization ───────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip() {
        let s = CellularSheaf::complete(3, 2).unwrap();
        let json = serde_json::to_string(&s).unwrap();
        let s2: CellularSheaf = serde_json::from_str(&json).unwrap();
        assert_eq!(s.node_count(), s2.node_count());
        assert_eq!(s.restriction_maps.len(), s2.restriction_maps.len());
    }
}
