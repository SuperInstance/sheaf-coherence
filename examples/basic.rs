//! Basic example: compute sheaf cohomology for a simple multi-agent cover.

use sheaf_coherence::cochain::compute_cohomology;
use sheaf_coherence::gluing::glue;
use sheaf_coherence::section::SectionFamily;
use sheaf_coherence::{LocalSection, ObstructionClass, OpenCover};

fn main() {
    // Three agents covering variables {0,1,2,3,4}:
    // Agent 0 sees {0, 1, 2}
    // Agent 1 sees {1, 2, 3}
    // Agent 2 sees {2, 3, 4}
    let cover = OpenCover::new(vec![vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 4]]);

    println!(
        "Open cover with {} sets, universe size {}",
        cover.num_sets(),
        cover.universe_size()
    );
    println!("Universe: {:?}", cover.universe());

    // Consistent sections: all agents agree on shared variables
    let sections = SectionFamily::new(vec![
        LocalSection::new(0, vec![1.0, 2.0, 3.0]),
        LocalSection::new(1, vec![2.0, 3.0, 4.0]),
        LocalSection::new(2, vec![3.0, 4.0, 5.0]),
    ]);

    // Check pairwise consistency
    let consistent = sections.is_consistent(&cover, 1e-10);
    println!("Sections consistent: {}", consistent);

    // Compute cohomology
    let (h0, h1) = compute_cohomology(&cover, 1e-10);
    println!(
        "H⁰ dimension: {} (globally consistent sections)",
        h0.dimension
    );
    println!("H¹ dimension: {} (obstructions)", h1.dimension);

    // Detect obstructions
    let obs = ObstructionClass::detect(&cover, &sections, 1e-10);
    println!("Obstruction class: {:?}", obs.classify());
    println!("Is consistent: {}", obs.is_consistent());

    // Glue sections into a global section
    let result = glue(&cover, &sections, 1e-10);
    match result {
        sheaf_coherence::gluing::GluingResult::Success(global) => {
            println!("Glued global section: {:?}", global);
        }
        sheaf_coherence::gluing::GluingResult::Failed { h1_dimension } => {
            println!("Gluing failed! H¹ dimension: {}", h1_dimension);
        }
    }
}
