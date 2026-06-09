//! Basic example: quick demo of spectral fingerprinting
//!
//! Run with: `cargo run --example basic`

use spectral_fingerprint::*;

fn main() {
    println!("=== Spectral Fingerprint: Code Similarity Demo ===\n");

    // Build graphs from s-expression ASTs
    let g1 = CodeGraph::from_sexp("(fn foo (if (return 1) (return 2)))").unwrap();
    let g2 = CodeGraph::from_sexp("(fn bar (if (return 1) (return 2)))").unwrap(); // same structure
    let g3 = CodeGraph::from_sexp("(fn baz (for (assign x)))").unwrap(); // different

    let fp1 = SpectralFingerprint::from_graph_labeled(&g1, "foo").unwrap();
    let fp2 = SpectralFingerprint::from_graph_labeled(&g2, "bar").unwrap();
    let fp3 = SpectralFingerprint::from_graph_labeled(&g3, "baz").unwrap();

    println!("foo ↔ bar (same structure):");
    println!("  Cosine: {:.4}  L2: {:.4}  Jaccard: {:.4}",
             fp1.cosine_similarity(&fp2),
             fp1.l2_distance(&fp2),
             fp1.jaccard_similarity(&fp2));

    println!("\nfoo ↔ baz (different structure):");
    println!("  Cosine: {:.4}  L2: {:.4}  Jaccard: {:.4}",
             fp1.cosine_similarity(&fp3),
             fp1.l2_distance(&fp3),
             fp1.jaccard_similarity(&fp3));

    // Structural diff
    println!("\n--- Structural Diff: foo → baz ---");
    let diff = StructuralDiff::diff(&g1, &g3);
    println!("{}", diff.report());

    // Batch duplicate detection
    println!("--- Batch Duplicate Detection ---");
    let mut bp = BatchProcessor::new();
    bp.register("foo", "a.rs", "(fn foo (if (return 1) (return 2)))");
    bp.register("bar", "b.rs", "(fn bar (if (return 1) (return 2)))");
    bp.register("baz", "c.rs", "(fn baz (for (assign x)))");

    let result = bp.process().unwrap();
    println!("  {} functions, {} duplicates found:", result.total_functions, result.duplicates.len());
    for d in &result.duplicates {
        println!("    {} ↔ {} ({:.2}%)", d.function_a, d.function_b, d.similarity * 100.0);
    }
}
