//! Tutorial: Spectral Fingerprinting for Code Similarity
//!
//! A guided walkthrough from AST to fingerprint to similarity search.
//!
//! Run with: `cargo run --example tutorial`

use spectral_fingerprint::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Spectral Fingerprint Tutorial: Code → Eigenvalues → Match ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── Lesson 1: Building Code Graphs ──────────────────────────
    println!("━━━ Lesson 1: From AST to Adjacency Matrix ━━━\n");
    println!("Each AST node becomes a vertex. Parent-child edges become\n");
    println!("adjacency matrix entries. The result is a graph that captures\n");
    println!("structural shape, not names or values.\n");

    let nodes = vec![
        CodeNode::new("fn", Some("compute".into()), 0),
        CodeNode::new("if", None, 1),
        CodeNode::new("return", None, 2),
        CodeNode::new("assign", None, 2),
    ];
    let parents = vec![None, Some(0), Some(1), Some(1)];
    let graph = CodeGraph::from_tree(nodes, &parents).unwrap();

    println!("  Graph: fn(compute) → if → {{return, assign}}");
    println!("  Nodes: {}", graph.node_count);
    println!("  Adjacency matrix ({}×{}):", graph.node_count, graph.node_count);
    for i in 0..graph.node_count {
        print!("    ");
        for j in 0..graph.node_count {
            let val = graph.get(i, j).unwrap();
            print!("{} ", if val > 0.0 { "1" } else { "·" });
        }
        println!();
    }
    println!();

    // ── Lesson 2: Computing Fingerprints ────────────────────────
    println!("━━━ Lesson 2: Eigenvalues as Structural DNA ━━━\n");
    println!("Power iteration finds the dominant eigenvalue. Deflation\n");
    println!("removes it and finds the next. After 64 iterations, we have\n");
    println!("a fixed-length vector that captures graph structure.\n");

    let fp = SpectralFingerprint::from_graph_labeled(&graph, "compute").unwrap();
    println!("  Fingerprint length: {}", fp.eigenvalues.len());
    println!("  First 10 eigenvalues:");
    for (i, &val) in fp.eigenvalues.iter().take(10).enumerate() {
        let bar_len = (val.abs() * 20.0) as usize;
        let bar: String = if val >= 0.0 {
            "█".repeat(bar_len)
        } else {
            "░".repeat(bar_len)
        };
        println!("    [{:2}] {:>8.4} {}", i, val, bar);
    }
    println!("    ... (remaining 54 are near zero for small graphs)");
    println!();

    // ── Lesson 3: Comparing Fingerprints ────────────────────────
    println!("━━━ Lesson 3: Three Similarity Metrics ━━━\n");

    // Two structurally identical functions
    let g1 = make_fn("foo", &["if", "return", "assign"]);
    let g2 = make_fn("bar", &["if", "return", "assign"]);
    let fp1 = SpectralFingerprint::from_graph(&g1).unwrap();
    let fp2 = SpectralFingerprint::from_graph(&g2).unwrap();

    println!("  Same structure, different names:");
    println!("    Cosine:  {:.6}", fp1.cosine_similarity(&fp2));
    println!("    L2 dist: {:.6}", fp1.l2_distance(&fp2));
    println!("    Jaccard: {:.6}", fp1.jaccard_similarity(&fp2));
    println!();

    // Different structures
    let g3 = make_fn("baz", &["for", "assign"]);
    let fp3 = SpectralFingerprint::from_graph(&g3).unwrap();
    println!("  Different structure (for+assign vs if+return+assign):");
    println!("    Cosine:  {:.6}", fp1.cosine_similarity(&fp3));
    println!("    L2 dist: {:.6}", fp1.l2_distance(&fp3));
    println!("    Jaccard: {:.6}", fp1.jaccard_similarity(&fp3));
    println!();

    // ── Lesson 4: Batch Processing ──────────────────────────────
    println!("━━━ Lesson 4: Finding Duplicates in a Codebase ━━━\n");

    let mut bp = BatchProcessor::new();
    bp.register("process_data", "src/main.rs", "(fn process_data (if (return 1) (return 2)))");
    bp.register("handle_data", "src/utils.rs", "(fn handle_data (if (return 1) (return 2)))");
    bp.register("compute", "src/math.rs", "(fn compute (for (assign x)))");
    bp.register("validate", "src/check.rs", "(fn validate (if (return ok)))");

    let result = bp.process().unwrap();
    println!("  Processed {} functions", result.total_functions);
    println!("  Unique: {}", result.unique_count);
    println!("  Clusters: {}", result.cluster_count);
    println!("  Duplicates:");
    for d in &result.duplicates {
        println!("    {} ↔ {} (similarity: {:.4})", d.function_a, d.function_b, d.similarity);
    }
    println!();

    // ── Lesson 5: Structural Diffing ────────────────────────────
    println!("━━━ Lesson 5: Structural Diff Between Versions ━━━\n");

    let old_graph = make_fn("v1", &["if", "return", "assign"]);
    let new_graph = make_fn("v2", &["if", "return", "assign", "log"]);

    let diff = StructuralDiff::diff(&old_graph, &new_graph);
    println!("{}", diff.report());
    println!();

    // ── Lesson 6: LSH for Large Codebases ───────────────────────
    println!("━━━ Lesson 6: LSH for Fast Approximate Search ━━━\n");
    println!("  For large codebases, O(n²) comparison is too slow.\n");
    println!("  LSH hashes fingerprints into buckets — similar code lands\n");
    println!("  in the same bucket, enabling sub-linear search.\n");

    let lsh = bp.build_lsh_index();
    let query = bp.fingerprints[0].clone();
    let matches = bp.query_approximate(&lsh, &query, 0.5);
    println!("  Query: {} — found {} candidates", query.label.as_deref().unwrap_or("?"), matches.len());
    for m in &matches {
        println!("    {} (score: {:.4})", m.label.as_deref().unwrap_or("?"), m.score);
    }

    println!("\n✅ Tutorial complete!");
}

fn make_fn(_name: &str, child_types: &[&str]) -> CodeGraph {
    let mut nodes = vec![CodeNode::new("fn", None, 0)];
    let mut parents = vec![None];
    for (i, &t) in child_types.iter().enumerate() {
        nodes.push(CodeNode::new(t, None, 1));
        parents.push(Some(0));
        // Connect siblings for more interesting structure
        if i > 0 {
            let _ = (); // siblings not connected by default
        }
    }
    CodeGraph::from_tree(nodes, &parents).unwrap()
}
