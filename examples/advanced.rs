//! Advanced Example: Real-world code similarity detection workflows
//!
//! Demonstrates duplicate detection, clustering, incremental indexing,
//! structural diffing, and LSH-based approximate search.
//!
//! Run with: `cargo run --example advanced`

use spectral_fingerprint::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Advanced Spectral Fingerprint: Real-World Workflows    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ── Application 1: Large-Scale Duplicate Detection ──────────
    println!("━━━ Application 1: Large-Scale Duplicate Detection ━━━\n");

    let mut bp = BatchProcessor::with_thresholds(0.98, 0.85);

    // Simulate a codebase with deliberate duplicates
    let functions = vec![
        ("parse_json", "src/parser.rs", "(fn parse_json (if (return obj) (return err)))"),
        ("parse_xml", "src/parser.rs", "(fn parse_xml (if (return obj) (return err)))"),
        ("parse_csv", "src/parser.rs", "(fn parse_csv (for (assign line) (return rows)))"),
        ("validate_input", "src/validate.rs", "(fn validate_input (if (return ok) (return err)))"),
        ("check_auth", "src/auth.rs", "(fn check_auth (if (if (return ok) (return err)) (return denied)))"),
        ("sanitize_data", "src/sanitize.rs", "(fn sanitize_data (for (assign clean) (return result)))"),
        ("format_output", "src/format.rs", "(fn format_output (for (assign line) (return rows)))"),
        ("handle_error", "src/error.rs", "(fn handle_error (if (return ok) (return err)))"),
    ];

    for (name, file, sexp) in &functions {
        bp.register(*name, *file, *sexp);
    }

    let result = bp.process().unwrap();
    println!("  Processed {} functions from {} files", result.total_functions, 5);
    println!("  Unique: {} | Clusters: {}", result.unique_count, result.cluster_count);
    println!("\n  Duplicate pairs:");
    for d in &result.duplicates {
        println!("    {:.2}%  {} ({}) ↔ {} ({})",
                 d.similarity * 100.0,
                 d.function_a, d.file_a,
                 d.function_b, d.file_b);
    }
    println!();

    // ── Application 2: Clustering ───────────────────────────────
    println!("━━━ Application 2: Code Structure Clustering ━━━\n");

    let mut index = SimilarityIndex::new();
    for fp in &bp.fingerprints {
        index.add(fp.clone());
    }

    let clusters = index.cluster(3, 50).unwrap();
    println!("  K-means clustering (k=3):\n");
    for (i, cluster) in clusters.iter().enumerate() {
        println!("  Cluster {} ({} members):", i + 1, cluster.members.len());
        for &member_idx in &cluster.members {
            let label = bp.fingerprints[member_idx].label.as_deref().unwrap_or("?");
            println!("    • {}", label);
        }
    }
    println!();

    // ── Application 3: Incremental Updates ──────────────────────
    println!("━━━ Application 3: Incremental Codebase Updates ━━━\n");
    println!("  Adding new functions and detecting duplicates incrementally...\n");

    let new_funcs = vec![
        FunctionEntry {
            name: "parse_yaml".into(),
            file: "src/parser.rs".into(),
            sexp: "(fn parse_yaml (if (return obj) (return err)))".into(),
        },
        FunctionEntry {
            name: "compress".into(),
            file: "src/util.rs".into(),
            sexp: "(fn compress (for (assign chunk) (return data)))".into(),
        },
    ];

    let new_dupes = bp.update(new_funcs).unwrap();
    println!("  New duplicate pairs found:");
    if new_dupes.is_empty() {
        println!("    (none)");
    } else {
        for d in &new_dupes {
            println!("    {:.2}%  {} ↔ {}",
                     d.similarity * 100.0,
                     d.function_a, d.function_b);
        }
    }
    println!("  Total functions now: {}", bp.functions.len());
    println!();

    // ── Application 4: Structural Diffing ───────────────────────
    println!("━━━ Application 4: Structural Diff Between Versions ━━━\n");

    let scenarios = vec![
        (
            "Unchanged",
            make_graph(&["fn", "if", "return", "assign"]),
            make_graph(&["fn", "if", "return", "assign"]),
        ),
        (
            "Added node",
            make_graph(&["fn", "if", "return"]),
            make_graph(&["fn", "if", "return", "log"]),
        ),
        (
            "Changed structure",
            make_graph(&["fn", "if", "return", "assign"]),
            make_graph(&["fn", "for", "assign", "assign", "return"]),
        ),
    ];

    for (name, old, new) in &scenarios {
        let diff = StructuralDiff::diff(old, new);
        println!("  {}:", name);
        println!("    Added: {} | Removed: {} | Unchanged: {}",
                 diff.added_count(), diff.removed_count(), diff.unchanged_count());
        println!("    Spectral similarity: {:.4}", diff.spectral_similarity);
        println!("    Edit distance: {:.2}", diff.edit_distance());
        println!();
    }

    // ── Application 5: LSH Performance ──────────────────────────
    println!("━━━ Application 5: LSH Approximate Search ━━━\n");

    let lsh = bp.build_lsh_index();
    let query = bp.fingerprints[0].clone();

    println!("  Query: {}", query.label.as_deref().unwrap_or("?"));
    println!();

    // Exact search (brute force)
    let mut index2 = SimilarityIndex::new();
    for fp in &bp.fingerprints {
        index2.add(fp.clone());
    }
    let exact = index2.query_top_k(&query, 5).unwrap();

    // Approximate search (LSH)
    let approx = bp.query_approximate(&lsh, &query, 0.0);

    println!("  Exact top-5 (brute force):");
    for m in &exact {
        println!("    {:.4}  {}", m.score, m.label.as_deref().unwrap_or("?"));
    }
    println!("\n  Approximate matches (LSH):");
    for m in &approx {
        println!("    {:.4}  {} (hits: {})", m.score, m.label.as_deref().unwrap_or("?"), 0);
    }
    println!();

    // ── Application 6: S-Expression Parsing ─────────────────────
    println!("━━━ Application 6: Parsing S-Expression ASTs ━━━\n");

    let sexps = vec![
        "(fn main (if (return 1) (return 2)))",
        "(fn helper (for (assign x) (return x)))",
        "(fn nested (if (if (return 1) (return 2)) (return 3)))",
    ];

    for sexp in &sexps {
        let graph = CodeGraph::from_sexp(sexp).unwrap();
        let fp = SpectralFingerprint::from_graph(&graph).unwrap();
        println!("  {:40} → {} nodes, top eigenvalue: {:.4}",
                 sexp, graph.node_count, fp.eigenvalues[0]);
    }

    println!("\n✅ Advanced examples complete.");
}

fn make_graph(types: &[&str]) -> CodeGraph {
    let nodes: Vec<CodeNode> = types.iter().enumerate().map(|(i, &t)| {
        CodeNode::new(t, if i == 0 { Some("func".into()) } else { None }, if i == 0 { 0 } else { 1 })
    }).collect();
    let parents: Vec<Option<usize>> = (0..types.len()).map(|i| if i == 0 { None } else { Some(0) }).collect();
    CodeGraph::from_tree(nodes, &parents).unwrap()
}
