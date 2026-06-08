# spectral-fingerprint

Spectral fingerprinting for code similarity — eigenvalue decomposition of AST adjacency matrices.

[![crates.io](https://img.shields.io/crates/v/spectral-fingerprint.svg)](https://crates.io/crates/spectral-fingerprint)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The Big Idea

Every function's abstract syntax tree has a **shape** — a topology of parent-child relationships between nodes. Two functions with similar structure (e.g., both have nested `if`-`return` patterns) should look similar even if they use different variable names or literals.

**Spectral fingerprinting** converts that shape into a fixed-length vector:

1. **AST → Adjacency Matrix**: Each AST node becomes a vertex; parent-child edges become entries in a symmetric adjacency matrix.
2. **Adjacency Matrix → Eigenvalues**: The eigenvalues of the adjacency matrix capture its structural properties — connectivity, branching factor, depth distribution.
3. **Eigenvalues → Fingerprint**: We extract the top 64 eigenvalues (by absolute value), pad with zeros if needed, and L2-normalize the result.

The result is a **64-dimensional vector** that serves as a structural "DNA barcode" for the function. Similar code → similar eigenvalues → similar fingerprints.

## Why Eigenvalues Capture Structure

An adjacency matrix `A` of a graph `G` encodes all connectivity information. Its eigenvalue decomposition `A = QΛQᵀ` reveals fundamental properties:

- **The largest eigenvalue** is bounded by the maximum degree and correlates with graph density.
- **The eigenvalue spectrum** determines the number of paths of each length between nodes. Two graphs with the same spectrum are **cospectral** — they share all path-count statistics.
- **The spectral gap** (difference between the two largest eigenvalues) relates to expansion properties and connectivity robustness.
- **Negative eigenvalues** indicate bipartite-like substructures (alternating node types in the AST).

Because AST topology changes smoothly with structural changes (adding a node shifts eigenvalues slightly, not randomly), eigenvalue fingerprints are **stable under small edits** while remaining **discriminative** between genuinely different structures.

## Power Iteration

We compute eigenvalues using **iterative deflation with power iteration** — no external linear algebra libraries needed:

1. Start with random vector `v`
2. Repeatedly compute `v ← A·v` and normalize (power iteration)
3. The Rayleigh quotient `vᵀAv / vᵀv` converges to the dominant eigenvalue `λ₁`
4. **Deflate**: set `A ← A - λ₁·vvᵀ` to remove the found component
5. Repeat to find `λ₂, λ₃, ...` until convergence or reaching 64 eigenvalues

This converges because the dominant eigenvector grows fastest under repeated multiplication. After each deflation, the next-largest eigenvalue becomes dominant.

## Similarity Measures

Three comparison methods are provided:

| Method | Formula | Properties |
|--------|---------|------------|
| **Cosine Similarity** | `dot(a,b) / (‖a‖·‖b‖)` | Scale-invariant, 1.0 = identical direction |
| **L2 Distance** | `‖a - b‖₂` | Magnitude-sensitive, 0.0 = identical |
| **Jaccard (binarized)** | `|A∩B| / |A∪B|` on sign bits | Robust to magnitude noise, discrete |

## Locality-Sensitive Hashing

For large codebases, brute-force O(n²) comparison is expensive. We provide **LSH via random projections**:

1. Generate 128 random projection vectors
2. Hash each fingerprint by computing `sign(projection · fingerprint)` for each projection
3. Split the 128-bit hash into 8 sub-hashes (bands)
4. Candidates sharing any sub-hash bucket are approximate neighbors
5. Verify with exact cosine similarity

This gives **sub-linear** candidate retrieval with high recall for similar fingerprints.

## Quick Start

```rust
use spectral_fingerprint::{CodeGraph, CodeNode, SpectralFingerprint};

// Build a code graph from an AST-like tree
let nodes = vec![
    CodeNode::new("fn", Some("my_func".into()), 0),
    CodeNode::new("if", None, 1),
    CodeNode::new("return", None, 2),
    CodeNode::new("return", None, 2),
];
let parents = vec![None, Some(0), Some(1), Some(1)];
let graph = CodeGraph::from_tree(nodes, &parents).unwrap();

// Compute a 64-dimensional spectral fingerprint
let fp = SpectralFingerprint::from_graph(&graph).unwrap();

// Compare two fingerprints
let similarity = fp.cosine_similarity(&other_fingerprint);
```

## Batch Processing

```rust
use spectral_fingerprint::BatchProcessor;

let mut bp = BatchProcessor::new();
bp.register("foo", "src/lib.rs", "(fn foo (if (return 1) (return 2)))");
bp.register("bar", "src/utils.rs", "(fn bar (if (return 1) (return 2)))");

let result = bp.process().unwrap();
println!("Found {} duplicate pairs", result.duplicates.len());
```

## Structural Diffing

```rust
use spectral_fingerprint::{CodeGraph, StructuralDiff};

let diff = StructuralDiff::diff(&old_graph, &new_graph);
println!("{}", diff.report());
// Structural Diff: 3 → 4 nodes
//   Added: 1  Removed: 0  Unchanged: 3
//   Spectral distance: 0.234
//   Spectral similarity: 0.972
//   Edit distance (approx): 0.23
// + [  3] assign
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Source Code (AST)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │CodeGraph│  ast_matrix.rs — adjacency matrix + node metadata
                    └────┬────┘
                         │ eigenvalue decomposition (power iteration + deflation)
                    ┌────▼─────────────┐
                    │SpectralFingerprint│  fingerprint.rs — 64-dim L2-normalized vector
                    └────┬─────────────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
        ┌─────▼────┐ ┌──▼───┐ ┌───▼──────┐
        │Similarity│ │LSH   │ │StructDiff│
        │  Index   │ │Hash  │ │          │
        └─────┬────┘ └──┬───┘ └──────────┘
              │         │
        ┌─────▼─────────▼──────┐
        │   BatchProcessor     │  batch.rs — directory-level duplicate detection
        └──────────────────────┘
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `ast_matrix` | `CodeGraph` and `CodeNode` — AST → adjacency matrix conversion |
| `fingerprint` | `SpectralFingerprint` — eigenvalue computation, similarity metrics |
| `similarity` | `SimilarityIndex` — top-K queries, threshold search, k-means clustering |
| `hash` | `SpectralHash` — locality-sensitive hashing for fast approximate search |
| `diff` | `StructuralDiff` — structural comparison of code graphs |
| `batch` | `BatchProcessor` — batch processing, incremental updates, LSH integration |

## Design Principles

- **Pure Rust** — no `unsafe`, no external math libraries (eigenvalue solver built from scratch)
- **Fixed-length fingerprints** — 64 eigenvalues regardless of input graph size
- **Serde-friendly** — all types implement `Serialize`/`Deserialize`
- **Zero-cost abstractions** — flat `Vec<f64>` adjacency matrices, no trait objects

## License

MIT
