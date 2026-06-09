//! # spectral-fingerprint
//!
//! **Spectral fingerprinting for code similarity — eigenvalue decomposition of AST adjacency matrices.**
//!
//! This crate converts code structure into compact, fixed-length "fingerprints" using
//! eigenvalue decomposition. A function's AST becomes an adjacency matrix; the eigenvalues
//! form a 64-dimensional vector that captures structural topology and is invariant to node
//! ordering. Two functions with similar structure produce similar fingerprints regardless
//! of variable names or literal values.
//!
//! ## The Key Insight
//!
//! An adjacency matrix `A` of a graph `G` encodes all connectivity information. Its
//! eigenvalue decomposition `A = QΛQᵀ` reveals fundamental structural properties:
//! the largest eigenvalue correlates with graph density, the eigenvalue spectrum determines
//! path counts between nodes, and the spectral gap relates to connectivity robustness.
//! Because AST topology changes smoothly with structural edits, eigenvalue fingerprints
//! are **stable under small changes** while remaining **discriminative** between different
//! structures.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      Source Code (AST)                       │
//! └────────────────────────┬────────────────────────────────────┘
//!                          │
//!                     ┌────▼────┐
//!                     │CodeGraph│  ast_matrix.rs — adjacency matrix + node metadata
//!                     └────┬────┘
//!                          │ eigenvalue decomposition (power iteration + deflation)
//!                     ┌────▼─────────────┐
//!                     │SpectralFingerprint│  fingerprint.rs — 64-dim L2-normalized vector
//!                     └────┬─────────────┘
//!                          │
//!               ┌──────────┼──────────┐
//!               │          │          │
//!         ┌─────▼────┐ ┌──▼───┐ ┌───▼──────┐
//!         │Similarity│ │LSH   │ │StructDiff│
//!         │  Index   │ │Hash  │ │          │
//!         └─────┬────┘ └──┬───┘ └──────────┘
//!               │         │
//!         ┌─────▼─────────▼──────┐
//!         │   BatchProcessor     │  batch.rs — directory-level duplicate detection
//!         └──────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```
//! use spectral_fingerprint::{CodeGraph, CodeNode, SpectralFingerprint};
//!
//! // Build a code graph from a tree of AST nodes.
//! let nodes = vec![
//!     CodeNode::new("fn", Some("my_func".into()), 0),
//!     CodeNode::new("if", None, 1),
//!     CodeNode::new("return", None, 2),
//! ];
//! let parents = vec![None, Some(0), Some(1)];
//! let graph = CodeGraph::from_tree(nodes, &parents).unwrap();
//!
//! // Compute a spectral fingerprint.
//! let fp = SpectralFingerprint::from_graph(&graph).unwrap();
//!
//! // Compare with another fingerprint.
//! let fp2 = SpectralFingerprint::from_graph(&graph).unwrap();
//! let similarity = fp.cosine_similarity(&fp2);
//! assert!(similarity > 0.99);
//! ```

pub mod ast_matrix;
pub mod batch;
pub mod diff;
pub mod fingerprint;
pub mod hash;
pub mod similarity;

// Re-export key types.
pub use ast_matrix::{CodeGraph, CodeNode};
pub use batch::{BatchProcessor, BatchResult, DuplicatePair, FunctionEntry};
pub use diff::{DiffEntry, StructuralDiff};
pub use fingerprint::{FINGERPRINT_LEN, SpectralFingerprint};
pub use hash::{SpectralHash, HASH_BITS, NUM_TABLES};
pub use similarity::{Cluster, SimilarityIndex, SimilarityMatch};
