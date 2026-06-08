//! # spectral-fingerprint
//!
//! Spectral fingerprinting for code similarity detection.
//!
//! This crate uses eigenvalue decomposition to create compact, fixed-length
//! "fingerprints" of code structure. A function's AST becomes an adjacency
//! matrix; the eigenvalues of this matrix form a fingerprint that is invariant
//! to node ordering and captures structural topology.
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
