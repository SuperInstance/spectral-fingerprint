//! Spectral fingerprinting via eigenvalue decomposition.
//!
//! A [`SpectralFingerprint`] is a fixed-length vector of eigenvalues derived
//! from a code graph's adjacency or Laplacian matrix. It supports comparison
//! via cosine similarity, L2 distance, and Jaccard on binarized values.

use crate::ast_matrix::CodeGraph;
use serde::{Deserialize, Serialize};

/// Fixed fingerprint length (number of eigenvalue slots).
pub const FINGERPRINT_LEN: usize = 64;

/// A spectral fingerprint: a fixed-length, L2-normalized vector of eigenvalues.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpectralFingerprint {
    /// The eigenvalue vector (length = `FINGERPRINT_LEN`).
    pub eigenvalues: Vec<f64>,
    /// Optional label identifying the source function.
    pub label: Option<String>,
}

/// Errors during fingerprint computation.
#[derive(Debug, thiserror::Error)]
pub enum FingerprintError {
    #[error("graph is empty, cannot compute fingerprint")]
    EmptyGraph,
}

impl SpectralFingerprint {
    /// Compute a spectral fingerprint from a [`CodeGraph`].
    ///
    /// Uses power iteration to find eigenvalues of the adjacency matrix,
    /// then normalizes to a fixed length by padding with zeros or truncating.
    /// Finally, L2-normalizes the vector.
    pub fn from_graph(graph: &CodeGraph) -> Result<Self, FingerprintError> {
        if graph.node_count == 0 {
            return Err(FingerprintError::EmptyGraph);
        }
        let n = graph.node_count;
        let eigenvalues = compute_eigenvalues(&graph.adjacency, n);
        let mut fp = normalize_to_length(eigenvalues, FINGERPRINT_LEN);
        l2_normalize(&mut fp);
        Ok(Self {
            eigenvalues: fp,
            label: None,
        })
    }

    /// Compute fingerprint with a label.
    pub fn from_graph_labeled(
        graph: &CodeGraph,
        label: impl Into<String>,
    ) -> Result<Self, FingerprintError> {
        let mut fp = Self::from_graph(graph)?;
        fp.label = Some(label.into());
        Ok(fp)
    }

    /// Create a fingerprint directly from eigenvalue data (for testing / deserialization).
    pub fn from_eigenvalues(eigenvalues: Vec<f64>) -> Self {
        let mut fp = normalize_to_length(eigenvalues, FINGERPRINT_LEN);
        l2_normalize(&mut fp);
        Self {
            eigenvalues: fp,
            label: None,
        }
    }

    /// Cosine similarity between two fingerprints (1.0 = identical direction).
    pub fn cosine_similarity(&self, other: &SpectralFingerprint) -> f64 {
        let dot: f64 = self
            .eigenvalues
            .iter()
            .zip(other.eigenvalues.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f64 = self.eigenvalues.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = other.eigenvalues.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// L2 (Euclidean) distance between two fingerprints.
    pub fn l2_distance(&self, other: &SpectralFingerprint) -> f64 {
        self.eigenvalues
            .iter()
            .zip(other.eigenvalues.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Jaccard similarity on binarized eigenvalues.
    ///
    /// Each eigenvalue is thresholded: > 0 → 1, else → 0.
    /// Jaccard = |intersection| / |union|.
    pub fn jaccard_similarity(&self, other: &SpectralFingerprint) -> f64 {
        let a_bits: Vec<bool> = self.eigenvalues.iter().map(|x| *x > 0.0).collect();
        let b_bits: Vec<bool> = other.eigenvalues.iter().map(|x| *x > 0.0).collect();
        let intersection = a_bits
            .iter()
            .zip(b_bits.iter())
            .filter(|(a, b)| **a && **b)
            .count();
        let union = a_bits
            .iter()
            .zip(b_bits.iter())
            .filter(|(a, b)| **a || **b)
            .count();
        if union == 0 {
            return 1.0; // both all-zero
        }
        intersection as f64 / union as f64
    }

    /// Raw eigenvalue slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.eigenvalues
    }
}

/// Compute eigenvalues via iterative deflation using power iteration.
///
/// Finds the top `min(n, FINGERPRINT_LEN)` eigenvalues by repeatedly:
/// 1. Running power iteration to find the dominant eigenvalue/eigenvector
/// 2. Deflating the matrix: A' = A - λ * v * v^T
fn compute_eigenvalues(matrix: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let max_eigs = n.min(FINGERPRINT_LEN);
    let mut a = matrix.to_vec();
    let mut eigenvalues = Vec::with_capacity(max_eigs);

    for _ in 0..max_eigs {
        let (eigenvalue, eigenvector) = power_iteration(&a, n, 100, 1e-10);
        if eigenvalue.abs() < 1e-12 {
            break;
        }
        eigenvalues.push(eigenvalue);
        // Deflate: A = A - λ * v * v^T
        deflate(&mut a, n, eigenvalue, &eigenvector);
    }

    // Sort by absolute value descending.
    eigenvalues.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Power iteration: find the dominant eigenvalue and eigenvector of an n×n matrix.
fn power_iteration(matrix: &[f64], n: usize, max_iters: usize, tolerance: f64) -> (f64, Vec<f64>) {
    // Start with a random-ish vector.
    let mut v = vec![1.0; n];
    for (i, vi) in v.iter_mut().enumerate().take(n) {
        *vi = (i as f64 * 0.1 + 1.0).sin();
    }
    l2_normalize(&mut v);

    let mut eigenvalue = 0.0_f64;
    for _ in 0..max_iters {
        // Multiply: w = A * v
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix[i * n + j] * v[j];
            }
        }
        let new_eigenvalue = dot(&v, &w);
        l2_normalize(&mut w);
        // Check convergence.
        if (new_eigenvalue - eigenvalue).abs() < tolerance {
            eigenvalue = new_eigenvalue;
            v = w;
            break;
        }
        eigenvalue = new_eigenvalue;
        v = w;
    }
    (eigenvalue, v)
}

/// Deflate a matrix by subtracting the rank-1 component λ * v * v^T.
fn deflate(matrix: &mut [f64], n: usize, eigenvalue: f64, eigenvector: &[f64]) {
    for i in 0..n {
        for j in 0..n {
            matrix[i * n + j] -= eigenvalue * eigenvector[i] * eigenvector[j];
        }
    }
}

/// Pad or truncate eigenvalues to a fixed length.
fn normalize_to_length(mut eigenvalues: Vec<f64>, target_len: usize) -> Vec<f64> {
    eigenvalues.resize(target_len, 0.0);
    eigenvalues
}

/// L2-normalize a vector in place. If the vector is all zeros, leave it.
fn l2_normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_matrix::{CodeGraph, CodeNode};

    fn make_simple_graph() -> CodeGraph {
        let nodes = vec![
            CodeNode::new("fn", Some("test".into()), 0),
            CodeNode::new("if", None, 1),
            CodeNode::new("return", None, 2),
            CodeNode::new("return", None, 2),
        ];
        let parents = vec![None, Some(0), Some(1), Some(1)];
        CodeGraph::from_tree(nodes, &parents).unwrap()
    }

    #[test]
    fn test_fingerprint_from_graph() {
        let g = make_simple_graph();
        let fp = SpectralFingerprint::from_graph(&g).unwrap();
        assert_eq!(fp.eigenvalues.len(), FINGERPRINT_LEN);
    }

    #[test]
    fn test_fingerprint_empty_graph() {
        let g = CodeGraph::new();
        assert!(SpectralFingerprint::from_graph(&g).is_err());
    }

    #[test]
    fn test_identical_graphs_identical_fingerprint() {
        let g1 = make_simple_graph();
        let g2 = make_simple_graph();
        let fp1 = SpectralFingerprint::from_graph(&g1).unwrap();
        let fp2 = SpectralFingerprint::from_graph(&g2).unwrap();
        // Eigenvalues should be identical.
        assert!((fp1.eigenvalues[0] - fp2.eigenvalues[0]).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let g = make_simple_graph();
        let fp = SpectralFingerprint::from_graph(&g).unwrap();
        let sim = fp.cosine_similarity(&fp);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_different() {
        let g1 = make_simple_graph();
        let nodes2 = vec![
            CodeNode::new("fn", Some("other".into()), 0),
            CodeNode::new("for", None, 1),
            CodeNode::new("assign", None, 2),
        ];
        let parents2 = vec![None, Some(0), Some(1)];
        let g2 = CodeGraph::from_tree(nodes2, &parents2).unwrap();
        let fp1 = SpectralFingerprint::from_graph(&g1).unwrap();
        let fp2 = SpectralFingerprint::from_graph(&g2).unwrap();
        let sim = fp1.cosine_similarity(&fp2);
        // Different graphs should have similarity < 1.0 but may not be 0.
        assert!(sim < 1.0);
    }

    #[test]
    fn test_l2_distance_identical() {
        let g = make_simple_graph();
        let fp = SpectralFingerprint::from_graph(&g).unwrap();
        let dist = fp.l2_distance(&fp);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_l2_distance_different() {
        let g1 = make_simple_graph();
        let nodes2 = vec![
            CodeNode::new("fn", Some("x".into()), 0),
            CodeNode::new("for", None, 1),
        ];
        let g2 = CodeGraph::from_tree(nodes2, &[None, Some(0)]).unwrap();
        let fp1 = SpectralFingerprint::from_graph(&g1).unwrap();
        let fp2 = SpectralFingerprint::from_graph(&g2).unwrap();
        assert!(fp1.l2_distance(&fp2) > 0.0);
    }

    #[test]
    fn test_jaccard_identical() {
        let g = make_simple_graph();
        let fp = SpectralFingerprint::from_graph(&g).unwrap();
        let sim = fp.jaccard_similarity(&fp);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_all_positive() {
        // If all eigenvalues > 0 in both, Jaccard should be 1.0.
        let fp1 = SpectralFingerprint::from_eigenvalues(vec![1.0, 2.0, 3.0]);
        let fp2 = SpectralFingerprint::from_eigenvalues(vec![3.0, 2.0, 1.0]);
        let sim = fp1.jaccard_similarity(&fp2);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_eigenvalues() {
        let fp = SpectralFingerprint::from_eigenvalues(vec![3.0, 1.0, 0.5]);
        assert_eq!(fp.eigenvalues.len(), FINGERPRINT_LEN);
    }

    #[test]
    fn test_labeled_fingerprint() {
        let g = make_simple_graph();
        let fp = SpectralFingerprint::from_graph_labeled(&g, "my_func").unwrap();
        assert_eq!(fp.label.as_deref(), Some("my_func"));
    }

    #[test]
    fn test_as_slice() {
        let fp = SpectralFingerprint::from_eigenvalues(vec![1.0]);
        assert_eq!(fp.as_slice().len(), FINGERPRINT_LEN);
    }

    #[test]
    fn test_power_iteration_2x2() {
        // [[2,1],[1,2]] has eigenvalues 3 and 1.
        let matrix = vec![2.0, 1.0, 1.0, 2.0];
        let (eigenvalue, _) = power_iteration(&matrix, 2, 200, 1e-12);
        assert!((eigenvalue - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-10);
        assert!((v[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_l2_normalize_zero() {
        let mut v = vec![0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }
}
