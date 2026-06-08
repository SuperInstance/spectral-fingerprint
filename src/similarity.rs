//! Similarity index: build, query, and cluster spectral fingerprints.
//!
//! [`SimilarityIndex`] stores a collection of fingerprints and supports:
//! - Top-K queries (brute-force cosine similarity search)
//! - Duplicate/near-duplicate detection at a configurable threshold
//! - K-means clustering on eigenvectors for similarity grouping

use crate::fingerprint::SpectralFingerprint;
use serde::{Deserialize, Serialize};

/// A match result from a similarity query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimilarityMatch {
    /// Index of the matched fingerprint in the index.
    pub index: usize,
    /// Cosine similarity score.
    pub score: f64,
    /// Label of the matched fingerprint (if available).
    pub label: Option<String>,
}

/// A cluster of fingerprints.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Cluster {
    /// Indices of fingerprints in this cluster.
    pub members: Vec<usize>,
    /// Centroid eigenvalues (average of member eigenvalues).
    pub centroid: Vec<f64>,
}

/// A similarity index for spectral fingerprints.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimilarityIndex {
    /// Stored fingerprints.
    pub fingerprints: Vec<SpectralFingerprint>,
}

/// Errors for similarity operations.
#[derive(Debug, thiserror::Error)]
pub enum SimilarityError {
    #[error("index is empty")]
    Empty,
    #[error("k ({requested}) exceeds available fingerprints ({available})")]
    KTooLarge { requested: usize, available: usize },
    #[error("invalid number of clusters: {message}")]
    InvalidClusters { message: String },
}

impl SimilarityIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            fingerprints: Vec::new(),
        }
    }

    /// Add a fingerprint to the index.
    pub fn add(&mut self, fp: SpectralFingerprint) -> usize {
        let idx = self.fingerprints.len();
        self.fingerprints.push(fp);
        idx
    }

    /// Number of fingerprints in the index.
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }

    /// Get a fingerprint by index.
    pub fn get(&self, index: usize) -> Option<&SpectralFingerprint> {
        self.fingerprints.get(index)
    }

    /// Query top-K most similar fingerprints by cosine similarity.
    pub fn query_top_k(
        &self,
        query: &SpectralFingerprint,
        k: usize,
    ) -> Result<Vec<SimilarityMatch>, SimilarityError> {
        if self.fingerprints.is_empty() {
            return Err(SimilarityError::Empty);
        }
        let effective_k = k.min(self.fingerprints.len());
        let mut scores: Vec<(usize, f64)> = self
            .fingerprints
            .iter()
            .enumerate()
            .map(|(i, fp)| (i, query.cosine_similarity(fp)))
            .collect();
        // Sort descending by score.
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores[..effective_k]
            .iter()
            .map(|(idx, score)| SimilarityMatch {
                index: *idx,
                score: *score,
                label: self.fingerprints[*idx].label.clone(),
            })
            .collect())
    }

    /// Find all fingerprints above a similarity threshold.
    pub fn query_threshold(
        &self,
        query: &SpectralFingerprint,
        threshold: f64,
    ) -> Vec<SimilarityMatch> {
        self.fingerprints
            .iter()
            .enumerate()
            .filter_map(|(i, fp)| {
                let score = query.cosine_similarity(fp);
                if score >= threshold {
                    Some(SimilarityMatch {
                        index: i,
                        score,
                        label: fp.label.clone(),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find all duplicate pairs above the given threshold.
    pub fn find_duplicates(&self, threshold: f64) -> Vec<(usize, usize, f64)> {
        let mut dupes = Vec::new();
        for i in 0..self.fingerprints.len() {
            for j in (i + 1)..self.fingerprints.len() {
                let sim = self.fingerprints[i].cosine_similarity(&self.fingerprints[j]);
                if sim >= threshold {
                    dupes.push((i, j, sim));
                }
            }
        }
        dupes
    }

    /// K-means clustering on the fingerprint eigenvalues.
    ///
    /// Returns `k` clusters with member indices and centroids.
    pub fn cluster(&self, k: usize, max_iters: usize) -> Result<Vec<Cluster>, SimilarityError> {
        if self.fingerprints.is_empty() {
            return Err(SimilarityError::Empty);
        }
        if k == 0 {
            return Err(SimilarityError::InvalidClusters {
                message: "k must be > 0".into(),
            });
        }
        if k > self.fingerprints.len() {
            return Err(SimilarityError::InvalidClusters {
                message: format!(
                    "k ({}) > number of fingerprints ({})",
                    k,
                    self.fingerprints.len()
                ),
            });
        }

        let dim = self.fingerprints[0].eigenvalues.len();

        // Initialize centroids: pick k distinct fingerprints as centroids.
        let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
        for fp in &self.fingerprints {
            if centroids.len() >= k {
                break;
            }
            let candidate = &fp.eigenvalues;
            let is_dup = centroids.iter().any(|c| {
                let dist: f64 = c.iter().zip(candidate.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
                dist < 1e-12
            });
            if !is_dup {
                centroids.push(candidate.clone());
            }
        }
        // If we couldn't find k distinct, pad with the last fingerprint.
        while centroids.len() < k {
            centroids.push(self.fingerprints.last().unwrap().eigenvalues.clone());
        }

        let mut assignments = vec![0usize; self.fingerprints.len()];

        for _ in 0..max_iters {
            // Assign each fingerprint to nearest centroid.
            let mut changed = false;
            for (i, fp) in self.fingerprints.iter().enumerate() {
                let best = nearest_centroid(&fp.eigenvalues, &centroids);
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }

            // Recompute centroids.
            let mut sums: Vec<Vec<f64>> = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];
            for (i, fp) in self.fingerprints.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (d, val) in fp.eigenvalues.iter().enumerate() {
                    sums[c][d] += val;
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for d in 0..dim {
                        centroids[c][d] = sums[c][d] / counts[c] as f64;
                    }
                }
            }
        }

        // Build result clusters.
        let mut clusters: Vec<Cluster> = (0..k)
            .map(|c| Cluster {
                members: Vec::new(),
                centroid: centroids[c].clone(),
            })
            .collect();
        for (i, &c) in assignments.iter().enumerate() {
            clusters[c].members.push(i);
        }

        // Remove empty clusters.
        Ok(clusters.into_iter().filter(|c| !c.members.is_empty()).collect())
    }
}

impl Default for SimilarityIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the index of the nearest centroid by squared Euclidean distance.
fn nearest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let dist: f64 = point
                .iter()
                .zip(c.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (i, dist)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::SpectralFingerprint;

    fn make_fp(eigenvalues: Vec<f64>) -> SpectralFingerprint {
        SpectralFingerprint::from_eigenvalues(eigenvalues)
    }

    fn make_labeled_fp(eigenvalues: Vec<f64>, label: &str) -> SpectralFingerprint {
        let mut fp = SpectralFingerprint::from_eigenvalues(eigenvalues);
        fp.label = Some(label.into());
        fp
    }

    #[test]
    fn test_empty_index() {
        let idx = SimilarityIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_add_and_get() {
        let mut idx = SimilarityIndex::new();
        let fp = make_fp(vec![1.0, 2.0]);
        let i = idx.add(fp);
        assert_eq!(i, 0);
        assert_eq!(idx.len(), 1);
        assert!(idx.get(0).is_some());
        assert!(idx.get(1).is_none());
    }

    #[test]
    fn test_query_top_k() {
        let mut idx = SimilarityIndex::new();
        let target = make_fp(vec![5.0, 5.0]);
        idx.add(make_fp(vec![4.9, 5.0])); // close
        idx.add(make_fp(vec![1.0, 0.0])); // far
        idx.add(make_fp(vec![5.0, 4.9])); // close
        let results = idx.query_top_k(&target, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn test_query_top_k_empty() {
        let idx = SimilarityIndex::new();
        let fp = make_fp(vec![1.0]);
        assert!(idx.query_top_k(&fp, 1).is_err());
    }

    #[test]
    fn test_query_threshold() {
        let mut idx = SimilarityIndex::new();
        let target = make_fp(vec![5.0, 5.0]);
        idx.add(make_fp(vec![5.0, 5.0])); // identical
        idx.add(make_fp(vec![0.0, 1.0])); // different
        let results = idx.query_threshold(&target, 0.99);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_find_duplicates() {
        let mut idx = SimilarityIndex::new();
        idx.add(make_fp(vec![3.0, 3.0]));
        idx.add(make_fp(vec![3.0, 3.0])); // duplicate
        idx.add(make_fp(vec![0.0, 1.0])); // different
        let dupes = idx.find_duplicates(0.99);
        assert_eq!(dupes.len(), 1);
        assert_eq!(dupes[0].0, 0);
        assert_eq!(dupes[0].1, 1);
    }

    #[test]
    fn test_find_no_duplicates() {
        let mut idx = SimilarityIndex::new();
        idx.add(make_fp(vec![3.0, 0.0]));
        idx.add(make_fp(vec![0.0, 3.0]));
        let dupes = idx.find_duplicates(0.99);
        assert!(dupes.is_empty());
    }

    #[test]
    fn test_cluster_basic() {
        let mut idx = SimilarityIndex::new();
        // Two groups: near (1,0) and near (0,1)
        for _ in 0..3 {
            idx.add(make_fp(vec![1.0, 0.0]));
        }
        for _ in 0..3 {
            idx.add(make_fp(vec![0.0, 1.0]));
        }
        let clusters = idx.cluster(2, 50).unwrap();
        assert_eq!(clusters.len(), 2);
        // Each cluster should have 3 members.
        let total: usize = clusters.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn test_cluster_too_many() {
        let mut idx = SimilarityIndex::new();
        idx.add(make_fp(vec![1.0]));
        let result = idx.cluster(5, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_cluster_zero_k() {
        let mut idx = SimilarityIndex::new();
        idx.add(make_fp(vec![1.0]));
        let result = idx.cluster(0, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_labeled_match() {
        let mut idx = SimilarityIndex::new();
        idx.add(make_labeled_fp(vec![5.0, 5.0], "func_a"));
        let query = make_fp(vec![5.0, 5.0]);
        let results = idx.query_top_k(&query, 1).unwrap();
        assert_eq!(results[0].label.as_deref(), Some("func_a"));
    }

    #[test]
    fn test_default() {
        let idx = SimilarityIndex::default();
        assert!(idx.is_empty());
    }
}
