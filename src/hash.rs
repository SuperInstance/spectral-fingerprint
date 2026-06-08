//! Locality-sensitive hashing (LSH) for fast approximate nearest-neighbor search.
//!
//! [`SpectralHash`] hashes eigenvalue vectors into fixed-length bit strings.
//! Fingerprints with similar eigenvalues tend to share hash buckets, enabling
//! sub-linear candidate retrieval before exact comparison.

use crate::fingerprint::{FINGERPRINT_LEN, SpectralFingerprint};
use serde::{Deserialize, Serialize};

/// Number of hash bits.
pub const HASH_BITS: usize = 128;

/// Number of hash tables (bands) for multi-probe LSH.
pub const NUM_TABLES: usize = 8;

/// A locality-sensitive hash bucket key.
pub type BucketKey = u128;

/// A spectral hash for fast approximate similarity search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpectralHash {
    /// The hash tables: each table maps bucket keys to fingerprint indices.
    pub tables: Vec<Vec<(BucketKey, usize)>>,
    /// Random projection vectors used for hashing (one per hash bit).
    pub projections: Vec<Vec<f64>>,
}

/// An LSH search result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LshCandidate {
    /// Index of the candidate fingerprint.
    pub index: usize,
    /// Number of hash tables where this candidate matched.
    pub hit_count: usize,
}

impl SpectralHash {
    /// Create a new `SpectralHash` with deterministic projections.
    pub fn new() -> Self {
        let projections = Self::generate_projections();
        let tables = vec![Vec::new(); NUM_TABLES];
        Self {
            tables,
            projections,
        }
    }

    /// Generate deterministic random projection vectors.
    /// Uses a simple LCG PRNG seeded for reproducibility.
    fn generate_projections() -> Vec<Vec<f64>> {
        let mut rng = LcgRng::new(42);
        (0..HASH_BITS)
            .map(|_| {
                (0..FINGERPRINT_LEN)
                    .map(|_| {
                    rng.next_f64() - 0.5
                    })
                    .collect()
            })
            .collect()
    }

    /// Hash a fingerprint into a `HASH_BITS`-bit value.
    pub fn hash(&self, fp: &SpectralFingerprint) -> BucketKey {
        let mut bits: BucketKey = 0;
        for (i, proj) in self.projections.iter().enumerate() {
            let dot: f64 = fp
                .eigenvalues
                .iter()
                .zip(proj.iter())
                .map(|(a, b)| a * b)
                .sum();
            if dot >= 0.0 {
                bits |= 1 << i;
            }
        }
        bits
    }

    /// Insert a fingerprint index into all hash tables.
    pub fn insert(&mut self, fp: &SpectralFingerprint, index: usize) {
        let full_hash = self.hash(fp);
        let bits_per_table = HASH_BITS / NUM_TABLES;
        for table_idx in 0..NUM_TABLES {
            // Extract a sub-key for this table.
            let shift = (table_idx * bits_per_table) as u128;
            let mask: BucketKey = (1u128 << bits_per_table) - 1;
            let sub_key = (full_hash >> shift) & mask;
            self.tables[table_idx].push((sub_key, index));
        }
    }

    /// Query for candidate fingerprint indices.
    ///
    /// Returns candidates that share at least one bucket with the query.
    pub fn query(&self, fp: &SpectralFingerprint) -> Vec<LshCandidate> {
        let full_hash = self.hash(fp);
        let bits_per_table = HASH_BITS / NUM_TABLES;
        let mut hit_counts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for table_idx in 0..NUM_TABLES {
            let shift = (table_idx * bits_per_table) as u128;
            let mask: BucketKey = (1u128 << bits_per_table) - 1;
            let sub_key = (full_hash >> shift) & mask;
            for (key, index) in &self.tables[table_idx] {
                if *key == sub_key {
                    *hit_counts.entry(*index).or_insert(0) += 1;
                }
            }
        }

        let mut candidates: Vec<LshCandidate> = hit_counts
            .into_iter()
            .map(|(index, hit_count)| LshCandidate { index, hit_count })
            .collect();
        candidates.sort_by_key(|b| std::cmp::Reverse(b.hit_count));
        candidates
    }

    /// Build an LSH index from a slice of fingerprints.
    pub fn build_index(fingerprints: &[SpectralFingerprint]) -> Self {
        let mut lsh = Self::new();
        for (i, fp) in fingerprints.iter().enumerate() {
            lsh.insert(fp, i);
        }
        lsh
    }
}

impl Default for SpectralHash {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple LCG PRNG for deterministic projections.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Constants from Numerical Recipes.
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::SpectralFingerprint;

    fn make_fp(eigenvalues: Vec<f64>) -> SpectralFingerprint {
        SpectralFingerprint::from_eigenvalues(eigenvalues)
    }

    #[test]
    fn test_hash_deterministic() {
        let lsh = SpectralHash::new();
        let fp = make_fp(vec![1.0, 2.0, 3.0]);
        let h1 = lsh.hash(&fp);
        let h2 = lsh.hash(&fp);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_identical_inputs_match() {
        let lsh = SpectralHash::new();
        let fp1 = make_fp(vec![5.0, 5.0, 5.0]);
        let fp2 = make_fp(vec![5.0, 5.0, 5.0]);
        assert_eq!(lsh.hash(&fp1), lsh.hash(&fp2));
    }

    #[test]
    fn test_hash_different_inputs_differ() {
        let lsh = SpectralHash::new();
        let fp1 = make_fp(vec![5.0, 5.0, 5.0]);
        let fp2 = make_fp(vec![0.0, -1.0, -2.0]);
        // It's possible but extremely unlikely that two very different
        // inputs hash identically.
        assert_ne!(lsh.hash(&fp1), lsh.hash(&fp2));
    }

    #[test]
    fn test_insert_and_query() {
        let mut lsh = SpectralHash::new();
        let target = make_fp(vec![5.0, 5.0]);
        lsh.insert(&target, 0);
        let results = lsh.query(&target);
        assert!(!results.is_empty());
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_query_similar_found() {
        let mut lsh = SpectralHash::new();
        let fp1 = make_fp(vec![1.0, 1.0]);
        let fp2 = make_fp(vec![1.01, 1.01]); // very similar
        let fp3 = make_fp(vec![100.0, -50.0]); // very different
        lsh.insert(&fp1, 0);
        lsh.insert(&fp2, 1);
        lsh.insert(&fp3, 2);
        let candidates = lsh.query(&fp1);
        // fp2 (index 1) should be a candidate; fp3 may or may not be.
        assert!(candidates.iter().any(|c| c.index == 1));
    }

    #[test]
    fn test_build_index() {
        let fps: Vec<SpectralFingerprint> = (0..10)
            .map(|i| make_fp(vec![i as f64, i as f64]))
            .collect();
        let lsh = SpectralHash::build_index(&fps);
        assert_eq!(lsh.tables.len(), NUM_TABLES);
        // Each table should have 10 entries.
        for table in &lsh.tables {
            assert_eq!(table.len(), 10);
        }
    }

    #[test]
    fn test_empty_query() {
        let lsh = SpectralHash::new();
        let fp = make_fp(vec![1.0]);
        let results = lsh.query(&fp);
        assert!(results.is_empty());
    }

    #[test]
    fn test_default() {
        let lsh = SpectralHash::default();
        assert_eq!(lsh.tables.len(), NUM_TABLES);
    }

    #[test]
    fn test_projections_dimension() {
        let lsh = SpectralHash::new();
        assert_eq!(lsh.projections.len(), HASH_BITS);
        for proj in &lsh.projections {
            assert_eq!(proj.len(), FINGERPRINT_LEN);
        }
    }
}
