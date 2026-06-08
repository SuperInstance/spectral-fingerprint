//! Batch processing: fingerprint directories of source files.
//!
//! [`BatchProcessor`] processes multiple "files" (represented as AST sexp strings),
//! builds fingerprints for all functions, and detects duplicates/near-duplicates.

use crate::ast_matrix::CodeGraph;
use crate::fingerprint::SpectralFingerprint;
use crate::hash::SpectralHash;
use crate::similarity::{SimilarityIndex, SimilarityMatch};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A function extracted from a source file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionEntry {
    /// Function name or identifier.
    pub name: String,
    /// Source file path (logical).
    pub file: String,
    /// The AST as an s-expression string.
    pub sexp: String,
}

/// A duplicate pair detected during batch processing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DuplicatePair {
    /// First function.
    pub function_a: String,
    /// Second function.
    pub function_b: String,
    /// Cosine similarity score.
    pub similarity: f64,
    /// File of function_a.
    pub file_a: String,
    /// File of function_b.
    pub file_b: String,
}

/// Results of a batch processing run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchResult {
    /// Total number of functions processed.
    pub total_functions: usize,
    /// Number of unique fingerprints.
    pub unique_count: usize,
    /// Detected duplicate pairs.
    pub duplicates: Vec<DuplicatePair>,
    /// Number of near-duplicate clusters.
    pub cluster_count: usize,
}

/// A batch processor for code similarity detection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchProcessor {
    /// Registered functions.
    pub functions: Vec<FunctionEntry>,
    /// Computed fingerprints.
    pub fingerprints: Vec<SpectralFingerprint>,
    /// Similarity threshold for duplicate detection.
    pub duplicate_threshold: f64,
    /// Near-duplicate threshold (lower than duplicate).
    pub near_duplicate_threshold: f64,
}

/// Errors for batch processing.
#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    #[error("failed to parse sexp for function '{name}': {error}")]
    ParseFailed { name: String, error: String },
    #[error("no functions registered")]
    NoFunctions,
}

impl BatchProcessor {
    /// Create a new batch processor with default thresholds.
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            fingerprints: Vec::new(),
            duplicate_threshold: 0.98,
            near_duplicate_threshold: 0.90,
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(duplicate: f64, near_duplicate: f64) -> Self {
        Self {
            functions: Vec::new(),
            fingerprints: Vec::new(),
            duplicate_threshold: duplicate,
            near_duplicate_threshold: near_duplicate,
        }
    }

    /// Register a function for processing.
    pub fn register(&mut self, name: impl Into<String>, file: impl Into<String>, sexp: impl Into<String>) {
        self.functions.push(FunctionEntry {
            name: name.into(),
            file: file.into(),
            sexp: sexp.into(),
        });
    }

    /// Process all registered functions: build graphs, compute fingerprints.
    pub fn process(&mut self) -> Result<BatchResult, BatchError> {
        if self.functions.is_empty() {
            return Err(BatchError::NoFunctions);
        }

        self.fingerprints.clear();

        for func in &self.functions {
            let graph = CodeGraph::from_sexp(&func.sexp).map_err(|e| BatchError::ParseFailed {
                name: func.name.clone(),
                error: e.to_string(),
            })?;
            let mut fp = SpectralFingerprint::from_graph(&graph).map_err(|_| {
                BatchError::ParseFailed {
                    name: func.name.clone(),
                    error: "empty graph".into(),
                }
            })?;
            fp.label = Some(format!("{}:{}", func.file, func.name));
            self.fingerprints.push(fp);
        }

        let mut index = SimilarityIndex::new();
        for fp in &self.fingerprints {
            index.add(fp.clone());
        }

        // Find duplicates.
        let dupe_pairs = index.find_duplicates(self.near_duplicate_threshold);
        let duplicates: Vec<DuplicatePair> = dupe_pairs
            .iter()
            .filter(|(_, _, sim)| *sim >= self.near_duplicate_threshold)
            .map(|(a, b, sim)| {
                let fa = &self.functions[*a];
                let fb = &self.functions[*b];
                DuplicatePair {
                    function_a: fa.name.clone(),
                    function_b: fb.name.clone(),
                    similarity: *sim,
                    file_a: fa.file.clone(),
                    file_b: fb.file.clone(),
                }
            })
            .collect();

        let exact_dupes: Vec<_> = duplicates
            .iter()
            .filter(|d| d.similarity >= self.duplicate_threshold)
            .collect();
        let unique_count = self.fingerprints.len() - (exact_dupes.len().min(self.fingerprints.len() / 2));

        // Simple cluster count: count connected components of near-duplicates.
        let cluster_count = count_clusters(&duplicates, self.functions.len());

        Ok(BatchResult {
            total_functions: self.functions.len(),
            unique_count,
            duplicates,
            cluster_count,
        })
    }

    /// Incrementally add functions and re-detect duplicates only for new entries.
    pub fn update(&mut self, new_functions: Vec<FunctionEntry>) -> Result<Vec<DuplicatePair>, BatchError> {
        let start_idx = self.functions.len();
        self.functions.extend(new_functions);

        // Compute fingerprints for new functions only.
        let new_fps: Result<Vec<_>, _> = self.functions[start_idx..]
            .iter()
            .map(|func| {
                let graph =
                    CodeGraph::from_sexp(&func.sexp).map_err(|e| BatchError::ParseFailed {
                        name: func.name.clone(),
                        error: e.to_string(),
                    })?;
                let mut fp = SpectralFingerprint::from_graph(&graph).map_err(|_| {
                    BatchError::ParseFailed {
                        name: func.name.clone(),
                        error: "empty graph".into(),
                    }
                })?;
                fp.label = Some(format!("{}:{}", func.file, func.name));
                Ok(fp)
            })
            .collect();

        let new_fps = new_fps?;
        self.fingerprints.extend(new_fps);

        // Check new fingerprints against all existing.
        let mut new_dupes = Vec::new();
        for new_i in start_idx..self.fingerprints.len() {
            for old_i in 0..new_i {
                let sim = self.fingerprints[old_i].cosine_similarity(&self.fingerprints[new_i]);
                if sim >= self.near_duplicate_threshold {
                    let fa = &self.functions[old_i];
                    let fb = &self.functions[new_i];
                    new_dupes.push(DuplicatePair {
                        function_a: fa.name.clone(),
                        function_b: fb.name.clone(),
                        similarity: sim,
                        file_a: fa.file.clone(),
                        file_b: fb.file.clone(),
                    });
                }
            }
        }

        Ok(new_dupes)
    }

    /// Build an LSH index for fast approximate queries.
    pub fn build_lsh_index(&self) -> SpectralHash {
        SpectralHash::build_index(&self.fingerprints)
    }

    /// Query the LSH index for approximate nearest neighbors.
    pub fn query_approximate(
        &self,
        lsh: &SpectralHash,
        query: &SpectralFingerprint,
        exact_threshold: f64,
    ) -> Vec<SimilarityMatch> {
        let candidates = lsh.query(query);
        candidates
            .into_iter()
            .filter_map(|c| {
                let fp = &self.fingerprints[c.index];
                let score = query.cosine_similarity(fp);
                if score >= exact_threshold {
                    Some(SimilarityMatch {
                        index: c.index,
                        score,
                        label: fp.label.clone(),
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Count connected components (clusters) from duplicate pairs using union-find.
fn count_clusters(duplicates: &[DuplicatePair], _total: usize) -> usize {
    if duplicates.is_empty() {
        return 0;
    }

    // Map function names to indices.
    let mut name_to_idx: HashMap<String, usize> = HashMap::new();
    let mut idx = 0;
    for d in duplicates {
        if !name_to_idx.contains_key(&d.function_a) {
            name_to_idx.insert(d.function_a.clone(), idx);
            idx += 1;
        }
        if !name_to_idx.contains_key(&d.function_b) {
            name_to_idx.insert(d.function_b.clone(), idx);
            idx += 1;
        }
    }

    let n = name_to_idx.len();
    let mut parent: Vec<usize> = (0..n).collect();

    let find = |parent: &mut Vec<usize>, x: usize| -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        // Path compression.
        let mut current = x;
        while parent[current] != root {
            let next = parent[current];
            parent[current] = root;
            current = next;
        }
        root
    };

    let union = |parent: &mut Vec<usize>, a: usize, b: usize| {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    };

    for d in duplicates {
        let a = name_to_idx[&d.function_a];
        let b = name_to_idx[&d.function_b];
        union(&mut parent, a, b);
    }

    // Count unique roots.
    let mut roots = std::collections::HashSet::new();
    for i in 0..n {
        roots.insert(find(&mut parent, i));
    }

    roots.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_processor() -> BatchProcessor {
        BatchProcessor::with_thresholds(0.98, 0.90)
    }

    #[test]
    fn test_empty_processor() {
        let mut bp = BatchProcessor::new();
        assert!(bp.process().is_err());
    }

    #[test]
    fn test_single_function() {
        let mut bp = make_processor();
        bp.register("foo", "test.rs", "(fn foo (return 1))");
        let result = bp.process().unwrap();
        assert_eq!(result.total_functions, 1);
        assert_eq!(result.duplicates.len(), 0);
    }

    #[test]
    fn test_identical_functions() {
        let mut bp = make_processor();
        bp.register("foo", "a.rs", "(fn foo (if (return 1) (return 2)))");
        bp.register("bar", "b.rs", "(fn bar (if (return 1) (return 2)))");
        let result = bp.process().unwrap();
        assert_eq!(result.total_functions, 2);
        assert!(!result.duplicates.is_empty());
    }

    #[test]
    fn test_different_functions() {
        let mut bp = make_processor();
        bp.register("foo", "a.rs", "(fn foo (return 1))");
        bp.register("bar", "b.rs", "(for (assign x))");
        let result = bp.process().unwrap();
        // These may or may not be duplicates depending on spectral similarity.
        assert_eq!(result.total_functions, 2);
    }

    #[test]
    fn test_incremental_update() {
        let mut bp = make_processor();
        bp.register("foo", "a.rs", "(fn foo (if (return 1) (return 2)))");
        let _ = bp.process().unwrap();

        let new_funcs = vec![FunctionEntry {
            name: "bar".into(),
            file: "b.rs".into(),
            sexp: "(fn bar (if (return 1) (return 2)))".into(),
        }];
        let dupes = bp.update(new_funcs).unwrap();
        // Should detect bar as duplicate of foo.
        assert!(!dupes.is_empty());
    }

    #[test]
    fn test_lsh_query() {
        let mut bp = make_processor();
        bp.register("foo", "a.rs", "(fn foo (if (return 1) (return 2)))");
        bp.register("bar", "b.rs", "(for (assign x))");
        bp.process().unwrap();

        let lsh = bp.build_lsh_index();
        // Query with foo's fingerprint.
        let query = bp.fingerprints[0].clone();
        let results = bp.query_approximate(&lsh, &query, 0.0);
        // Should get at least the exact match.
        assert!(!results.is_empty());
    }

    #[test]
    fn test_batch_result_report() {
        let mut bp = make_processor();
        bp.register("foo", "a.rs", "(fn foo (return 1))");
        bp.register("bar", "b.rs", "(fn bar (return 1))");
        let result = bp.process().unwrap();
        // Verify the result structure.
        assert_eq!(result.total_functions, 2);
        assert!(result.unique_count <= 2);
    }

    #[test]
    fn test_register_multiple() {
        let mut bp = make_processor();
        for i in 0..5 {
            bp.register(format!("func_{}", i), "test.rs", format!("(fn f{})", i));
        }
        let result = bp.process().unwrap();
        assert_eq!(result.total_functions, 5);
    }

    #[test]
    fn test_default_processor() {
        let bp = BatchProcessor::default();
        assert!(bp.functions.is_empty());
        assert_eq!(bp.duplicate_threshold, 0.98);
    }

    #[test]
    fn test_cluster_count_no_dupes() {
        let count = count_clusters(&[], 5);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_cluster_count_one_cluster() {
        let dupes = vec![DuplicatePair {
            function_a: "a".into(),
            function_b: "b".into(),
            similarity: 0.95,
            file_a: "x.rs".into(),
            file_b: "y.rs".into(),
        }];
        let count = count_clusters(&dupes, 2);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_cluster_count_two_clusters() {
        let dupes = vec![
            DuplicatePair {
                function_a: "a".into(),
                function_b: "b".into(),
                similarity: 0.95,
                file_a: "x.rs".into(),
                file_b: "y.rs".into(),
            },
            DuplicatePair {
                function_a: "c".into(),
                function_b: "d".into(),
                similarity: 0.92,
                file_a: "z.rs".into(),
                file_b: "w.rs".into(),
            },
        ];
        let count = count_clusters(&dupes, 4);
        assert_eq!(count, 2);
    }
}
