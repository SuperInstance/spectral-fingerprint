//! Structural diff: compare two code graphs and identify changes.
//!
//! [`StructuralDiff`] computes a diff between two [`CodeGraph`] instances,
//! identifying added/removed nodes, estimating edit distance from eigenvalue
//! changes, and generating human-readable reports.

use crate::ast_matrix::CodeGraph;
use crate::fingerprint::SpectralFingerprint;
use serde::{Deserialize, Serialize};

/// A single diff entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DiffEntry {
    /// Node added in the new graph.
    Added {
        index: usize,
        node_type: String,
        label: Option<String>,
    },
    /// Node removed from the old graph.
    Removed {
        index: usize,
        node_type: String,
        label: Option<String>,
    },
    /// Node present in both (matched by position).
    Unchanged {
        index: usize,
        node_type: String,
    },
}

/// A structural diff result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StructuralDiff {
    /// Individual diff entries.
    pub entries: Vec<DiffEntry>,
    /// Number of nodes in the old graph.
    pub old_node_count: usize,
    /// Number of nodes in the new graph.
    pub new_node_count: usize,
    /// Eigenvalue-based edit distance approximation.
    pub spectral_distance: f64,
    /// Cosine similarity of the spectral fingerprints.
    pub spectral_similarity: f64,
}

impl StructuralDiff {
    /// Compute a structural diff between two code graphs.
    pub fn diff(old: &CodeGraph, new: &CodeGraph) -> Self {
        let entries = compute_entries(old, new);

        let old_fp = SpectralFingerprint::from_graph(old).ok();
        let new_fp = SpectralFingerprint::from_graph(new).ok();

        let (spectral_distance, spectral_similarity) = match (&old_fp, &new_fp) {
            (Some(o), Some(n)) => (o.l2_distance(n), o.cosine_similarity(n)),
            _ => (f64::INFINITY, 0.0),
        };

        Self {
            entries,
            old_node_count: old.node_count,
            new_node_count: new.node_count,
            spectral_distance,
            spectral_similarity,
        }
    }

    /// Count added nodes.
    pub fn added_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e, DiffEntry::Added { .. }))
            .count()
    }

    /// Count removed nodes.
    pub fn removed_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e, DiffEntry::Removed { .. }))
            .count()
    }

    /// Count unchanged nodes.
    pub fn unchanged_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e, DiffEntry::Unchanged { .. }))
            .count()
    }

    /// Approximate edit distance: added + removed + 0.5 * structural change.
    pub fn edit_distance(&self) -> f64 {
        let added = self.added_count() as f64;
        let removed = self.removed_count() as f64;
        added + removed + self.spectral_distance
    }

    /// Generate a human-readable diff report.
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Structural Diff: {} → {} nodes",
            self.old_node_count, self.new_node_count
        ));
        lines.push(format!(
            "  Added: {}  Removed: {}  Unchanged: {}",
            self.added_count(),
            self.removed_count(),
            self.unchanged_count()
        ));
        lines.push(format!(
            "  Spectral distance: {:.6}",
            self.spectral_distance
        ));
        lines.push(format!(
            "  Spectral similarity: {:.6}",
            self.spectral_similarity
        ));
        lines.push(format!("  Edit distance (approx): {:.2}", self.edit_distance()));
        lines.push(String::new());

        for entry in &self.entries {
            match entry {
                DiffEntry::Added {
                    index,
                    node_type,
                    label,
                } => {
                    let lbl = label
                        .as_deref()
                        .map_or(String::new(), |l| format!(" ({})", l));
                    lines.push(format!("+ [{:3}] {}{}", index, node_type, lbl));
                }
                DiffEntry::Removed {
                    index,
                    node_type,
                    label,
                } => {
                    let lbl = label
                        .as_deref()
                        .map_or(String::new(), |l| format!(" ({})", l));
                    lines.push(format!("- [{:3}] {}{}", index, node_type, lbl));
                }
                DiffEntry::Unchanged { index, node_type } => {
                    lines.push(format!("  [{:3}] {}", index, node_type));
                }
            }
        }

        lines.join("\n")
    }
}

/// Compute diff entries by matching nodes by type and position.
fn compute_entries(old: &CodeGraph, new: &CodeGraph) -> Vec<DiffEntry> {
    let mut entries = Vec::new();
    let max_len = old.node_count.max(new.node_count);

    for i in 0..max_len {
        match (old.nodes.get(i), new.nodes.get(i)) {
            (Some(o), Some(n)) => {
                if o.node_type == n.node_type {
                    entries.push(DiffEntry::Unchanged {
                        index: i,
                        node_type: o.node_type.clone(),
                    });
                } else {
                    entries.push(DiffEntry::Removed {
                        index: i,
                        node_type: o.node_type.clone(),
                        label: o.label.clone(),
                    });
                    entries.push(DiffEntry::Added {
                        index: i,
                        node_type: n.node_type.clone(),
                        label: n.label.clone(),
                    });
                }
            }
            (Some(o), None) => {
                entries.push(DiffEntry::Removed {
                    index: i,
                    node_type: o.node_type.clone(),
                    label: o.label.clone(),
                });
            }
            (None, Some(n)) => {
                entries.push(DiffEntry::Added {
                    index: i,
                    node_type: n.node_type.clone(),
                    label: n.label.clone(),
                });
            }
            (None, None) => unreachable!(),
        }
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_matrix::{CodeGraph, CodeNode};

    fn make_graph_1() -> CodeGraph {
        let nodes = vec![
            CodeNode::new("fn", Some("foo".into()), 0),
            CodeNode::new("if", None, 1),
            CodeNode::new("return", None, 2),
        ];
        let parents = vec![None, Some(0), Some(1)];
        CodeGraph::from_tree(nodes, &parents).unwrap()
    }

    fn make_graph_2() -> CodeGraph {
        // Same structure, different function name
        let nodes = vec![
            CodeNode::new("fn", Some("bar".into()), 0),
            CodeNode::new("if", None, 1),
            CodeNode::new("return", None, 2),
        ];
        let parents = vec![None, Some(0), Some(1)];
        CodeGraph::from_tree(nodes, &parents).unwrap()
    }

    fn make_graph_3() -> CodeGraph {
        // Different structure
        let nodes = vec![
            CodeNode::new("fn", Some("baz".into()), 0),
            CodeNode::new("for", None, 1),
            CodeNode::new("assign", None, 2),
            CodeNode::new("return", None, 2),
        ];
        let parents = vec![None, Some(0), Some(1), Some(1)];
        CodeGraph::from_tree(nodes, &parents).unwrap()
    }

    #[test]
    fn test_identical_graphs() {
        let g1 = make_graph_1();
        let g2 = make_graph_1();
        let diff = StructuralDiff::diff(&g1, &g2);
        assert_eq!(diff.added_count(), 0);
        assert_eq!(diff.removed_count(), 0);
        assert_eq!(diff.unchanged_count(), 3);
        assert!((diff.spectral_similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similar_graphs() {
        let g1 = make_graph_1();
        let g2 = make_graph_2();
        let diff = StructuralDiff::diff(&g1, &g2);
        // Same structure → high similarity.
        assert!(diff.spectral_similarity > 0.9);
        // First node has different label, but same type → unchanged.
        assert_eq!(diff.unchanged_count(), 3);
    }

    #[test]
    fn test_different_graphs() {
        let g1 = make_graph_1();
        let g3 = make_graph_3();
        let diff = StructuralDiff::diff(&g1, &g3);
        assert!(diff.added_count() > 0);
        assert!(diff.spectral_similarity < 1.0);
    }

    #[test]
    fn test_edit_distance_identical() {
        let g1 = make_graph_1();
        let g2 = make_graph_1();
        let diff = StructuralDiff::diff(&g1, &g2);
        assert!(diff.edit_distance().abs() < 1e-10);
    }

    #[test]
    fn test_edit_distance_increases_with_changes() {
        let g1 = make_graph_1();
        let g2 = make_graph_2();
        let g3 = make_graph_3();
        let d12 = StructuralDiff::diff(&g1, &g2).edit_distance();
        let d13 = StructuralDiff::diff(&g1, &g3).edit_distance();
        // More structural changes → larger edit distance.
        assert!(d13 > d12);
    }

    #[test]
    fn test_report() {
        let g1 = make_graph_1();
        let g3 = make_graph_3();
        let diff = StructuralDiff::diff(&g1, &g3);
        let report = diff.report();
        assert!(report.contains("Structural Diff"));
        assert!(report.contains("Added"));
        assert!(report.contains("Removed"));
    }

    #[test]
    fn test_empty_to_nonempty() {
        let empty = CodeGraph::new();
        let g = make_graph_1();
        let diff = StructuralDiff::diff(&empty, &g);
        assert_eq!(diff.old_node_count, 0);
        assert_eq!(diff.new_node_count, 3);
        assert_eq!(diff.added_count(), 3);
    }

    #[test]
    fn test_nonempty_to_empty() {
        let g = make_graph_1();
        let empty = CodeGraph::new();
        let diff = StructuralDiff::diff(&g, &empty);
        assert_eq!(diff.removed_count(), 3);
        assert_eq!(diff.added_count(), 0);
    }
}
