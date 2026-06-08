//! Convert AST-like structures into adjacency matrices.
//!
//! A [`CodeGraph`] represents a simplified AST as an undirected adjacency matrix.
//! Each AST node becomes a [`CodeNode`] with a type label, optional text label,
//! and depth in the tree. Edges connect parent-child relationships.

use serde::{Deserialize, Serialize};

/// A single node in a code graph, representing an AST node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodeNode {
    /// The node type (e.g., "function", "if", "return", "literal").
    pub node_type: String,
    /// An optional label (e.g., variable name, function name).
    pub label: Option<String>,
    /// Depth in the AST (0 = root).
    pub depth: usize,
}

impl CodeNode {
    /// Create a new `CodeNode`.
    pub fn new(node_type: impl Into<String>, label: Option<String>, depth: usize) -> Self {
        Self {
            node_type: node_type.into(),
            label,
            depth,
        }
    }
}

/// A code graph: an adjacency matrix plus metadata for each node.
///
/// The adjacency matrix is stored as a flat `Vec<f64>` in row-major order.
/// For an undirected graph, `matrix[i * n + j] == matrix[j * n + i]`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodeGraph {
    /// Number of nodes.
    pub node_count: usize,
    /// Flat adjacency matrix (row-major), size `node_count * node_count`.
    pub adjacency: Vec<f64>,
    /// Metadata for each node.
    pub nodes: Vec<CodeNode>,
}

/// Errors for graph operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("node index {index} out of bounds (node_count={node_count})")]
    IndexOutOfBounds { index: usize, node_count: usize },
    #[error("empty graph has no nodes")]
    EmptyGraph,
    #[error("invalid edge: {message}")]
    InvalidEdge { message: String },
}

impl CodeGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            node_count: 0,
            adjacency: Vec::new(),
            nodes: Vec::new(),
        }
    }

    /// Create a graph with `n` placeholder nodes and a zero adjacency matrix.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            node_count: n,
            adjacency: vec![0.0; n * n],
            nodes: (0..n)
                .map(|_| CodeNode::new("placeholder", None, 0))
                .collect(),
        }
    }

    /// Add a node and return its index.
    pub fn add_node(&mut self, node: CodeNode) -> usize {
        let idx = self.node_count;
        self.node_count += 1;
        self.nodes.push(node);
        // Extend the adjacency matrix by one row and one column.
        let old = self.node_count - 1;
        let mut new_mat = vec![0.0; self.node_count * self.node_count];
        for r in 0..old {
            for c in 0..old {
                new_mat[r * self.node_count + c] = self.adjacency[r * old + c];
            }
        }
        self.adjacency = new_mat;
        idx
    }

    /// Add an undirected edge between `a` and `b`.
    pub fn add_edge(&mut self, a: usize, b: usize) -> Result<(), GraphError> {
        if a >= self.node_count || b >= self.node_count {
            return Err(GraphError::IndexOutOfBounds {
                index: a.max(b),
                node_count: self.node_count,
            });
        }
        self.adjacency[a * self.node_count + b] = 1.0;
        self.adjacency[b * self.node_count + a] = 1.0;
        Ok(())
    }

    /// Get the adjacency value at (row, col).
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.node_count || col >= self.node_count {
            None
        } else {
            Some(self.adjacency[row * self.node_count + col])
        }
    }

    /// Degree of node `i` (sum of row i in the adjacency matrix).
    pub fn degree(&self, i: usize) -> f64 {
        (0..self.node_count)
            .map(|j| self.adjacency[i * self.node_count + j])
            .sum()
    }

    /// Compute the degree matrix as a flat vector (diagonal only).
    pub fn degree_matrix(&self) -> Vec<f64> {
        (0..self.node_count).map(|i| self.degree(i)).collect()
    }

    /// Compute the Laplacian matrix: L = D - A.
    pub fn laplacian(&self) -> Vec<f64> {
        let n = self.node_count;
        let degrees = self.degree_matrix();
        let mut lap = vec![0.0; n * n];
        for i in 0..n {
            lap[i * n + i] = degrees[i];
            for j in 0..n {
                lap[i * n + j] -= self.adjacency[i * n + j];
            }
        }
        lap
    }

    /// Build a `CodeGraph` from a tree represented as a flat list of nodes
    /// where `parent[i]` is `None` for the root or `Some(parent_index)`.
    pub fn from_tree(nodes: Vec<CodeNode>, parents: &[Option<usize>]) -> Result<Self, GraphError> {
        if nodes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }
        if nodes.len() != parents.len() {
            return Err(GraphError::InvalidEdge {
                message: format!(
                    "nodes len ({}) != parents len ({})",
                    nodes.len(),
                    parents.len()
                ),
            });
        }
        let n = nodes.len();
        let mut graph = Self {
            node_count: n,
            adjacency: vec![0.0; n * n],
            nodes,
        };
        for (child, maybe_parent) in parents.iter().enumerate() {
            if let Some(parent) = maybe_parent {
                if *parent >= n {
                    return Err(GraphError::IndexOutOfBounds {
                        index: *parent,
                        node_count: n,
                    });
                }
                graph.add_edge(child, *parent)?;
            }
        }
        Ok(graph)
    }

    /// Build a `CodeGraph` from a simplified S-expression AST string.
    ///
    /// Parses nested parentheses like `(fn foo (if (return 1) (return 2))`.
    /// Each `(` opens a new node; the first token is the node type.
    pub fn from_sexp(sexp: &str) -> Result<Self, GraphError> {
        let mut nodes: Vec<CodeNode> = Vec::new();
        let mut parents: Vec<Option<usize>> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();
        let mut depth = 0usize;
        let tokens = sexp.split_whitespace().peekable();

        for tok in tokens {
            // Strip leading '(' characters.
            let opens = tok.chars().filter(|c| *c == '(').count();
            let cleaned: String = tok.chars().filter(|c| *c != '(' && *c != ')').collect();

            for _ in 0..opens {
                depth += 1;
            }

            if !cleaned.is_empty() {
                let parent = stack.last().copied();
                let idx = nodes.len();
                let label = if cleaned.starts_with('"') {
                    Some(cleaned.trim_matches('"').to_string())
                } else {
                    None
                };
                nodes.push(CodeNode::new(&cleaned, label, depth));
                parents.push(parent);
                // This node becomes the current context for subsequent children.
                // Only push to stack if it was opened with '('.
                // For simplicity, every token is a potential parent.
                if opens > 0 {
                    stack.push(idx);
                }
            }

            // Count closing ')' to pop stack.
            let closes = tok.chars().filter(|c| *c == ')').count();
            for _ in 0..closes {
                if !stack.is_empty() {
                    stack.pop();
                }
                depth = depth.saturating_sub(1);
            }
        }

        if nodes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }

        Self::from_tree(nodes, &parents)
    }
}

impl Default for CodeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let g = CodeGraph::new();
        assert_eq!(g.node_count, 0);
    }

    #[test]
    fn test_add_nodes() {
        let mut g = CodeGraph::new();
        let a = g.add_node(CodeNode::new("fn", Some("main".into()), 0));
        let b = g.add_node(CodeNode::new("return", None, 1));
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(g.node_count, 2);
    }

    #[test]
    fn test_add_edge() {
        let mut g = CodeGraph::new();
        let a = g.add_node(CodeNode::new("fn", None, 0));
        let b = g.add_node(CodeNode::new("block", None, 1));
        g.add_edge(a, b).unwrap();
        assert_eq!(g.get(a, b), Some(1.0));
        assert_eq!(g.get(b, a), Some(1.0)); // undirected
        assert_eq!(g.get(a, a), Some(0.0)); // no self-loop
    }

    #[test]
    fn test_edge_out_of_bounds() {
        let mut g = CodeGraph::with_capacity(2);
        assert!(g.add_edge(0, 5).is_err());
    }

    #[test]
    fn test_degree() {
        let mut g = CodeGraph::new();
        let a = g.add_node(CodeNode::new("a", None, 0));
        let b = g.add_node(CodeNode::new("b", None, 0));
        let c = g.add_node(CodeNode::new("c", None, 0));
        g.add_edge(a, b).unwrap();
        g.add_edge(a, c).unwrap();
        assert_eq!(g.degree(a), 2.0);
        assert_eq!(g.degree(b), 1.0);
    }

    #[test]
    fn test_laplacian() {
        let mut g = CodeGraph::new();
        let a = g.add_node(CodeNode::new("a", None, 0));
        let b = g.add_node(CodeNode::new("b", None, 0));
        g.add_edge(a, b).unwrap();
        let lap = g.laplacian();
        // L = [[1, -1], [-1, 1]]
        assert_eq!(lap[0 * 2 + 0], 1.0);
        assert_eq!(lap[0 * 2 + 1], -1.0);
        assert_eq!(lap[1 * 2 + 0], -1.0);
        assert_eq!(lap[1 * 2 + 1], 1.0);
    }

    #[test]
    fn test_from_tree() {
        let nodes = vec![
            CodeNode::new("fn", Some("foo".into()), 0),
            CodeNode::new("return", None, 1),
            CodeNode::new("if", None, 1),
        ];
        let parents = vec![None, Some(0), Some(0)];
        let g = CodeGraph::from_tree(nodes, &parents).unwrap();
        assert_eq!(g.node_count, 3);
        assert_eq!(g.get(0, 1), Some(1.0));
        assert_eq!(g.get(0, 2), Some(1.0));
        assert_eq!(g.get(1, 2), Some(0.0)); // siblings not connected
    }

    #[test]
    fn test_from_tree_empty() {
        let result = CodeGraph::from_tree(vec![], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_tree_mismatched_lengths() {
        let nodes = vec![CodeNode::new("a", None, 0)];
        let parents: Vec<Option<usize>> = vec![];
        let result = CodeGraph::from_tree(nodes, &parents);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_capacity() {
        let g = CodeGraph::with_capacity(5);
        assert_eq!(g.node_count, 5);
        assert_eq!(g.adjacency.len(), 25);
        assert!(g.adjacency.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_default() {
        let g = CodeGraph::default();
        assert_eq!(g.node_count, 0);
    }
}
