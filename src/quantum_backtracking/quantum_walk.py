"""Exact discrete-time quantum walk simulator on rooted backtracking trees.

Implements the Hilbert space, local diffusion operators D_x, bipartition
reflections R_A and R_B, and walk operator W = R_B R_A from Section 2 of
Montanaro (2015) / Belovs (2013).

Because R_A and R_B act as direct sums of Householder reflections on disjoint
star subspaces H_x = span({|x>} U {|y> : x -> y}), each walk step executes in
O(T) arithmetic operations and O(T) memory without forming dense T x T matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .backtracking import BacktrackingTree

ComplexVector = List[complex]


@dataclass
class QuantumWalkOperators:
    """Encapsulates Montanaro's tree quantum walk operators R_A, R_B, and W = R_B R_A.

    Attributes:
      tree: The rooted backtracking tree with T vertices (root is node 0).
      depth_param: Positive weighting parameter n used in the root diffusion
        operator D_r = I - 2 |psi_r><psi_r|.
      even_nodes: Vertices in bipartition A (even distance from root, including root).
      odd_nodes: Vertices in bipartition B (odd distance from root).
    """

    tree: BacktrackingTree
    depth_param: float
    even_nodes: List[int]
    odd_nodes: List[int]

    @property
    def dim(self) -> int:
        return self.tree.num_vertices

    def root_state(self) -> ComplexVector:
        """Return the initial basis state |r> = |0>."""
        vec = [0j] * self.dim
        vec[0] = 1.0 + 0j
        return vec

    def _apply_diffusion_in_place(self, state: ComplexVector, node_id: int) -> None:
        """Apply local diffusion operator D_x to `state` in place.

        - If x is marked: D_x = I (no-op).
        - If x == root (and unmarked):
            |psi_r> = (|r> + sqrt(n) * sum_{y: r->y} |y>) / sqrt(1 + d_r * n)
            D_r = I - 2 |psi_r><psi_r|
        - If x != root (and unmarked):
            |psi_x> = (|x> + sum_{y: x->y} |y>) / sqrt(d_x)
            D_x = I - 2 |psi_x><psi_x|
          where d_x = |{y: x->y}| + 1.
        """
        node = self.tree.nodes[node_id]
        if node.is_marked:
            return

        children = node.children_ids
        if node.is_root:
            d_r = len(children)
            if d_r == 0:
                # Root with no children: |psi_r> = |r>, D_r |r> = -|r>
                state[0] = -state[0]
                return
            sqrt_n = math.sqrt(self.depth_param)
            norm_sq = 1.0 + d_r * self.depth_param
            inner = state[0] + sqrt_n * sum(state[c] for c in children)
            factor = 2.0 * inner / norm_sq
            state[0] -= factor
            for c in children:
                state[c] -= factor * sqrt_n
        else:
            d_x = len(children) + 1  # includes parent edge contribution (1 for |x>)
            inner = state[node_id] + sum(state[c] for c in children)
            factor = 2.0 * inner / float(d_x)
            state[node_id] -= factor
            for c in children:
                state[c] -= factor

    def apply_ra(self, state: Sequence[complex]) -> ComplexVector:
        """Apply R_A = direct_sum_{x in A} D_x to a statevector."""
        out = list(state)
        for node_id in self.even_nodes:
            self._apply_diffusion_in_place(out, node_id)
        return out

    def apply_rb(self, state: Sequence[complex]) -> ComplexVector:
        """Apply R_B = |r><r| + direct_sum_{x in B} D_x to a statevector."""
        out = list(state)
        for node_id in self.odd_nodes:
            self._apply_diffusion_in_place(out, node_id)
        return out

    def apply_walk(self, state: Sequence[complex]) -> ComplexVector:
        """Apply one quantum walk step W = R_B R_A."""
        return self.apply_rb(self.apply_ra(state))

    def build_matrix(self, operator: str = "W") -> List[List[complex]]:
        """Materialize a dense T x T matrix for 'R_A', 'R_B', or 'W' (for testing/analysis)."""
        t = self.dim
        cols: List[ComplexVector] = []
        apply_fn = {
            "R_A": self.apply_ra,
            "R_B": self.apply_rb,
            "W": self.apply_walk,
        }.get(operator)
        if apply_fn is None:
            raise ValueError(f"Unknown operator '{operator}'. Expected 'R_A', 'R_B', or 'W'.")

        for j in range(t):
            basis = [0j] * t
            basis[j] = 1.0 + 0j
            cols.append(apply_fn(basis))

        return [[cols[col][row] for col in range(t)] for row in range(t)]


def build_quantum_walk(
    tree: BacktrackingTree,
    depth_param: Optional[float] = None,
) -> QuantumWalkOperators:
    """Construct Montanaro's quantum walk operators for a BacktrackingTree.

    Args:
      tree: The rooted backtracking tree.
      depth_param: Weighting parameter n for D_r. Defaults to max(1, tree.max_depth_bound).
    """
    n_param = float(depth_param if depth_param is not None else max(1, tree.max_depth_bound))
    even_nodes = [node.node_id for node in tree.nodes if node.depth % 2 == 0]
    odd_nodes = [node.node_id for node in tree.nodes if node.depth % 2 == 1]
    return QuantumWalkOperators(
        tree=tree,
        depth_param=n_param,
        even_nodes=even_nodes,
        odd_nodes=odd_nodes,
    )


def vector_norm(vec: Sequence[complex]) -> float:
    """Euclidean (L2) norm of a complex vector."""
    return math.sqrt(sum(abs(v) ** 2 for v in vec))


def inner_product(a: Sequence[complex], b: Sequence[complex]) -> complex:
    """Standard Hermitian inner product <a|b>."""
    return sum(x.conjugate() * y for x, y in zip(a, b))


def construct_marked_eigenvector(
    tree: BacktrackingTree,
    marked_node_id: int,
    depth_param: Optional[float] = None,
) -> ComplexVector:
    """Construct the normalized eigenvalue-1 eigenvector |phi'> from Montanaro Eq. (1).

    For a marked vertex x_0 != root:
      |phi> = sqrt(n) |r> + sum_{x != r, x on path(r -> x_0)} (-1)^{l(x)} |x>
      |phi'> = |phi> / || |phi> ||
    """
    node = tree.nodes[marked_node_id]
    if not node.is_marked:
        raise ValueError(f"Node {marked_node_id} is not marked.")

    n_param = float(depth_param if depth_param is not None else max(1, tree.max_depth_bound))
    vec = [0j] * tree.num_vertices
    path = tree.path_to_root(marked_node_id)
    vec[0] = math.sqrt(n_param) + 0j
    for vid in path[1:]:
        depth = tree.nodes[vid].depth
        vec[vid] = complex((-1) ** depth, 0.0)

    norm = vector_norm(vec)
    return [v / norm for v in vec]


def verify_eigenvector_invariance(
    ops: QuantumWalkOperators,
    marked_node_id: int,
) -> Dict[str, float]:
    """Verify Lemma 6's eigenvector properties for a marked vertex x_0.

    Returns a dictionary containing:
      - ra_residual: || R_A |phi'> - |phi'> ||_2
      - rb_residual: || R_B |phi'> - |phi'> ||_2
      - walk_residual: || W |phi'> - |phi'> ||_2
      - root_overlap: |<r | phi'>| (guaranteed >= 1/sqrt(2) when l(x_0) <= n)
      - root_overlap_sq: |<r | phi'>|^2 (guaranteed >= 1/2 when l(x_0) <= n)
    """
    phi = construct_marked_eigenvector(ops.tree, marked_node_id, ops.depth_param)
    ra_phi = ops.apply_ra(phi)
    rb_phi = ops.apply_rb(phi)
    w_phi = ops.apply_walk(phi)

    ra_res = vector_norm([a - b for a, b in zip(ra_phi, phi)])
    rb_res = vector_norm([a - b for a, b in zip(rb_phi, phi)])
    w_res = vector_norm([a - b for a, b in zip(w_phi, phi)])
    overlap = abs(phi[0])

    return {
        "ra_residual": ra_res,
        "rb_residual": rb_res,
        "walk_residual": w_res,
        "root_overlap": overlap,
        "root_overlap_sq": overlap ** 2,
    }


def verify_unitarity(matrix: Sequence[Sequence[complex]], tol: float = 1e-10) -> Tuple[bool, float]:
    """Check whether a dense matrix U satisfies || U^dagger U - I ||_F <= tol."""
    n = len(matrix)
    sq_err = 0.0
    for i in range(n):
        for j in range(n):
            val = sum(matrix[k][i].conjugate() * matrix[k][j] for k in range(n))
            target = 1.0 + 0j if i == j else 0j
            sq_err += abs(val - target) ** 2
    frob_err = math.sqrt(sq_err)
    return (frob_err <= tol), frob_err
