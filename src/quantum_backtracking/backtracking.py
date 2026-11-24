"""Classical backtracking tree construction (Algorithm 1 of Montanaro 2015).

Explores the rooted tree of valid partial assignments defined by a CSP predicate
P and branching heuristic h, tracking tree size T, maximum depth n, marked
solution vertices, and classical DFS visit counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .csp import CSPInstance, HeuristicFn, PartialAssignment, PredicateResult


@dataclass
class TreeNode:
    """A vertex in the backtracking tree corresponding to a valid partial assignment."""

    node_id: int
    assignment: PartialAssignment
    depth: int
    parent_id: Optional[int]
    children_ids: List[int] = field(default_factory=list)
    is_marked: bool = False

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def degree(self) -> int:
        """Undirected graph degree d_x: |children| + 1 for x != root, |children| for root."""
        return len(self.children_ids) if self.is_root else len(self.children_ids) + 1


@dataclass
class BacktrackingTree:
    """Rooted tree of valid partial assignments explored by Algorithm 1.

    Attributes:
      nodes: List of TreeNode objects indexed by node_id (0 is always the root).
      max_depth_bound: Problem variable count n (upper bound on tree depth).
      oracle_calls: Total predicate evaluations P(x) performed by classical Algorithm 1.
      first_solution_visits: Number of valid vertices visited before the first marked
        vertex is discovered in depth-first order (None if unsatisfiable).
    """

    nodes: List[TreeNode]
    max_depth_bound: int
    oracle_calls: int = 0
    first_solution_visits: Optional[int] = None

    @property
    def num_vertices(self) -> int:
        """Total number of valid vertices T in the backtracking tree."""
        return len(self.nodes)

    @property
    def actual_depth(self) -> int:
        """Maximum distance from the root to any leaf in this tree."""
        return max((node.depth for node in self.nodes), default=0)

    @property
    def marked_nodes(self) -> List[int]:
        """List of node IDs x such that P(x) == TRUE."""
        return [node.node_id for node in self.nodes if node.is_marked]

    @property
    def has_marked(self) -> bool:
        return any(node.is_marked for node in self.nodes)

    def path_to_root(self, node_id: int) -> List[int]:
        """Return vertex IDs along the unique path from root (0) down to node_id."""
        path: List[int] = []
        curr: Optional[int] = node_id
        while curr is not None:
            path.append(curr)
            curr = self.nodes[curr].parent_id
        path.reverse()
        return path

    def subtree(self, root_id: int) -> Tuple["BacktrackingTree", Dict[int, int]]:
        """Extract the rooted subtree descending from root_id.

        Returns:
          (subtree, new_to_old_id_map) where node 0 of subtree corresponds to root_id.
        """
        new_nodes: List[TreeNode] = []
        old_to_new: Dict[int, int] = {}
        new_to_old: Dict[int, int] = {}
        root_depth = self.nodes[root_id].depth

        def dfs(old_id: int, new_parent: Optional[int]) -> int:
            old_node = self.nodes[old_id]
            new_id = len(new_nodes)
            old_to_new[old_id] = new_id
            new_to_old[new_id] = old_id
            copied = TreeNode(
                node_id=new_id,
                assignment=old_node.assignment,
                depth=old_node.depth - root_depth,
                parent_id=new_parent,
                children_ids=[],
                is_marked=old_node.is_marked,
            )
            new_nodes.append(copied)
            for child_old in old_node.children_ids:
                child_new = dfs(child_old, new_id)
                copied.children_ids.append(child_new)
            return new_id

        dfs(root_id, None)
        remaining_depth = max(1, self.max_depth_bound - root_depth)
        return (
            BacktrackingTree(nodes=new_nodes, max_depth_bound=remaining_depth),
            new_to_old,
        )

    def truncate_depth(self, depth_limit: int) -> Tuple["BacktrackingTree", Dict[int, int]]:
        """Return the subtree containing all vertices at distance <= depth_limit from root."""
        new_nodes: List[TreeNode] = []
        new_to_old: Dict[int, int] = {}

        def dfs(old_id: int, new_parent: Optional[int]) -> int:
            old_node = self.nodes[old_id]
            new_id = len(new_nodes)
            new_to_old[new_id] = old_id
            copied = TreeNode(
                node_id=new_id,
                assignment=old_node.assignment,
                depth=old_node.depth,
                parent_id=new_parent,
                children_ids=[],
                is_marked=old_node.is_marked,
            )
            new_nodes.append(copied)
            if old_node.depth < depth_limit:
                for child_old in old_node.children_ids:
                    child_new = dfs(child_old, new_id)
                    copied.children_ids.append(child_new)
            return new_id

        dfs(0, None)
        return (
            BacktrackingTree(nodes=new_nodes, max_depth_bound=max(1, depth_limit)),
            new_to_old,
        )


def build_backtracking_tree(
    csp: CSPInstance,
    heuristic: Optional[HeuristicFn] = None,
    max_nodes: Optional[int] = None,
) -> BacktrackingTree:
    """Construct the exact backtracking tree explored by Algorithm 1.

    Per Montanaro Section 3, the tree contains only valid partial assignments
    (P(x) != FALSE). Internal nodes have P(x) == INDETERMINATE; leaves are either
    marked solutions (P(x) == TRUE) or dead ends whose extensions are all invalid.

    Args:
      csp: The CSP instance defining P, n, and d.
      heuristic: Optional branching heuristic h(x). Defaults to csp.default_heuristic.
      max_nodes: Optional safety cap on the number of valid vertices.

    Returns:
      The constructed BacktrackingTree.
    """
    branch_fn = heuristic or csp.default_heuristic
    nodes: List[TreeNode] = []
    oracle_calls = 0
    first_solution_visits: Optional[int] = None

    root_assignment = PartialAssignment()
    root_res = csp.predicate(root_assignment)
    oracle_calls += 1

    if root_res == PredicateResult.FALSE:
        return BacktrackingTree(
            nodes=[TreeNode(node_id=0, assignment=root_assignment, depth=0, parent_id=None, is_marked=False)],
            max_depth_bound=csp.num_vars,
            oracle_calls=oracle_calls,
            first_solution_visits=None,
        )

    root_marked = root_res == PredicateResult.TRUE
    nodes.append(
        TreeNode(
            node_id=0,
            assignment=root_assignment,
            depth=0,
            parent_id=None,
            is_marked=root_marked,
        )
    )
    if root_marked:
        first_solution_visits = 1
        return BacktrackingTree(
            nodes=nodes,
            max_depth_bound=csp.num_vars,
            oracle_calls=oracle_calls,
            first_solution_visits=first_solution_visits,
        )

    def expand(node_id: int) -> None:
        nonlocal oracle_calls, first_solution_visits
        node = nodes[node_id]
        if node.is_marked or node.assignment.is_complete(csp.num_vars):
            return
        if max_nodes is not None and len(nodes) >= max_nodes:
            return

        var_idx = branch_fn(node.assignment)
        for val in range(csp.domain_size):
            if max_nodes is not None and len(nodes) >= max_nodes:
                return
            child_assign = node.assignment.extend(var_idx, val)
            res = csp.predicate(child_assign)
            oracle_calls += 1
            if res == PredicateResult.FALSE:
                continue

            child_id = len(nodes)
            is_marked = res == PredicateResult.TRUE
            child_node = TreeNode(
                node_id=child_id,
                assignment=child_assign,
                depth=node.depth + 1,
                parent_id=node_id,
                is_marked=is_marked,
            )
            nodes.append(child_node)
            node.children_ids.append(child_id)

            if is_marked and first_solution_visits is None:
                first_solution_visits = len(nodes)

            if not is_marked and not child_assign.is_complete(csp.num_vars):
                expand(child_id)

    expand(0)
    return BacktrackingTree(
        nodes=nodes,
        max_depth_bound=csp.num_vars,
        oracle_calls=oracle_calls,
        first_solution_visits=first_solution_visits,
    )
