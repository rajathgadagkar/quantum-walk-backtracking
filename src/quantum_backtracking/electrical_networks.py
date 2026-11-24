"""Belovs' electrical network framework for quantum walks on trees.

Implements exact electrical network computations on rooted backtracking trees
following Belovs (2013, arXiv:1302.3143) and Jarret & Wan (2018, Phys. Rev. A
97, 022337):
  - Exact effective resistance R_eff(r, M) between the root r and the set of
    marked vertices M via series-parallel tree reduction.
  - Minimum-energy unit electrical flow f: E -> R from r to M, verifying
    Kirchhoff's current law and Thomson's principle (Energy(f) == R_eff(r, M)).
  - Theoretical query complexity comparisons: Classical O(T), Grover O(sqrt(d^n)),
    Montanaro Detection O(sqrt(T * n)), and Belovs/Jarret-Wan O(sqrt(T * R_eff)).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

from .backtracking import BacktrackingTree


@dataclass
class ElectricalNetworkAnalysis:
    """Summary of electrical-network parameters and query bounds for a tree."""

    num_vertices: int
    actual_depth: int
    max_depth_bound: int
    num_marked: int
    effective_resistance: float  # math.inf if no marked vertices exist
    flow_energy: float  # equals effective_resistance by Thomson's principle
    kirchhoff_max_residual: float
    classical_tree_queries: int
    grover_queries: float
    montanaro_detection_scale: float  # sqrt(T * n)
    belovs_resistance_scale: float  # sqrt(T * R_eff) (or sqrt(T * n) if UNSAT)


def compute_effective_resistance_to_marked(
    tree: BacktrackingTree,
    edge_resistance: float = 1.0,
) -> Tuple[float, Dict[int, float]]:
    """Compute exact effective resistance R_eff(r, M) on a rooted tree.

    Because a tree has no cycles, series-parallel reduction is exact:
      - If u in M: R(u) = 0.
      - Otherwise: 1 / R(u) = sum_{v: u -> v} 1 / (edge_resistance + R(v)),
        where branches with R(v) = inf contribute 0 conductance.

    Returns:
      (R_eff(root, M), node_resistances_dict)
    """
    node_res: Dict[int, float] = {}

    def dfs(node_id: int) -> float:
        node = tree.nodes[node_id]
        if node.is_marked:
            node_res[node_id] = 0.0
            return 0.0
        conductance = 0.0
        for child_id in node.children_ids:
            child_r = dfs(child_id)
            if math.isfinite(child_r):
                conductance += 1.0 / (edge_resistance + child_r)
        r_u = (1.0 / conductance) if conductance > 0.0 else math.inf
        node_res[node_id] = r_u
        return r_u

    root_r = dfs(0)
    return root_r, node_res


def construct_unit_flow_to_marked(
    tree: BacktrackingTree,
    edge_resistance: float = 1.0,
) -> Tuple[Dict[Tuple[int, int], float], float, float]:
    """Construct the unique minimum-energy unit electrical flow from root r to M.

    Returns:
      (edge_flows, energy, max_kirchhoff_residual)
      where edge_flows maps directed edge (parent, child) -> current f(parent -> child).
      If no marked vertices exist, returns ({}, math.inf, 0.0).
    """
    root_r, node_res = compute_effective_resistance_to_marked(tree, edge_resistance)
    if not math.isfinite(root_r):
        return {}, math.inf, 0.0

    edge_flows: Dict[Tuple[int, int], float] = {}

    def distribute(node_id: int, incoming_current: float) -> None:
        node = tree.nodes[node_id]
        if node.is_marked or incoming_current == 0.0:
            return
        total_cond = 0.0
        branch_conds: Dict[int, float] = {}
        for child_id in node.children_ids:
            child_r = node_res[child_id]
            if math.isfinite(child_r):
                g = 1.0 / (edge_resistance + child_r)
                branch_conds[child_id] = g
                total_cond += g
        if total_cond == 0.0:
            return
        for child_id, g in branch_conds.items():
            flow = incoming_current * (g / total_cond)
            edge_flows[(node_id, child_id)] = flow
            distribute(child_id, flow)

    distribute(0, 1.0)

    energy = sum(edge_resistance * (f ** 2) for f in edge_flows.values())

    # Verify Kirchhoff's current law at every non-marked internal vertex
    max_residual = 0.0
    for node in tree.nodes:
        if node.is_marked:
            continue
        inflow = 1.0 if node.is_root else edge_flows.get((node.parent_id, node.node_id), 0.0)
        outflow = sum(edge_flows.get((node.node_id, c), 0.0) for c in node.children_ids)
        max_residual = max(max_residual, abs(inflow - outflow))

    return edge_flows, energy, max_residual


def analyze_tree_electrical_network(
    tree: BacktrackingTree,
    domain_size: int = 2,
    num_vars: Optional[int] = None,
) -> ElectricalNetworkAnalysis:
    """Compute full electrical-network metrics and theoretical query scales for a tree."""
    n_vars = num_vars if num_vars is not None else tree.max_depth_bound
    t_size = tree.num_vertices
    r_eff, _ = compute_effective_resistance_to_marked(tree)
    _, energy, kcl_res = construct_unit_flow_to_marked(tree)

    grover = math.sqrt(float(domain_size) ** n_vars)
    montanaro_scale = math.sqrt(float(t_size) * float(max(1, n_vars)))
    belovs_scale = (
        math.sqrt(float(t_size) * r_eff) if math.isfinite(r_eff) else montanaro_scale
    )

    return ElectricalNetworkAnalysis(
        num_vertices=t_size,
        actual_depth=tree.actual_depth,
        max_depth_bound=n_vars,
        num_marked=len(tree.marked_nodes),
        effective_resistance=r_eff,
        flow_energy=energy,
        kirchhoff_max_residual=kcl_res,
        classical_tree_queries=t_size,
        grover_queries=grover,
        montanaro_detection_scale=montanaro_scale,
        belovs_resistance_scale=belovs_scale,
    )
