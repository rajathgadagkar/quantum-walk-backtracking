"""Quantum Walk Speedup of Backtracking Algorithms.

A clean, zero-dependency Python implementation and analytical toolkit for
Ashley Montanaro's discrete-time quantum walk speedup of classical backtracking
algorithms (arXiv:1509.02374 / Theory of Computing 14(15), 2018), together with
Belovs' electrical network analysis, phase estimation detection/search, and
heavy-tailed average-case complexity simulations.

Author: Rajath Gadagkar <rajathgadagkar@gmail.com>
"""

from .average_case import (
    AverageCaseComparison,
    NaiveBtBounds,
    naive_bt_expected_tree_size_bounds,
    power_law_separation_experiment,
    varying_density_3sat_separation,
)
from .backtracking import BacktrackingTree, TreeNode, build_backtracking_tree
from .csp import (
    CSPInstance,
    GraphColoringInstance,
    KSATInstance,
    NQueensInstance,
    PartialAssignment,
    PredicateResult,
    parse_dimacs_cnf,
)
from .electrical_networks import (
    ElectricalNetworkAnalysis,
    analyze_tree_electrical_network,
    compute_effective_resistance_to_marked,
    construct_unit_flow_to_marked,
)
from .phase_estimation import (
    DetectionResult,
    SearchResult,
    detect_marked_vertex,
    find_marked_vertex_binary_search,
    find_unique_marked_vertex,
    qpe_zero_phase_probability,
)
from .quantum_walk import (
    QuantumWalkOperators,
    build_quantum_walk,
    construct_marked_eigenvector,
    verify_eigenvector_invariance,
    verify_unitarity,
)

__all__ = [
    "AverageCaseComparison",
    "BacktrackingTree",
    "CSPInstance",
    "DetectionResult",
    "ElectricalNetworkAnalysis",
    "GraphColoringInstance",
    "KSATInstance",
    "NQueensInstance",
    "NaiveBtBounds",
    "PartialAssignment",
    "PredicateResult",
    "QuantumWalkOperators",
    "SearchResult",
    "TreeNode",
    "analyze_tree_electrical_network",
    "build_backtracking_tree",
    "build_quantum_walk",
    "compute_effective_resistance_to_marked",
    "construct_marked_eigenvector",
    "construct_unit_flow_to_marked",
    "detect_marked_vertex",
    "find_marked_vertex_binary_search",
    "find_unique_marked_vertex",
    "naive_bt_expected_tree_size_bounds",
    "parse_dimacs_cnf",
    "power_law_separation_experiment",
    "qpe_zero_phase_probability",
    "varying_density_3sat_separation",
    "verify_eigenvector_invariance",
    "verify_unitarity",
]
