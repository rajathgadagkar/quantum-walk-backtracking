"""Unit tests for Montanaro's quantum walk operators, eigenvectors, QPE, and search."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_backtracking import (
    KSATInstance,
    NQueensInstance,
    PredicateResult,
    build_backtracking_tree,
    build_quantum_walk,
    detect_marked_vertex,
    find_marked_vertex_binary_search,
    find_unique_marked_vertex,
    qpe_zero_phase_probability,
    verify_eigenvector_invariance,
    verify_unitarity,
)


class TestQuantumWalkAndQPE(unittest.TestCase):
    def test_operators_are_unitary(self) -> None:
        tree = build_backtracking_tree(NQueensInstance(4))
        ops = build_quantum_walk(tree)
        for name in ("R_A", "R_B", "W"):
            mat = ops.build_matrix(name)
            is_u, err = verify_unitarity(mat)
            self.assertTrue(is_u, f"{name} is not unitary (Frobenius error={err})")

    def test_marked_eigenvector_exact_invariance_and_root_overlap(self) -> None:
        tree = build_backtracking_tree(NQueensInstance(4))
        ops = build_quantum_walk(tree, depth_param=4)
        for mid in tree.marked_nodes:
            res = verify_eigenvector_invariance(ops, mid)
            self.assertLess(res["ra_residual"], 1e-12)
            self.assertLess(res["rb_residual"], 1e-12)
            self.assertLess(res["walk_residual"], 1e-12)
            self.assertGreaterEqual(res["root_overlap_sq"], 0.5 - 1e-12)

    def test_detection_sat_vs_unsat(self) -> None:
        bench = Path(__file__).resolve().parent.parent / "benchmarks"
        sat_tree = build_backtracking_tree(KSATInstance.from_dimacs_file(bench / "sample_3sat_sat.cnf"))
        unsat_tree = build_backtracking_tree(KSATInstance.from_dimacs_file(bench / "sample_3sat_unsat.cnf"))

        sat_det = detect_marked_vertex(sat_tree, delta=0.05, beta=0.25, deterministic=True)
        unsat_det = detect_marked_vertex(unsat_tree, delta=0.05, beta=0.25, deterministic=True)

        self.assertTrue(sat_det.detected)
        self.assertGreaterEqual(sat_det.zero_phase_prob, 0.5 - 1e-9)
        self.assertFalse(unsat_det.detected)
        self.assertLess(unsat_det.zero_phase_prob, 0.25)

    def test_qpe_spectrum_sums_to_one(self) -> None:
        tree = build_backtracking_tree(NQueensInstance(4))
        ops = build_quantum_walk(tree)
        qpe = qpe_zero_phase_probability(ops, ancilla_qubits=4, compute_full_spectrum=True)
        assert qpe.phase_probabilities is not None
        total_prob = sum(p for _, p in qpe.phase_probabilities)
        self.assertAlmostEqual(total_prob, 1.0, places=10)

    def test_binary_search_finds_valid_solution_and_rejects_unsat(self) -> None:
        csp_sat = NQueensInstance(4)
        tree_sat = build_backtracking_tree(csp_sat)
        res_sat = find_marked_vertex_binary_search(tree_sat, num_vars=4, known_t_bound=tree_sat.num_vertices)
        self.assertTrue(res_sat.found)
        assert res_sat.node_id is not None
        self.assertEqual(csp_sat.predicate(tree_sat.nodes[res_sat.node_id].assignment), PredicateResult.TRUE)

        bench = Path(__file__).resolve().parent.parent / "benchmarks"
        tree_unsat = build_backtracking_tree(KSATInstance.from_dimacs_file(bench / "sample_3sat_unsat.cnf"))
        res_unsat = find_marked_vertex_binary_search(
            tree_unsat, num_vars=5, known_t_bound=tree_unsat.num_vertices
        )
        self.assertFalse(res_unsat.found)
        self.assertIsNone(res_unsat.node_id)

    def test_unique_marked_vertex_eigenvector_search(self) -> None:
        # 4-variable 3-SAT with a unique satisfying assignment (1, 0, 1, 1)
        clauses = [
            (1, 2, 3),
            (1, -2, 3),
            (1, 2, -3),
            (1, -2, -3),  # forces x1 = 1
            (-1, -2, 4),
            (-1, -2, -4),  # forces x2 = 0
            (-1, 2, 3),   # forces x3 = 1
            (-1, 2, 4),   # forces x4 = 1
        ]
        csp = KSATInstance(num_vars=4, clauses=clauses, use_unit_clause=False)
        tree = build_backtracking_tree(csp)
        self.assertEqual(len(tree.marked_nodes), 1)

        res = find_unique_marked_vertex(tree, num_vars=4, seed=123)
        self.assertTrue(res.found)
        assert res.node_id is not None
        self.assertEqual(csp.predicate(tree.nodes[res.node_id].assignment), PredicateResult.TRUE)


if __name__ == "__main__":
    unittest.main()
