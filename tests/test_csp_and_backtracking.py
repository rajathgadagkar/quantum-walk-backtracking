"""Unit tests for CSP models, DIMACS CNF parsing, and classical backtracking trees."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_backtracking import (
    GraphColoringInstance,
    KSATInstance,
    NQueensInstance,
    PartialAssignment,
    PredicateResult,
    build_backtracking_tree,
)


class TestCSPAndBacktracking(unittest.TestCase):
    def test_dimacs_sat_and_unsat(self) -> None:
        bench = Path(__file__).resolve().parent.parent / "benchmarks"
        sat_csp = KSATInstance.from_dimacs_file(bench / "sample_3sat_sat.cnf")
        unsat_csp = KSATInstance.from_dimacs_file(bench / "sample_3sat_unsat.cnf")

        sat_tree = build_backtracking_tree(sat_csp)
        unsat_tree = build_backtracking_tree(unsat_csp)

        self.assertTrue(sat_tree.has_marked)
        self.assertFalse(unsat_tree.has_marked)
        for mid in sat_tree.marked_nodes:
            self.assertEqual(sat_csp.predicate(sat_tree.nodes[mid].assignment), PredicateResult.TRUE)

    def test_n_queens_known_solution_counts(self) -> None:
        tree_4 = build_backtracking_tree(NQueensInstance(4))
        tree_5 = build_backtracking_tree(NQueensInstance(5))
        self.assertEqual(len(tree_4.marked_nodes), 2)
        self.assertEqual(len(tree_5.marked_nodes), 10)
        self.assertLess(tree_4.num_vertices, 4 ** 4)

    def test_graph_coloring_triangle_and_k4(self) -> None:
        triangle = GraphColoringInstance(3, [(1, 2), (2, 3), (3, 1)], num_colors=3)
        k4 = GraphColoringInstance(4, [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)], num_colors=3)
        self.assertEqual(len(build_backtracking_tree(triangle).marked_nodes), 6)
        self.assertFalse(build_backtracking_tree(k4).has_marked)

    def test_unit_clause_prunes_vs_naive(self) -> None:
        csp_unit = KSATInstance.random_ksat(8, 34, k=3, seed=7, use_unit_clause=True)
        csp_naive = KSATInstance.random_ksat(8, 34, k=3, seed=7, use_unit_clause=False)
        t_unit = build_backtracking_tree(csp_unit)
        t_naive = build_backtracking_tree(csp_naive)
        self.assertEqual(t_unit.has_marked, t_naive.has_marked)
        self.assertLessEqual(t_unit.num_vertices, t_naive.num_vertices)

    def test_partial_assignment_format(self) -> None:
        pa = PartialAssignment(((2, 1), (4, 0)))
        self.assertEqual(pa.level, 2)
        self.assertEqual(pa.format_vector(4), "*1*0")


if __name__ == "__main__":
    unittest.main()
