"""Unit tests for Belovs' electrical networks and Section 4 average-case bounds."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_backtracking import (
    KSATInstance,
    NQueensInstance,
    analyze_tree_electrical_network,
    build_backtracking_tree,
    compute_effective_resistance_to_marked,
    construct_unit_flow_to_marked,
    naive_bt_expected_tree_size_bounds,
    power_law_separation_experiment,
    varying_density_3sat_separation,
)


class TestElectricalNetworksAndAverageCase(unittest.TestCase):
    def test_thomson_principle_and_kirchhoff_law(self) -> None:
        tree = build_backtracking_tree(NQueensInstance(5))
        r_eff, _ = compute_effective_resistance_to_marked(tree)
        _, energy, kcl_res = construct_unit_flow_to_marked(tree)

        self.assertTrue(math.isfinite(r_eff))
        self.assertLessEqual(r_eff, tree.actual_depth)
        self.assertAlmostEqual(r_eff, energy, places=10)
        self.assertLess(kcl_res, 1e-12)

    def test_unsat_tree_has_infinite_resistance(self) -> None:
        bench = Path(__file__).resolve().parent.parent / "benchmarks"
        unsat_tree = build_backtracking_tree(KSATInstance.from_dimacs_file(bench / "sample_3sat_unsat.cnf"))
        analysis = analyze_tree_electrical_network(unsat_tree, domain_size=2, num_vars=5)
        self.assertTrue(math.isinf(analysis.effective_resistance))
        self.assertEqual(analysis.num_marked, 0)

    def test_naive_bt_proposition_8_bounds_bracket_exact(self) -> None:
        bounds = naive_bt_expected_tree_size_bounds(num_vars=20, alpha=6.0, k=3)
        self.assertGreater(bounds.exact_expected_vertices, 1.0)
        self.assertLessEqual(bounds.upper_exponent_c, 0.907 / math.sqrt(6.0) + 1e-3)
        self.assertGreaterEqual(bounds.lower_exponent_c_prime, 0.906 / math.sqrt(6.0) - 0.142 / 36.0 - 1e-3)
        self.assertLessEqual(bounds.exact_expected_vertices, bounds.upper_bound_vertices * 1.05)

    def test_power_law_exponential_separation(self) -> None:
        pts = power_law_separation_experiment(beta=-1.75, max_exponents=[6, 14, 22, 30, 40])
        # Classical expectation grows as Omega(M^{0.25}) while quantum expectation stays bounded
        self.assertGreater(pts[-1].expected_classical, 100.0 * pts[0].expected_classical)
        self.assertLess(pts[-1].expected_quantum, 5.0 * pts[0].expected_quantum)
        self.assertGreater(pts[-1].speedup_ratio, 100.0)

    def test_varying_density_3sat_separation(self) -> None:
        pts = varying_density_3sat_separation([10, 20, 30])
        self.assertEqual(len(pts), 3)
        self.assertGreater(pts[-1].expected_classical, pts[0].expected_classical)


if __name__ == "__main__":
    unittest.main()
