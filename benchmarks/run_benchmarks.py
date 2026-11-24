#!/usr/bin/env python3
"""Reproducible benchmark suite and SVG figure generator for quantum-walk-backtracking.

Runs exact quantum walk simulations, electrical network analyses, QPE spectral
profiles, and heavy-tailed average-case experiments, saving JSON results to
`results/benchmark_summary.json` and SVG charts to `docs/figures/`.

Usage:
    PYTHONPATH=src python3 benchmarks/run_benchmarks.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_backtracking import (
    GraphColoringInstance,
    KSATInstance,
    NQueensInstance,
    analyze_tree_electrical_network,
    build_backtracking_tree,
    build_quantum_walk,
    detect_marked_vertex,
    find_marked_vertex_binary_search,
    naive_bt_expected_tree_size_bounds,
    power_law_separation_experiment,
    qpe_zero_phase_probability,
    varying_density_3sat_separation,
    verify_eigenvector_invariance,
)


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">\n'
        f'  <rect width="{width}" height="{height}" rx="12" fill="#0f172a"/>\n'
    )


def generate_scaling_svg(records: List[Dict[str, Any]], out_path: Path) -> None:
    """Generate grouped log-scale bar chart comparing query complexities."""
    w, h = 920, 460
    pad_l, pad_r, pad_t, pad_b = 85, 35, 75, 95
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    series_meta = [
        ("classical_T", "Classical Tree (T)", "#f97316"),
        ("grover_sqrt_dn", "Grover (sqrt(d^n))", "#94a3b8"),
        ("montanaro_sqrt_Tn", "Montanaro (sqrt(T·n))", "#38bdf8"),
        ("belovs_sqrt_TReff", "Belovs (sqrt(T·R_eff))", "#34d399"),
    ]

    all_vals = [
        max(1.0, float(rec[key]))
        for rec in records
        for key, _, _ in series_meta
    ]
    max_log = math.ceil(math.log10(max(all_vals) * 1.25))
    min_log = 0.0

    def y_pos(val: float) -> float:
        lv = max(min_log, math.log10(max(1.0, val)))
        frac = (lv - min_log) / max(1e-9, max_log - min_log)
        return pad_t + plot_h * (1.0 - frac)

    lines = [_svg_header(w, h)]
    lines.append(
        '  <text x="460" y="34" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="700">'
        "Query Complexity Scaling Across CSP Benchmarks (Log10 Scale)</text>\n"
    )
    lines.append(
        '  <text x="460" y="54" text-anchor="middle" fill="#94a3b8" font-size="12">'
        "Classical Backtracking Tree Size T vs. Grover, Montanaro Quantum Walk, and Belovs Effective Resistance</text>\n"
    )

    # Horizontal grid lines
    for p in range(int(min_log), int(max_log) + 1):
        yp = y_pos(10.0 ** p)
        lines.append(
            f'  <line x1="{pad_l}" y1="{yp:.1f}" x2="{w - pad_r}" y2="{yp:.1f}" '
            f'stroke="#1e293b" stroke-width="1"/>\n'
        )
        lines.append(
            f'  <text x="{pad_l - 10}" y="{yp + 4:.1f}" text-anchor="end" fill="#cbd5e1" font-size="11">'
            f"10^{p}</text>\n"
        )

    # Axes
    lines.append(
        f'  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#475569" stroke-width="1.5"/>\n'
    )
    lines.append(
        f'  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{w - pad_r}" y2="{pad_t + plot_h}" stroke="#475569" stroke-width="1.5"/>\n'
    )

    n_groups = len(records)
    group_w = plot_w / max(1, n_groups)
    bar_w = min(22.0, (group_w * 0.76) / len(series_meta))
    gap = 3.0

    for g_idx, rec in enumerate(records):
        cx = pad_l + (g_idx + 0.5) * group_w
        total_bars_w = len(series_meta) * bar_w + (len(series_meta) - 1) * gap
        start_x = cx - total_bars_w / 2.0
        for s_idx, (key, _, color) in enumerate(series_meta):
            val = float(rec[key])
            bx = start_x + s_idx * (bar_w + gap)
            by = y_pos(val)
            bh = max(2.0, (pad_t + plot_h) - by)
            lines.append(
                f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                f'rx="3" fill="{color}" opacity="0.92"/>\n'
            )
        label = rec["short_name"]
        lines.append(
            f'  <text x="{cx:.1f}" y="{pad_t + plot_h + 22}" text-anchor="middle" '
            f'fill="#e2e8f0" font-size="11" font-weight="600">{label}</text>\n'
        )
        status = "SAT" if rec["satisfiable"] else "UNSAT"
        lines.append(
            f'  <text x="{cx:.1f}" y="{pad_t + plot_h + 38}" text-anchor="middle" '
            f'fill="#94a3b8" font-size="10">({status}, T={rec["classical_T"]})</text>\n'
        )

    # Legend
    leg_y = h - 24
    leg_x = pad_l + 20
    for _, title, color in series_meta:
        lines.append(f'  <rect x="{leg_x}" y="{leg_y - 10}" width="12" height="12" rx="2" fill="{color}"/>\n')
        lines.append(f'  <text x="{leg_x + 18}" y="{leg_y}" fill="#e2e8f0" font-size="11">{title}</text>\n')
        leg_x += 195

    lines.append("</svg>\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def generate_qpe_spectrum_svg(
    sat_spectrum: List[Tuple[float, float]],
    unsat_spectrum: List[Tuple[float, float]],
    out_path: Path,
) -> None:
    """Generate side-by-side QPE phase distribution chart for SAT vs. UNSAT trees."""
    w, h = 920, 420
    lines = [_svg_header(w, h)]
    lines.append(
        '  <text x="460" y="34" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="700">'
        "Exact QPE Phase Spectrum of W = R_B R_A on Satisfiable vs. Unsatisfiable 3-SAT Trees</text>\n"
    )
    lines.append(
        '  <text x="460" y="54" text-anchor="middle" fill="#94a3b8" font-size="12">'
        "Algorithm 2 distinguishes eigenvalue-1 probability >= 1/2 (SAT) from <= 1/4 (UNSAT) using threshold 3/8</text>\n"
    )

    panels = [
        ("Satisfiable 3-SAT (Marked Vertex Exists)", sat_spectrum, 65, 430, "#34d399"),
        ("Unsatisfiable 3-SAT (No Marked Vertex)", unsat_spectrum, 505, 870, "#f43f5e"),
    ]
    top_y, bot_y = 95, 345
    plot_h = bot_y - top_y
    max_p = 1.0

    for title, spec, x0, x1, color in panels:
        pw = x1 - x0
        lines.append(
            f'  <text x="{(x0 + x1) / 2:.1f}" y="{top_y - 14}" text-anchor="middle" '
            f'fill="#e2e8f0" font-size="13" font-weight="600">{title}</text>\n'
        )
        # Grid & y-axis ticks
        for tick in (0.0, 0.25, 0.375, 0.5, 0.75, 1.0):
            yy = bot_y - (tick / max_p) * plot_h
            is_thresh = abs(tick - 0.375) < 1e-6
            stroke = "#facc15" if is_thresh else "#1e293b"
            dash = ' stroke-dasharray="5,4"' if is_thresh else ""
            lines.append(
                f'  <line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" '
                f'stroke="{stroke}" stroke-width="{"1.5" if is_thresh else "1"}"{dash}/>\n'
            )
            lbl = "3/8" if is_thresh else f"{tick:.2f}"
            lbl_col = "#facc15" if is_thresh else "#94a3b8"
            lines.append(
                f'  <text x="{x0 - 8}" y="{yy + 4:.1f}" text-anchor="end" fill="{lbl_col}" font-size="10">{lbl}</text>\n'
            )

        lines.append(f'  <line x1="{x0}" y1="{top_y}" x2="{x0}" y2="{bot_y}" stroke="#475569" stroke-width="1.5"/>\n')
        lines.append(f'  <line x1="{x0}" y1="{bot_y}" x2="{x1}" y2="{bot_y}" stroke="#475569" stroke-width="1.5"/>\n')

        m = len(spec)
        bar_w = max(2.0, (pw / float(m)) * 0.75)
        for idx, (_, prob) in enumerate(spec):
            bx = x0 + (idx + 0.12) * (pw / float(m))
            bh = (prob / max_p) * plot_h
            by = bot_y - bh
            fill = "#38bdf8" if idx == 0 else color
            lines.append(
                f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{max(1.0, bh):.1f}" '
                f'rx="2" fill="{fill}"/>\n'
            )

        p0 = spec[0][1]
        lines.append(
            f'  <text x="{(x0 + x1) / 2:.1f}" y="{bot_y + 26}" text-anchor="middle" fill="#cbd5e1" font-size="11">'
            f"Ancilla Phase Index j (0 .. {m - 1})  |  Pr[j=0] = {p0:.4f}</text>\n"
        )

    lines.append(
        '  <text x="460" y="400" text-anchor="middle" fill="#facc15" font-size="11">'
        "Dashed line: Chernoff acceptance threshold 3/8 (0.375)  |  Blue bar: zero-phase bin j = 0 (eigenvalue 1)</text>\n"
    )
    lines.append("</svg>\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def generate_power_law_svg(
    points: List[Dict[str, float]],
    beta: float,
    out_path: Path,
) -> None:
    """Generate log-scale plot of E[T] vs E[sqrt(T)] under power-law runtime distribution."""
    w, h = 920, 420
    pad_l, pad_r, pad_t, pad_b = 90, 45, 75, 75
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    xs = [p["log2_M"] for p in points]
    y_cl = [math.log10(max(1.0, p["expected_classical"])) for p in points]
    y_qu = [math.log10(max(1.0, p["expected_quantum"])) for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, math.ceil(max(max(y_cl), max(y_qu)) * 1.15)

    def sx(val: float) -> float:
        return pad_l + ((val - x_min) / max(1e-9, x_max - x_min)) * plot_w

    def sy(log_val: float) -> float:
        return pad_t + plot_h * (1.0 - (log_val - y_min) / max(1e-9, y_max - y_min))

    lines = [_svg_header(w, h)]
    lines.append(
        f'  <text x="460" y="34" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="700">'
        f"Exponential Average-Case Separation Under Heavy-Tailed Power-Law Runtimes (beta = {beta})</text>\n"
    )
    lines.append(
        '  <text x="460" y="54" text-anchor="middle" fill="#94a3b8" font-size="12">'
        "For Pr[T = t] ~ t^beta with -2 < beta < -3/2, E[T] grows exponentially in log2(M) while E[sqrt(T)] converges to O(1)</text>\n"
    )

    for yt in range(int(y_min), int(y_max) + 1):
        yy = sy(float(yt))
        lines.append(f'  <line x1="{pad_l}" y1="{yy:.1f}" x2="{w - pad_r}" y2="{yy:.1f}" stroke="#1e293b" stroke-width="1"/>\n')
        lines.append(f'  <text x="{pad_l - 10}" y="{yy + 4:.1f}" text-anchor="end" fill="#cbd5e1" font-size="11">10^{yt}</text>\n')

    lines.append(f'  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#475569" stroke-width="1.5"/>\n')
    lines.append(f'  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{w - pad_r}" y2="{pad_t + plot_h}" stroke="#475569" stroke-width="1.5"/>\n')

    cl_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, y_cl))
    qu_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, y_qu))
    lines.append(f'  <polyline fill="none" stroke="#f97316" stroke-width="3" points="{cl_pts}"/>\n')
    lines.append(f'  <polyline fill="none" stroke="#38bdf8" stroke-width="3" points="{qu_pts}"/>\n')

    for x, yc, yq in zip(xs, y_cl, y_qu):
        lines.append(f'  <circle cx="{sx(x):.1f}" cy="{sy(yc):.1f}" r="4.5" fill="#f97316"/>\n')
        lines.append(f'  <circle cx="{sx(x):.1f}" cy="{sy(yq):.1f}" r="4.5" fill="#38bdf8"/>\n')
        lines.append(
            f'  <text x="{sx(x):.1f}" y="{pad_t + plot_h + 22}" text-anchor="middle" fill="#cbd5e1" font-size="11">'
            f"2^{int(x)}</text>\n"
        )

    lines.append(
        f'  <text x="460" y="{h - 24}" text-anchor="middle" fill="#cbd5e1" font-size="12">'
        "Support Cutoff M = max(T)</text>\n"
    )
    # Legend box
    lines.append('  <rect x="115" y="92" width="265" height="54" rx="8" fill="#1e293b" opacity="0.9"/>\n')
    lines.append('  <line x1="130" y1="112" x2="158" y2="112" stroke="#f97316" stroke-width="3"/>\n')
    lines.append('  <text x="166" y="116" fill="#f8fafc" font-size="11">Classical Expected E[T] = Omega(M^(beta+2))</text>\n')
    lines.append('  <line x1="130" y1="132" x2="158" y2="132" stroke="#38bdf8" stroke-width="3"/>\n')
    lines.append('  <text x="166" y="136" fill="#f8fafc" font-size="11">Quantum Expected E[sqrt(T)] = O(1)</text>\n')

    lines.append("</svg>\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    benchmarks_dir = ROOT_DIR / "benchmarks"
    results_dir = ROOT_DIR / "results"
    figures_dir = ROOT_DIR / "docs" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    instances = [
        ("3-SAT (DIMACS SAT)", KSATInstance.from_dimacs_file(benchmarks_dir / "sample_3sat_sat.cnf", use_unit_clause=True)),
        ("3-SAT (DIMACS UNSAT)", KSATInstance.from_dimacs_file(benchmarks_dir / "sample_3sat_unsat.cnf", use_unit_clause=True)),
        ("Rand 3-SAT n=8", KSATInstance.random_ksat(8, 34, k=3, seed=7, use_unit_clause=True)),
        ("Rand 3-SAT n=10", KSATInstance.random_ksat(10, 42, k=3, seed=19, use_unit_clause=True)),
        ("4-Queens", NQueensInstance(4)),
        ("5-Queens", NQueensInstance(5)),
        (
            "3-Coloring (K4-e+)",
            GraphColoringInstance(
                num_vertices=6,
                edges=[(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 4), (1, 5)],
                num_colors=3,
                name="6-Vertex-3Coloring",
                use_mrv=True,
            ),
        ),
    ]

    benchmark_records: List[Dict[str, Any]] = []
    for short_name, csp in instances:
        tree = build_backtracking_tree(csp)
        elec = analyze_tree_electrical_network(tree, domain_size=csp.domain_size, num_vars=csp.num_vars)
        det = detect_marked_vertex(tree, delta=0.05, beta=0.25, deterministic=True)

        eig_check = None
        if tree.marked_nodes:
            ops = build_quantum_walk(tree, depth_param=max(1, csp.num_vars))
            eig_check = verify_eigenvector_invariance(ops, tree.marked_nodes[0])

        search_res = find_marked_vertex_binary_search(
            tree,
            num_vars=csp.num_vars,
            known_t_bound=tree.num_vertices,
            delta=0.05,
            beta=0.25,
            deterministic=True,
        )

        record: Dict[str, Any] = {
            "short_name": short_name,
            "name": csp.name,
            "num_vars": csp.num_vars,
            "domain_size": csp.domain_size,
            "classical_T": tree.num_vertices,
            "actual_depth": tree.actual_depth,
            "num_marked": len(tree.marked_nodes),
            "satisfiable": tree.has_marked,
            "effective_resistance": (
                round(elec.effective_resistance, 4) if math.isfinite(elec.effective_resistance) else None
            ),
            "grover_sqrt_dn": round(elec.grover_queries, 2),
            "montanaro_sqrt_Tn": round(elec.montanaro_detection_scale, 2),
            "belovs_sqrt_TReff": round(elec.belovs_resistance_scale, 2),
            "qpe_zero_phase_prob": round(det.zero_phase_prob, 5),
            "qpe_detected": det.detected,
            "qpe_ancilla_qubits": det.ancilla_qubits,
            "search_found": search_res.found,
            "search_assignment": search_res.assignment_str,
            "eigenvector_walk_residual": (
                eig_check["walk_residual"] if eig_check is not None else None
            ),
        }
        benchmark_records.append(record)

    # Full QPE phase spectra for SAT vs UNSAT DIMACS 3-SAT
    sat_tree = build_backtracking_tree(
        KSATInstance.from_dimacs_file(benchmarks_dir / "sample_3sat_sat.cnf", use_unit_clause=True)
    )
    unsat_tree = build_backtracking_tree(
        KSATInstance.from_dimacs_file(benchmarks_dir / "sample_3sat_unsat.cnf", use_unit_clause=True)
    )
    sat_spec = qpe_zero_phase_probability(build_quantum_walk(sat_tree), ancilla_qubits=5, compute_full_spectrum=True)
    unsat_spec = qpe_zero_phase_probability(build_quantum_walk(unsat_tree), ancilla_qubits=5, compute_full_spectrum=True)

    # Power-law & Proposition 8 experiments
    beta = -1.75
    power_law_raw = power_law_separation_experiment(beta=beta, max_exponents=[6, 10, 14, 18, 22, 26, 30])
    power_law_data = [
        {
            "log2_M": p.parameter_value,
            "expected_classical": round(p.expected_classical, 4),
            "expected_quantum": round(p.expected_quantum, 4),
            "speedup_ratio": round(p.speedup_ratio, 4),
        }
        for p in power_law_raw
    ]

    naive_bt_checks = [
        naive_bt_expected_tree_size_bounds(num_vars=20, alpha=a, k=3).__dict__
        for a in (4.0, 6.0, 10.0, 20.0)
    ]
    varying_density = [
        {
            "n": item.parameter_value,
            "expected_classical_lb": item.expected_classical,
            "expected_quantum_ub": item.expected_quantum,
            "speedup_ratio": item.speedup_ratio,
        }
        for item in varying_density_3sat_separation([10, 20, 30, 40])
    ]

    summary = {
        "benchmarks": benchmark_records,
        "qpe_spectrum_sat_p0": sat_spec.zero_phase_prob,
        "qpe_spectrum_unsat_p0": unsat_spec.zero_phase_prob,
        "power_law_beta": beta,
        "power_law_separation": power_law_data,
        "naive_bt_proposition_8": naive_bt_checks,
        "varying_density_3sat": varying_density,
    }

    (results_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    generate_scaling_svg(benchmark_records, figures_dir / "quantum_vs_classical_scaling.svg")
    assert sat_spec.phase_probabilities is not None and unsat_spec.phase_probabilities is not None
    generate_qpe_spectrum_svg(
        sat_spec.phase_probabilities,
        unsat_spec.phase_probabilities,
        figures_dir / "phase_estimation_spectrum.svg",
    )
    generate_power_law_svg(power_law_data, beta, figures_dir / "power_law_runtime_separation.svg")

    print("Benchmarks completed successfully.")
    for rec in benchmark_records:
        print(
            f"  {rec['short_name']:20s} | SAT={str(rec['satisfiable']):5s} | "
            f"T={rec['classical_T']:4d} | sqrt(Tn)={rec['montanaro_sqrt_Tn']:6.2f} | "
            f"p0={rec['qpe_zero_phase_prob']:.4f} | detected={rec['qpe_detected']} | "
            f"search_ok={rec['search_found'] == rec['satisfiable']}"
        )


if __name__ == "__main__":
    main()
