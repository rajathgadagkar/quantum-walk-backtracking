"""Average-case complexity analysis and exponential quantum-classical separations.

Implements Section 4 and Appendix B of Montanaro (2015):
  1. Exact expected tree size E(n, m, k) and Proposition 8 asymptotic bounds
     (2^{C' n} <= E <= O(n 2^{C n})) for the NaïveBt algorithm on random k-SAT.
  2. Power-law (heavy-tailed) runtime distribution experiment showing how
     Pr[T = t] ~ t^beta with -2 < beta < -3/2 yields an exponential average-case
     separation between E[T] = Omega(M^{beta+2}) and E[sqrt(T)] = O(1).
  3. Varying-density random 3-SAT distribution (Section 4.1) demonstrating
     C_n = Omega(2^{0.094 n}) vs Q_n = poly(n).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List


@dataclass
class NaiveBtBounds:
    """Exact and Proposition 8 bounds for NaïveBt on random k-SAT."""

    num_vars: int
    num_clauses: int
    alpha: float
    k: int
    exact_expected_vertices: float
    lower_exponent_c_prime: float
    upper_exponent_c: float
    lower_bound_vertices: float
    upper_bound_vertices: float


@dataclass
class AverageCaseComparison:
    """Comparison of expected classical vs. quantum runtime under a distribution."""

    parameter_value: float
    expected_classical: float
    expected_quantum: float
    speedup_ratio: float


def naive_bt_expected_tree_size_bounds(
    num_vars: int,
    alpha: float,
    k: int = 3,
) -> NaiveBtBounds:
    """Compute exact E and Proposition 8 bounds for NaïveBt on random k-SAT.

    From Appendix B of Montanaro (2015):
      E = sum_{l=0}^{n} 2^l * (1 - C(l, k) / (2^k * C(n, k)))^m
      C  = ((2^k * ln 2) / (alpha * k))^(1 / (k - 1)) * (1 - 1/k)
      C' = ((2^k * ln 2) / (alpha * k))^(1 / (k - 1)) * (1 - 1/k - (ln 2)/(alpha * k^2) * ((2^k * ln 2)/(alpha * k))^(1/(k-1)))
    """
    if alpha <= 0:
        raise ValueError("Clause density alpha must be positive.")
    if k < 2 or k > num_vars:
        raise ValueError(f"Require 2 <= k <= num_vars; got k={k}, num_vars={num_vars}.")

    m = int(round(alpha * num_vars))
    denom = float((1 << k) * math.comb(num_vars, k))

    exact_e = 0.0
    for level in range(num_vars + 1):
        p_fail = math.comb(level, k) / denom if level >= k else 0.0
        exact_e += (2.0 ** level) * ((1.0 - p_fail) ** m)

    base = ((float(1 << k) * math.log(2.0)) / (alpha * float(k))) ** (1.0 / float(k - 1))
    c_upper = base * (1.0 - 1.0 / float(k))
    c_lower = base * (1.0 - 1.0 / float(k) - (math.log(2.0) / (alpha * float(k * k))) * base)

    lower_v = 2.0 ** (c_lower * num_vars) if c_lower > 0 else 1.0
    upper_v = float(num_vars + 1) * (2.0 ** (c_upper * num_vars))

    return NaiveBtBounds(
        num_vars=num_vars,
        num_clauses=m,
        alpha=alpha,
        k=k,
        exact_expected_vertices=exact_e,
        lower_exponent_c_prime=c_lower,
        upper_exponent_c=c_upper,
        lower_bound_vertices=lower_v,
        upper_bound_vertices=upper_v,
    )


def power_law_separation_experiment(
    beta: float = -1.75,
    max_exponents: List[int] | None = None,
) -> List[AverageCaseComparison]:
    """Evaluate E[T] vs E[sqrt(T)] for a truncated power-law distribution Pr[T = t] ~ t^beta.

    Uses exact summation over logarithmically binned or integer support t in {1, ..., 2^e}
    (exact sum for small e, Euler-Maclaurin / exact integral tail for large e) so
    that M = 2^e can scale up to 2^40 cleanly without floating-point overflow.

    For -2 < beta < -3/2:
      - Normalization Z(M) = sum_{t=1}^M t^beta converges to zeta(-beta) = O(1).
      - E[T] = (1/Z) sum_{t=1}^M t^{beta+1} = Omega(M^{beta+2}) (exponential in log2 M).
      - E[sqrt(T)] = (1/Z) sum_{t=1}^M t^{beta+0.5} converges to O(1)!
    """
    if max_exponents is None:
        max_exponents = [6, 10, 14, 18, 22, 26, 30]

    exact_cutoff = 4096
    results: List[AverageCaseComparison] = []

    def truncated_p_series(power: float, upper: int) -> float:
        """Compute sum_{t=1}^{upper} t^{power} accurately for arbitrarily large upper."""
        if upper <= exact_cutoff:
            return sum(float(t) ** power for t in range(1, upper + 1))
        head = sum(float(t) ** power for t in range(1, exact_cutoff + 1))
        # Euler-Maclaurin approximation for sum_{t=a+1}^{b} t^p
        a = float(exact_cutoff)
        b = float(upper)
        if abs(power + 1.0) < 1e-12:
            tail = math.log(b / a) + 0.5 * (1.0 / b - 1.0 / a)
        else:
            p1 = power + 1.0
            tail = (b ** p1 - a ** p1) / p1 + 0.5 * (b ** power - a ** power)
        return head + tail

    for exp_val in max_exponents:
        m_cap = 1 << exp_val
        z_norm = truncated_p_series(beta, m_cap)
        e_classical = truncated_p_series(beta + 1.0, m_cap) / z_norm
        e_quantum = truncated_p_series(beta + 0.5, m_cap) / z_norm
        results.append(
            AverageCaseComparison(
                parameter_value=float(exp_val),
                expected_classical=e_classical,
                expected_quantum=e_quantum,
                speedup_ratio=e_classical / max(1e-12, e_quantum),
            )
        )
    return results


def varying_density_3sat_separation(
    n_values: List[int] | None = None,
) -> List[AverageCaseComparison]:
    """Evaluate Montanaro Section 4.1's varying-density 3-SAT separation.

    For each n, defines p_m proportional to 2^{-0.454 * n^{3/2} / sqrt(m)} over
    integer clause counts ceil(16 n / ln 2) < m <= n^3, and computes the
    Proposition 8 lower bound on E[C_n] vs the upper bound on E[Q_n] (omitting
    the poly(n) overhead factor for clean comparison alongside n^{3/2}).
    """
    if n_values is None:
        n_values = [10, 20, 30, 40, 50, 60]

    results: List[AverageCaseComparison] = []
    for n in n_values:
        m_min = math.floor(16.0 * n / math.log(2.0)) + 1
        m_max = n ** 3
        # Evaluate in log-space for numerical stability across large exponents
        # Step over m (exact for small ranges, uniform grid for large m_max)
        if m_max - m_min <= 4000:
            m_grid = list(range(m_min, m_max + 1))
            weights_step = [1.0] * len(m_grid)
        else:
            # Fine near m_min where the exponential mass concentrates, coarser in tail
            near_end = min(m_max, m_min + 2000)
            m_grid = list(range(m_min, near_end + 1))
            weights_step = [1.0] * len(m_grid)
            num_tail = 1000
            step = (m_max - near_end) / float(num_tail)
            for idx in range(1, num_tail + 1):
                m_grid.append(int(round(near_end + idx * step)))
                weights_step.append(step)

        log_unnorm = [-0.454 * (n ** 1.5) / math.sqrt(float(m)) * math.log(2.0) for m in m_grid]
        max_lu = max(log_unnorm)
        unnorm = [math.exp(lu - max_lu) * ws for lu, ws in zip(log_unnorm, weights_step)]
        z_norm = sum(unnorm)

        # Classical lower bound term: p_m * 2^{n * (0.906*sqrt(n/m) - 0.142*(n/m)^2)}
        # Quantum upper bound term: p_m * 2^{0.454 * n^{3/2}/sqrt(m)} * (n^{1.5})
        e_classical = 0.0
        e_quantum = 0.0
        for m, lu, ws in zip(m_grid, log_unnorm, weights_step):
            alpha = float(m) / float(n)
            c_lower_exp = n * (0.906 / math.sqrt(alpha) - 0.142 / (alpha ** 2))
            q_upper_exp = 0.454 * (n ** 1.5) / math.sqrt(float(m))

            log_p_m = (lu - max_lu) - math.log(z_norm)
            e_classical += ws * math.exp(log_p_m + c_lower_exp * math.log(2.0))
            e_quantum += ws * math.exp(log_p_m + q_upper_exp * math.log(2.0)) * (n ** 1.5)

        results.append(
            AverageCaseComparison(
                parameter_value=float(n),
                expected_classical=e_classical,
                expected_quantum=e_quantum,
                speedup_ratio=e_classical / max(1e-12, e_quantum),
            )
        )
    return results
