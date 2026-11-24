"""Constraint Satisfaction Problem (CSP) abstractions and benchmark families.

Implements the oracle model from Section 1 and Section 3 of Montanaro (2015):
  - Predicate P: D -> {TRUE, FALSE, INDETERMINATE}
  - Branching heuristic h: D -> {1, ..., n}
where partial assignments are represented as ordered sequences of
(variable_index, value) pairs (1-indexed variables, 0-indexed domain values).

Includes canonical CSP families:
  1. k-SAT (with DIMACS CNF parser, NaïveBt ordered heuristic, and DPLL
     unit-clause heuristic).
  2. N-Queens.
  3. Graph k-Coloring (with ordered and minimum-remaining-values heuristics).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


class PredicateResult(Enum):
    """Three-valued predicate output P(x) for a partial assignment x."""

    TRUE = "true"
    FALSE = "false"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class PartialAssignment:
    """Immutable partial assignment represented as a sequence (i_1, v_1), ..., (i_l, v_l).

    Variables are 1-indexed (1..n) and domain values are 0-indexed (0..d-1),
    matching Montanaro's Hilbert-space vertex encoding in Section 3.
    """

    assignments: Tuple[Tuple[int, int], ...] = ()

    @property
    def level(self) -> int:
        """Depth l of this partial assignment in the backtracking tree."""
        return len(self.assignments)

    def to_dict(self) -> Dict[int, int]:
        """Return a mapping from assigned 1-indexed variable to its value."""
        return dict(self.assignments)

    def is_complete(self, num_vars: int) -> bool:
        """Return True iff all n variables have been assigned."""
        return len(self.assignments) == num_vars

    def extend(self, var_index: int, value: int) -> "PartialAssignment":
        """Return a new partial assignment extending this one with (var_index, value)."""
        return PartialAssignment(self.assignments + ((var_index, value),))

    def format_vector(self, num_vars: int) -> str:
        """Format as a string over [d] U {'*'} of length n."""
        mapping = self.to_dict()
        return "".join(str(mapping[i]) if i in mapping else "*" for i in range(1, num_vars + 1))


HeuristicFn = Callable[[PartialAssignment], int]


class CSPInstance(ABC):
    """Abstract base class for a Constraint Satisfaction Problem over [d]^n."""

    num_vars: int
    domain_size: int
    name: str

    @abstractmethod
    def predicate(self, assignment: PartialAssignment) -> PredicateResult:
        """Evaluate the partial-assignment predicate P(x)."""

    def naive_heuristic(self, assignment: PartialAssignment) -> int:
        """Return the lowest unassigned variable index in {1, ..., n}."""
        assigned = {var for var, _ in assignment.assignments}
        for idx in range(1, self.num_vars + 1):
            if idx not in assigned:
                return idx
        raise ValueError("Cannot branch on a complete assignment.")

    def default_heuristic(self, assignment: PartialAssignment) -> int:
        """Default branching heuristic h(x) for this CSP."""
        return self.naive_heuristic(assignment)


class KSATInstance(CSPInstance):
    """Boolean k-SAT instance represented in Conjunctive Normal Form (CNF).

    Each clause is a tuple of nonzero signed integers:
      +i represents literal x_i = 1 (True)
      -i represents literal x_i = 0 (False)
    where i in {1, ..., num_vars}.
    """

    def __init__(
        self,
        num_vars: int,
        clauses: Sequence[Sequence[int]],
        name: str = "k-SAT",
        use_unit_clause: bool = True,
    ) -> None:
        self.num_vars = num_vars
        self.domain_size = 2
        self.clauses: Tuple[Tuple[int, ...], ...] = tuple(tuple(c) for c in clauses)
        self.name = name
        self.use_unit_clause = use_unit_clause

    def predicate(self, assignment: PartialAssignment) -> PredicateResult:
        """Evaluate CNF formula under partial assignment x.

        Returns:
          - FALSE if any clause is falsified (all its literals are False).
          - TRUE if all clauses are satisfied.
          - INDETERMINATE otherwise (provided x is not complete).
        """
        values = assignment.to_dict()
        all_satisfied = True

        for clause in self.clauses:
            clause_sat = False
            has_unassigned = False
            for lit in clause:
                var = abs(lit)
                if var not in values:
                    has_unassigned = True
                else:
                    val = values[var]
                    if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                        clause_sat = True
                        break
            if not clause_sat:
                if not has_unassigned:
                    return PredicateResult.FALSE
                all_satisfied = False

        if all_satisfied:
            return PredicateResult.TRUE
        if assignment.is_complete(self.num_vars):
            return PredicateResult.FALSE
        return PredicateResult.INDETERMINATE

    def unit_clause_heuristic(self, assignment: PartialAssignment) -> int:
        """DPLL unit-clause branching heuristic.

        If there exists an unsatisfied clause with exactly one unassigned literal,
        branch on that literal's variable immediately (forcing unit propagation
        in the backtracking tree). Otherwise fall back to the lowest unassigned
        variable index.
        """
        values = assignment.to_dict()
        for clause in self.clauses:
            clause_sat = False
            unassigned_vars: List[int] = []
            for lit in clause:
                var = abs(lit)
                if var not in values:
                    unassigned_vars.append(var)
                else:
                    val = values[var]
                    if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                        clause_sat = True
                        break
            if not clause_sat and len(unassigned_vars) == 1:
                return unassigned_vars[0]
        return self.naive_heuristic(assignment)

    def default_heuristic(self, assignment: PartialAssignment) -> int:
        if self.use_unit_clause:
            return self.unit_clause_heuristic(assignment)
        return self.naive_heuristic(assignment)

    @classmethod
    def random_ksat(
        cls,
        num_vars: int,
        num_clauses: int,
        k: int = 3,
        seed: Optional[int] = None,
        use_unit_clause: bool = True,
        name: Optional[str] = None,
    ) -> "KSATInstance":
        """Generate a uniformly random k-SAT instance with m clauses on n variables.

        Each clause contains k distinct variables, each negated independently with
        probability 1/2 (matching Appendix B of Montanaro 2015).
        """
        if k > num_vars:
            raise ValueError(f"Clause width k={k} cannot exceed num_vars={num_vars}.")
        rng = random.Random(seed)
        clauses: List[Tuple[int, ...]] = []
        variables = list(range(1, num_vars + 1))
        for _ in range(num_clauses):
            chosen = rng.sample(variables, k)
            clause = tuple(v if rng.random() < 0.5 else -v for v in chosen)
            clauses.append(clause)
        label = name or f"Random-{k}SAT(n={num_vars},m={num_clauses},seed={seed})"
        return cls(num_vars=num_vars, clauses=clauses, name=label, use_unit_clause=use_unit_clause)

    @classmethod
    def from_dimacs(
        cls,
        text: str,
        name: str = "DIMACS-CNF",
        use_unit_clause: bool = True,
    ) -> "KSATInstance":
        """Parse a DIMACS CNF formatted string into a KSATInstance."""
        num_vars, clauses = parse_dimacs_cnf(text)
        return cls(num_vars=num_vars, clauses=clauses, name=name, use_unit_clause=use_unit_clause)

    @classmethod
    def from_dimacs_file(
        cls,
        path: str | Path,
        use_unit_clause: bool = True,
    ) -> "KSATInstance":
        """Load a DIMACS CNF file from disk."""
        file_path = Path(path)
        return cls.from_dimacs(
            file_path.read_text(encoding="utf-8"),
            name=file_path.stem,
            use_unit_clause=use_unit_clause,
        )


def parse_dimacs_cnf(text: str) -> Tuple[int, List[Tuple[int, ...]]]:
    """Parse standard DIMACS CNF text into (num_vars, clauses)."""
    num_vars = 0
    clauses: List[Tuple[int, ...]] = []
    current_clause: List[int] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("c") or line.startswith("%"):
            continue
        if line.startswith("p"):
            parts = line.split()
            if len(parts) >= 4 and parts[1].lower() == "cnf":
                num_vars = int(parts[2])
            continue
        for token in line.split():
            lit = int(token)
            if lit == 0:
                if current_clause:
                    clauses.append(tuple(current_clause))
                    current_clause = []
            else:
                current_clause.append(lit)
                num_vars = max(num_vars, abs(lit))

    if current_clause:
        clauses.append(tuple(current_clause))

    return num_vars, clauses


class NQueensInstance(CSPInstance):
    """N-Queens CSP: variable r in {1..N} specifies the column in {0..N-1} for row r."""

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("Board size n must be at least 1.")
        self.num_vars = n
        self.domain_size = n
        self.name = f"{n}-Queens"

    def predicate(self, assignment: PartialAssignment) -> PredicateResult:
        items = assignment.assignments
        for idx_a in range(len(items)):
            r1, c1 = items[idx_a]
            for idx_b in range(idx_a + 1, len(items)):
                r2, c2 = items[idx_b]
                if c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                    return PredicateResult.FALSE
        if assignment.is_complete(self.num_vars):
            return PredicateResult.TRUE
        return PredicateResult.INDETERMINATE


class GraphColoringInstance(CSPInstance):
    """Graph k-coloring CSP over vertices {1..n} and undirected edges."""

    def __init__(
        self,
        num_vertices: int,
        edges: Iterable[Tuple[int, int]],
        num_colors: int = 3,
        name: str = "GraphColoring",
        use_mrv: bool = True,
    ) -> None:
        self.num_vars = num_vertices
        self.domain_size = num_colors
        self.edges: Tuple[Tuple[int, int], ...] = tuple(
            (min(u, v), max(u, v)) for u, v in edges if u != v
        )
        self.adjacency: Dict[int, List[int]] = {v: [] for v in range(1, num_vertices + 1)}
        for u, v in self.edges:
            self.adjacency[u].append(v)
            self.adjacency[v].append(u)
        self.name = name
        self.use_mrv = use_mrv

    def predicate(self, assignment: PartialAssignment) -> PredicateResult:
        colors = assignment.to_dict()
        for u, v in self.edges:
            if u in colors and v in colors and colors[u] == colors[v]:
                return PredicateResult.FALSE
        if assignment.is_complete(self.num_vars):
            return PredicateResult.TRUE
        return PredicateResult.INDETERMINATE

    def mrv_heuristic(self, assignment: PartialAssignment) -> int:
        """Minimum-Remaining-Values (MRV) heuristic for graph coloring."""
        colors = assignment.to_dict()
        best_var = -1
        best_remaining = self.domain_size + 1
        for v in range(1, self.num_vars + 1):
            if v in colors:
                continue
            used = {colors[nb] for nb in self.adjacency[v] if nb in colors}
            remaining = self.domain_size - len(used)
            if remaining < best_remaining:
                best_remaining = remaining
                best_var = v
        if best_var == -1:
            raise ValueError("Cannot branch on a complete assignment.")
        return best_var

    def default_heuristic(self, assignment: PartialAssignment) -> int:
        if self.use_mrv:
            return self.mrv_heuristic(assignment)
        return self.naive_heuristic(assignment)
