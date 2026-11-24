"""Quantum Phase Estimation (QPE), Detection (Algorithm 2), and Search (Theorem 2).

Implements:
  - Exact QPE simulation on W = R_B R_A starting from |r>, computing both the
    eigenvalue-1 acceptance probability and the full phase distribution.
  - Algorithm 2 (Theorem 1): Marked vertex detection via majority vote over
    K = ceil(gamma * log(1/delta)) QPE trials with precision beta / sqrt(T * n).
  - Section 2.1 (Theorem 2): General marked vertex search via subtree binary
    search with exponential doubling on unknown T.
  - Section 2.2 (Theorem 2, unique solution): Faster O(sqrt(T*n) log^3 n) search
    via depth binary search and sampling from the QPE-prepared eigenvector |phi'>.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Tuple

from .backtracking import BacktrackingTree
from .quantum_walk import ComplexVector, QuantumWalkOperators, build_quantum_walk, vector_norm


@dataclass
class QPEResult:
    """Exact outcome statistics of s-qubit QPE on W starting from |r>."""

    ancilla_qubits: int
    walk_steps: int
    zero_phase_prob: float
    collapsed_zero_state: ComplexVector
    phase_probabilities: Optional[List[Tuple[float, float]]] = None


@dataclass
class DetectionResult:
    """Result of Algorithm 2 (Detecting a marked vertex)."""

    detected: bool
    zero_phase_prob: float
    acceptances: int
    trials: int
    ancilla_qubits: int
    total_walk_steps: int


@dataclass
class SearchResult:
    """Result of quantum backtracking search (Theorem 2)."""

    found: bool
    node_id: Optional[int]
    assignment_str: Optional[str]
    total_walk_steps: int
    qpe_invocations: int
    method: str


def qpe_zero_phase_probability(
    ops: QuantumWalkOperators,
    ancilla_qubits: int,
    compute_full_spectrum: bool = False,
) -> QPEResult:
    """Simulate s-qubit QPE on W = R_B R_A with input state |r>.

    For M = 2^s:
      After Hadamards, controlled-W^k, and inverse QFT on the ancilla register,
      the unnormalized state paired with ancilla outcome |0^s> is:
        |v_0> = (1 / M) * sum_{k=0}^{M-1} W^k |r>
      so Pr[outcome = 0^s] = || |v_0> ||_2^2 and the post-measurement state is
      |v_0> / || |v_0> ||_2.

    Args:
      ops: QuantumWalkOperators for the tree.
      ancilla_qubits: Number of QPE precision qubits s (uses M = 2^s walk steps).
      compute_full_spectrum: If True, also compute Pr[outcome = j] for all
        j in {0, ..., M-1} (returned as (phase_angle_in_radians, probability)).
    """
    m = 1 << ancilla_qubits
    curr = ops.root_state()
    sum_vec = list(curr)

    history: Optional[List[ComplexVector]] = [list(curr)] if compute_full_spectrum else None

    for _ in range(1, m):
        curr = ops.apply_walk(curr)
        for idx in range(ops.dim):
            sum_vec[idx] += curr[idx]
        if history is not None:
            history.append(list(curr))

    inv_m = 1.0 / float(m)
    scaled_zero = [v * inv_m for v in sum_vec]
    norm_zero = vector_norm(scaled_zero)
    prob_zero = norm_zero ** 2
    if norm_zero > 1e-15:
        collapsed = [v / norm_zero for v in scaled_zero]
    else:
        collapsed = [0j] * ops.dim

    spectrum: Optional[List[Tuple[float, float]]] = None
    if history is not None:
        spectrum = []
        for j in range(m):
            # |v_j> = (1/M) sum_{k=0}^{M-1} exp(-2*pi*i*j*k/M) W^k |r>
            vj = [0j] * ops.dim
            for k in range(m):
                phase = cmath.exp(-2j * math.pi * j * k / float(m))
                hk = history[k]
                for idx in range(ops.dim):
                    vj[idx] += phase * hk[idx]
            prob_j = (vector_norm(vj) * inv_m) ** 2
            angle = 2.0 * math.pi * j / float(m)
            spectrum.append((angle, prob_j))

    return QPEResult(
        ancilla_qubits=ancilla_qubits,
        walk_steps=m,
        zero_phase_prob=prob_zero,
        collapsed_zero_state=collapsed,
        phase_probabilities=spectrum,
    )


def required_ancilla_qubits(tree_size_bound: int, depth_bound: int, beta: float = 0.25) -> int:
    """Compute s = ceil(log2(sqrt(T * n) / beta)) for QPE precision beta / sqrt(T * n)."""
    target_steps = math.sqrt(max(1, tree_size_bound) * max(1, depth_bound)) / beta
    return max(2, math.ceil(math.log2(max(2.0, target_steps))))


def detect_marked_vertex(
    tree: BacktrackingTree,
    tree_size_bound: Optional[int] = None,
    depth_bound: Optional[int] = None,
    delta: float = 0.05,
    beta: float = 0.25,
    gamma: float = 8.0,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> DetectionResult:
    """Execute Algorithm 2 (Montanaro 2015) to detect whether `tree` has a marked vertex.

    Checks whether the root itself is marked first (per Section 3). Otherwise,
    runs K = ceil(gamma * ln(1/delta)) independent trials of QPE with precision
    beta / sqrt(T * n) and accepts if at least 3K/8 trials return eigenvalue 1.

    Args:
      tree: The rooted backtracking tree.
      tree_size_bound: Upper bound T on tree size (defaults to tree.num_vertices).
      depth_bound: Upper bound n on tree depth (defaults to max(1, tree.max_depth_bound)).
      delta: Target failure probability in (0, 1).
      beta: Precision constant (smaller -> higher QPE precision).
      gamma: Repetition constant for Chernoff majority vote.
      seed: Optional RNG seed when deterministic=False.
      deterministic: If True, evaluates the exact expectation threshold (p0 >= 3/8)
        while still reporting K trials and exact step counts; if False, samples
        K Bernoulli(p0) outcomes.
    """
    if tree.nodes[0].is_marked:
        return DetectionResult(
            detected=True,
            zero_phase_prob=1.0,
            acceptances=1,
            trials=1,
            ancilla_qubits=0,
            total_walk_steps=0,
        )

    t_bound = tree_size_bound if tree_size_bound is not None else tree.num_vertices
    n_bound = depth_bound if depth_bound is not None else max(1, tree.max_depth_bound)

    ops = build_quantum_walk(tree, depth_param=n_bound)
    s = required_ancilla_qubits(t_bound, n_bound, beta=beta)
    qpe = qpe_zero_phase_probability(ops, ancilla_qubits=s)

    trials = max(1, math.ceil(gamma * math.log(1.0 / max(1e-12, delta))))
    if deterministic:
        acceptances = round(qpe.zero_phase_prob * trials)
        detected = qpe.zero_phase_prob >= 0.375
    else:
        rng = random.Random(seed)
        acceptances = sum(1 for _ in range(trials) if rng.random() < qpe.zero_phase_prob)
        detected = acceptances >= (3.0 * trials / 8.0)

    return DetectionResult(
        detected=detected,
        zero_phase_prob=qpe.zero_phase_prob,
        acceptances=acceptances,
        trials=trials,
        ancilla_qubits=s,
        total_walk_steps=qpe.walk_steps * trials,
    )


def find_marked_vertex_binary_search(
    tree: BacktrackingTree,
    num_vars: int,
    known_t_bound: Optional[int] = None,
    delta: float = 0.05,
    beta: float = 0.25,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> SearchResult:
    """Find a marked vertex using Section 2.1's subtree descent + exponential doubling.

    Matches Theorem 2's general search algorithm:
      - If T is not known in advance, doubles a guess T_guess = 1, 2, 4, ...
      - Descends from the root by running Algorithm 2 on each child's subtree
        (keeping T_guess unchanged, per Section 2.1) until reaching a marked vertex.
      - Verifies the returned vertex is actually marked before terminating.
    """
    if tree.nodes[0].is_marked:
        return SearchResult(
            found=True,
            node_id=0,
            assignment_str=tree.nodes[0].assignment.format_vector(num_vars),
            total_walk_steps=0,
            qpe_invocations=0,
            method="binary_search",
        )

    total_steps = 0
    invocations = 0
    n_bound = max(1, tree.max_depth_bound)
    sub_delta = delta / max(4.0, float(n_bound ** 2))

    if known_t_bound is not None:
        t_guesses = [known_t_bound]
    else:
        t_guesses = []
        g = 1
        cap = max(2, 2 * tree.num_vertices)
        while g <= cap:
            t_guesses.append(g)
            g *= 2

    rng = random.Random(seed)

    for t_guess in t_guesses:
        root_det = detect_marked_vertex(
            tree,
            tree_size_bound=t_guess,
            depth_bound=n_bound,
            delta=sub_delta,
            beta=beta,
            deterministic=deterministic,
            seed=rng.randint(0, 10**9) if not deterministic else None,
        )
        total_steps += root_det.total_walk_steps
        invocations += 1

        if not root_det.detected:
            return SearchResult(
                found=False,
                node_id=None,
                assignment_str=None,
                total_walk_steps=total_steps,
                qpe_invocations=invocations,
                method="binary_search",
            )

        # Descend through subtrees
        curr_id = 0
        while not tree.nodes[curr_id].is_marked:
            children = tree.nodes[curr_id].children_ids
            if not children:
                break
            next_id: Optional[int] = None
            for child_id in children:
                if tree.nodes[child_id].is_marked:
                    next_id = child_id
                    break
                sub_tree, _ = tree.subtree(child_id)
                sub_det = detect_marked_vertex(
                    sub_tree,
                    tree_size_bound=t_guess,
                    depth_bound=n_bound,
                    delta=sub_delta,
                    beta=beta,
                    deterministic=deterministic,
                    seed=rng.randint(0, 10**9) if not deterministic else None,
                )
                total_steps += sub_det.total_walk_steps
                invocations += 1
                if sub_det.detected:
                    next_id = child_id
                    break
            if next_id is None:
                break
            curr_id = next_id

        if tree.nodes[curr_id].is_marked:
            return SearchResult(
                found=True,
                node_id=curr_id,
                assignment_str=tree.nodes[curr_id].assignment.format_vector(num_vars),
                total_walk_steps=total_steps,
                qpe_invocations=invocations,
                method="binary_search",
            )

    return SearchResult(
        found=False,
        node_id=None,
        assignment_str=None,
        total_walk_steps=total_steps,
        qpe_invocations=invocations,
        method="binary_search",
    )


def find_unique_marked_vertex(
    tree: BacktrackingTree,
    num_vars: int,
    tree_size_bound: Optional[int] = None,
    beta: float = 0.25,
    seed: Optional[int] = 42,
    max_rounds: int = 64,
) -> SearchResult:
    """Find a unique marked vertex x_0 using Section 2.2's eigenvector-sampling method.

    1. Binary search on depth i in {1, ..., n} to find exact depth l(x_0).
    2. Truncate tree to depth l(x_0).
    3. Repeatedly prepare |tilde{phi}'> via QPE conditioned on measuring |0^s>,
       sample a vertex y != root from |tilde{phi}'> in the computational basis
       (uniformly distributed along the path from root to x_0), and recurse on
       the subtree rooted at y until y == x_0.
    """
    if tree.nodes[0].is_marked:
        return SearchResult(
            found=True,
            node_id=0,
            assignment_str=tree.nodes[0].assignment.format_vector(num_vars),
            total_walk_steps=0,
            qpe_invocations=0,
            method="unique_eigenvector_sampling",
        )

    rng = random.Random(seed)
    t_bound = tree_size_bound if tree_size_bound is not None else tree.num_vertices
    total_steps = 0
    invocations = 0

    # Step 1: Binary search for minimal depth containing x_0
    lo, hi = 1, max(1, tree.actual_depth)
    target_depth = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        trunc, _ = tree.truncate_depth(mid)
        det = detect_marked_vertex(
            trunc,
            tree_size_bound=t_bound,
            depth_bound=mid,
            delta=0.05 / max(2.0, math.log2(max(2, tree.actual_depth))),
            beta=beta,
            deterministic=True,
        )
        total_steps += det.total_walk_steps
        invocations += 1
        if det.detected:
            target_depth = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # Step 2: Truncate tree to target_depth and descend by sampling |tilde{phi}'>
    curr_tree, id_map = tree.truncate_depth(target_depth)
    curr_remaining_depth = target_depth

    for _ in range(max_rounds):
        if curr_tree.nodes[0].is_marked:
            orig_id = id_map[0]
            return SearchResult(
                found=True,
                node_id=orig_id,
                assignment_str=tree.nodes[orig_id].assignment.format_vector(num_vars),
                total_walk_steps=total_steps,
                qpe_invocations=invocations,
                method="unique_eigenvector_sampling",
            )

        ops = build_quantum_walk(curr_tree, depth_param=max(1, curr_remaining_depth))
        # Use higher precision so || |tilde{phi}'> - |phi'> || is small
        s = required_ancilla_qubits(t_bound, max(1, curr_remaining_depth), beta=beta / 2.0)
        qpe = qpe_zero_phase_probability(ops, ancilla_qubits=s)
        total_steps += qpe.walk_steps
        invocations += 1

        # Sample computational basis state from collapsed state |tilde{phi}'>
        probs = [abs(amp) ** 2 for amp in qpe.collapsed_zero_state]
        total_p = sum(probs)
        if total_p <= 1e-12:
            break
        r = rng.random() * total_p
        cum = 0.0
        sampled = 0
        for idx, p in enumerate(probs):
            cum += p
            if r <= cum:
                sampled = idx
                break

        if sampled == 0:
            # Measured root (happens with probability ~1/2); retry
            continue

        orig_sampled = id_map[sampled]
        if tree.nodes[orig_sampled].is_marked:
            return SearchResult(
                found=True,
                node_id=orig_sampled,
                assignment_str=tree.nodes[orig_sampled].assignment.format_vector(num_vars),
                total_walk_steps=total_steps,
                qpe_invocations=invocations,
                method="unique_eigenvector_sampling",
            )

        # Recurse on subtree rooted at sampled vertex (strictly closer to x_0)
        sub_tree, sub_map = curr_tree.subtree(sampled)
        new_id_map: Dict[int, int] = {new_i: id_map[old_i] for new_i, old_i in sub_map.items()}
        curr_remaining_depth = max(1, target_depth - tree.nodes[orig_sampled].depth)
        curr_tree = sub_tree
        id_map = new_id_map

    return SearchResult(
        found=False,
        node_id=None,
        assignment_str=None,
        total_walk_steps=total_steps,
        qpe_invocations=invocations,
        method="unique_eigenvector_sampling",
    )
