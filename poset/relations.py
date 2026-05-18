"""
relations.py - Binary relation property checkers and closures.

All public functions accept relations as lists of ``(str, str)`` pairs
and a ground set as a list of ``str``.  Internally, heavy operations
(transitivity check, transitive closure) are dispatched to NumPy
matrix operations.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

Relation = List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rel_to_set(rel: Relation) -> Set[Tuple[str, str]]:
    return {(str(a), str(b)) for a, b in rel}


def _build_matrix(
    ground_set: List[str],
    rel: Relation,
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Build a boolean adjacency matrix from a relation over a ground set.

    Returns (matrix, elem_to_idx, elements).
    """
    elements = list(ground_set)
    idx = {e: i for i, e in enumerate(elements)}
    n = len(elements)
    mat = np.zeros((n, n), dtype=bool)
    for a, b in rel:
        a, b = str(a), str(b)
        if a in idx and b in idx:
            mat[idx[a], idx[b]] = True
    return mat, idx, elements


def _matrix_to_rel(
    mat: np.ndarray,
    elements: List[str],
) -> Relation:
    """Extract a sorted relation list from a boolean matrix."""
    rows, cols = np.where(mat)
    pairs = [(elements[i], elements[j]) for i, j in zip(rows, cols)]
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Property checkers
# ---------------------------------------------------------------------------

def IsReflexive(ground_set: List[str], rel: Relation) -> bool:
    """
    Check whether a binary relation is reflexive.

    Examples
    --------
    >>> IsReflexive(['a','b'], [('a','a'),('b','b'),('a','b')])
    True
    """
    s = _rel_to_set(rel)
    return all((e, e) in s for e in ground_set)


def IsSymmetric(rel: Relation) -> bool:
    """
    Check whether a binary relation is symmetric.

    Examples
    --------
    >>> IsSymmetric([('a','b'),('b','a')])
    True
    """
    s = _rel_to_set(rel)
    return all((b, a) in s for a, b in s)


def IsAntisymmetric(rel: Relation) -> bool:
    """
    Check whether a binary relation is antisymmetric:
    (a,b) ∈ R and (b,a) ∈ R  implies  a = b.

    Examples
    --------
    >>> IsAntisymmetric([('a','b'),('a','a')])
    True
    """
    s = _rel_to_set(rel)
    for a, b in s:
        if a != b and (b, a) in s:
            return False
    return True


def IsTransitive(rel: Relation) -> bool:
    """
    Check whether a binary relation is transitive.

    Uses matrix multiplication: R is transitive iff R² ⊆ R
    (i.e. every pair reachable in two steps is already in R).

    The ground set is inferred from the relation elements.

    Examples
    --------
    >>> IsTransitive([('a','b'),('b','c'),('a','c')])
    True
    """
    # Infer ground set from relation
    ground = sorted({e for pair in rel for e in pair})
    if not ground:
        return True

    mat, _, _ = _build_matrix(ground, rel)

    # R² via boolean matrix multiply
    reachable_2 = (mat.astype(np.uint8) @ mat.astype(np.uint8)) > 0

    # Transitive iff every pair in R² is also in R
    return bool(np.all(~reachable_2 | mat))


def IsPreorder(ground_set: List[str], rel: Relation) -> bool:
    """
    Check whether a relation is a pre-order (reflexive + transitive).

    Examples
    --------
    >>> IsPreorder(['a','b','c'], [('a','a'),('b','b'),('c','c'),('a','b')])
    True
    """
    return IsReflexive(ground_set, rel) and IsTransitive(rel)


def IsPartialOrder(ground_set: List[str], rel: Relation) -> bool:
    """
    Check whether a relation is a partial order
    (reflexive, antisymmetric, transitive).

    Examples
    --------
    >>> rels = [('a','a'),('b','b'),('c','c'),('a','b'),('a','c'),('b','c')]
    >>> IsPartialOrder(['a','b','c'], rels)
    True
    """
    return (
        IsReflexive(ground_set, rel)
        and IsAntisymmetric(rel)
        and IsTransitive(rel)
    )


# ---------------------------------------------------------------------------
# Closures
# ---------------------------------------------------------------------------

def TransitiveClosure(rel: Relation) -> Relation:
    """
    Compute the transitive closure of a binary relation.

    Uses the vectorized Floyd-Warshall algorithm on a boolean matrix.

    Returns
    -------
    list of (str, str)
        Sorted list of all pairs in the transitive closure.

    Examples
    --------
    >>> TransitiveClosure([('a','b'),('b','c')])
    [('a', 'b'), ('a', 'c'), ('b', 'c')]
    """
    ground = sorted({e for pair in rel for e in pair})
    if not ground:
        return []

    mat, _, elements = _build_matrix(ground, rel)

    # Floyd-Warshall transitive closure
    n = len(elements)
    for k in range(n):
        mat |= mat[:, k : k + 1] & mat[k : k + 1, :]

    return _matrix_to_rel(mat, elements)


def ReflexiveClosure(ground_set: List[str], rel: Relation) -> Relation:
    """
    Compute the reflexive closure of a binary relation.

    Returns
    -------
    list of (str, str)

    Examples
    --------
    >>> ReflexiveClosure(['a','b','c'], [('a','b')])
    [('a', 'a'), ('a', 'b'), ('b', 'b'), ('c', 'c')]
    """
    s = _rel_to_set(rel)
    for e in ground_set:
        s.add((e, e))
    return sorted(s)
