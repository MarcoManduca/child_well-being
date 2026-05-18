"""
poset_ops.py - POSet algebraic construction operations.

All construction functions return a new POSet instance.
Where possible, dominance matrices are built directly via
NumPy operations rather than enumerating element pairs.
"""

from __future__ import annotations

from itertools import product as iproduct
from typing import List, Tuple, Optional

import numpy as np

from .poset import POSet, LinearPOSet


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------

def ProductPOSet(*posets: POSet) -> POSet:
    """
    Cartesian product of posets P1 × … × Pk.

    (a1,…,ak) ≤ (b1,…,bk) iff ai ≤i bi for all i.

    Uses Kronecker-style matrix construction: the product dominance
    matrix is the element-wise AND of the individual dominance matrices
    expanded to the product space.

    Examples
    --------
    >>> p1 = POSet(['a','b'], [('a','b')])
    >>> p2 = LinearPOSet(['x','y'])
    >>> pp = ProductPOSet(p1, p2)
    """
    if len(posets) < 2:
        raise ValueError("ProductPOSet requires at least 2 posets.")

    elem_lists = [p.elements for p in posets]
    combos = list(iproduct(*elem_lists))
    labels = [_tuple_label(c) for c in combos]
    N = len(combos)

    # Build index mappings: for each factor, which index does each
    # combo element have in that factor?
    # combo_factor_idx[f][combo_i] = index of combo_i's f-th component
    combo_factor_idx = []
    for f, p in enumerate(posets):
        idx_map = p._idx
        indices = np.array([idx_map[combos[i][f]] for i in range(N)], dtype=np.int32)
        combo_factor_idx.append(indices)

    # Product dominance: dom[i,j] = AND over all factors of factor_mat[ci_f, cj_f]
    dom_mat = np.ones((N, N), dtype=bool)
    for f, p in enumerate(posets):
        fi = combo_factor_idx[f]
        # factor_dom[i,j] = p._mat[fi[i], fi[j]] for all i,j
        factor_dom = p._mat[fi[:, None], fi[None, :]]
        dom_mat &= factor_dom

    # Build POSet with pre-computed matrix
    pos = POSet(labels, dom=None)
    pos._mat = dom_mat
    pos._strict = dom_mat & ~dom_mat.T
    return pos


# ---------------------------------------------------------------------------
# Disjoint sum
# ---------------------------------------------------------------------------

def DisjointSumPOSet(*posets: POSet) -> POSet:
    """
    Disjoint sum: a ≤ b iff both are in the same component and a ≤ b there.

    The resulting dominance matrix is block-diagonal.

    Examples
    --------
    >>> p1 = POSet(['a','b'], [('a','b')])
    >>> p2 = POSet(['c','d'], [('c','d')])
    >>> ds = DisjointSumPOSet(p1, p2)
    """
    _check_disjoint(posets)

    all_elems = []
    for p in posets:
        all_elems.extend(p.elements)

    N = len(all_elems)
    dom_mat = np.eye(N, dtype=bool)

    offset = 0
    for p in posets:
        n = p.n
        dom_mat[offset : offset + n, offset : offset + n] = p._mat
        offset += n

    pos = POSet(all_elems, dom=None)
    pos._mat = dom_mat
    pos._strict = dom_mat & ~dom_mat.T
    return pos


# ---------------------------------------------------------------------------
# Linear sum
# ---------------------------------------------------------------------------

def LinearSumPOSet(*posets: POSet) -> POSet:
    """
    Linear sum: stack posets bottom-to-top;
    all elements of Pi < all elements of Pj for i < j.

    Examples
    --------
    >>> p1 = POSet(['a','b'], [('a','b')])
    >>> p2 = POSet(['c','d'], [('c','d')])
    >>> ls = LinearSumPOSet(p1, p2)
    """
    _check_disjoint(posets)

    all_elems = []
    for p in posets:
        all_elems.extend(p.elements)

    N = len(all_elems)
    dom_mat = np.eye(N, dtype=bool)

    # Block-diagonal: within-component order
    offset = 0
    offsets = []
    for p in posets:
        n = p.n
        dom_mat[offset : offset + n, offset : offset + n] = p._mat
        offsets.append((offset, n))
        offset += n

    # Cross-component: Pi < Pj for i < j
    for i in range(len(posets)):
        oi, ni = offsets[i]
        for j in range(i + 1, len(posets)):
            oj, nj = offsets[j]
            dom_mat[oi : oi + ni, oj : oj + nj] = True

    # Transitive closure is already satisfied by construction
    pos = POSet(all_elems, dom=None)
    pos._mat = dom_mat
    pos._strict = dom_mat & ~dom_mat.T
    return pos


# ---------------------------------------------------------------------------
# Dual
# ---------------------------------------------------------------------------

def DualPOSet(poset: POSet) -> POSet:
    """
    Dual of a poset: reverse all dominances.

    a ≤_d b  iff  b ≤ a in the original.

    Examples
    --------
    >>> p = POSet(['a','b','c'], [('a','b'),('b','c')])
    >>> dp = DualPOSet(p)
    """
    pos = POSet(poset.elements, dom=None)
    pos._mat = poset._mat.T.copy()
    pos._strict = pos._mat & ~pos._mat.T
    return pos


# ---------------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------------

def IntersectionPOSet(*posets: POSet) -> POSet:
    """
    Intersection of posets on the same ground set.

    a ≤∩ b  iff  a ≤i b for all i.

    The intersection matrix is the element-wise AND of all dominance
    matrices (after reindexing to a common element order).

    Examples
    --------
    >>> p1 = POSet(['a','b','c'], [('a','b'),('b','c')])
    >>> p2 = POSet(['a','b','c'], [('a','b')])
    >>> pi = IntersectionPOSet(p1, p2)
    """
    if len(posets) < 2:
        raise ValueError("IntersectionPOSet requires at least 2 posets.")

    ref_elems = posets[0].elements
    ref_set = set(ref_elems)
    for p in posets[1:]:
        if set(p.elements) != ref_set:
            raise ValueError("All posets must have the same ground set.")

    # Reindex all matrices to the first poset's element order
    n = len(ref_elems)
    dom_mat = np.ones((n, n), dtype=bool)

    for p in posets:
        # Permutation: perm[i] = index in p of ref_elems[i]
        perm = np.array([p._idx[e] for e in ref_elems], dtype=np.int32)
        reindexed = p._mat[np.ix_(perm, perm)]
        dom_mat &= reindexed

    pos = POSet(ref_elems, dom=None)
    pos._mat = dom_mat
    pos._strict = dom_mat & ~dom_mat.T
    return pos


# ---------------------------------------------------------------------------
# Lifting (add bottom element)
# ---------------------------------------------------------------------------

def LiftingPOSet(poset: POSet, element: str) -> POSet:
    """
    Add a new bottom element to the poset (below all existing elements).

    Examples
    --------
    >>> p = POSet(['a','b'], [('a','b')])
    >>> lp = LiftingPOSet(p, 'bot')
    """
    if element in poset.elements:
        raise ValueError(f"Element '{element}' already in poset.")

    n = poset.n
    N = n + 1
    elems = [element] + poset.elements

    dom_mat = np.eye(N, dtype=bool)
    # Copy original order into bottom-right block
    dom_mat[1:, 1:] = poset._mat
    # New element dominates everything (row 0, all columns)
    dom_mat[0, :] = True

    pos = POSet(elems, dom=None)
    pos._mat = dom_mat
    pos._strict = dom_mat & ~dom_mat.T
    return pos


# ---------------------------------------------------------------------------
# Lexicographic product
# ---------------------------------------------------------------------------

def LexicographicProductPOSet(*posets: POSet) -> POSet:
    """
    Lexicographic product of posets.

    (a1,…,ak) ≤_lex (b1,…,bk) iff there exists an index m such that
    am <m bm and ai = bi for all i < m, or a = b.

    Examples
    --------
    >>> p1 = LinearPOSet(['a','b','c'])
    >>> p2 = LinearPOSet(['x','y'])
    >>> lp = LexicographicProductPOSet(p1, p2)
    """
    if len(posets) < 2:
        raise ValueError("LexicographicProductPOSet requires at least 2 posets.")

    elem_lists = [p.elements for p in posets]
    combos = list(iproduct(*elem_lists))
    labels = [_tuple_label(c) for c in combos]
    N = len(combos)
    K = len(posets)

    # Pre-compute factor indices for each combo
    combo_factor_idx = []
    for f, p in enumerate(posets):
        idx_map = p._idx
        indices = np.array(
            [idx_map[combos[i][f]] for i in range(N)], dtype=np.int32
        )
        combo_factor_idx.append(indices)

    # Pre-compute per-factor matrices
    # eq[f][i,j]    = (combo_i component f) == (combo_j component f)
    # strict[f][i,j] = (combo_i component f) < (combo_j component f)
    # leq[f][i,j]   = (combo_i component f) <= (combo_j component f)
    eq = []
    strict = []
    for f, p in enumerate(posets):
        fi = combo_factor_idx[f]
        f_mat = p._mat
        f_leq = f_mat[fi[:, None], fi[None, :]]
        f_eq = f_leq & f_leq.T  # both directions → equal
        f_strict = f_leq & ~f_eq
        eq.append(f_eq)
        strict.append(f_strict)

    # Lex order: a ≤_lex b iff a = b, or exists m such that
    # a[0]=b[0], ..., a[m-1]=b[m-1], a[m] <m b[m]
    dom_mat = np.eye(N, dtype=bool)  # reflexive

    # For each priority level m = 0, ..., K-1:
    #   condition_m = eq[0] & eq[1] & ... & eq[m-1] & strict[m]
    prefix_eq = np.ones((N, N), dtype=bool)
    for m in range(K):
        dom_mat |= prefix_eq & strict[m]
        prefix_eq &= eq[m]

    # Transitive closure (may be needed for non-linear factors)
    dom_mat = POSet._transitive_closure(dom_mat)

    pos = POSet(labels, dom=None)
    pos._mat = dom_mat
    pos._strict = dom_mat & ~dom_mat.T
    return pos


# ---------------------------------------------------------------------------
# Crown
# ---------------------------------------------------------------------------

def CrownPOSet(elements_1: List[str], elements_2: List[str]) -> POSet:
    """
    Crown poset over two disjoint collections of the same size.

    ai ≤ bj for all i ≠ j;  ai ∥ aj;  bi ∥ bj;  ai ∥ bi.

    Examples
    --------
    >>> crown = CrownPOSet(['a1','a2','a3'], ['b1','b2','b3'])
    """
    n = len(elements_1)
    if len(elements_2) != n:
        raise ValueError("Both element collections must have the same size.")

    N = 2 * n
    all_elems = list(elements_1) + list(elements_2)
    dom_mat = np.eye(N, dtype=bool)

    # ai ≤ bj for i ≠ j
    cross = np.ones((n, n), dtype=bool)
    np.fill_diagonal(cross, False)
    dom_mat[:n, n:] = cross

    # Transitive closure (needed because ai ≤ bj, but no further chains)
    dom_mat = POSet._transitive_closure(dom_mat)

    pos = POSet(all_elems, dom=None)
    pos._mat = dom_mat
    pos._strict = dom_mat & ~dom_mat.T
    return pos


# ---------------------------------------------------------------------------
# Fence
# ---------------------------------------------------------------------------

def FencePOSet(elements: List[str], orientation: str = "upFirst") -> POSet:
    """
    Fence poset (zigzag) over elements.

    Parameters
    ----------
    elements : list of str
    orientation : 'upFirst' or 'downFirst'

    Examples
    --------
    >>> fence = FencePOSet(['a','b','c','d','e'])
    """
    n = len(elements)
    dom = []
    if orientation == "upFirst":
        for i in range(0, n - 1, 2):
            dom.append((elements[i], elements[i + 1]))
        for i in range(2, n, 2):
            dom.append((elements[i], elements[i - 1]))
    else:  # downFirst
        for i in range(0, n - 1, 2):
            dom.append((elements[i + 1], elements[i]))
        for i in range(2, n, 2):
            dom.append((elements[i - 1], elements[i]))
    return POSet(elements, dom)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tuple_label(t: tuple) -> str:
    return "(" + ",".join(str(x) for x in t) + ")"


def _check_disjoint(posets):
    seen = set()
    for p in posets:
        s = set(p.elements)
        if seen & s:
            raise ValueError("Posets must have disjoint ground sets.")
        seen |= s
