"""
poset_ops.py - POSet algebraic construction operations.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from .poset import POSet, LinearPOSet


def ProductPOSet(*posets: POSet) -> POSet:
    """
    Cartesian product of posets P1 × ... × Pk.

    (a1,...,ak) ≤ (b1,...,bk) iff ai ≤i bi for all i.

    Examples
    --------
    >>> p1 = POSet(['a','b'], [('a','b')])
    >>> p2 = LinearPOSet(['x','y'])
    >>> pp = ProductPOSet(p1, p2)
    """
    if len(posets) < 2:
        raise ValueError("ProductPOSet requires at least 2 posets.")

    from itertools import product as iproduct

    elem_lists = [p.elements for p in posets]

    # All combinations as tuples
    combos = list(iproduct(*elem_lists))
    labels = [_tuple_label(c) for c in combos]

    dom = []
    for i, ci in enumerate(combos):
        for j, cj in enumerate(combos):
            if i != j and all(posets[k].dominates(ci[k], cj[k])
                              for k in range(len(posets))):
                dom.append((labels[i], labels[j]))

    return POSet(labels, dom)


def DisjointSumPOSet(*posets: POSet) -> POSet:
    """
    Disjoint sum: a ≤ b iff both are in the same component and a ≤ b there.

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

    dom = []
    for p in posets:
        dom.extend(p.order_relation())

    return POSet(all_elems, dom)


def LinearSumPOSet(*posets: POSet) -> POSet:
    """
    Linear sum: stack posets bottom-to-top; all elements of Pi < all elements of Pi+1.

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

    dom = []
    for p in posets:
        dom.extend(p.order_relation())

    # Add cross-relations: every element of Pi ≤ every element of Pj for i < j
    for i in range(len(posets)):
        for j in range(i+1, len(posets)):
            for a in posets[i].elements:
                for b in posets[j].elements:
                    dom.append((a, b))

    return POSet(all_elems, dom)


def DualPOSet(poset: POSet) -> POSet:
    """
    Dual of a poset: reverse all dominances.

    a ≤_d b iff b ≤ a in the original.

    Examples
    --------
    >>> p = POSet(['a','b','c'], [('a','b'),('b','c')])
    >>> dp = DualPOSet(p)
    """
    elems = poset.elements
    dom = [(b, a) for a, b in poset.order_relation() if a != b]
    return POSet(elems, dom)


def IntersectionPOSet(*posets: POSet) -> POSet:
    """
    Intersection of posets on the same ground set.

    a ≤∩ b iff a ≤i b for all i.

    Examples
    --------
    >>> p1 = POSet(['a','b','c'], [('a','b'),('b','c')])
    >>> p2 = POSet(['a','b','c'], [('a','b')])
    >>> pi = IntersectionPOSet(p1, p2)
    """
    if len(posets) < 2:
        raise ValueError("IntersectionPOSet requires at least 2 posets.")
    ref_elems = set(posets[0].elements)
    for p in posets[1:]:
        if set(p.elements) != ref_elems:
            raise ValueError("All posets must have the same ground set.")

    elems = posets[0].elements
    dom = []
    for a in elems:
        for b in elems:
            if a != b and all(p.dominates(a, b) for p in posets):
                dom.append((a, b))
    return POSet(elems, dom)


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
    elems = [element] + poset.elements
    dom = list(poset.order_relation())
    for e in poset.elements:
        dom.append((element, e))
    return POSet(elems, dom)


def LexicographicProductPOSet(*posets: POSet) -> POSet:
    """
    Lexicographic product of posets.

    (a1,...,ak) ≤_lex (b1,...,bk) iff a1 < b1,
    or a1=b1 and a2 < b2, ... etc.

    Examples
    --------
    >>> p1 = LinearPOSet(['a','b','c'])
    >>> p2 = LinearPOSet(['x','y'])
    >>> lp = LexicographicProductPOSet(p1, p2)
    """
    from itertools import product as iproduct

    elem_lists = [p.elements for p in posets]
    combos = list(iproduct(*elem_lists))
    labels = [_tuple_label(c) for c in combos]

    dom = []
    for i, ci in enumerate(combos):
        for j, cj in enumerate(combos):
            if i == j:
                continue
            if _lex_le(ci, cj, posets):
                dom.append((labels[i], labels[j]))

    return POSet(labels, dom)


def CrownPOSet(elements_1: List[str], elements_2: List[str]) -> POSet:
    """
    Crown poset over two disjoint collections of the same size.

    ai ≤ bj for all i ≠ j; ai || aj; bi || bj; ai || bi.

    Examples
    --------
    >>> crown = CrownPOSet(['a1','a2','a3'], ['b1','b2','b3'])
    """
    n = len(elements_1)
    if len(elements_2) != n:
        raise ValueError("Both element collections must have the same size.")
    all_elems = list(elements_1) + list(elements_2)
    dom = []
    for i, a in enumerate(elements_1):
        for j, b in enumerate(elements_2):
            if i != j:
                dom.append((a, b))
    return POSet(all_elems, dom)


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
        for i in range(0, n-1, 2):
            dom.append((elements[i], elements[i+1]))
        for i in range(2, n, 2):
            dom.append((elements[i], elements[i-1]))
    else:  # downFirst
        for i in range(0, n-1, 2):
            dom.append((elements[i+1], elements[i]))
        for i in range(2, n, 2):
            dom.append((elements[i-1], elements[i]))
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


def _lex_le(a: tuple, b: tuple, posets) -> bool:
    """Return True if a ≤_lex b."""
    for k, p in enumerate(posets):
        if a[k] == b[k]:
            continue
        return p.dominates(a[k], b[k])
    return False  # equal tuples
