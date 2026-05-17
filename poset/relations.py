"""
relations.py - Binary relation property checkers and closures.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np


Relation = List[Tuple[str, str]]


def _rel_to_set(rel: Relation):
    return {(str(a), str(b)) for a, b in rel}


def IsReflexive(ground_set: List[str], rel: Relation) -> bool:
    """
    Check whether a binary relation is reflexive.

    Parameters
    ----------
    ground_set : list of str
    rel : list of (str, str)

    Returns
    -------
    bool

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
    (a,b) in R and (b,a) in R implies a == b.

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

    Examples
    --------
    >>> IsTransitive([('a','b'),('b','c'),('a','c')])
    True
    """
    s = _rel_to_set(rel)
    for a, b in s:
        for c, d in s:
            if b == c and (a, d) not in s:
                return False
    return True


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
    Check whether a relation is a partial order (reflexive, antisymmetric, transitive).

    Examples
    --------
    >>> rels = [('a','a'),('b','b'),('c','c'),('a','b'),('a','c'),('b','c')]
    >>> IsPartialOrder(['a','b','c'], rels)
    True
    """
    return (IsReflexive(ground_set, rel)
            and IsAntisymmetric(rel)
            and IsTransitive(rel))


def TransitiveClosure(rel: Relation) -> Relation:
    """
    Compute the transitive closure of a binary relation.

    Returns
    -------
    list of (str, str)

    Examples
    --------
    >>> TransitiveClosure([('a','b'),('b','c')])
    [('a', 'b'), ('b', 'c'), ('a', 'c')]
    """
    s = _rel_to_set(rel)
    changed = True
    while changed:
        changed = False
        new = set()
        for a, b in s:
            for c, d in s:
                if b == c and (a, d) not in s:
                    new.add((a, d))
                    changed = True
        s |= new
    return sorted(s)


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
