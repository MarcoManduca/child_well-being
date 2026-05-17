"""
poset_query.py - Free-standing query functions mirroring the R API.
All functions accept a POSet object as first argument.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from .poset import POSet


def POSetElements(poset: POSet) -> List[str]:
    """Return the ground-set elements of the poset."""
    return poset.elements


def DominanceMatrix(poset: POSet) -> np.ndarray:
    """
    Boolean dominance matrix Z (n×n).
    Z[i,j] = True iff j-th element ≥ i-th element.
    """
    return poset.dominance_matrix()


def Dominates(poset: POSet, element1: str, element2: str) -> bool:
    """Return True if element1 ≤ element2 in the poset."""
    return poset.dominates(element1, element2)


def IsDominatedBy(poset: POSet, element1: str, element2: str) -> bool:
    """Return True if element1 ≤ element2 in the poset."""
    return poset.is_dominated_by(element1, element2)


def IsComparableWith(poset: POSet, element1: str, element2: str) -> bool:
    """Return True if element1 and element2 are comparable."""
    return poset.is_comparable(element1, element2)


def IsIncomparableWith(poset: POSet, element1: str, element2: str) -> bool:
    """Return True if element1 and element2 are incomparable."""
    return poset.is_incomparable(element1, element2)


def ComparabilitySetOf(poset: POSet, element: str) -> List[str]:
    """Return elements comparable with the given element."""
    return poset.comparability_set(element)


def IncomparabilitySetOf(poset: POSet, element: str) -> List[str]:
    """Return elements incomparable with the given element."""
    return poset.incomparability_set(element)


def IncomparabilityRelation(poset: POSet) -> List[Tuple[str, str]]:
    """Return all incomparable pairs (a, b) with a != b."""
    return poset.incomparability_relation()


def OrderRelation(poset: POSet) -> List[Tuple[str, str]]:
    """Return all pairs (a, b) such that a ≤ b."""
    return poset.order_relation()


def UpsetOf(poset: POSet, elements: List[str]) -> List[str]:
    """Return the upset (up-closure) of a set of elements."""
    return poset.upset_of(elements)


def DownsetOf(poset: POSet, elements: List[str]) -> List[str]:
    """Return the downset (down-closure) of a set of elements."""
    return poset.downset_of(elements)


def IsUpset(poset: POSet, elements: List[str]) -> bool:
    """Return True if elements form an upset."""
    return poset.is_upset(elements)


def IsDownset(poset: POSet, elements: List[str]) -> bool:
    """Return True if elements form a downset."""
    return poset.is_downset(elements)


def POSetMaximals(poset: POSet) -> List[str]:
    """Return the maximal elements of the poset."""
    return poset.maximals()


def POSetMinimals(poset: POSet) -> List[str]:
    """Return the minimal elements of the poset."""
    return poset.minimals()


def IsMaximal(poset: POSet, element: str) -> bool:
    """Return True if element is maximal."""
    return poset.is_maximal(element)


def IsMinimal(poset: POSet, element: str) -> bool:
    """Return True if element is minimal."""
    return poset.is_minimal(element)


def CoverRelation(poset: POSet) -> List[Tuple[str, str]]:
    """Return cover pairs (a, b): b covers a."""
    return poset.cover_relation()


def CoverMatrix(poset: POSet) -> np.ndarray:
    """Boolean cover matrix C where C[i,j]=True iff j-th covers i-th."""
    return poset.cover_matrix()


def POSetMeet(poset: POSet, elements: List[str]) -> Optional[str]:
    """Return the greatest lower bound of elements, or None."""
    return poset.meet(elements)


def POSetJoin(poset: POSet, elements: List[str]) -> Optional[str]:
    """Return the least upper bound of elements, or None."""
    return poset.join(elements)


def IsExtensionOf(poset1: POSet, poset2: POSet) -> bool:
    """Return True if poset1 is an extension of poset2."""
    return poset1.is_extension_of(poset2)
