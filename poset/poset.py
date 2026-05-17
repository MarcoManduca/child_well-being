"""
poset.py - Core POSet data structures
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Set, Dict
import numpy as np


class POSet:
    """
    Partially Ordered Set P = (V, ≤).

    Parameters
    ----------
    elements : list of str
        Labels of the ground set V.
    dom : list of (str, str) or np.ndarray of shape (n, 2), optional
        Dominance pairs. Each pair (a, b) means a ≤ b.
        Reflexive pairs are added automatically.

    Examples
    --------
    >>> pos = POSet(['a', 'b', 'c', 'd'],
    ...             dom=[('a','b'), ('c','b'), ('b','d')])
    """

    def __init__(
        self,
        elements: List[str],
        dom: Optional[List[Tuple[str, str]]] = None,
    ):
        if len(elements) != len(set(elements)):
            raise ValueError("Element labels must be unique.")
        self._elements: List[str] = list(elements)
        self._idx: Dict[str, int] = {e: i for i, e in enumerate(elements)}
        n = len(elements)

        # Internal adjacency matrix (strict dominances + reflexive)
        self._mat = np.zeros((n, n), dtype=bool)
        # Reflexive
        for i in range(n):
            self._mat[i, i] = True

        if dom is not None:
            dom_arr = np.asarray(dom)
            if dom_arr.ndim == 1 and len(dom_arr) == 0:
                pass
            elif dom_arr.ndim == 2:
                for row in dom_arr:
                    a, b = str(row[0]), str(row[1])
                    if a not in self._idx:
                        raise ValueError(f"Element '{a}' not in ground set.")
                    if b not in self._idx:
                        raise ValueError(f"Element '{b}' not in ground set.")
                    self._mat[self._idx[a], self._idx[b]] = True

        # Compute transitive closure (Floyd-Warshall)
        self._mat = self._transitive_closure(self._mat)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transitive_closure(mat: np.ndarray) -> np.ndarray:
        """Floyd-Warshall transitive closure over a boolean matrix."""
        n = mat.shape[0]
        m = mat.copy()
        for k in range(n):
            m = m | (m[:, k:k+1] & m[k:k+1, :])
        return m

    def _validate_element(self, e: str) -> int:
        if e not in self._idx:
            raise ValueError(f"Element '{e}' not in poset.")
        return self._idx[e]

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def elements(self) -> List[str]:
        """Return the list of poset elements."""
        return list(self._elements)

    @property
    def n(self) -> int:
        return len(self._elements)

    def dominance_matrix(self) -> np.ndarray:
        """
        Boolean dominance matrix Z where Z[i,j]=True iff j-th element ≥ i-th element.
        Row/col order matches self.elements.
        """
        return self._mat.copy()

    def order_relation(self) -> List[Tuple[str, str]]:
        """Return all pairs (a, b) such that a ≤ b."""
        pairs = []
        el = self._elements
        for i in range(self.n):
            for j in range(self.n):
                if self._mat[i, j]:
                    pairs.append((el[i], el[j]))
        return pairs

    def cover_relation(self) -> List[Tuple[str, str]]:
        """Return cover pairs (a, b): b covers a (a <· b)."""
        covers = []
        el = self._elements
        n = self.n
        for i in range(n):
            for j in range(n):
                if i == j or not self._mat[i, j]:
                    continue
                # b=el[j] covers a=el[i] if no intermediate c exists
                covered = True
                for k in range(n):
                    if k != i and k != j and self._mat[i, k] and self._mat[k, j]:
                        covered = False
                        break
                if covered:
                    covers.append((el[i], el[j]))
        return covers

    def cover_matrix(self) -> np.ndarray:
        """Boolean cover matrix C where C[i,j]=True iff j-th element covers i-th."""
        n = self.n
        C = np.zeros((n, n), dtype=bool)
        for a, b in self.cover_relation():
            C[self._idx[a], self._idx[b]] = True
        return C

    # ------------------------------------------------------------------
    # Element-level queries
    # ------------------------------------------------------------------

    def dominates(self, a: str, b: str) -> bool:
        """Return True if a ≤ b."""
        return bool(self._mat[self._validate_element(a), self._validate_element(b)])

    def is_dominated_by(self, a: str, b: str) -> bool:
        """Return True if a ≤ b (same as dominates but named from a's perspective)."""
        return self.dominates(a, b)

    def is_comparable(self, a: str, b: str) -> bool:
        ia, ib = self._validate_element(a), self._validate_element(b)
        return bool(self._mat[ia, ib] or self._mat[ib, ia])

    def is_incomparable(self, a: str, b: str) -> bool:
        return not self.is_comparable(a, b)

    def comparability_set(self, e: str) -> List[str]:
        ie = self._validate_element(e)
        return [el for el in self._elements
                if el != e and self.is_comparable(e, el)]

    def incomparability_set(self, e: str) -> List[str]:
        ie = self._validate_element(e)
        return [el for el in self._elements
                if el != e and self.is_incomparable(e, el)]

    def incomparability_relation(self) -> List[Tuple[str, str]]:
        pairs = []
        el = self._elements
        n = self.n
        for i in range(n):
            for j in range(i+1, n):
                if not self._mat[i, j] and not self._mat[j, i]:
                    pairs.append((el[i], el[j]))
        return pairs

    def upset_of(self, elems: List[str]) -> List[str]:
        """Return the upset (up-closure) of a set of elements."""
        indices = {self._validate_element(e) for e in elems}
        result = set()
        for ie in indices:
            for j in range(self.n):
                if self._mat[ie, j]:
                    result.add(j)
        return [self._elements[j] for j in sorted(result)]

    def downset_of(self, elems: List[str]) -> List[str]:
        """Return the downset (down-closure) of a set of elements."""
        indices = {self._validate_element(e) for e in elems}
        result = set()
        for ie in indices:
            for i in range(self.n):
                if self._mat[i, ie]:
                    result.add(i)
        return [self._elements[i] for i in sorted(result)]

    def is_upset(self, elems: List[str]) -> bool:
        s = set(elems)
        return set(self.upset_of(elems)) == s

    def is_downset(self, elems: List[str]) -> bool:
        s = set(elems)
        return set(self.downset_of(elems)) == s

    def minimals(self) -> List[str]:
        """Elements with no strictly smaller element."""
        result = []
        for i, e in enumerate(self._elements):
            if not any(self._mat[j, i] and j != i for j in range(self.n)):
                result.append(e)
        return result

    def maximals(self) -> List[str]:
        """Elements with no strictly larger element."""
        result = []
        for i, e in enumerate(self._elements):
            if not any(self._mat[i, j] and j != i for j in range(self.n)):
                result.append(e)
        return result

    def is_minimal(self, e: str) -> bool:
        return e in self.minimals()

    def is_maximal(self, e: str) -> bool:
        return e in self.maximals()

    def meet(self, elems: List[str]) -> Optional[str]:
        """Greatest lower bound of elements, or None if it doesn't exist."""
        # Lower bounds: elements dominated by all in elems
        target_indices = [self._validate_element(e) for e in elems]
        lower = []
        for i in range(self.n):
            if all(self._mat[i, t] for t in target_indices):
                lower.append(i)
        if not lower:
            return None
        # Find the greatest among lower bounds
        for candidate in lower:
            if all(self._mat[candidate2, candidate] for candidate2 in lower):
                return self._elements[candidate]
        return None

    def join(self, elems: List[str]) -> Optional[str]:
        """Least upper bound of elements, or None if it doesn't exist."""
        target_indices = [self._validate_element(e) for e in elems]
        upper = []
        for j in range(self.n):
            if all(self._mat[t, j] for t in target_indices):
                upper.append(j)
        if not upper:
            return None
        for candidate in upper:
            if all(self._mat[candidate, candidate2] for candidate2 in upper):
                return self._elements[candidate]
        return None

    def is_extension_of(self, other: "POSet") -> bool:
        """Return True if self is an extension of other (other ≤ self in terms of order)."""
        if set(self._elements) != set(other._elements):
            return False
        for a, b in other.order_relation():
            if not self.dominates(a, b):
                return False
        return True

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        covers = self.cover_relation()
        return (f"POSet(elements={self._elements}, "
                f"covers={covers})")

    def __len__(self) -> int:
        return self.n


# ---------------------------------------------------------------------------
# LinearPOSet
# ---------------------------------------------------------------------------

class LinearPOSet(POSet):
    """
    Linearly (totally) ordered set.

    Parameters
    ----------
    elements : list of str
        Elements in ascending order: elements[h] ≤ elements[k] iff h ≤ k.

    Examples
    --------
    >>> lp = LinearPOSet(['a', 'b', 'c', 'd'])
    """

    def __init__(self, elements: List[str]):
        n = len(elements)
        dom = [(elements[i], elements[j])
               for i in range(n) for j in range(i, n)]
        super().__init__(elements, dom)


# ---------------------------------------------------------------------------
# BinaryVariablePOSet
# ---------------------------------------------------------------------------

class BinaryVariablePOSet(POSet):
    """
    Component-wise poset on binary vectors {0,1}^k.

    Given k binary variable names, builds the poset whose elements are all 2^k
    binary vectors, ordered component-wise.

    Parameters
    ----------
    variables : list of str
        Names of the k binary variables.

    Examples
    --------
    >>> bp = BinaryVariablePOSet(['var1', 'var2', 'var3'])
    """

    def __init__(self, variables: List[str]):
        k = len(variables)
        # Generate all 2^k binary vectors as tuples
        profiles = [tuple((i >> b) & 1 for b in range(k))
                    for i in range(2 ** k)]

        def vec_label(v):
            return "(" + ",".join(str(x) for x in v) + ")"

        labels = [vec_label(p) for p in profiles]
        self._profiles = profiles
        self._variables = list(variables)

        dom = []
        for i, pi in enumerate(profiles):
            for j, pj in enumerate(profiles):
                if i != j and all(pi[b] <= pj[b] for b in range(k)):
                    dom.append((labels[i], labels[j]))

        super().__init__(labels, dom)

    @property
    def variables(self):
        return list(self._variables)
