"""
poset.py - Core POSet data structures.

Classes:
  - POSet              : general partial order
  - LinearPOSet        : total (linear) order
  - BinaryVariablePOSet: component-wise order on {0,1}^k
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# POSet
# ---------------------------------------------------------------------------

class POSet:
    """
    Partially Ordered Set  P = (V, ≤).

    Parameters
    ----------
    elements : list of str
        Labels of the ground set V.
    dom : list of (str, str) or np.ndarray of shape (m, 2), optional
        Dominance pairs.  Each pair (a, b) means a ≤ b.
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

        # Internal adjacency matrix (dominances + reflexive)
        self._mat = np.eye(n, dtype=bool)

        if dom is not None:
            dom_arr = np.asarray(dom)
            if dom_arr.ndim == 2 and dom_arr.shape[0] > 0:
                for row in dom_arr:
                    a, b = str(row[0]), str(row[1])
                    if a not in self._idx:
                        raise ValueError(f"Element '{a}' not in ground set.")
                    if b not in self._idx:
                        raise ValueError(f"Element '{b}' not in ground set.")
                    self._mat[self._idx[a], self._idx[b]] = True

        # Transitive closure (vectorized Floyd-Warshall)
        self._mat = self._transitive_closure(self._mat)

        # Pre-compute strict-order matrix: _strict[i,j] = True iff i < j
        self._strict = self._mat & ~self._mat.T

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transitive_closure(mat: np.ndarray) -> np.ndarray:
        """Floyd-Warshall transitive closure over a boolean matrix."""
        n = mat.shape[0]
        m = mat.copy()
        for k in range(n):
            m |= m[:, k : k + 1] & m[k : k + 1, :]
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
        Boolean dominance matrix Z where Z[i,j]=True iff
        elements[i] ≤ elements[j].
        """
        return self._mat.copy()

    def order_relation(self) -> List[Tuple[str, str]]:
        """Return all pairs (a, b) such that a ≤ b."""
        el = self._elements
        rows, cols = np.where(self._mat)
        return [(el[i], el[j]) for i, j in zip(rows, cols)]

    def cover_relation(self) -> List[Tuple[str, str]]:
        """
        Return cover pairs (a, b): b covers a  (a <· b).

        Vectorized: the cover matrix is the strict-order matrix with
        all "shortcutable" pairs removed.
        """
        C = self._cover_matrix_bool()
        el = self._elements
        rows, cols = np.where(C)
        return [(el[i], el[j]) for i, j in zip(rows, cols)]

    def cover_matrix(self) -> np.ndarray:
        """Boolean cover matrix C[i,j]=True iff elements[j] covers elements[i]."""
        return self._cover_matrix_bool().copy()

    def _cover_matrix_bool(self) -> np.ndarray:
        """
        Compute the cover matrix from the strict-order matrix.

        C = strict  AND NOT  (strict @ strict)

        strict @ strict [i,j] is True iff there exists an intermediate k
        such that i < k < j  →  j does NOT cover i.
        """
        S = self._strict
        # Boolean matrix multiplication: S² via float then threshold
        has_intermediate = (S.astype(np.uint8) @ S.astype(np.uint8)) > 0
        return S & ~has_intermediate

    # ------------------------------------------------------------------
    # Element-level queries
    # ------------------------------------------------------------------

    def dominates(self, a: str, b: str) -> bool:
        """Return True if a ≤ b."""
        return bool(
            self._mat[self._validate_element(a), self._validate_element(b)]
        )

    def leq(self, a: str, b: str) -> bool:
        """Alias for dominates: a ≤ b."""
        return self.dominates(a, b)

    def is_dominated_by(self, a: str, b: str) -> bool:
        """Return True if a ≤ b."""
        return self.dominates(a, b)

    def is_comparable(self, a: str, b: str) -> bool:
        ia, ib = self._validate_element(a), self._validate_element(b)
        return bool(self._mat[ia, ib] or self._mat[ib, ia])

    def is_incomparable(self, a: str, b: str) -> bool:
        return not self.is_comparable(a, b)

    def comparability_set(self, e: str) -> List[str]:
        ie = self._validate_element(e)
        comparable = self._mat[ie, :] | self._mat[:, ie]
        comparable[ie] = False
        return [self._elements[j] for j in np.where(comparable)[0]]

    def incomparability_set(self, e: str) -> List[str]:
        ie = self._validate_element(e)
        comparable = self._mat[ie, :] | self._mat[:, ie]
        comparable[ie] = True  # exclude self
        return [self._elements[j] for j in np.where(~comparable)[0]]

    def incomparability_relation(self) -> List[Tuple[str, str]]:
        """Return all unordered pairs (a, b) that are incomparable."""
        el = self._elements
        incomp = ~self._mat & ~self._mat.T
        # Upper triangle only (avoid duplicates)
        rows, cols = np.where(np.triu(incomp, k=1))
        return [(el[i], el[j]) for i, j in zip(rows, cols)]

    # ------------------------------------------------------------------
    # Up-sets / down-sets
    # ------------------------------------------------------------------

    def upset_of(self, elems: List[str]) -> List[str]:
        """Return the upset (up-closure) of a set of elements."""
        indices = np.array([self._validate_element(e) for e in elems])
        # Union of rows: element j is in the upset iff any seed i ≤ j
        mask = self._mat[indices, :].any(axis=0)
        return [self._elements[j] for j in np.where(mask)[0]]

    def downset_of(self, elems: List[str]) -> List[str]:
        """Return the downset (down-closure) of a set of elements."""
        indices = np.array([self._validate_element(e) for e in elems])
        # Union of columns: element i is in the downset iff i ≤ any seed j
        mask = self._mat[:, indices].any(axis=1)
        return [self._elements[i] for i in np.where(mask)[0]]

    def is_upset(self, elems: List[str]) -> bool:
        return set(self.upset_of(elems)) == set(elems)

    def is_downset(self, elems: List[str]) -> bool:
        return set(self.downset_of(elems)) == set(elems)

    # ------------------------------------------------------------------
    # Extremal elements
    # ------------------------------------------------------------------

    def minimals(self) -> List[str]:
        """Elements with no strictly smaller element."""
        # Minimal i: no j != i with j < i, i.e. _strict[:, i] is all False
        has_predecessor = self._strict.any(axis=0)
        return [self._elements[i] for i in np.where(~has_predecessor)[0]]

    def maximals(self) -> List[str]:
        """Elements with no strictly larger element."""
        # Maximal i: no j != i with i < j, i.e. _strict[i, :] is all False
        has_successor = self._strict.any(axis=1)
        return [self._elements[i] for i in np.where(~has_successor)[0]]

    def is_minimal(self, e: str) -> bool:
        ie = self._validate_element(e)
        return not self._strict[:, ie].any()

    def is_maximal(self, e: str) -> bool:
        ie = self._validate_element(e)
        return not self._strict[ie, :].any()

    # ------------------------------------------------------------------
    # Lattice operations
    # ------------------------------------------------------------------

    def meet(self, elems: List[str]) -> Optional[str]:
        """Greatest lower bound of elements, or None if it doesn't exist."""
        targets = np.array([self._validate_element(e) for e in elems])

        # Lower bounds of targets: i such that i ≤ t for all t in targets
        # _mat[i, t] = True for all t  →  row-wise AND across target columns
        is_lower = self._mat[:, targets].all(axis=1)
        lower_idx = np.where(is_lower)[0]

        if len(lower_idx) == 0:
            return None

        # Greatest lower bound: candidate c such that every other lower
        # bound l satisfies l ≤ c  →  _mat[l, c] = True for all l
        for c in lower_idx:
            if self._mat[lower_idx, c].all():
                return self._elements[c]

        return None

    def join(self, elems: List[str]) -> Optional[str]:
        """Least upper bound of elements, or None if it doesn't exist."""
        targets = np.array([self._validate_element(e) for e in elems])

        # Upper bounds: j such that t ≤ j for all t in targets
        is_upper = self._mat[targets, :].all(axis=0)
        upper_idx = np.where(is_upper)[0]

        if len(upper_idx) == 0:
            return None

        # Least upper bound: candidate c such that c ≤ every other upper bound
        for c in upper_idx:
            if self._mat[c, upper_idx].all():
                return self._elements[c]

        return None

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------

    def is_extension_of(self, other: "POSet") -> bool:
        """True if self is an extension of other (other ⊆ self as order)."""
        if set(self._elements) != set(other._elements):
            return False
        # Reindex other's matrix to self's element order
        perm = np.array([other._idx[e] for e in self._elements])
        other_reindexed = other._mat[np.ix_(perm, perm)]
        # self is extension iff other._mat → self._mat (implication)
        return bool(np.all(~other_reindexed | self._mat))

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        covers = self.cover_relation()
        return f"POSet(elements={self._elements}, covers={covers})"

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
        # Build dominance matrix directly (upper-triangular + diagonal)
        mat = np.zeros((n, n), dtype=bool)
        for i in range(n):
            mat[i, i:] = True
        # Bypass pair-list construction: call POSet.__init__ with no dom,
        # then set _mat directly (already transitively closed).
        super().__init__(elements, dom=None)
        self._mat = mat
        self._strict = mat & ~mat.T


# ---------------------------------------------------------------------------
# BinaryVariablePOSet
# ---------------------------------------------------------------------------

class BinaryVariablePOSet(POSet):
    """
    Component-wise poset on binary vectors {0,1}^k.

    Given k binary variable names, builds the poset whose elements are
    all 2^k binary vectors, ordered component-wise.

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
        n = 2 ** k

        # Generate all 2^k binary vectors
        profiles = np.array(
            [[(i >> b) & 1 for b in range(k)] for i in range(n)],
            dtype=np.int8,
        )

        labels = [
            "(" + ",".join(str(x) for x in profiles[i]) + ")"
            for i in range(n)
        ]

        self._profiles = profiles
        self._variables = list(variables)

        # Vectorized component-wise dominance:
        # dom[i,j] = True iff profiles[i] <= profiles[j] component-wise
        # profiles[:, None, :] shape (n, 1, k)
        # profiles[None, :, :] shape (1, n, k)
        dom_mat = np.all(
            profiles[:, None, :] <= profiles[None, :, :], axis=2
        )

        # Initialise POSet with no dom pairs, then inject the matrix
        super().__init__(labels, dom=None)
        self._mat = dom_mat
        self._strict = dom_mat & ~dom_mat.T

    @property
    def variables(self):
        return list(self._variables)
