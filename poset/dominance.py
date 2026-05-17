"""
dominance.py - BLS dominance matrix.

Reference:
    Brueggemann R., Lerche D.B., Sørensen P.B. (2003).
    First attempts to relate structures of Hasse diagrams with mutual probabilities.
    NERI Technical Report No. 479.
"""

from __future__ import annotations
import numpy as np
from .poset import POSet


def BLSDominance(poset: POSet) -> np.ndarray:
    """
    Compute the BLS dominance matrix of a poset.

    The BLS (Brueggemann-Lerche-Sørensen) score for a pair (a, b) is:

        BLS(a,b) = |downset(b) ∩ upset(a)| / n

    where n is the number of poset elements. This gives a value in [0,1]
    representing how strongly b dominates a in the poset structure.

    Parameters
    ----------
    poset : POSet

    Returns
    -------
    np.ndarray (n×n), BLS[i,j] = BLS dominance of j over i.

    Examples
    --------
    >>> pos = POSet(['a','b','c','d'], [('a','b'),('c','b'),('b','d')])
    >>> D = BLSDominance(pos)
    """
    elements = poset.elements
    n = poset.n
    mat = poset.dominance_matrix()  # mat[i,j] = True iff i ≤ j

    bls = np.zeros((n, n))

    for i in range(n):
        # upset of elements[i]: all j such that i ≤ j
        up_i = set(j for j in range(n) if mat[i, j])
        for k in range(n):
            # downset of elements[k]: all j such that j ≤ k
            down_k = set(j for j in range(n) if mat[j, k])
            bls[i, k] = len(up_i & down_k) / n

    return bls
