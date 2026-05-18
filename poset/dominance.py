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
    n = poset.n
    mat = poset.dominance_matrix().astype(np.float64)  # mat[i,j] = 1.0 iff i ≤ j

    # upset(i)  = mat[i, :]   → row i,  shape (n,)
    # downset(k) = mat[:, k]  → col k,  shape (n,)
    # |upset(i) ∩ downset(k)| = dot product of row i with col k
    # BLS[i,k] = (mat[i,:] · mat[:,k]) / n  for all i,k
    # → BLS = mat @ mat / n

    bls = (mat @ mat) / n

    return bls
