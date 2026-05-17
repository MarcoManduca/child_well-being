"""
fuzzy.py - Fuzzy in-betweenness and fuzzy separation.

All functions take a dominance matrix (n×n) as input, which can come from
BLSDominance(), ExactMRP()['MRP'], or BubleyDyerMRP()['MRP'].

Reference:
    Fattore M., De Capitani L., Avellone A., Suardi A. (2024).
    A fuzzy posetic toolbox for multi-criteria evaluation on ordinal data systems.
    Annals of Operations Research. doi:10.1007/s10479-024-06352-3
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional
import numpy as np

_INBET_TYPES = {"symmetric", "asymmetricLower", "asymmetricUpper"}
_SEP_TYPES = {"symmetric", "asymmetricLower", "asymmetricUpper", "vertical", "horizontal"}


# ---------------------------------------------------------------------------
# In-betweenness
# ---------------------------------------------------------------------------

def FuzzyInBetweenness(
    dom: np.ndarray,
    norm: Callable,
    conorm: Callable,
    *types: str,
) -> Dict[str, np.ndarray]:
    """
    Compute fuzzy in-betweenness arrays with user-supplied t-norm and t-conorm.

    finb(pi, pj, pk) for symmetric in-betweenness:
        T(dom(pi,pj), dom(pj,pk)) ⊕ T(dom(pk,pj), dom(pj,pi))

    Parameters
    ----------
    dom : np.ndarray (n×n)
        Dominance degree matrix (values in [0,1]).
    norm : callable (x, y) → float
        t-norm (e.g. min, product).
    conorm : callable (x, y) → float
        t-conorm (e.g. max, probabilistic sum).
    *types : str
        'symmetric', 'asymmetricLower', 'asymmetricUpper'.

    Returns
    -------
    dict mapping type name → 3D array of shape (n, n, n).

    Examples
    --------
    >>> D = BLSDominance(pos)
    >>> tnorm = lambda x, y: x * y
    >>> tconorm = lambda x, y: x + y - x * y
    >>> result = FuzzyInBetweenness(D, tnorm, tconorm, 'symmetric')
    >>> result['symmetric'].shape   # (n, n, n)
    """
    _validate_inbet_types(types)
    n = dom.shape[0]
    result = {}

    for t in types:
        arr = np.zeros((n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if t == 'symmetric':
                        # finb(pi, pj, pk) = T(dom_ij, dom_jk) ⊕ T(dom_kj, dom_ji)
                        v = conorm(
                            norm(dom[i, j], dom[j, k]),
                            norm(dom[k, j], dom[j, i])
                        )
                    elif t == 'asymmetricLower':
                        # pi < pj < pk
                        v = norm(dom[i, j], dom[j, k])
                    else:  # asymmetricUpper: pk < pj < pi
                        v = norm(dom[k, j], dom[j, i])
                    arr[i, j, k] = v
        result[t] = arr

    return result


def FuzzyInBetweennessMinMax(dom: np.ndarray, *types: str) -> Dict[str, np.ndarray]:
    """
    Fuzzy in-betweenness with minimum t-norm and maximum t-conorm.

    Examples
    --------
    >>> result = FuzzyInBetweennessMinMax(D, 'symmetric', 'asymmetricLower')
    """
    return FuzzyInBetweenness(
        dom,
        norm=lambda x, y: min(x, y),
        conorm=lambda x, y: max(x, y),
        *types,
    )


def FuzzyInBetweennessProbabilistic(dom: np.ndarray, *types: str) -> Dict[str, np.ndarray]:
    """
    Fuzzy in-betweenness with product t-norm and probabilistic-sum t-conorm.

    Examples
    --------
    >>> result = FuzzyInBetweennessProbabilistic(D, 'symmetric')
    """
    return FuzzyInBetweenness(
        dom,
        norm=lambda x, y: x * y,
        conorm=lambda x, y: x + y - x * y,
        *types,
    )


# ---------------------------------------------------------------------------
# Fuzzy Separation
# ---------------------------------------------------------------------------

def FuzzySeparation(
    dom: np.ndarray,
    norm: Callable,
    conorm: Callable,
    *types: str,
) -> Dict[str, np.ndarray]:
    """
    Compute fuzzy separation matrices with user-supplied t-norm and t-conorm.

    Fuzzy symmetric separation between i and j:
        FuzSep(i,j) = T(dom_ij, 1-dom_ji) ⊕ T(dom_ji, 1-dom_ij)

    Fuzzy asymmetric lower (i < j):
        FuzAL(i,j) = T(dom_ij, 1-dom_ji)

    Fuzzy asymmetric upper (j < i):
        FuzAU(i,j) = T(dom_ji, 1-dom_ij)

    Vertical:   |FuzAL(i,j) - FuzAU(i,j)|
    Horizontal: FuzSep(i,j) - Vertical(i,j)

    Parameters
    ----------
    dom : np.ndarray (n×n)
    norm : callable
    conorm : callable
    *types : str

    Returns
    -------
    dict mapping type name → np.ndarray (n×n)

    Examples
    --------
    >>> D = BLSDominance(pos)
    >>> tnorm = lambda x, y: x * y
    >>> tconorm = lambda x, y: x + y - x * y
    >>> result = FuzzySeparation(D, tnorm, tconorm, 'symmetric', 'vertical')
    """
    _validate_sep_types(types)
    n = dom.shape[0]
    result = {}

    sym = np.zeros((n, n))
    aL = np.zeros((n, n))
    aU = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            comp_ij = 1.0 - dom[j, i]  # complement: degree to which j does NOT dominate i
            comp_ji = 1.0 - dom[i, j]
            al = norm(dom[i, j], comp_ij)
            au = norm(dom[j, i], comp_ji)
            s = conorm(al, au)
            aL[i, j] = al
            aU[i, j] = au
            sym[i, j] = s

    vert = np.abs(aL - aU)
    horiz = sym - vert

    for t in types:
        if t == 'symmetric':
            result[t] = sym
        elif t == 'asymmetricLower':
            result[t] = aL
        elif t == 'asymmetricUpper':
            result[t] = aU
        elif t == 'vertical':
            result[t] = vert
        elif t == 'horizontal':
            result[t] = horiz

    return result


def FuzzySeparationMinMax(dom: np.ndarray, *types: str) -> Dict[str, np.ndarray]:
    """
    Fuzzy separation with minimum t-norm and maximum t-conorm.

    Examples
    --------
    >>> result = FuzzySeparationMinMax(D, 'symmetric', 'vertical')
    """
    return FuzzySeparation(
        dom,
        norm=lambda x, y: min(x, y),
        conorm=lambda x, y: max(x, y),
        *types,
    )


def FuzzySeparationProbabilistic(dom: np.ndarray, *types: str) -> Dict[str, np.ndarray]:
    """
    Fuzzy separation with product t-norm and probabilistic-sum t-conorm.

    Examples
    --------
    >>> result = FuzzySeparationProbabilistic(D, 'symmetric', 'asymmetricLower')
    """
    return FuzzySeparation(
        dom,
        norm=lambda x, y: x * y,
        conorm=lambda x, y: x + y - x * y,
        *types,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_inbet_types(types):
    for t in types:
        if t not in _INBET_TYPES:
            raise ValueError(f"Unknown in-betweenness type '{t}'. Choose from {_INBET_TYPES}.")
    if not types:
        raise ValueError("Specify at least one in-betweenness type.")


def _validate_sep_types(types):
    for t in types:
        if t not in _SEP_TYPES:
            raise ValueError(f"Unknown separation type '{t}'. Choose from {_SEP_TYPES}.")
    if not types:
        raise ValueError("Specify at least one separation type.")
