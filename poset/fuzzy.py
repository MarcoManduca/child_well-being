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

# Sentinel objects identifying the built-in norm/conorm pairs so we can
# dispatch to fully-vectorized implementations instead of element-wise
# Python callables.
_NORM_MIN = "min"
_NORM_PRODUCT = "product"
_CONORM_MAX = "max"
_CONORM_PROBSUM = "probsum"


# ---------------------------------------------------------------------------
# Vectorized core: Separation
# ---------------------------------------------------------------------------

def _separation_vectorized(
    dom: np.ndarray,
    norm_id: str,
    conorm_id: str,
) -> Dict[str, np.ndarray]:
    """
    Fully vectorized separation for known norm/conorm pairs.

    comp_ij = 1 - dom[j, i]   (complement: degree j does NOT dominate i)
    comp_ji = 1 - dom[i, j]

    aL[i,j] = norm(dom[i,j], comp_ij)
    aU[i,j] = norm(dom[j,i], comp_ji)
    sym[i,j] = conorm(aL[i,j], aU[i,j])
    """
    d = dom
    dT = dom.T  # dT[i,j] = dom[j,i]

    comp_ij = 1.0 - dT   # 1 - dom[j,i]
    comp_ji = 1.0 - d     # 1 - dom[i,j]

    if norm_id == _NORM_MIN:
        aL = np.minimum(d, comp_ij)
        aU = np.minimum(dT, comp_ji)
    else:  # product
        aL = d * comp_ij
        aU = dT * comp_ji

    if conorm_id == _CONORM_MAX:
        sym = np.maximum(aL, aU)
    else:  # probabilistic sum
        sym = aL + aU - aL * aU

    vert = np.abs(aL - aU)
    horiz = sym - vert

    return {
        "asymmetricLower": aL,
        "asymmetricUpper": aU,
        "symmetric": sym,
        "vertical": vert,
        "horizontal": horiz,
    }


def _separation_generic(
    dom: np.ndarray,
    norm: Callable,
    conorm: Callable,
) -> Dict[str, np.ndarray]:
    """
    Fallback for arbitrary user-supplied norm/conorm.

    Vectorized where possible: builds aL/aU via np.vectorize,
    then derives sym/vert/horiz with array ops.
    """
    n = dom.shape[0]
    dT = dom.T

    comp_ij = 1.0 - dT
    comp_ji = 1.0 - dom

    vnorm = np.vectorize(norm)
    vconorm = np.vectorize(conorm)

    aL = vnorm(dom, comp_ij)
    aU = vnorm(dT, comp_ji)
    sym = vconorm(aL, aU)

    vert = np.abs(aL - aU)
    horiz = sym - vert

    return {
        "asymmetricLower": aL,
        "asymmetricUpper": aU,
        "symmetric": sym,
        "vertical": vert,
        "horizontal": horiz,
    }


# ---------------------------------------------------------------------------
# Vectorized core: In-betweenness
# ---------------------------------------------------------------------------

def _inbetweenness_vectorized(
    dom: np.ndarray,
    norm_id: str,
    conorm_id: str,
    types: tuple,
) -> Dict[str, np.ndarray]:
    """
    Fully vectorized in-betweenness for known norm/conorm pairs.

    Uses 3D broadcasting: for indices (i, j, k),
        asymmetricLower[i,j,k] = norm(dom[i,j], dom[j,k])
        asymmetricUpper[i,j,k] = norm(dom[k,j], dom[j,i])
        symmetric[i,j,k]       = conorm(lower, upper)

    Shapes:
        dom[i,j] → dom[:, :, None]   broadcast over k
        dom[j,k] → dom[None, :, :]   broadcast over i
        dom[k,j] → dom.T[None, :, :] = dom[:, :, None].T  (need care)
        dom[j,i] → dom.T[:, :, None]  broadcast over k
    """
    n = dom.shape[0]

    # dom_ij[i,j,k] = dom[i,j]  (constant over k)
    dom_ij = dom[:, :, None]  # (n, n, 1) → broadcasts to (n, n, n)
    # dom_jk[i,j,k] = dom[j,k]  (constant over i)
    dom_jk = dom[None, :, :]  # (1, n, n)
    # dom_kj[i,j,k] = dom[k,j]  (constant over i)
    dom_kj = dom.T[None, :, :]  # (1, n, n)  — dom.T[j,k] = dom[k,j]
    # dom_ji[i,j,k] = dom[j,i]  (constant over k)
    dom_ji = dom.T[:, :, None]  # (n, n, 1)  — dom.T[i,j] = dom[j,i]

    result = {}

    need_lower = "asymmetricLower" in types or "symmetric" in types
    need_upper = "asymmetricUpper" in types or "symmetric" in types

    if norm_id == _NORM_MIN:
        if need_lower:
            lower = np.minimum(dom_ij, dom_jk)
        if need_upper:
            upper = np.minimum(dom_kj, dom_ji)
    else:  # product
        if need_lower:
            lower = dom_ij * dom_jk
        if need_upper:
            upper = dom_kj * dom_ji

    if "asymmetricLower" in types:
        result["asymmetricLower"] = lower

    if "asymmetricUpper" in types:
        result["asymmetricUpper"] = upper

    if "symmetric" in types:
        if conorm_id == _CONORM_MAX:
            result["symmetric"] = np.maximum(lower, upper)
        else:  # probabilistic sum
            result["symmetric"] = lower + upper - lower * upper

    return result


def _inbetweenness_generic(
    dom: np.ndarray,
    norm: Callable,
    conorm: Callable,
    types: tuple,
) -> Dict[str, np.ndarray]:
    """
    Fallback for arbitrary user-supplied norm/conorm.

    Still uses 3D broadcasting via np.vectorize for the element-wise
    callables, which is faster than a triple Python loop.
    """
    n = dom.shape[0]

    dom_ij = dom[:, :, None]
    dom_jk = dom[None, :, :]
    dom_kj = dom.T[None, :, :]
    dom_ji = dom.T[:, :, None]

    vnorm = np.vectorize(norm)
    vconorm = np.vectorize(conorm)

    result = {}

    need_lower = "asymmetricLower" in types or "symmetric" in types
    need_upper = "asymmetricUpper" in types or "symmetric" in types

    if need_lower:
        lower = vnorm(dom_ij, dom_jk)
    if need_upper:
        upper = vnorm(dom_kj, dom_ji)

    if "asymmetricLower" in types:
        result["asymmetricLower"] = lower
    if "asymmetricUpper" in types:
        result["asymmetricUpper"] = upper
    if "symmetric" in types:
        result["symmetric"] = vconorm(lower, upper)

    return result


# ---------------------------------------------------------------------------
# Public API: In-betweenness
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
    return _inbetweenness_generic(dom, norm, conorm, types)


def FuzzyInBetweennessMinMax(
    dom: np.ndarray, *types: str
) -> Dict[str, np.ndarray]:
    """
    Fuzzy in-betweenness with minimum t-norm and maximum t-conorm.

    Fully vectorized (no Python loops).

    Examples
    --------
    >>> result = FuzzyInBetweennessMinMax(D, 'symmetric', 'asymmetricLower')
    """
    _validate_inbet_types(types)
    return _inbetweenness_vectorized(dom, _NORM_MIN, _CONORM_MAX, types)


def FuzzyInBetweennessProbabilistic(
    dom: np.ndarray, *types: str
) -> Dict[str, np.ndarray]:
    """
    Fuzzy in-betweenness with product t-norm and probabilistic-sum t-conorm.

    Fully vectorized (no Python loops).

    Examples
    --------
    >>> result = FuzzyInBetweennessProbabilistic(D, 'symmetric')
    """
    _validate_inbet_types(types)
    return _inbetweenness_vectorized(dom, _NORM_PRODUCT, _CONORM_PROBSUM, types)


# ---------------------------------------------------------------------------
# Public API: Separation
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
        'symmetric', 'asymmetricLower', 'asymmetricUpper',
        'vertical', 'horizontal'.

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
    all_mats = _separation_generic(dom, norm, conorm)
    return {t: all_mats[t] for t in types}


def FuzzySeparationMinMax(
    dom: np.ndarray, *types: str
) -> Dict[str, np.ndarray]:
    """
    Fuzzy separation with minimum t-norm and maximum t-conorm.

    Fully vectorized (no Python loops).

    Examples
    --------
    >>> result = FuzzySeparationMinMax(D, 'symmetric', 'vertical')
    """
    _validate_sep_types(types)
    all_mats = _separation_vectorized(dom, _NORM_MIN, _CONORM_MAX)
    return {t: all_mats[t] for t in types}


def FuzzySeparationProbabilistic(
    dom: np.ndarray, *types: str
) -> Dict[str, np.ndarray]:
    """
    Fuzzy separation with product t-norm and probabilistic-sum t-conorm.

    Fully vectorized (no Python loops).

    Examples
    --------
    >>> result = FuzzySeparationProbabilistic(D, 'symmetric', 'asymmetricLower')
    """
    _validate_sep_types(types)
    all_mats = _separation_vectorized(dom, _NORM_PRODUCT, _CONORM_PROBSUM)
    return {t: all_mats[t] for t in types}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_inbet_types(types):
    if not types:
        raise ValueError("Specify at least one in-betweenness type.")
    for t in types:
        if t not in _INBET_TYPES:
            raise ValueError(
                f"Unknown in-betweenness type '{t}'.  "
                f"Choose from {_INBET_TYPES}."
            )


def _validate_sep_types(types):
    if not types:
        raise ValueError("Specify at least one separation type.")
    for t in types:
        if t not in _SEP_TYPES:
            raise ValueError(
                f"Unknown separation type '{t}'.  "
                f"Choose from {_SEP_TYPES}."
            )
