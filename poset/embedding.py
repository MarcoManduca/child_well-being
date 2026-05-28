"""
embedding.py - Bidimensional representation of multidimensional ordinal binary data.

Implements:
  - BidimensionalPosetRepresentation  : representation from a given variable priority
  - OptimalBidimensionalEmbedding     : find the optimal representation over all permutations

Reference:
    Arcagni A., Fattore M. (2014).
    PARSEC: An R Package for Poset-Based Analysis of Multidimensional Poverty.
    Journal of Statistical Software, 62(4).
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def BidimensionalPosetRepresentation(
    profile: np.ndarray,
    weights: np.ndarray,
    variables_priority: List[int],
) -> Dict:
    """
    Bidimensional representation from a reversed pair of lexicographic extensions.

    Parameters
    ----------
    profile : np.ndarray (m × k), binary (0/1)
        Unique observed profiles (one per row).
    weights : np.ndarray (m,)
        Frequencies / weights of each profile.
    variables_priority : list of int (length k)
        Permutation of 0..k-1 defining the priority.
        First linear extension sorts by variable priority[0] first, etc.
        Second (reversed) sorts in reverse order.

    Returns
    -------
    dict with keys:
        'LossValue' : float (global error)
        'Representation' : dict with arrays 'profiles', 'x', 'y', 'weights', 'error'

    Examples
    --------
    >>> k = 4
    >>> profiles = np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[1,1,0,0]])
    >>> weights = np.array([10, 5, 7, 3])
    >>> vp = [0, 1, 2, 3]
    >>> result = BidimensionalPosetRepresentation(profiles, weights, vp)
    """
    profile = np.asarray(profile, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float64)
    m, k = profile.shape
    perm = list(variables_priority)

    # Build two linear extensions (forward and reversed priority)
    le1_order = _lex_sort_order(profile, perm)
    le2_order = _lex_sort_order(profile, perm[::-1])

    # Coordinates = rank position in each linear extension
    x_coords = np.empty(m, dtype=np.int32)
    y_coords = np.empty(m, dtype=np.int32)
    x_coords[le1_order] = np.arange(m)
    y_coords[le2_order] = np.arange(m)

    # Compute per-profile approximation error (vectorized)
    errors = _compute_errors(profile, weights, x_coords, y_coords)
    total_loss = float(np.dot(weights, errors) / weights.sum())

    # Profile labels: base-10 representation
    base10 = _profiles_to_base10(profile)

    return {
        "LossValue": total_loss,
        "Representation": {
            "profiles": base10,
            "x": x_coords,
            "y": y_coords,
            "weights": weights,
            "error": errors,
        },
    }


def OptimalBidimensionalEmbedding(
    profile: np.ndarray,
    weights: np.ndarray,
    output_every_sec: Optional[int] = None,
    thread_share: float = 1.0,
) -> Dict:
    """
    Find the optimal bidimensional embedding over all k!/2 reversed lex pairs.

    Parameters
    ----------
    profile : np.ndarray (m × k), binary (0/1)
    weights : np.ndarray (m,)
    output_every_sec : int, optional
    thread_share : float in (0,1], ignored (Python version)

    Returns
    -------
    dict with keys:
        'allLoss'               : np.ndarray (k!/2,) of loss values
        'variablesPriority'     : np.ndarray (k!/2, k) of permutations
        'bestLossValue'         : float
        'bestVariablePriority'  : list of int
        'bestRepresentation'    : same structure as BidimensionalPosetRepresentation

    Examples
    --------
    >>> k = 4
    >>> profiles = np.array(list(itertools.product([0,1], repeat=k)))
    >>> weights = np.random.randint(1, 100, len(profiles))
    >>> result = OptimalBidimensionalEmbedding(profiles, weights)
    """
    profile = np.asarray(profile, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float64)
    m, k = profile.shape

    # Pre-compute the component-wise comparability matrices once
    # (shared across all permutations — the true order doesn't change)
    true_le, true_ge = _precompute_true_order(profile)

    all_perms = list(itertools.permutations(range(k)))
    # Each reversed pair counted once
    half = all_perms[: len(all_perms) // 2 + len(all_perms) % 2]

    all_loss = np.empty(len(half), dtype=np.float64)
    all_vp = np.empty((len(half), k), dtype=np.int32)

    best_loss = np.inf
    best_vp = None
    best_repr = None

    start = time.time()
    last_print = start

    for idx, perm in enumerate(half):
        vp = list(perm)

        le1_order = _lex_sort_order(profile, vp)
        le2_order = _lex_sort_order(profile, vp[::-1])

        x = np.empty(m, dtype=np.int32)
        y = np.empty(m, dtype=np.int32)
        x[le1_order] = np.arange(m)
        y[le2_order] = np.arange(m)

        errors = _compute_errors_precomputed(
            true_le, true_ge, weights, x, y
        )
        loss = float(np.dot(weights, errors) / weights.sum())

        all_loss[idx] = loss
        all_vp[idx] = vp

        if loss < best_loss:
            best_loss = loss
            best_vp = vp
            # Store full representation for the best
            base10 = _profiles_to_base10(profile)
            best_repr = {
                "LossValue": loss,
                "Representation": {
                    "profiles": base10,
                    "x": x.copy(),
                    "y": y.copy(),
                    "weights": weights,
                    "error": errors,
                },
            }

        if output_every_sec is not None:
            now = time.time()
            if now - last_print >= output_every_sec:
                print(
                    f"  Analyzed {idx + 1}/{len(half)} pairs "
                    f"(best loss so far: {best_loss:.4f})..."
                )
                last_print = now

    return {
        "allLoss": all_loss,
        "variablesPriority": all_vp,
        "bestLossValue": best_loss,
        "bestVariablePriority": best_vp,
        "bestRepresentation": best_repr,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lex_sort_order(profile: np.ndarray, perm: List[int]) -> np.ndarray:
    """Return sort indices for profiles under lex order given by *perm*."""
    keys = tuple(profile[:, p] for p in reversed(perm))
    return np.lexsort(keys)


def _profiles_to_base10(profile: np.ndarray) -> np.ndarray:
    """Convert binary rows to base-10 integers."""
    k = profile.shape[1]
    powers = 2 ** np.arange(k)
    return profile @ powers


# ---------------------------------------------------------------------------
# True-order pre-computation (shared across permutations)
# ---------------------------------------------------------------------------

def _precompute_true_order(
    profile: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute pairwise component-wise comparability.

    Returns
    -------
    true_le : np.ndarray (m × m) bool
        true_le[i, j] = True iff profile[i] ≤ profile[j] component-wise.
    true_ge : np.ndarray (m × m) bool
        true_ge[i, j] = True iff profile[j] ≤ profile[i] component-wise.
    """
    m, k = profile.shape

    # profile[:, None, :] shape (m, 1, k)
    # profile[None, :, :] shape (1, m, k)
    # comparison broadcasts to (m, m, k), then .all(axis=2) → (m, m)
    true_le = np.all(
        profile[:, None, :] <= profile[None, :, :], axis=2
    )
    # true_ge[i,j] = true_le[j,i]
    true_ge = true_le.T

    return true_le, true_ge


# ---------------------------------------------------------------------------
# Error computation (vectorized)
# ---------------------------------------------------------------------------

def _compute_errors(
    profile: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Per-profile approximation error (standalone version).

    For each profile i, measures how much the 2D embedding disagrees
    with the true component-wise partial order, weighted by profile
    frequencies.

    True order:  i ≤_cmp j  iff all components of profile[i] ≤ profile[j].
    2D order:    i ≤_2d  j  iff x[i] ≤ x[j] and y[i] ≤ y[j].

    Error sources:
      - Truly comparable pair rendered incomparable in 2D.
      - Truly incomparable pair rendered strictly comparable in 2D.
    """
    true_le, true_ge = _precompute_true_order(profile)
    return _compute_errors_precomputed(true_le, true_ge, weights, x, y)


def _compute_errors_precomputed(
    true_le: np.ndarray,
    true_ge: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Per-profile approximation error using pre-computed true order matrices.

    Parameters
    ----------
    true_le : (m, m) bool — true_le[i,j] iff i ≤ j component-wise
    true_ge : (m, m) bool — true_ge[i,j] iff j ≤ i component-wise
    weights : (m,) float
    x, y    : (m,) int — 2D coordinates

    Returns
    -------
    errors : (m,) float — per-profile weighted error
    """
    m = len(weights)
    total_weight = weights.sum()

    # 2D comparability matrices (vectorized)
    # x2d_le[i,j] = True iff x[i] <= x[j] AND y[i] <= y[j]
    x2d_le = (x[:, None] <= x[None, :]) & (y[:, None] <= y[None, :])
    x2d_ge = x2d_le.T

    # True comparable: at least one direction holds
    true_comparable = true_le | true_ge

    # 2D comparable: at least one direction holds
    x2d_comparable = x2d_le | x2d_ge

    # 2D strictly comparable (one direction but not both, i.e. not equal)
    x2d_strict = x2d_comparable & ~(x2d_le & x2d_ge)

    # Error type 1: truly comparable but 2D incomparable
    err1 = true_comparable & ~x2d_comparable

    # Error type 2: truly incomparable but 2D strictly comparable
    err2 = ~true_comparable & x2d_strict

    # Combined error matrix (exclude diagonal)
    err_matrix = (err1 | err2).astype(np.float64)
    np.fill_diagonal(err_matrix, 0.0)

    # Weighted error per profile: sum over j of weights[j] * err[i,j]
    errors = (err_matrix @ weights) / total_weight

    return errors
