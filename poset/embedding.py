"""
embedding.py - Bidimensional representation of multidimensional ordinal binary data.

Implements:
  - BidimentionalPosetRepresentation  : representation from a given variable priority
  - OptimalBidimensionalEmbedding     : find the optimal representation over all permutations
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
import itertools
import time


def BidimentionalPosetRepresentation(
    profile: np.ndarray,
    weights: np.ndarray,
    variables_priority: List[int],
) -> dict:
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
    >>> result = BidimentionalPosetRepresentation(profiles, weights, vp)
    """
    profile = np.asarray(profile, dtype=int)
    weights = np.asarray(weights, dtype=float)
    m, k = profile.shape
    perm = list(variables_priority)

    # Build two linear extensions
    # Forward: sort by perm[0] first, perm[1] second, ...
    le1_order = _lex_sort_order(profile, perm)
    le2_order = _lex_sort_order(profile, perm[::-1])

    # x-coordinate = position in le1, y-coordinate = position in le2
    x_coords = np.argsort(le1_order)
    y_coords = np.argsort(le2_order)

    # Compute per-profile approximation error
    # Error L(b | Din, p) = weighted share of pairs where the 2D order violates the true order
    errors = _compute_errors(profile, weights, x_coords, y_coords)
    total_loss = float(np.sum(weights * errors) / np.sum(weights))

    # Profile labels: base-10 representation
    base10 = _profiles_to_base10(profile)

    return {
        'LossValue': total_loss,
        'Representation': {
            'profiles': base10,
            'x': x_coords,
            'y': y_coords,
            'weights': weights,
            'error': errors,
        }
    }


def OptimalBidimensionalEmbedding(
    profile: np.ndarray,
    weights: np.ndarray,
    output_every_sec: Optional[int] = None,
    thread_share: float = 1.0,
) -> dict:
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
        'allLoss' : np.ndarray (k!/2,) of loss values
        'variablesPriority' : np.ndarray (k!/2, k) of permutations
        'bestLossValue' : float
        'bestVariablePriority' : list of int
        'bestRepresentation' : same structure as BidimentionalPosetRepresentation output

    Examples
    --------
    >>> k = 4
    >>> profiles = np.array(list(itertools.product([0,1], repeat=k)))
    >>> weights = np.random.randint(1, 100, len(profiles))
    >>> result = OptimalBidimensionalEmbedding(profiles, weights)
    """
    profile = np.asarray(profile, dtype=int)
    weights = np.asarray(weights, dtype=float)
    m, k = profile.shape

    all_perms = list(itertools.permutations(range(k)))
    # Use only the first half (each reversed pair counted once)
    half = all_perms[:len(all_perms)//2 + len(all_perms) % 2]

    all_loss = []
    all_vp = []
    start = time.time()
    last_print = start

    best_loss = np.inf
    best_vp = None
    best_repr = None

    for idx, perm in enumerate(half):
        vp = list(perm)
        res = BidimentionalPosetRepresentation(profile, weights, vp)
        loss = res['LossValue']
        all_loss.append(loss)
        all_vp.append(vp)

        if loss < best_loss:
            best_loss = loss
            best_vp = vp
            best_repr = res

        if output_every_sec is not None:
            now = time.time()
            if now - last_print >= output_every_sec:
                print(f"  Analyzed {idx+1}/{len(half)} pairs "
                      f"(best loss so far: {best_loss:.4f})...")
                last_print = now

    return {
        'allLoss': np.array(all_loss),
        'variablesPriority': np.array(all_vp),
        'bestLossValue': best_loss,
        'bestVariablePriority': best_vp,
        'bestRepresentation': best_repr,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lex_sort_order(profile: np.ndarray, perm: List[int]) -> np.ndarray:
    """Return sort indices for profiles under lex order given by perm."""
    keys = tuple(profile[:, p] for p in reversed(perm))
    return np.lexsort(keys)


def _profiles_to_base10(profile: np.ndarray) -> np.ndarray:
    """Convert binary rows to base-10 integers."""
    k = profile.shape[1]
    powers = 2 ** np.arange(k)
    return profile @ powers


def _compute_errors(
    profile: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Per-profile approximation error.

    For each profile b, count the share of other profiles b' where
    the true order (component-wise) is violated in the 2D representation.

    True order: b ≤_cmp b' iff all components of b ≤ b'.
    2D representation order: b ≤_2d b' iff x[b] ≤ x[b'] and y[b] ≤ y[b'].

    Error for profile b = fraction of pairs where true order and 2D order disagree.
    """
    m, k = profile.shape
    total_weight = weights.sum()
    errors = np.zeros(m)

    for i in range(m):
        err = 0.0
        for j in range(m):
            if i == j:
                continue
            # True order: i ≤_cmp j?
            true_le = bool(np.all(profile[i] <= profile[j]))
            true_ge = bool(np.all(profile[j] <= profile[i]))
            # 2D order
            x2d_le = (x[i] <= x[j]) and (y[i] <= y[j])
            x2d_ge = (x[j] <= x[i]) and (y[j] <= y[i])

            # Error: true comparable but 2D incomparable, or vice versa
            if (true_le or true_ge) and not (x2d_le or x2d_ge):
                err += weights[j]
            elif not (true_le or true_ge) and (x2d_le or x2d_ge) and i != j:
                # Only count strictly comparable in 2D (not equal)
                if x2d_le and not x2d_ge or x2d_ge and not x2d_le:
                    err += weights[j]

        errors[i] = err / total_weight

    return errors
