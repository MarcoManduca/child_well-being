"""
mrp.py - Mutual Ranking Probabilities (MRP).

MRP(a, b) = share of linear extensions where b dominates a.

Implements:
  - ExactMRP           : over all linear extensions
  - BubleyDyerMRPGenerator / BubleyDyerMRP : approximate via MCMC
  - LexMRP             : over lexicographic linear extensions
"""

from __future__ import annotations
from typing import Optional, List, Dict, Union
import numpy as np
import math
import time
import itertools

from .poset import POSet
from .linear_extensions import LEGenerator, LEBubleyDyer, LEGet


# ---------------------------------------------------------------------------
# Exact MRP
# ---------------------------------------------------------------------------

def ExactMRP(
    poset: POSet,
    output_every_sec: Optional[int] = None,
) -> Dict:
    """
    Compute the exact MRP matrix over all linear extensions.

    MRP[i,j] = P(element_j dominates element_i) across all linear extensions.

    Parameters
    ----------
    poset : POSet
    output_every_sec : int, optional

    Returns
    -------
    dict with keys:
        'MRP'  : np.ndarray (n×n) of MRP values
        'n_extensions' : int
        'elements' : list of str

    Examples
    --------
    >>> result = ExactMRP(pos)
    >>> result['MRP']
    """
    gen = LEGenerator(poset)
    n_elem = poset.n
    elements = poset.elements
    mrp = np.zeros((n_elem, n_elem))

    gen.reset()
    start = time.time()
    last_print = start
    count = 0

    all_le = gen.get_batch()  # all extensions
    for le in all_le:
        count += 1
        pos_map = {e: idx for idx, e in enumerate(le)}
        for i in range(n_elem):
            for j in range(n_elem):
                if pos_map[elements[j]] >= pos_map[elements[i]]:
                    mrp[i, j] += 1
        if output_every_sec is not None:
            now = time.time()
            if now - last_print >= output_every_sec:
                print(f"  Processed {count} linear extensions...")
                last_print = now

    if count > 0:
        mrp /= count

    return {
        'MRP': mrp,
        'n_extensions': count,
        'elements': elements,
    }


# ---------------------------------------------------------------------------
# Bubley-Dyer MRP Generator + Computer
# ---------------------------------------------------------------------------

class _BubleyDyerMRPGen:
    """Internal state holder for incremental MRP computation."""
    def __init__(self, poset: POSet, seed: Optional[int] = None):
        self.poset = poset
        self._sampler = LEBubleyDyer(poset, seed=seed)
        n = poset.n
        self._mrp_sum = np.zeros((n, n))
        self._total_count = 0

    def update(
        self,
        n: Optional[int] = None,
        error: Optional[float] = None,
        output_every_sec: Optional[int] = None,
    ) -> Dict:
        E = self.poset.n
        elements = self.poset.elements

        if error is not None and n is None:
            ln_E = math.log(E) if E > 1 else 1
            n_eps = int(E**4 * ln_E**2 + E**3 * ln_E * math.log(1 / error))
            additional = max(n_eps - self._total_count, 0)
            if additional == 0:
                print("Warning: desired precision already achieved; no new extensions generated.")
                mrp = self._mrp_sum / self._total_count if self._total_count else self._mrp_sum
                return {'MRP': mrp, 'n_extensions': self._total_count, 'elements': elements}
            n = additional

        batch = self._sampler.sample_batch(n=n, error=None,
                                           output_every_sec=output_every_sec)
        n_elem = E
        for le in batch:
            pos_map = {e: idx for idx, e in enumerate(le)}
            for i in range(n_elem):
                for j in range(n_elem):
                    if pos_map[elements[j]] >= pos_map[elements[i]]:
                        self._mrp_sum[i, j] += 1
        self._total_count += len(batch)

        mrp = self._mrp_sum / self._total_count if self._total_count else self._mrp_sum
        return {'MRP': mrp, 'n_extensions': self._total_count, 'elements': elements}


def BubleyDyerMRPGenerator(poset: POSet, seed: Optional[int] = None) -> _BubleyDyerMRPGen:
    """
    Create a Bubley-Dyer MRP generator.

    Parameters
    ----------
    poset : POSet
    seed : int, optional

    Returns
    -------
    _BubleyDyerMRPGen object (pass to BubleyDyerMRP)

    Examples
    --------
    >>> gen = BubleyDyerMRPGenerator(pos)
    >>> result = BubleyDyerMRP(gen, n=10000)
    """
    return _BubleyDyerMRPGen(poset, seed=seed)


def BubleyDyerMRP(
    generator: _BubleyDyerMRPGen,
    n: Optional[int] = None,
    error: Optional[float] = None,
    output_every_sec: Optional[int] = None,
) -> Dict:
    """
    Compute (or refine) an approximate MRP matrix via Bubley-Dyer sampling.

    Parameters
    ----------
    generator : _BubleyDyerMRPGen
        Created by BubleyDyerMRPGenerator().
    n : int, optional
        Number of extensions to sample.
    error : float in (0,1), optional
        Desired distance from uniformity.
    output_every_sec : int, optional

    Returns
    -------
    dict with keys 'MRP', 'n_extensions', 'elements'

    Examples
    --------
    >>> gen = BubleyDyerMRPGenerator(pos)
    >>> result = BubleyDyerMRP(gen, n=50000)
    >>> result = BubleyDyerMRP(gen, n=10000)   # refinement
    """
    return generator.update(n=n, error=error, output_every_sec=output_every_sec)


# ---------------------------------------------------------------------------
# LexMRP
# ---------------------------------------------------------------------------

def LexMRP(nvar: int, deg) -> np.ndarray:
    """
    MRP matrix computed over lexicographic linear extensions of the
    component-wise Boolean lattice on k ordinal variables.

    Parameters
    ----------
    nvar : int
        Number of ordinal variables k.
    deg : int | list of str | list of int | list of list of str
        Degrees specification:
        - int m  → all variables have degrees 0,1,...,m-1
        - list of str → custom labels, same for all variables
        - list of int (length k) → per-variable degree counts
        - list of list of str → per-variable custom labels

    Returns
    -------
    np.ndarray (N×N) MRP matrix, where N = prod(deg_i)

    Examples
    --------
    >>> MRP = LexMRP(3, 4)          # 3 vars, 4 levels each
    >>> MRP = LexMRP(3, ['a','b','c','d'])
    """
    labels_per_var = _parse_deg(nvar, deg)
    profiles, profile_labels = _build_profiles(labels_per_var)
    N = len(profiles)

    # Build two reversed lexicographic extensions for each priority
    # Average MRP over all k!/2 pairs (as in the R package)
    perms = list(itertools.permutations(range(nvar)))
    # Use only half (reversed pairs counted once)
    half_perms = perms[:len(perms)//2 + (len(perms) % 2)]

    mrp_sum = np.zeros((N, N))
    count = 0

    for perm in perms:
        # Forward lex extension: sort profiles by perm[0], then perm[1], ...
        le_fwd = sorted(range(N), key=lambda idx: tuple(profiles[idx][k] for k in perm))
        le_rev = le_fwd[::-1]

        for le in [le_fwd, le_rev]:
            pos_map = {le[i]: i for i in range(N)}
            for i in range(N):
                for j in range(N):
                    if pos_map[j] >= pos_map[i]:
                        mrp_sum[i, j] += 1
            count += 1

    mrp = mrp_sum / count
    return mrp, profile_labels


# ---------------------------------------------------------------------------
# Helpers shared with separation.py
# ---------------------------------------------------------------------------

def _parse_deg(nvar: int, deg):
    """Return list of lists of labels, one list per variable."""
    if isinstance(deg, int):
        return [list(range(deg)) for _ in range(nvar)]
    elif isinstance(deg, (list, tuple)):
        if len(deg) == 0:
            raise ValueError("deg must not be empty.")
        first = deg[0]
        if isinstance(first, int):
            # list of ints → per-variable degree counts
            if len(deg) == nvar:
                return [list(range(d)) for d in deg]
            else:
                # single int in a list? treat as common degrees
                return [list(deg) for _ in range(nvar)]
        elif isinstance(first, str):
            # Common labels for all variables
            return [list(deg) for _ in range(nvar)]
        elif isinstance(first, (list, tuple)):
            # Per-variable labels
            return [list(d) for d in deg]
    raise ValueError(f"Cannot parse deg={deg!r}.")


def _build_profiles(labels_per_var):
    """Return (profiles as index tuples, profile label strings)."""
    import itertools
    combos = list(itertools.product(*[range(len(lv)) for lv in labels_per_var]))
    labels = []
    for combo in combos:
        lbl = "(" + ",".join(str(labels_per_var[k][v]) for k, v in enumerate(combo)) + ")"
        labels.append(lbl)
    return combos, labels
