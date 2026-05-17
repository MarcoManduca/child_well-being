"""
separation.py - Separation matrices computation.

Types of separation:
  - symmetric       : mean |pos(a) - pos(b)|
  - asymmetricLower : mean (pos(b)-pos(a)) * I(a < b in LE)
  - asymmetricUpper : mean (pos(a)-pos(b)) * I(b < a in LE)
  - vertical        : |asymLower - asymUpper|
  - horizontal      : symmetric - vertical

Reference:
    Fattore et al. (2024). Annals of Operations Research.
"""

from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np
import math
import time
import itertools

from .poset import POSet
from .linear_extensions import LEGenerator, LEBubleyDyer
from .mrp import _parse_deg, _build_profiles

_VALID_TYPES = {"symmetric", "asymmetricLower", "asymmetricUpper", "vertical", "horizontal"}


def _accumulate_sep(le: List[str], elements: List[str], n: int, accum: Dict):
    """Update separation accumulators from one linear extension."""
    pos_map = {e: idx for idx, e in enumerate(le)}
    for i in range(n):
        for j in range(n):
            a, b = elements[i], elements[j]
            pi, pj = pos_map[a], pos_map[b]
            diff = pj - pi  # positive if b is after a
            accum['sym'][i, j] += abs(diff)
            if diff > 0:  # a < b in LE
                accum['aL'][i, j] += diff
            elif diff < 0:  # b < a in LE
                accum['aU'][i, j] += (-diff)


def _finalize_sep(accum: Dict, count: int, types: List[str]) -> Dict:
    sym = accum['sym'] / count
    aL = accum['aL'] / count
    aU = accum['aU'] / count
    vert = np.abs(aL - aU)
    horiz = sym - vert

    result = {}
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


def _empty_accum(n: int) -> Dict:
    return {k: np.zeros((n, n)) for k in ('sym', 'aL', 'aU')}


# ---------------------------------------------------------------------------
# ExactSeparation
# ---------------------------------------------------------------------------

def ExactSeparation(
    poset: POSet,
    *types: str,
    output_every_sec: Optional[int] = None,
) -> Dict:
    """
    Compute exact separation matrices over all linear extensions.

    Parameters
    ----------
    poset : POSet
    *types : str
        One or more of: 'symmetric', 'asymmetricLower', 'asymmetricUpper',
        'vertical', 'horizontal'.
    output_every_sec : int, optional

    Returns
    -------
    dict with separation matrices and 'n_extensions'.

    Examples
    --------
    >>> result = ExactSeparation(pos, 'symmetric', 'vertical')
    >>> result['symmetric']
    """
    _validate_types(types)
    elements = poset.elements
    n = poset.n
    accum = _empty_accum(n)

    gen = LEGenerator(poset)
    gen.reset()
    all_le = gen.get_batch()
    start = time.time()
    last_print = start

    for k, le in enumerate(all_le):
        _accumulate_sep(le, elements, n, accum)
        if output_every_sec is not None:
            now = time.time()
            if now - last_print >= output_every_sec:
                print(f"  Separation: processed {k+1} extensions...")
                last_print = now

    count = len(all_le)
    result = _finalize_sep(accum, max(count, 1), list(types))
    result['n_extensions'] = count
    result['elements'] = elements
    return result


# ---------------------------------------------------------------------------
# Bubley-Dyer Separation
# ---------------------------------------------------------------------------

class _BubleyDyerSepGen:
    """Internal state for incremental Bubley-Dyer separation."""
    def __init__(self, poset: POSet, seed: Optional[int], types: List[str]):
        self.poset = poset
        self.types = types
        self._sampler = LEBubleyDyer(poset, seed=seed)
        n = poset.n
        self._accum = _empty_accum(n)
        self._total = 0

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
            n_eps = int(E**4 * ln_E**2 + E**3 * ln_E * math.log(1/error))
            additional = max(n_eps - self._total, 0)
            if additional == 0:
                print("Warning: precision already achieved.")
                count = max(self._total, 1)
                result = _finalize_sep(self._accum, count, self.types)
                result['n_extensions'] = self._total
                result['elements'] = elements
                return result
            n = additional

        batch = self._sampler.sample_batch(n=n, output_every_sec=output_every_sec)
        nv = self.poset.n
        for le in batch:
            _accumulate_sep(le, elements, nv, self._accum)
        self._total += len(batch)

        count = max(self._total, 1)
        result = _finalize_sep(self._accum, count, self.types)
        result['n_extensions'] = self._total
        result['elements'] = elements
        return result


def BuildBubleyDyerSeparationGenerator(
    poset: POSet,
    seed: Optional[int] = None,
    *types: str,
) -> _BubleyDyerSepGen:
    """
    Create a Bubley-Dyer separation generator.

    Parameters
    ----------
    poset : POSet
    seed : int, optional
    *types : str
        Types of separation to compute.

    Returns
    -------
    _BubleyDyerSepGen

    Examples
    --------
    >>> gen = BuildBubleyDyerSeparationGenerator(pos, None,
    ...           'symmetric', 'asymmetricUpper', 'vertical')
    >>> result = BubleyDyerSeparation(gen, n=10000)
    """
    _validate_types(types)
    return _BubleyDyerSepGen(poset, seed, list(types))


def BubleyDyerSeparation(
    generator: _BubleyDyerSepGen,
    n: Optional[int] = None,
    error: Optional[float] = None,
    output_every_sec: Optional[int] = None,
) -> Dict:
    """
    Compute (or refine) approximate separation matrices via Bubley-Dyer.

    Examples
    --------
    >>> gen = BuildBubleyDyerSeparationGenerator(pos, None, 'symmetric')
    >>> result = BubleyDyerSeparation(gen, n=10000)
    """
    return generator.update(n=n, error=error, output_every_sec=output_every_sec)


# ---------------------------------------------------------------------------
# LexSeparation
# ---------------------------------------------------------------------------

def LexSeparation(nvar: int, deg, *types: str) -> Dict:
    """
    Separation matrices over lexicographic linear extensions.

    Parameters
    ----------
    nvar : int
    deg : see LexMRP
    *types : str

    Returns
    -------
    dict of separation matrices keyed by type name, plus 'elements'.

    Examples
    --------
    >>> result = LexSeparation(3, 4, 'symmetric', 'asymmetricLower')
    """
    _validate_types(types)
    labels_per_var = _parse_deg(nvar, deg)
    profiles, profile_labels = _build_profiles(labels_per_var)
    N = len(profiles)
    elements = profile_labels

    perms = list(itertools.permutations(range(nvar)))
    accum = _empty_accum(N)
    count = 0

    for perm in perms:
        le_fwd = sorted(range(N), key=lambda idx: tuple(profiles[idx][k] for k in perm))
        le_rev = le_fwd[::-1]

        for le_indices in [le_fwd, le_rev]:
            le = [elements[i] for i in le_indices]
            _accumulate_sep(le, elements, N, accum)
            count += 1

    result = _finalize_sep(accum, count, list(types))
    result['elements'] = elements
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_types(types):
    for t in types:
        if t not in _VALID_TYPES:
            raise ValueError(f"Unknown separation type '{t}'. "
                             f"Choose from {_VALID_TYPES}.")
    if not types:
        raise ValueError("Specify at least one separation type.")
