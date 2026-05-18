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

import itertools
import math
import time
from typing import Dict, List, Optional

import numpy as np

from .poset import POSet
from .linear_extensions import LEGenerator, LEBubleyDyer
from .mrp import _parse_deg, _build_profiles

_VALID_TYPES = {
    "symmetric", "asymmetricLower", "asymmetricUpper",
    "vertical", "horizontal",
}


# ---------------------------------------------------------------------------
# Vectorized accumulation
# ---------------------------------------------------------------------------

def _empty_accum(n: int) -> Dict[str, np.ndarray]:
    return {k: np.zeros((n, n), dtype=np.float64) for k in ("sym", "aL", "aU")}


def _accumulate_sep_vec(
    le: List[str],
    elem_to_idx: Dict[str, int],
    n: int,
    pos: np.ndarray,
    accum: Dict[str, np.ndarray],
) -> None:
    """
    Vectorized separation accumulation from one linear extension.

    Parameters
    ----------
    le : list of str — the linear extension
    elem_to_idx : dict — element label → index
    n : int — number of elements
    pos : np.ndarray (n,) — pre-allocated work array (mutated in-place)
    accum : dict with 'sym', 'aL', 'aU' arrays
    """
    # Build position vector: pos[element_index] = rank in this LE
    for rank, e in enumerate(le):
        pos[elem_to_idx[e]] = rank

    # diff[i, j] = pos[j] - pos[i]   (positive if j is after i)
    diff = pos[None, :] - pos[:, None]

    accum["sym"] += np.abs(diff)
    # asymmetricLower: accumulate diff where diff > 0 (i before j)
    mask_pos = diff > 0
    accum["aL"] += np.where(mask_pos, diff, 0.0)
    # asymmetricUpper: accumulate -diff where diff < 0 (j before i)
    mask_neg = diff < 0
    accum["aU"] += np.where(mask_neg, -diff, 0.0)


def _finalize_sep(
    accum: Dict[str, np.ndarray],
    count: int,
    types: List[str],
) -> Dict[str, np.ndarray]:
    """Compute requested separation matrices from accumulated sums."""
    sym = accum["sym"] / count
    aL = accum["aL"] / count
    aU = accum["aU"] / count
    vert = np.abs(aL - aU)
    horiz = sym - vert

    _map = {
        "symmetric": sym,
        "asymmetricLower": aL,
        "asymmetricUpper": aU,
        "vertical": vert,
        "horizontal": horiz,
    }
    return {t: _map[t] for t in types}


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
    dict with separation matrices and 'n_extensions', 'elements'.

    Examples
    --------
    >>> result = ExactSeparation(pos, 'symmetric', 'vertical')
    >>> result['symmetric']
    """
    _validate_types(types)

    elements = poset.elements
    n = poset.n
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    accum = _empty_accum(n)
    pos = np.empty(n, dtype=np.float64)

    gen = LEGenerator(poset)
    gen.reset()
    all_le = gen.get_batch()

    start = time.time()
    last_print = start

    for k, le in enumerate(all_le):
        _accumulate_sep_vec(le, elem_to_idx, n, pos, accum)

        if output_every_sec is not None:
            now = time.time()
            if now - last_print >= output_every_sec:
                print(f"  Separation: processed {k + 1} extensions...")
                last_print = now

    count = max(len(all_le), 1)
    result = _finalize_sep(accum, count, list(types))
    result["n_extensions"] = len(all_le)
    result["elements"] = elements
    return result


# ---------------------------------------------------------------------------
# Bubley-Dyer Separation
# ---------------------------------------------------------------------------

class _BubleyDyerSepGen:
    """
    Internal state for incremental Bubley-Dyer separation computation.

    Supports the same three modes as _BubleyDyerMRPGen:
    fixed count, theoretical bound, and empirical convergence.
    """

    def __init__(self, poset: POSet, seed: Optional[int], types: List[str]):
        self.poset = poset
        self.types = types
        self._sampler = LEBubleyDyer(poset, seed=seed)

        n = poset.n
        self._accum = _empty_accum(n)
        self._total_count: int = 0
        self._elem_to_idx = {e: i for i, e in enumerate(poset.elements)}
        self._pos = np.empty(n, dtype=np.float64)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        n: Optional[int] = None,
        error: Optional[float] = None,
        converge_tol: Optional[float] = None,
        converge_check_every: int = 10_000,
        max_samples: int = 1_000_000,
        output_every_sec: Optional[int] = None,
    ) -> Dict:
        """
        Add samples and return updated separation matrices.

        Three modes (pick one):

        1. **Fixed count** (``n``): add exactly *n* new samples.
        2. **Theoretical bound** (``error``): very conservative.
        3. **Empirical convergence** (``converge_tol``): stop when
           max |Δsep| between checkpoints < tol.
        """
        E = self.poset.n

        if converge_tol is not None:
            return self._update_until_converged(
                tol=converge_tol,
                check_every=converge_check_every,
                max_samples=max_samples,
                output_every_sec=output_every_sec,
            )

        if error is not None and n is None:
            ln_E = math.log(E) if E > 1 else 1.0
            n_eps = int(
                E ** 4 * ln_E ** 2
                + E ** 3 * ln_E * math.log(1.0 / error)
            )
            additional = max(n_eps - self._total_count, 0)
            if additional == 0:
                print(
                    "Warning: desired precision already achieved; "
                    "no new extensions generated."
                )
                return self._build_result()
            n = additional

        if n is None or n <= 0:
            raise ValueError(
                "Specify one of: n (int > 0), error (float), "
                "or converge_tol (float)."
            )

        self._sample_and_accumulate(n, output_every_sec=output_every_sec)
        return self._build_result()

    # ------------------------------------------------------------------
    # Convergence mode
    # ------------------------------------------------------------------

    def _update_until_converged(
        self,
        tol: float,
        check_every: int,
        max_samples: int,
        output_every_sec: Optional[int],
    ) -> Dict:
        added = 0
        prev_sep = self._current_sep_snapshot() if self._total_count > 0 else None
        converged = False
        max_delta = float("inf")

        start = time.time()
        last_print = start

        while added < max_samples:
            batch_size = min(check_every, max_samples - added)
            self._sample_and_accumulate(batch_size, output_every_sec=None)
            added += batch_size

            current_sep = self._current_sep_snapshot()

            if prev_sep is not None:
                max_delta = max(
                    float(np.max(np.abs(current_sep[k] - prev_sep[k])))
                    for k in ("sym", "aL", "aU")
                )

                if output_every_sec is not None:
                    now = time.time()
                    if now - last_print >= output_every_sec:
                        print(
                            f"  {self._total_count:>10,} samples | "
                            f"max |Δsep| = {max_delta:.6f} "
                            f"(target < {tol})"
                        )
                        last_print = now

                if max_delta < tol:
                    converged = True
                    break

            prev_sep = current_sep

        result = self._build_result()
        result["converged"] = converged
        result["max_delta"] = max_delta

        if converged:
            print(
                f"  Converged after {self._total_count:,} total samples "
                f"(max |Δsep| = {max_delta:.6f} < {tol})"
            )
        else:
            print(
                f"  Budget exhausted ({max_samples:,} new samples). "
                f"max |Δsep| = {max_delta:.6f} (target was < {tol})"
            )

        return result

    # ------------------------------------------------------------------
    # Sampling engine
    # ------------------------------------------------------------------

    def _sample_and_accumulate(
        self,
        n: int,
        output_every_sec: Optional[int] = None,
    ) -> None:
        batch = self._sampler.sample_batch(
            n=n, error=None, output_every_sec=output_every_sec,
        )
        n_elem = self.poset.n
        elem_to_idx = self._elem_to_idx
        pos = self._pos

        for le in batch:
            _accumulate_sep_vec(le, elem_to_idx, n_elem, pos, self._accum)

        self._total_count += len(batch)

    # ------------------------------------------------------------------
    # Properties and helpers
    # ------------------------------------------------------------------

    def _current_sep_snapshot(self) -> Dict[str, np.ndarray]:
        c = max(self._total_count, 1)
        return {k: self._accum[k] / c for k in ("sym", "aL", "aU")}

    @property
    def total_samples(self) -> int:
        return self._total_count

    def reset(self) -> None:
        n = self.poset.n
        self._accum = _empty_accum(n)
        self._total_count = 0

    def _build_result(self) -> Dict:
        count = max(self._total_count, 1)
        result = _finalize_sep(self._accum, count, self.types)
        result["n_extensions"] = self._total_count
        result["elements"] = self.poset.elements
        return result

    def __repr__(self) -> str:
        return (
            f"BubleyDyerSeparation("
            f"elements={self.poset.n}, "
            f"types={self.types}, "
            f"samples={self._total_count:,})"
        )


# ---------------------------------------------------------------------------
# Public factory + wrapper
# ---------------------------------------------------------------------------

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

    Returns
    -------
    _BubleyDyerSepGen

    Examples
    --------
    >>> gen = BuildBubleyDyerSeparationGenerator(pos, None,
    ...           'symmetric', 'asymmetricUpper', 'vertical')
    >>> result = BubleyDyerSeparation(gen, n=10_000)
    >>> # or with convergence:
    >>> result = gen.update(converge_tol=0.01, output_every_sec=5)
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
    >>> result = BubleyDyerSeparation(gen, n=10_000)
    >>> result = BubleyDyerSeparation(gen, n=5_000)   # refinement
    """
    return generator.update(
        n=n, error=error, output_every_sec=output_every_sec
    )


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

    elem_to_idx = {e: i for i, e in enumerate(elements)}

    perms = list(itertools.permutations(range(nvar)))
    accum = _empty_accum(N)
    pos = np.empty(N, dtype=np.float64)
    count = 0

    for perm in perms:
        le_fwd_idx = sorted(
            range(N),
            key=lambda idx: tuple(profiles[idx][k] for k in perm),
        )
        le_rev_idx = le_fwd_idx[::-1]

        for le_indices in (le_fwd_idx, le_rev_idx):
            le = [elements[i] for i in le_indices]
            _accumulate_sep_vec(le, elem_to_idx, N, pos, accum)
            count += 1

    result = _finalize_sep(accum, count, list(types))
    result["elements"] = elements
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_types(types):
    if not types:
        raise ValueError("Specify at least one separation type.")
    for t in types:
        if t not in _VALID_TYPES:
            raise ValueError(
                f"Unknown separation type '{t}'.  "
                f"Choose from {_VALID_TYPES}."
            )
