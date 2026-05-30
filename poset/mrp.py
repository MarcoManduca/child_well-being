"""
mrp.py - Mutual Ranking Probabilities (MRP).

MRP(a, b) = share of linear extensions where b dominates a.

Implements:
  - ExactMRP           : over all linear extensions
  - BubleyDyerMRPGenerator / BubleyDyerMRP : approximate via MCMC
  - LexMRP             : over lexicographic linear extensions
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

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
        Print progress every N seconds.

    Returns
    -------
    dict with keys:
        'MRP'          : np.ndarray (n×n) of MRP values
        'n_extensions' : int
        'elements'     : list of str

    Examples
    --------
    >>> result = ExactMRP(pos)
    >>> result['MRP']
    """
    gen = LEGenerator(poset)
    n_elem = poset.n
    elements = poset.elements

    # Pre-compute element-to-index mapping
    elem_to_idx = {e: i for i, e in enumerate(elements)}

    mrp = np.zeros((n_elem, n_elem), dtype=np.float64)
    pos = np.empty(n_elem, dtype=np.int32)

    gen.reset()
    start = time.time()
    last_print = start
    count = 0

    all_le = gen.get_batch()

    for le in all_le:
        count += 1

        # Build position vector: pos[element_index] = rank in this LE
        for rank, e in enumerate(le):
            pos[elem_to_idx[e]] = rank

        # Vectorized dominance count:
        # MRP[i,j] += 1 iff element j is ranked >= element i
        mrp += (pos[None, :] >= pos[:, None])

        if output_every_sec is not None:
            now = time.time()
            if now - last_print >= output_every_sec:
                print(f"  Processed {count} linear extensions...")
                last_print = now

    if count > 0:
        mrp /= count

    return {
        "MRP": mrp,
        "n_extensions": count,
        "elements": elements,
    }


# ---------------------------------------------------------------------------
# Bubley-Dyer MRP Generator
# ---------------------------------------------------------------------------

class _BubleyDyerMRPGen:
    """
    Internal state holder for incremental MRP computation.

    Maintains a running sum of dominance counts across all sampled
    linear extensions, allowing progressive refinement of MRP estimates.
    """

    def __init__(self, poset: POSet, seed: Optional[int] = None):
        self.poset = poset
        self._sampler = LEBubleyDyer(poset, seed=seed)

        n = poset.n
        self._mrp_sum = np.zeros((n, n), dtype=np.float64)
        self._total_count: int = 0

        # Pre-compute element-to-index mapping (used in vectorized update)
        self._elem_to_idx: Dict[str, int] = {
            e: i for i, e in enumerate(poset.elements)
        }

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
        Add samples and return the updated MRP matrix.

        Three modes (pick one):

        1. **Fixed count** (``n``):
           Add exactly *n* new samples.

        2. **Theoretical bound** (``error``):
           Compute the number of samples from the Bubley-Dyer
           mixing-time bound.  Very conservative — prefer mode 3.

        3. **Empirical convergence** (``converge_tol``):
           Keep sampling in batches of ``converge_check_every`` until the
           max absolute change in MRP between consecutive checkpoints
           falls below ``converge_tol``, or ``max_samples`` is reached.

        Parameters
        ----------
        n : int, optional
            Number of new samples to add.
        error : float, optional
            Desired precision (theoretical bound, very conservative).
        converge_tol : float, optional
            Stop when max |ΔMRP| between checkpoints < tol.
        converge_check_every : int
            Batch size between convergence checks (default: 10 000).
        max_samples : int
            Safety cap for convergence mode (default: 1 000 000).
        output_every_sec : int, optional
            Print progress every N seconds.

        Returns
        -------
        dict with keys:
            'MRP'          : np.ndarray (n_elem × n_elem)
            'n_extensions' : int  — total samples collected so far
            'elements'     : list[str]
            'converged'    : bool  (only in convergence mode)
            'max_delta'    : float (only in convergence mode)
        """
        E = self.poset.n

        # ----- Mode selection -----
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

        # ----- Fixed-count sampling -----
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
        """Sample in batches until MRP stabilises or budget is exhausted."""
        added = 0
        prev_mrp = self.mrp.copy() if self._total_count > 0 else None
        converged = False
        max_delta = float("inf")

        start = time.time()
        last_print = start

        while added < max_samples:
            batch_size = min(check_every, max_samples - added)
            self._sample_and_accumulate(batch_size, output_every_sec=None)
            added += batch_size

            current_mrp = self.mrp

            if prev_mrp is not None:
                max_delta = float(np.max(np.abs(current_mrp - prev_mrp)))

                if output_every_sec is not None:
                    now = time.time()
                    if now - last_print >= output_every_sec:
                        print(
                            f"  {self._total_count:>10,} samples | "
                            f"max |ΔMRP| = {max_delta:.6f} "
                            f"(target < {tol})"
                        )
                        last_print = now

                if max_delta < tol:
                    converged = True
                    break

            prev_mrp = current_mrp.copy()

        result = self._build_result()
        result["converged"] = converged
        result["max_delta"] = max_delta

        if converged:
            print(
                f"  Converged after {self._total_count:,} total samples "
                f"(max |ΔMRP| = {max_delta:.6f} < {tol})"
            )
        else:
            print(
                f"  Budget exhausted ({max_samples:,} new samples). "
                f"max |ΔMRP| = {max_delta:.6f} (target was < {tol})"
            )

        return result

    # ------------------------------------------------------------------
    # Sampling engine (vectorized)
    # ------------------------------------------------------------------

    def _sample_and_accumulate(
        self,
        n: int,
        output_every_sec: Optional[int] = None,
    ) -> None:
        """
        Draw *n* linear extensions and update the running MRP sum.

        Uses NumPy broadcasting instead of a Python double-loop:
        for each sampled LE, build a position vector and compute
        the full n×n dominance matrix in one vectorized operation.
        """
        n_elem = self.poset.n
        elem_to_idx = self._elem_to_idx

        batch = self._sampler.sample_batch(
            n=n,
            error=None,
            output_every_sec=output_every_sec,
        )

        # Pre-allocate position array (reused each iteration)
        pos = np.empty(n_elem, dtype=np.int32)

        for le in batch:
            # Build position vector: pos[element_index] = rank in this LE
            for rank, e in enumerate(le):
                pos[elem_to_idx[e]] = rank

            # Vectorized dominance: MRP[i,j] counts how often
            # element j is ranked >= element i  (j dominates i)
            self._mrp_sum += (pos[None, :] >= pos[:, None])

        self._total_count += len(batch)

    # ------------------------------------------------------------------
    # Properties and helpers
    # ------------------------------------------------------------------

    @property
    def mrp(self) -> np.ndarray:
        """Current MRP matrix estimate."""
        if self._total_count == 0:
            return np.zeros_like(self._mrp_sum)
        return self._mrp_sum / self._total_count

    @property
    def total_samples(self) -> int:
        """Total number of linear extensions sampled so far."""
        return self._total_count

    def mrp_score(self) -> Dict[str, float]:
        """
        Average MRP score per element (column-mean of MRP matrix).

        MRP[i,j] = P(j dominates i), so column j mean = P(j dominates a random
        element) = domination score of j.  Higher score → element dominates more
        others → better performer.
        Returns a dict ``{element_name: score}``.
        """
        m = self.mrp
        scores = m.mean(axis=0)
        return {
            e: float(scores[i])
            for i, e in enumerate(self.poset.elements)
        }

    def reset(self) -> None:
        """Clear all accumulated samples and start fresh."""
        n = self.poset.n
        self._mrp_sum = np.zeros((n, n), dtype=np.float64)
        self._total_count = 0

    def _build_result(self) -> Dict:
        return {
            "MRP": self.mrp,
            "n_extensions": self._total_count,
            "elements": self.poset.elements,
        }

    def __repr__(self) -> str:
        return (
            f"BubleyDyerMRP("
            f"elements={self.poset.n}, "
            f"samples={self._total_count:,})"
        )


# ---------------------------------------------------------------------------
# Public factory + convenience wrapper
# ---------------------------------------------------------------------------

def BubleyDyerMRPGenerator(
    poset: POSet,
    seed: Optional[int] = None,
) -> _BubleyDyerMRPGen:
    """
    Create a Bubley-Dyer MRP generator.

    Parameters
    ----------
    poset : POSet
    seed : int, optional

    Returns
    -------
    _BubleyDyerMRPGen

    Examples
    --------
    >>> gen = BubleyDyerMRPGenerator(pos, seed=42)

    >>> # Mode 1: fixed count
    >>> result = gen.update(n=100_000, output_every_sec=10)

    >>> # Mode 2: add more samples incrementally
    >>> result = gen.update(n=100_000)

    >>> # Mode 3: auto-converge
    >>> result = gen.update(converge_tol=0.005, output_every_sec=5)

    >>> # Inspect scores
    >>> gen.mrp_score()
    {'SWE_2015': 0.92, 'EST_2015': 0.08, ...}
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

    Thin wrapper around ``generator.update()`` for backward compatibility.

    Parameters
    ----------
    generator : _BubleyDyerMRPGen
        Created by ``BubleyDyerMRPGenerator()``.
    n : int, optional
        Number of extensions to sample.
    error : float in (0,1), optional
        Desired distance from uniformity.
    output_every_sec : int, optional

    Returns
    -------
    dict with keys ``'MRP'``, ``'n_extensions'``, ``'elements'``

    Examples
    --------
    >>> gen = BubleyDyerMRPGenerator(pos)
    >>> result = BubleyDyerMRP(gen, n=50_000)
    >>> result = BubleyDyerMRP(gen, n=10_000)   # refinement
    """
    return generator.update(
        n=n, error=error, output_every_sec=output_every_sec
    )


# ---------------------------------------------------------------------------
# LexMRP
# ---------------------------------------------------------------------------

def LexMRP(nvar: int, deg) -> Tuple[np.ndarray, List[str]]:
    """
    MRP matrix computed over lexicographic linear extensions of the
    component-wise Boolean lattice on *k* ordinal variables.

    Parameters
    ----------
    nvar : int
        Number of ordinal variables *k*.
    deg : int | list of str | list of int | list of list of str
        Degrees specification:

        - ``int m``  → all variables have degrees 0, 1, …, m-1
        - ``list of str`` → custom labels, same for all variables
        - ``list of int`` (length k) → per-variable degree counts
        - ``list of list of str`` → per-variable custom labels

    Returns
    -------
    (np.ndarray, list[str])
        ``(MRP, profile_labels)`` where MRP is N×N and N = prod(deg_i).

    Examples
    --------
    >>> MRP, labels = LexMRP(3, 4)          # 3 vars, 4 levels each
    >>> MRP, labels = LexMRP(3, ['a','b','c','d'])
    """
    labels_per_var = _parse_deg(nvar, deg)
    profiles, profile_labels = _build_profiles(labels_per_var)
    N = len(profiles)

    perms = list(itertools.permutations(range(nvar)))

    mrp_sum = np.zeros((N, N), dtype=np.float64)
    inv_pos = np.empty(N, dtype=np.int32)
    count = 0

    for perm in perms:
        # Forward lex extension: sort profiles by perm[0], then perm[1], ...
        le_fwd = sorted(
            range(N),
            key=lambda idx: tuple(profiles[idx][k] for k in perm),
        )
        le_rev = le_fwd[::-1]

        for le in (le_fwd, le_rev):
            # inv_pos[element_index] = position in this linear extension
            for position, elem_idx in enumerate(le):
                inv_pos[elem_idx] = position

            # Vectorized dominance count
            mrp_sum += (inv_pos[None, :] >= inv_pos[:, None])
            count += 1

    mrp = mrp_sum / count
    return mrp, profile_labels


# ---------------------------------------------------------------------------
# Helpers (shared with separation.py)
# ---------------------------------------------------------------------------

def _parse_deg(nvar: int, deg):
    """Return list of lists of labels, one list per variable."""
    if isinstance(deg, int):
        return [list(range(deg)) for _ in range(nvar)]

    if isinstance(deg, (list, tuple)):
        if len(deg) == 0:
            raise ValueError("deg must not be empty.")

        first = deg[0]

        if isinstance(first, int):
            # list of ints → per-variable degree counts
            if len(deg) == nvar:
                return [list(range(d)) for d in deg]
            # single int in a list → treat as common degrees
            return [list(deg) for _ in range(nvar)]

        if isinstance(first, str):
            # Common labels for all variables
            return [list(deg) for _ in range(nvar)]

        if isinstance(first, (list, tuple)):
            # Per-variable labels
            return [list(d) for d in deg]

    raise ValueError(f"Cannot parse deg={deg!r}.")


def _build_profiles(labels_per_var):
    """Return (profiles as index tuples, profile label strings)."""
    combos = list(
        itertools.product(*[range(len(lv)) for lv in labels_per_var])
    )
    labels = []
    for combo in combos:
        lbl = (
            "("
            + ",".join(
                str(labels_per_var[k][v]) for k, v in enumerate(combo)
            )
            + ")"
        )
        labels.append(lbl)
    return combos, labels
