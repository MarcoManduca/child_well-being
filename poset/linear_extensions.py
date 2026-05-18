"""
linear_extensions.py - Linear extension generation.

Implements:
  - LEGenerator  : exact enumeration via recursive backtracking
  - LEBubleyDyer : Bubley-Dyer MCMC sampler (1999)
  - LEGet        : unified interface

References:
    Bubley R., Dyer M. (1999). Faster random generation of linear extensions.
    Discrete Mathematics, 201, 81-88.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Exact generator
# ---------------------------------------------------------------------------

class LEGenerator:
    """
    Exact linear extension generator.

    Generates all linear extensions of a poset via recursive backtracking,
    always selecting from the current set of minimal elements.

    Parameters
    ----------
    poset : POSet

    Examples
    --------
    >>> gen = LEGenerator(pos)
    >>> matrix = LEGet(gen)   # all extensions as columns
    """

    def __init__(self, poset):
        self.poset = poset
        self._extensions: List[List[str]] = []
        self._generated = False
        self._cursor = 0

        # Pre-compute strict-order matrix: _strict[i,j] = True iff i < j
        n = poset.n
        mat = poset._mat
        self._strict = mat & ~mat.T

        # Pre-compute initial in-degrees from the strict order
        self._base_indegree = self._strict.sum(axis=0).astype(int)

    def _generate_all(self):
        if self._generated:
            return

        elements = self.poset.elements
        n = len(elements)
        strict = self._strict
        results = []

        def backtrack(remaining: List[int], current: List[int], indeg: np.ndarray):
            if not remaining:
                results.append([elements[i] for i in current])
                return

            available = [i for i in remaining if indeg[i] == 0]

            for choice in available:
                # Decrement in-degree for successors of choice
                successors = []
                for j in remaining:
                    if j != choice and strict[choice, j]:
                        indeg[j] -= 1
                        successors.append(j)

                new_rem = [x for x in remaining if x != choice]
                backtrack(new_rem, current + [choice], indeg)

                # Restore in-degrees
                for j in successors:
                    indeg[j] += 1

        # Use mutable indegree array with restore instead of copy per branch
        backtrack(list(range(n)), [], self._base_indegree.copy())
        self._extensions = results
        self._generated = True
        self._cursor = 0

    def reset(self):
        self._generate_all()
        self._cursor = 0

    def get_batch(self, n: Optional[int] = None) -> List[List[str]]:
        self._generate_all()
        if n is None:
            batch = self._extensions[self._cursor:]
            self._cursor = len(self._extensions)
        else:
            batch = self._extensions[self._cursor : self._cursor + n]
            self._cursor += len(batch)
        return batch

    @property
    def total(self) -> int:
        self._generate_all()
        return len(self._extensions)


# ---------------------------------------------------------------------------
# Bubley-Dyer MCMC sampler
# ---------------------------------------------------------------------------

class LEBubleyDyer:
    """
    Bubley-Dyer MCMC sampler for linear extensions.

    Uses the adjacent transposition Markov chain: at each step, pick a
    random adjacent pair in the current permutation and swap if the
    result is still a valid linear extension.

    Optimisations over a naive implementation:

    - **Pre-computed incomparability matrix** so that swap validation is
      an O(1) array lookup instead of a method call.
    - **Batch random-number generation** via NumPy: all swap positions
      for the burn-in / thinning phase are drawn at once, avoiding
      per-step Python-level ``random.randint`` overhead.
    - **In-place mutation** of the current permutation and its inverse
      position map.

    Parameters
    ----------
    poset : POSet
    seed : int, optional

    Examples
    --------
    >>> gen = LEBubleyDyer(pos, seed=42)
    >>> matrix = LEGet(gen, n=1000)
    """

    def __init__(self, poset, seed: Optional[int] = None):
        self.poset = poset
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        n = poset.n
        elements = poset.elements

        # Pre-compute strict order and incomparability as dense bool matrices
        # indexed by *element index* (not label).
        mat = poset._mat
        strict = mat & ~mat.T
        # incomp[i,j] = True iff i and j are incomparable
        self._incomp = ~mat & ~mat.T
        np.fill_diagonal(self._incomp, False)

        # Element label ↔ index mappings
        self._elem_to_idx = {e: i for i, e in enumerate(elements)}
        self._idx_to_elem = elements  # list, index-accessible

        # Strict-order matrix for initial topological sort
        self._strict = strict
        self._base_indegree = strict.sum(axis=0).astype(int)

        # Current state: a valid linear extension (as list of labels)
        self._current = self._topo_sort()

    # ------------------------------------------------------------------
    # Topological sort (initial state)
    # ------------------------------------------------------------------

    def _topo_sort(self) -> List[str]:
        """Random topological sort as starting state."""
        n = self.poset.n
        elements = self.poset.elements
        strict = self._strict
        indeg = self._base_indegree.copy()

        result = []
        remaining = list(range(n))

        while remaining:
            avail = [i for i in remaining if indeg[i] == 0]
            choice = self._rng.choice(avail)
            result.append(elements[choice])
            remaining.remove(choice)
            for j in remaining:
                if strict[choice, j]:
                    indeg[j] -= 1

        return result

    # ------------------------------------------------------------------
    # Mixing parameters
    # ------------------------------------------------------------------

    @staticmethod
    def _mixing_steps(n_elem: int) -> int:
        """Number of Markov steps per sample (n³ heuristic)."""
        return max(n_elem ** 3, 100)

    # ------------------------------------------------------------------
    # Single-sample draw
    # ------------------------------------------------------------------

    def sample_one(self, burn_in: Optional[int] = None) -> List[str]:
        """
        Advance the chain and return one sampled linear extension.

        Uses batch random-number generation and O(1) swap validation.
        """
        n = self.poset.n
        steps = burn_in if burn_in is not None else self._mixing_steps(n)
        le = self._current  # mutate in place
        elem_to_idx = self._elem_to_idx
        incomp = self._incomp

        # Draw all swap positions at once
        positions = self._np_rng.integers(0, n - 1, size=steps)

        for pos in positions:
            a_idx = elem_to_idx[le[pos]]
            b_idx = elem_to_idx[le[pos + 1]]
            # Swap iff the two elements are incomparable
            if incomp[a_idx, b_idx]:
                le[pos], le[pos + 1] = le[pos + 1], le[pos]

        self._current = le
        return list(le)

    # ------------------------------------------------------------------
    # Batch sampling
    # ------------------------------------------------------------------

    def sample_batch(
        self,
        n: Optional[int] = None,
        error: Optional[float] = None,
        output_every_sec: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Sample multiple linear extensions.

        Parameters
        ----------
        n : int, optional
            Number of extensions to sample.
        error : float in (0,1), optional
            Desired distance from uniformity.  Determines *n* via the
            Bubley-Dyer bound:
            ``n_eps = E⁴·(ln E)² + E³·ln E·ln(1/ε)``
        output_every_sec : int, optional
        """
        E = self.poset.n

        if n is None and error is None:
            raise ValueError("Specify either n or error.")

        if error is not None and n is None:
            ln_E = math.log(E) if E > 1 else 1.0
            n = int(E ** 4 * ln_E ** 2 + E ** 3 * ln_E * math.log(1.0 / error))
            n = max(n, 1)

        results = []
        burn = self._mixing_steps(E)

        start = time.time()
        last_print = start

        for i in range(n):
            results.append(self.sample_one(burn_in=burn))

            if output_every_sec is not None:
                now = time.time()
                if now - last_print >= output_every_sec:
                    print(f"  Generated {i + 1}/{n} linear extensions...")
                    last_print = now

        return results


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def LEGet(
    generator,
    from_start: bool = True,
    n: Optional[int] = None,
    error: Optional[float] = None,
    output_every_sec: Optional[int] = None,
) -> np.ndarray:
    """
    Generate linear extensions from a generator.

    Parameters
    ----------
    generator : LEGenerator or LEBubleyDyer
    from_start : bool
        If True, reset generator before generating.
    n : int, optional
    error : float, optional
        Only for LEBubleyDyer.
    output_every_sec : int, optional

    Returns
    -------
    np.ndarray of shape (n_elements, n_extensions)
        Each column is one linear extension (element labels).

    Examples
    --------
    >>> gen = LEGenerator(pos)
    >>> mat = LEGet(gen)            # all extensions
    >>> gen2 = LEBubleyDyer(pos, seed=42)
    >>> mat2 = LEGet(gen2, n=500)
    """
    if isinstance(generator, LEGenerator):
        if from_start:
            generator.reset()
        batch = generator.get_batch(n)

    elif isinstance(generator, LEBubleyDyer):
        if from_start:
            generator._current = generator._topo_sort()
        batch = generator.sample_batch(
            n=n, error=error, output_every_sec=output_every_sec
        )

    else:
        raise TypeError("generator must be LEGenerator or LEBubleyDyer.")

    if not batch:
        return np.empty((generator.poset.n, 0), dtype=object)

    return np.array(batch, dtype=object).T
