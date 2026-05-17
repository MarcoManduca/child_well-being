"""
linear_extensions.py - Linear extension generation.

Implements:
  - LEGenerator  : exact enumeration (Habib et al. 2001 approach via topological sort)
  - LEBubleyDyer : Bubley-Dyer MCMC sampler (1999)
  - LEGet        : unified interface to generate extensions from either generator
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np
import math
import random
import time


class LEGenerator:
    """
    Exact linear extension generator.

    Generates all linear extensions of a poset via a recursive
    backtracking algorithm that always selects from the current
    minimals.

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

    def _generate_all(self):
        if self._generated:
            return
        elements = self.poset.elements
        n = len(elements)
        # Build in-degree vector (number of elements strictly below each)
        indegree = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j and self.poset._mat[i, j] and not self.poset._mat[j, i]:
                    indegree[j] += 1  # i < j, j has one more predecessor

        results = []

        def backtrack(remaining: List[int], current: List[int], indeg: np.ndarray):
            if not remaining:
                results.append([elements[i] for i in current])
                return
            # Available: remaining with indegree 0 among remaining
            rem_set = set(remaining)
            available = [i for i in remaining if indeg[i] == 0]
            for choice in available:
                new_indeg = indeg.copy()
                # Remove 'choice': decrement indegree of its successors
                for j in remaining:
                    if j != choice and self.poset._mat[choice, j] and not self.poset._mat[j, choice]:
                        new_indeg[j] -= 1
                new_rem = [x for x in remaining if x != choice]
                backtrack(new_rem, current + [choice], new_indeg)

        backtrack(list(range(n)), [], indegree)
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
            batch = self._extensions[self._cursor:self._cursor + n]
            self._cursor += len(batch)
        return batch

    @property
    def total(self) -> int:
        self._generate_all()
        return len(self._extensions)


class LEBubleyDyer:
    """
    Bubley-Dyer MCMC sampler for linear extensions.

    Uses the adjacent transposition Markov chain on linear extensions.
    At each step, pick a random adjacent pair and swap if the result
    is still a linear extension.

    Parameters
    ----------
    poset : POSet
    seed : int, optional

    References
    ----------
    Bubley R., Dyer M. (1999). Faster random generation of linear extensions.
    Discrete Mathematics, 201, 81-88.

    Examples
    --------
    >>> gen = LEBubleyDyer(pos, seed=42)
    >>> matrix = LEGet(gen, n=1000)
    """

    def __init__(self, poset, seed: Optional[int] = None):
        self.poset = poset
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        elements = poset.elements
        n = len(elements)
        # Start with a valid linear extension (topological sort)
        self._current = self._topo_sort()

    def _topo_sort(self) -> List[str]:
        """Deterministic topological sort as starting state."""
        p = self.poset
        n = p.n
        indeg = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j and p._mat[i, j] and not p._mat[j, i]:
                    indeg[j] += 1
        result = []
        remaining = list(range(n))
        while remaining:
            avail = [i for i in remaining if indeg[i] == 0]
            choice = self._rng.choice(avail)
            result.append(p.elements[choice])
            remaining.remove(choice)
            for j in remaining:
                if p._mat[choice, j] and not p._mat[j, choice]:
                    indeg[j] -= 1
        return result

    def _mixing_steps(self, n_elem: int) -> int:
        """Number of Markov steps for one sample."""
        # n^3 steps to approximately reach stationarity
        return max(n_elem ** 3, 100)

    def _is_valid_swap(self, le: List[str], pos: int) -> bool:
        """Check if swapping le[pos] and le[pos+1] gives a valid extension."""
        a, b = le[pos], le[pos+1]
        # Valid iff a and b are incomparable
        return self.poset.is_incomparable(a, b)

    def sample_one(self, burn_in: Optional[int] = None) -> List[str]:
        """Return one sampled linear extension."""
        n = self.poset.n
        steps = burn_in if burn_in is not None else self._mixing_steps(n)
        le = list(self._current)
        for _ in range(steps):
            pos = self._rng.randint(0, n - 2)
            if self._is_valid_swap(le, pos):
                le[pos], le[pos+1] = le[pos+1], le[pos]
        self._current = le
        return list(le)

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
            Desired distance from uniformity. Determines n via
            n_eps = E^4*(ln E)^2 + E^3*ln(E)*ln(1/epsilon).
        output_every_sec : int, optional
            Print progress every this many seconds.
        """
        E = self.poset.n
        if n is None and error is None:
            raise ValueError("Specify either n or error.")
        if error is not None and n is None:
            ln_E = math.log(E) if E > 1 else 1
            n = int(E**4 * ln_E**2 + E**3 * ln_E * math.log(1 / error))
            n = max(n, 1)

        results = []
        start = time.time()
        last_print = start
        burn = self._mixing_steps(E)

        for i in range(n):
            results.append(self.sample_one(burn_in=burn))
            if output_every_sec is not None:
                now = time.time()
                if now - last_print >= output_every_sec:
                    print(f"  Generated {i+1}/{n} linear extensions...")
                    last_print = now

        return results


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
        Number of extensions.
    error : float, optional
        Only for LEBubleyDyer. Distance from uniformity.
    output_every_sec : int, optional

    Returns
    -------
    np.ndarray of shape (n_elements, n_extensions)
        Each column is one linear extension.

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
        batch = generator.sample_batch(n=n, error=error,
                                       output_every_sec=output_every_sec)
    else:
        raise TypeError("generator must be LEGenerator or LEBubleyDyer.")

    if not batch:
        return np.empty((generator.poset.n, 0), dtype=object)
    return np.array(batch, dtype=object).T
