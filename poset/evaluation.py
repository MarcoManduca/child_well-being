"""
evaluation.py - Mean value computation of functions over linear extensions.

Implements:
  - ExactEvaluation
  - BuildBubleyDyerEvaluationGenerator / BubleyDyerEvaluation
"""

from __future__ import annotations
from typing import Optional, Callable, List, Dict
import numpy as np
import math
import time

from .poset import POSet
from .linear_extensions import LEGenerator, LEBubleyDyer


def ExactEvaluation(
    poset: POSet,
    *functions: Callable,
    output_every_sec: Optional[int] = None,
) -> Dict:
    """
    Compute the mean value of functions over all linear extensions.

    Each function receives one linear extension (list of element labels)
    and must return a numeric np.ndarray.

    Parameters
    ----------
    poset : POSet
    *functions : callable
        Functions f(le) → np.ndarray.
    output_every_sec : int, optional

    Returns
    -------
    dict with keys 'averages' (list of arrays), 'n_extensions'.

    Examples
    --------
    >>> def rank_a(le): return np.array([le.index('a')])
    >>> result = ExactEvaluation(pos, rank_a)
    >>> result['averages']
    """
    gen = LEGenerator(poset)
    gen.reset()
    all_le = gen.get_batch()

    sums = [None] * len(functions)
    start = time.time()
    last_print = start
    count = 0

    for le in all_le:
        count += 1
        for fi, f in enumerate(functions):
            val = np.asarray(f(le), dtype=float)
            if sums[fi] is None:
                sums[fi] = val.copy()
            else:
                sums[fi] += val
        if output_every_sec is not None:
            now = time.time()
            if now - last_print >= output_every_sec:
                print(f"  Evaluation: processed {count} extensions...")
                last_print = now

    averages = [s / count if count > 0 else s for s in sums]
    return {'averages': averages, 'n_extensions': count}


# ---------------------------------------------------------------------------
# Bubley-Dyer Evaluation
# ---------------------------------------------------------------------------

class _BubleyDyerEvalGen:
    def __init__(self, poset: POSet, seed: Optional[int], functions: List[Callable]):
        self.poset = poset
        self.functions = functions
        self._sampler = LEBubleyDyer(poset, seed=seed)
        self._sums = [None] * len(functions)
        self._total = 0

    def update(
        self,
        n: Optional[int] = None,
        error: Optional[float] = None,
        output_every_sec: Optional[int] = None,
    ) -> Dict:
        E = self.poset.n

        if error is not None and n is None:
            ln_E = math.log(E) if E > 1 else 1
            n_eps = int(E**4 * ln_E**2 + E**3 * ln_E * math.log(1/error))
            additional = max(n_eps - self._total, 0)
            if additional == 0:
                print("Warning: precision already achieved; no new extensions generated.")
                avgs = [s / self._total if self._total else s for s in self._sums]
                return {'averages': avgs, 'n_extensions': self._total}
            n = additional

        batch = self._sampler.sample_batch(n=n, output_every_sec=output_every_sec)
        for le in batch:
            for fi, f in enumerate(self.functions):
                val = np.asarray(f(le), dtype=float)
                if self._sums[fi] is None:
                    self._sums[fi] = val.copy()
                else:
                    self._sums[fi] += val
        self._total += len(batch)

        avgs = [s / self._total if self._total else s for s in self._sums]
        return {'averages': avgs, 'n_extensions': self._total}


def BuildBubleyDyerEvaluationGenerator(
    poset: POSet,
    seed: Optional[int] = None,
    *functions: Callable,
) -> _BubleyDyerEvalGen:
    """
    Create a Bubley-Dyer function evaluation generator.

    Parameters
    ----------
    poset : POSet
    seed : int, optional
    *functions : callable
        Functions f(le) → np.ndarray.

    Returns
    -------
    _BubleyDyerEvalGen

    Examples
    --------
    >>> def median_pos(le): return np.array([le[len(le)//2] == 'a'])
    >>> gen = BuildBubleyDyerEvaluationGenerator(pos, None, median_pos)
    >>> result = BubleyDyerEvaluation(gen, n=10000)
    """
    return _BubleyDyerEvalGen(poset, seed, list(functions))


def BubleyDyerEvaluation(
    generator: _BubleyDyerEvalGen,
    n: Optional[int] = None,
    error: Optional[float] = None,
    output_every_sec: Optional[int] = None,
) -> Dict:
    """
    Estimate (or refine) function averages over linear extensions.

    Examples
    --------
    >>> gen = BuildBubleyDyerEvaluationGenerator(pos, None, my_func)
    >>> result = BubleyDyerEvaluation(gen, n=40000)
    >>> result = BubleyDyerEvaluation(gen, n=10000)   # refinement
    """
    return generator.update(n=n, error=error, output_every_sec=output_every_sec)
