"""
evaluation.py - Mean value computation of functions over linear extensions.

Implements:
  - ExactEvaluation
  - BuildBubleyDyerEvaluationGenerator / BubleyDyerEvaluation
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from .poset import POSet
from .linear_extensions import LEGenerator, LEBubleyDyer


# ---------------------------------------------------------------------------
# Exact Evaluation
# ---------------------------------------------------------------------------

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
    dict with keys:
        'averages'     : list of np.ndarray
        'n_extensions' : int

    Examples
    --------
    >>> def rank_a(le): return np.array([le.index('a')])
    >>> result = ExactEvaluation(pos, rank_a)
    >>> result['averages']
    """
    gen = LEGenerator(poset)
    gen.reset()
    all_le = gen.get_batch()

    n_funcs = len(functions)
    sums: List[Optional[np.ndarray]] = [None] * n_funcs

    start = time.time()
    last_print = start
    count = 0

    for le in all_le:
        count += 1
        for fi, f in enumerate(functions):
            val = np.asarray(f(le), dtype=np.float64)
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
    return {"averages": averages, "n_extensions": count}


# ---------------------------------------------------------------------------
# Bubley-Dyer Evaluation Generator
# ---------------------------------------------------------------------------

class _BubleyDyerEvalGen:
    """
    Internal state holder for incremental function evaluation
    over sampled linear extensions.

    Maintains running sums of function outputs, allowing progressive
    refinement of average estimates.
    """

    def __init__(
        self,
        poset: POSet,
        seed: Optional[int],
        functions: List[Callable],
    ):
        self.poset = poset
        self.functions = functions
        self._sampler = LEBubleyDyer(poset, seed=seed)
        self._n_funcs = len(functions)
        self._sums: List[Optional[np.ndarray]] = [None] * self._n_funcs
        self._total_count: int = 0

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
        Add samples and return updated function averages.

        Three modes (pick one):

        1. **Fixed count** (``n``):
           Add exactly *n* new samples.

        2. **Theoretical bound** (``error``):
           Compute the number of samples from the Bubley-Dyer
           mixing-time bound.  Very conservative — prefer mode 3.

        3. **Empirical convergence** (``converge_tol``):
           Keep sampling in batches of ``converge_check_every`` until the
           max absolute change in any average between consecutive
           checkpoints falls below ``converge_tol``, or ``max_samples``
           is reached.

        Parameters
        ----------
        n : int, optional
            Number of new samples to add.
        error : float, optional
            Desired precision (theoretical bound, very conservative).
        converge_tol : float, optional
            Stop when max |Δavg| between checkpoints < tol.
        converge_check_every : int
            Batch size between convergence checks (default: 10 000).
        max_samples : int
            Safety cap for convergence mode (default: 1 000 000).
        output_every_sec : int, optional
            Print progress every N seconds.

        Returns
        -------
        dict with keys:
            'averages'     : list of np.ndarray
            'n_extensions' : int (total samples so far)
            'converged'    : bool (only in convergence mode)
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
        """Sample in batches until averages stabilise or budget is exhausted."""
        added = 0
        prev_avgs = self.averages if self._total_count > 0 else None
        converged = False
        max_delta = float("inf")

        start = time.time()
        last_print = start

        while added < max_samples:
            batch_size = min(check_every, max_samples - added)
            self._sample_and_accumulate(batch_size, output_every_sec=None)
            added += batch_size

            current_avgs = self.averages

            if prev_avgs is not None:
                max_delta = max(
                    (
                        float(np.max(np.abs(ca - pa)))
                        for ca, pa in zip(current_avgs, prev_avgs)
                        if ca is not None and pa is not None
                    ),
                    default=float("inf"),
                )

                if output_every_sec is not None:
                    now = time.time()
                    if now - last_print >= output_every_sec:
                        print(
                            f"  {self._total_count:>10,} samples | "
                            f"max |Δavg| = {max_delta:.6f} "
                            f"(target < {tol})"
                        )
                        last_print = now

                if max_delta < tol:
                    converged = True
                    break

            prev_avgs = [a.copy() if a is not None else None for a in current_avgs]

        result = self._build_result()
        result["converged"] = converged
        result["max_delta"] = max_delta

        if converged:
            print(
                f"  Converged after {self._total_count:,} total samples "
                f"(max |Δavg| = {max_delta:.6f} < {tol})"
            )
        else:
            print(
                f"  Budget exhausted ({max_samples:,} new samples). "
                f"max |Δavg| = {max_delta:.6f} (target was < {tol})"
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
        """Draw *n* linear extensions and update running sums."""
        batch = self._sampler.sample_batch(
            n=n,
            error=None,
            output_every_sec=output_every_sec,
        )

        for le in batch:
            for fi, f in enumerate(self.functions):
                val = np.asarray(f(le), dtype=np.float64)
                if self._sums[fi] is None:
                    self._sums[fi] = val.copy()
                else:
                    self._sums[fi] += val

        self._total_count += len(batch)

    # ------------------------------------------------------------------
    # Properties and helpers
    # ------------------------------------------------------------------

    @property
    def averages(self) -> List[Optional[np.ndarray]]:
        """Current average estimates for each function."""
        if self._total_count == 0:
            return [None] * self._n_funcs
        return [
            s / self._total_count if s is not None else None
            for s in self._sums
        ]

    @property
    def total_samples(self) -> int:
        """Total number of linear extensions sampled so far."""
        return self._total_count

    def reset(self) -> None:
        """Clear all accumulated samples and start fresh."""
        self._sums = [None] * self._n_funcs
        self._total_count = 0

    def _build_result(self) -> Dict:
        return {
            "averages": self.averages,
            "n_extensions": self._total_count,
        }

    def __repr__(self) -> str:
        return (
            f"BubleyDyerEvaluation("
            f"elements={self.poset.n}, "
            f"functions={self._n_funcs}, "
            f"samples={self._total_count:,})"
        )


# ---------------------------------------------------------------------------
# Public factory + convenience wrapper
# ---------------------------------------------------------------------------

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
    >>> result = BubleyDyerEvaluation(gen, n=10_000)
    """
    return _BubleyDyerEvalGen(poset, seed, list(functions))


def BubleyDyerEvaluation(
    generator: Union[_BubleyDyerEvalGen, POSet],
    n: Optional[int] = None,
    error: Optional[float] = None,
    output_every_sec: Optional[int] = None,
) -> Dict:
    """
    Estimate (or refine) function averages over linear extensions.

    Accepts either a ``_BubleyDyerEvalGen`` (created by
    ``BuildBubleyDyerEvaluationGenerator``) or a ``POSet`` directly.
    If a ``POSet`` is passed, a generator is created on the fly
    (but note: no functions will be registered — use the generator
    pattern for function evaluation).

    Examples
    --------
    >>> gen = BuildBubleyDyerEvaluationGenerator(pos, None, my_func)
    >>> result = BubleyDyerEvaluation(gen, n=40_000)
    >>> result = BubleyDyerEvaluation(gen, n=10_000)   # refinement
    """
    if isinstance(generator, POSet):
        generator = _BubleyDyerEvalGen(generator, seed=None, functions=[])

    return generator.update(
        n=n, error=error, output_every_sec=output_every_sec
    )
