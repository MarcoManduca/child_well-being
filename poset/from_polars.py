"""
from_polars.py — Build POSet from Polars DataFrames with null-as-uncertainty.

Authors: Simone Caglio & Marco Manduca

Conceptual model
----------------
Each statistical unit does not occupy a *point* in the hyperlattice
of indicators, but a *region*: the set of all points compatible with
the observed values.  A null on indicator k means that the true value
could be any admissible value on that dimension.

Formally, each unit *a* is represented by an **interval**
    [a_lo, a_hi]
where:
  a_lo[k] = observed value if non-null, otherwise min_k  (lower bound)
  a_hi[k] = observed value if non-null, otherwise max_k  (upper bound)

Types of dominance between intervals (parameter ``dominance_mode``)
-------------------------------------------------------------------
'certain'
    a ≤_certain b  iff  a_hi[k] ≤ b_lo[k]  for every k
    "a is certainly below b in every possible scenario"

'possible'
    a ≤_possible b  iff  a_lo[k] ≤ b_hi[k]  for every k
    "there exists at least one scenario in which a is below b"

'certain_or_possible'  (default)
    Returns both POSet + confidence for every pair.

Construction of the ID
----------------------
The ID is constructed as  col1 + sep + col2.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

DominanceMode = Literal["certain", "possible", "certain_or_possible"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def poset_from_polars(
    df: "pl.DataFrame",
    id_col: Optional[str] = None,
    indicator_cols: Optional[List[str]] = None,
    higher_is_better: bool = True,
    dominance_mode: DominanceMode = "certain_or_possible",
    value_range: Optional[Tuple[float, float]] = None,
    max_null_frac: float = 1.0,
    unit_sep: str = "_",
    col1: Optional[str] = None,
    col2: Optional[str] = None,
) -> Dict:
    """
    Construct a POSet from a Polars DataFrame with null-as-uncertainty.

    Null values are neither imputed nor ignored: they represent
    **structural uncertainty**.  Each unit with at least one null
    occupies a *region* in the hyperlattice instead of a precise point.
    Dominance is evaluated on intervals ``[a_lo, a_hi]``.

    Parameters
    ----------
    df : pl.DataFrame
    id_col : str, optional
        Pre-existing ID column.  If None the ID is built from
        ``col1 + unit_sep + col2``.
    indicator_cols : list of str, optional
        Indicator columns.  If None → all integer columns
        (excluding id / time / area).
    higher_is_better : bool
        True (default): high value = best position in the poset.
    dominance_mode : str
        ``'certain'`` | ``'possible'`` | ``'certain_or_possible'``
        (default).
    value_range : (min, max), optional
        Global admissible range.  If None, derived per indicator.
    max_null_frac : float in [0, 1]
        Maximum fraction of null values to include a unit.
    unit_sep : str
        Separator in ID label.
    col1, col2 : str, optional
        Columns used to build the ID.

    Returns
    -------
    dict with keys:
        'elements', 'dom_certain', 'dom_possible',
        'poset_certain', 'poset_possible',
        'intervals', 'null_mask', 'confidence',
        'indicator_cols', 'excluded', 'value_ranges'
    """
    if not HAS_POLARS:
        raise ImportError("polars is not installed.  Use `pip install polars`")

    # 1. ID column
    df = _build_id_col(df, id_col, col2, col1, unit_sep)
    id_col_actual = "__unit_id__"

    # 2. Indicator columns
    indicator_cols = _resolve_indicator_cols(
        df, indicator_cols, id_col_actual, col2, col1, id_col
    )

    # 3. Raw matrix (float, NaN where null)
    all_ids = df[id_col_actual].to_list()
    mat_raw = df.select(indicator_cols).to_numpy().astype(np.float64)

    # 4. Filter by max_null_frac
    null_frac = np.isnan(mat_raw).mean(axis=1)
    keep_mask = null_frac <= max_null_frac
    ids = [all_ids[i] for i in range(len(all_ids)) if keep_mask[i]]
    excluded = [all_ids[i] for i in range(len(all_ids)) if not keep_mask[i]]
    mat = mat_raw[keep_mask, :]

    if len(ids) == 0:
        raise ValueError("No units survive the max_null_frac threshold.")

    # 5. Flip sign if lower is better
    if not higher_is_better:
        mat = -mat

    # 6. Per-indicator value ranges (used to fill null intervals)
    vranges = _compute_value_ranges(mat, value_range, higher_is_better)

    # 7. Build intervals [lo, hi] for each unit
    n, k = mat.shape
    null_mask_arr = np.isnan(mat)

    lo = mat.copy()
    hi = mat.copy()
    for j in range(k):
        lo[null_mask_arr[:, j], j] = vranges[j, 0]
        hi[null_mask_arr[:, j], j] = vranges[j, 1]

    # 8. Interval dominance (vectorized)
    dom_certain, dom_possible, confidence = _build_interval_dom(
        ids, lo, hi, null_mask_arr
    )

    # 9. Build the two POSets
    from .poset import POSet

    pos_certain = POSet(ids, dom_certain)
    pos_possible = POSet(ids, dom_possible)

    # 10. Output dictionaries
    intervals = {ids[i]: {"lo": lo[i], "hi": hi[i]} for i in range(n)}
    null_masks = {ids[i]: null_mask_arr[i] for i in range(n)}

    return {
        "elements": ids,
        "dom_certain": dom_certain,
        "dom_possible": dom_possible,
        "poset_certain": pos_certain,
        "poset_possible": pos_possible,
        "intervals": intervals,
        "null_mask": null_masks,
        "confidence": confidence,
        "indicator_cols": indicator_cols,
        "excluded": excluded,
        "value_ranges": vranges,
    }


def interval_summary(result: Dict) -> "pl.DataFrame":
    """
    Polars DataFrame summarising each unit's interval properties.

    Columns: unit, n_null, null_frac, interval_volume, is_point,
    n_certain_dom, n_possible_dom, mean_confidence
    """
    ids = result["elements"]
    ivs = result["intervals"]
    nmask = result["null_mask"]
    k = len(result["indicator_cols"])
    conf = result["confidence"]

    # Pre-build sets for O(1) lookup
    dom_cert_set = set(result["dom_certain"])
    dom_poss_set = set(result["dom_possible"])

    # Group confidence values by source unit
    conf_by_unit: Dict[str, List[float]] = {uid: [] for uid in ids}
    for (a, b), c in conf.items():
        if a in conf_by_unit:
            conf_by_unit[a].append(c)

    rows = []
    for uid in ids:
        nm = nmask[uid]
        lo = ivs[uid]["lo"]
        hi = ivs[uid]["hi"]
        n_null = int(nm.sum())
        vol = float(np.prod(hi - lo + 1))

        n_cert = sum(1 for b in ids if b != uid and (uid, b) in dom_cert_set)
        n_poss = sum(1 for b in ids if b != uid and (uid, b) in dom_poss_set)

        confs = conf_by_unit[uid]
        mean_c = float(np.mean(confs)) if confs else None

        rows.append(
            {
                "unit": uid,
                "n_null": n_null,
                "null_frac": round(n_null / k, 3),
                "interval_volume": round(vol, 1),
                "is_point": n_null == 0,
                "n_certain_dom": n_cert,
                "n_possible_dom": n_poss,
                "mean_confidence": round(mean_c, 3) if mean_c is not None else None,
            }
        )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_id_col(df, id_col, time_col, area_col, unit_sep):
    """Create ``__unit_id__`` column.  Order: area + sep + time."""
    import polars as pl

    if id_col is not None and id_col in df.columns:
        return df.with_columns(
            pl.col(id_col).cast(pl.Utf8).alias("__unit_id__")
        )

    if area_col is not None and time_col is not None:
        if area_col in df.columns and time_col in df.columns:
            return df.with_columns(
                (
                    pl.col(area_col).cast(pl.Utf8)
                    + unit_sep
                    + pl.col(time_col).cast(pl.Utf8)
                ).alias("__unit_id__")
            )

    if area_col is not None and area_col in df.columns:
        return df.with_columns(
            pl.col(area_col).cast(pl.Utf8).alias("__unit_id__")
        )

    if time_col is not None and time_col in df.columns:
        return df.with_columns(
            pl.col(time_col).cast(pl.Utf8).alias("__unit_id__")
        )

    return df.with_columns(
        pl.Series("__unit_id__", [f"unit_{i}" for i in range(len(df))])
    )


def _resolve_indicator_cols(df, indicator_cols, id_col_actual, time_col, area_col, id_col):
    """Detect or validate indicator columns."""
    import polars as pl

    if indicator_cols is not None:
        return list(indicator_cols)

    exclude = {id_col_actual, time_col, area_col, id_col} - {None}
    cols = [
        c
        for c in df.columns
        if c not in exclude
        and df[c].dtype
        in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        )
    ]
    if not cols:
        raise ValueError(
            "No indicator columns found.  Specify indicator_cols explicitly."
        )
    return cols


def _compute_value_ranges(
    mat: np.ndarray,
    value_range: Optional[Tuple[float, float]],
    higher_is_better: bool,
) -> np.ndarray:
    """Return array (k, 2) with ``[min_k, max_k]`` for each indicator."""
    k = mat.shape[1]
    vranges = np.zeros((k, 2), dtype=np.float64)

    if value_range is not None:
        lo_v, hi_v = value_range
        if not higher_is_better:
            lo_v, hi_v = -hi_v, -lo_v
        vranges[:, 0] = lo_v
        vranges[:, 1] = hi_v
    else:
        for j in range(k):
            col = mat[:, j]
            obs = col[~np.isnan(col)]
            if len(obs) == 0:
                vranges[j] = [0.0, 1.0]
            else:
                vranges[j, 0] = obs.min()
                vranges[j, 1] = obs.max()

    return vranges


def _build_interval_dom(
    ids: List[str],
    lo: np.ndarray,
    hi: np.ndarray,
    null_mask: np.ndarray,
) -> Tuple[List, List, Dict]:
    """
    Vectorized interval dominance computation.

    Certain:   a ≤_cert b  iff  hi[a, k] ≤ lo[b, k]  ∀ k
    Possible:  a ≤_poss b  iff  lo[a, k] ≤ hi[b, k]  ∀ k

    Confidence(a → b) = fraction of indicators where neither a nor b
    has a null value.

    All pairwise comparisons are done via NumPy broadcasting
    instead of a Python double-loop.
    """
    n, k = lo.shape

    # hi[i] vs lo[j] for all pairs: shape (n, n, k)
    # certain[i, j] = True iff hi[i, :] <= lo[j, :] component-wise
    certain_mat = np.all(hi[:, None, :] <= lo[None, :, :], axis=2)

    # lo[i] vs hi[j] for all pairs
    possible_mat = np.all(lo[:, None, :] <= hi[None, :, :], axis=2)

    # Exclude diagonal (self-dominance)
    np.fill_diagonal(certain_mat, False)
    np.fill_diagonal(possible_mat, False)

    # Confidence: fraction of indicators with no null on either unit
    # neither_null[i, j, k] = True iff indicator k is observed for both i and j
    # confidence[i, j] = mean over k
    not_null = ~null_mask  # (n, k) bool
    # not_null[i, k] AND not_null[j, k] → (n, n, k) via broadcasting
    neither_null_sum = not_null.astype(np.float64) @ not_null.astype(np.float64).T
    # each entry [i, j] = count of indicators where both are non-null
    confidence_mat = neither_null_sum / k

    # Extract pairs as lists
    dom_certain = []
    dom_possible = []
    confidence = {}

    # Use np.argwhere for sparse extraction (typically much fewer pairs than n²)
    cert_pairs = np.argwhere(certain_mat)
    poss_pairs = np.argwhere(possible_mat)

    # Gather all pairs that are either certain or possible (for confidence dict)
    either_mat = certain_mat | possible_mat

    for i, j in cert_pairs:
        dom_certain.append((ids[i], ids[j]))

    for i, j in poss_pairs:
        dom_possible.append((ids[i], ids[j]))

    # Build confidence dict only for pairs with at least one relation
    either_pairs = np.argwhere(either_mat)
    for i, j in either_pairs:
        confidence[(ids[i], ids[j])] = float(confidence_mat[i, j])

    return dom_certain, dom_possible, confidence
