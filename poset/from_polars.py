"""
Implementations to work with Polars dataframes and handle null values as uncertainty intervals in the hyperlattice.
Authors: Simone Caglio

Conceptual model
-------------------
Each statistical unit does not occupy a *point* in the hyperlattice
of indicators, but a *region*: the set of all points compatible with
the observed values. A null on indicator k means that the true value
could be any admissible value on that dimension.

Formally, each unit a is represented by an **interval**
    [a_lo, a_hi]
where:
  a_lo[k] = observed value if non-null, otherwise min_k (lower bound)
  a_hi[k] = observed value if non-null, otherwise max_k (upper bound)

Types of dominance between intervals (parameter `dominance_mode`)
-------------------------------------------------------------
'certain'
    a ≤_certain b  iff  a_hi[k] ≤ b_lo[k]  for every k
    "a is certainly below b in every possible scenario"

'possible'
    a ≤_possible b  iff  a_lo[k] ≤ b_hi[k]  for every k
    "there exists at least one scenario in which a is below b"

'certain_or_possible'  (default)
    Returns both POSet + confidence for every pair.

Construction of the ID
-------------------
The ID is constructed as  col1 + sep + col2.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Literal
import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

DominanceMode = Literal['certain', 'possible', 'certain_or_possible']



def poset_from_polars(
    df: "pl.DataFrame",
    id_col: Optional[str] = None,
    indicator_cols: Optional[List[str]] = None,
    higher_is_better: bool = True,
    dominance_mode: DominanceMode = 'certain_or_possible',
    value_range: Optional[Tuple[float, float]] = None,
    max_null_frac: float = 1.0,
    unit_sep: str = "_",
    col1: Optional[str] = None,
    col2: Optional[str] = None,
) -> Dict:
    """
    Constructs a POSet from a Polars DataFrame with null values representing indeterminacy.

    Null values are neither imputed nor ignored: they represent **structural uncertainty**.
    Each unit with at least one null occupies a *region* in the hyperlattice instead
    of a precise point. Dominance is evaluated on intervals [a_lo, a_hi].

    Parameters
    ----------
    df : pl.DataFrame
    id_col : str, optional
        Pre-existing ID column. If None, the ID is constructed from col1 + col2.
    indicator_cols : list of str, optional
        Indicator columns. If None → all integer columns (excluding id/time/area).
    higher_is_better : bool
        True (default): high value = best position in the poset.
        False: low value = best position.
    dominance_mode : str
        'certain'             → a ≤ b only if certain in every scenario
        'possible'            → a ≤ b if there is at least one scenario
        'certain_or_possible' → returns both (default)
    value_range : (min, max), optional
        Global admissible range. If None, derived from the data for each indicator.
    max_null_frac : float in [0,1]
        Maximum fraction of null values to include a unit. Default = 1.0.
    unit_sep : str
        Separator in ID label. Default = '_'.
    col1 : str, optional
        First column to build the ID.
    col2 : str, optional
        Second column to build the ID.

    Returns
    -------
    dict with keys:
        'elements'       : list of str
        'dom_certain'    : list of (str,str) — couples certainly dominated
        'dom_possible'   : list of (str,str) — couples possibly dominated
        'poset_certain'  : POSet on dom_certain
        'poset_possible' : POSet on dom_possible
        'intervals'      : {unit: {'lo': array, 'hi': array}}
        'null_mask'      : {unit: bool array (True = null)}
        'confidence'     : {(a,b): float} — 1.0 = no null on a or b, 0.0 = all null on a or b
        'indicator_cols' : list of str
        'excluded'       : list of str — units excluded for too many nulls
        'value_ranges'   : np.ndarray (k, 2) — [min_k, max_k] per indicator

    Examples
    --------
    >>> result = poset_from_polars(
    ...     df,
    ...     area_col='REF_AREA',
    ...     time_col='TIME_PERIOD',
    ... )
    >>> pos = result['poset_certain']
    >>> conf = result['confidence']   # {('SWE_2018','DNK_2018'): 0.87, ...}
    """
    if not HAS_POLARS:
        raise ImportError("polars is not installed. Use `pip install polars`")

    # 1. ID: col1 + sep + col2  (col1 prima)
    df = _build_id_col(df, id_col, col2, col1, unit_sep)
    id_col_actual = "__unit_id__"

    # 2. Indicatori
    indicator_cols = _resolve_indicator_cols(
        df, indicator_cols, id_col_actual, col2, col1, id_col
    )

    # 3. Matrice raw (float, NaN dove null)
    all_ids  = df[id_col_actual].to_list()
    mat_raw  = df.select(indicator_cols).to_numpy().astype(float)

    # 4. Filtra per max_null_frac
    null_frac  = np.isnan(mat_raw).mean(axis=1)
    keep_mask  = null_frac <= max_null_frac
    ids        = [all_ids[i] for i in range(len(all_ids)) if keep_mask[i]]
    excluded   = [all_ids[i] for i in range(len(all_ids)) if not keep_mask[i]]
    mat        = mat_raw[keep_mask, :]

    if len(ids) == 0:
        raise ValueError("No units exceed the max_null_frac threshold.")

    # 5. Inversione se higher_is_better=False
    if not higher_is_better:
        mat = -mat

    # 6. Range per dimensione (usato per costruire gli intervalli)
    vranges = _compute_value_ranges(mat, value_range, higher_is_better)

    # 7. Costruisci intervalli [lo, hi] per ogni unità
    n, k       = mat.shape
    lo         = mat.copy()
    hi         = mat.copy()
    null_mask_arr = np.isnan(mat)
    for j in range(k):
        lo[null_mask_arr[:, j], j] = vranges[j, 0]
        hi[null_mask_arr[:, j], j] = vranges[j, 1]

    # 8. Dominanza per intervalli
    dom_certain, dom_possible, confidence = _build_interval_dom(
        ids, lo, hi, null_mask_arr
    )

    # 9. Costruisci i due POSet
    from poset.poset import POSet
    pos_certain  = POSet(ids, dom_certain)
    pos_possible = POSet(ids, dom_possible)

    # 10. Dizionari di output
    intervals  = {ids[i]: {'lo': lo[i], 'hi': hi[i]} for i in range(n)}
    null_masks = {ids[i]: null_mask_arr[i] for i in range(n)}

    return {
        'elements'       : ids,
        'dom_certain'    : dom_certain,
        'dom_possible'   : dom_possible,
        'poset_certain'  : pos_certain,
        'poset_possible' : pos_possible,
        'intervals'      : intervals,
        'null_mask'      : null_masks,
        'confidence'     : confidence,
        'indicator_cols' : indicator_cols,
        'excluded'       : excluded,
        'value_ranges'   : vranges,
    }



def interval_summary(result: Dict) -> "pl.DataFrame":
    """
    DataFrame Polars of interval summary for each unit.

    Columns
    -------
    unit, n_null, null_frac, interval_volume, is_point,
    n_certain_dom, n_possible_dom, mean_confidence
    """
    ids   = result['elements']
    ivs   = result['intervals']
    nmask = result['null_mask']
    k     = len(result['indicator_cols'])
    conf  = result['confidence']

    dom_cert_set = set(result['dom_certain'])
    dom_poss_set = set(result['dom_possible'])

    rows = []
    for uid in ids:
        lo = ivs[uid]['lo']
        hi = ivs[uid]['hi']
        nm = nmask[uid]
        n_null = int(nm.sum())
        widths = hi - lo
        # Volume come prodotto delle larghezze (unità discrete: +1 per contare entrambi estremi)
        vol = float(np.prod(widths + 1))

        n_cert = sum(1 for b in ids if b != uid and (uid, b) in dom_cert_set)
        n_poss = sum(1 for b in ids if b != uid and (uid, b) in dom_poss_set)
        confs  = [conf[(uid, b)] for b in ids if b != uid and (uid, b) in conf]
        mean_c = float(np.mean(confs)) if confs else float('nan')

        rows.append({
            'unit'            : uid,
            'n_null'          : n_null,
            'null_frac'       : round(n_null / k, 3),
            'interval_volume' : round(vol, 1),
            'is_point'        : n_null == 0,
            'n_certain_dom'   : n_cert,
            'n_possible_dom'  : n_poss,
            'mean_confidence' : round(mean_c, 3) if not np.isnan(mean_c) else None,
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_id_col(df, id_col, time_col, area_col, unit_sep):
    """Make __unit_id__. Order: col1 + sep + col2."""
    import polars as pl

    if id_col is not None and id_col in df.columns:
        return df.with_columns(
            pl.col(id_col).cast(pl.Utf8).alias("__unit_id__")
        )

    # area prima, anno dopo
    if area_col is not None and time_col is not None:
        if area_col in df.columns and time_col in df.columns:
            return df.with_columns(
                (pl.col(area_col).cast(pl.Utf8)
                 + unit_sep
                 + pl.col(time_col).cast(pl.Utf8)).alias("__unit_id__")
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
    import polars as pl
    if indicator_cols is not None:
        return list(indicator_cols)
    exclude = {id_col_actual, time_col, area_col, id_col} - {None}
    cols = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]
    if not cols:
        raise ValueError(
            "No indicator columns found. Specify indicator_cols."
        )
    return cols


def _compute_value_ranges(
    mat: np.ndarray,
    value_range: Optional[Tuple[float, float]],
    higher_is_better: bool,
) -> np.ndarray:
    """Return array(k, 2) with [min_k, max_k] for each indicator."""
    k = mat.shape[1]
    vranges = np.zeros((k, 2))

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
    Dominance of certain and possible intervals.

    Certain:    a ≤_cert b  iff  hi[a,k] ≤ lo[b,k]  ∀k
    Possible: a ≤_poss b  iff  lo[a,k] ≤ hi[b,k]  ∀k

    Confidence(a→b) = fraction of indicators without null on either of the two units.
    """
    n, k = lo.shape
    dom_certain  = []
    dom_possible = []
    confidence   = {}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            certain  = bool(np.all(hi[i] <= lo[j]))
            possible = bool(np.all(lo[i] <= hi[j]))

            if certain:
                dom_certain.append((ids[i], ids[j]))

            if possible:
                dom_possible.append((ids[i], ids[j]))

            if certain or possible:
                neither_null = ~null_mask[i] & ~null_mask[j]
                confidence[(ids[i], ids[j])] = float(neither_null.sum()) / k

    return dom_certain, dom_possible, confidence
