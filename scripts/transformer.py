"""
transformer.py — Data transformation utilities for the child well-being pipeline.

Provides min-max normalisation, ordinal discretisation, and MRP-based cascade
aggregation at two levels (sub-dimension → dimension → macro-dimension).
"""

from __future__ import annotations

import time
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import polars as pl

from poset.from_polars import poset_from_polars
from poset.mrp import ExactMRP, BubleyDyerMRPGenerator


def normalize_minmax(
    df: pl.DataFrame,
    indicators: dict[str, dict] | None = None,
    suffix: str = "_norm",
    use_robust_bounds: bool = False,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pl.DataFrame:
    """
    Min-Max normalization of all indicators in the DataFrame.

    Parameters
    ---------
    df : pl.DataFrame
        DataFrame with columns corresponding to the indicator codes.
    indicators : dict, optional
        Configuration dictionary.
    suffix : str
        Suffix for the normalized columns (default: "_norm").
    use_robust_bounds : bool
        If True, uses quantiles instead of min/max to limit the effect of outliers.
    lower_quantile / upper_quantile : float
        Quantiles to use as bounds if use_robust_bounds=True.

    Returns
    -------
    pl.DataFrame
        DataFrame original + normalized columns with suffix.
        All _norm columns are oriented: 1 = maximum well-being, 0 = minimum well-being.
    """
    exprs = []

    for col_name, meta in indicators.items():
        if col_name not in df.columns:
            continue

        col = pl.col(col_name)

        if use_robust_bounds:
            col_min = df[col_name].quantile(lower_quantile)
            col_max = df[col_name].quantile(upper_quantile)
        else:
            col_min = df[col_name].drop_nulls().min()
            col_max = df[col_name].drop_nulls().max()

        if col_max == col_min:
            exprs.append(pl.lit(0.5).alias(f"{col_name}{suffix}"))
            continue

        denom = col_max - col_min

        if meta["direction"] == "negative":
            normalized = (pl.lit(col_max) - col) / denom
        else:
            normalized = (col - pl.lit(col_min)) / denom

        normalized = normalized.clip(0.0, 1.0)
        exprs.append(normalized.alias(f"{col_name}{suffix}"))

    return df.with_columns(exprs)


def get_norm_columns(df: pl.DataFrame, suffix: str = "_norm") -> list[str]:
    """Returns a list of column names in the DataFrame that end with the specified suffix."""
    return [c for c in df.columns if c.endswith(suffix)]


def discretize(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    n_levels: int = 3,
    method: str = "quantile",
    normalized_suffix: str = "_norm",
    suffix: str = "_ord",
) -> pl.DataFrame:
    """
    Discretizes continuous columns into ordinal levels {1, 2, ..., n_levels}.

    Parameters
    ---------
    columns : list[str]
        Columns to discretize. Default: all columns ending with "_norm".
    n_levels : int
        Number of ordinal levels (default: 3).
    method : str
        - "quantile": cut-point based on the quantiles of the distribution
        - "equal_width": equidistant cut-point on [0, 1]
    suffix : str
        Suffix for the discretized columns (default: "_ord").

    Returns
    -------
    pl.DataFrame with discretized columns.
    """
    if columns is None:
        columns = get_norm_columns(df, normalized_suffix)

    exprs = []

    for col_name in columns:
        if col_name not in df.columns:
            continue

        if method == "quantile":
            breakpoints = [
                df[col_name].drop_nulls().quantile(q)
                for q in [i / n_levels for i in range(1, n_levels)]
            ]
        elif method == "equal_width":
            breakpoints = [i / n_levels for i in range(1, n_levels)]
        else:
            raise ValueError(f"Method '{method}' not supported. Use 'quantile' or 'equal_width'.")

        col = pl.col(col_name)
        expr = pl.when(col.is_null()).then(None)

        for i, bp in enumerate(breakpoints):
            expr = expr.when(col <= bp).then(i + 1)

        expr = expr.otherwise(n_levels)

        out_name = col_name.replace("_norm", "") + suffix if "_norm" in col_name else col_name + suffix
        exprs.append(expr.alias(out_name))

    return df.with_columns(exprs)


def cascade_aggregate(
    df: pl.DataFrame,
    indicators: Dict[str, dict],
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    unit_sep: str = "_",
    mrp_mode: str = "exact",
    mrp_n_samples: int = 100_000,
    mrp_converge_tol: Optional[float] = 0.005,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Level 1 cascade on **pre-discretised** data.

    Parameters
    ----------
    df : pl.DataFrame
        Data with col1, col2, and indicator columns containing
        **ordinal integer values** (e.g. 1, 2, 3, 4).
    indicators : dict
        Indicator configuration.  Each entry must have keys
        'type' (``'indicator'`` or ``'public_expenditure'``)
        and ``'subdimension'``.
    col1, col2 : str, optional
        ID columns preserved in output (e.g. 'REF_AREA', 'TIME_PERIOD').
    unit_sep : str
        Separator for internal ID.
    mrp_mode : str
        ``'exact'`` or ``'approximate'``.
    mrp_n_samples : int
    mrp_converge_tol : float, optional
    seed : int
    verbose : bool

    Returns
    -------
    dict with keys:
        'indicator'          : pl.DataFrame
        'public_expenditure' : pl.DataFrame
        'sub_poset_details'  : dict
        'pass_through'       : dict
    """
    df = _ensure_id_col(df, col1, col2, unit_sep)
    units = df["__unit_id__"].to_list()

    available = [k for k in indicators if k in df.columns]
    print(f"Available indicators: {len(available)} / {len(indicators)}")
    print(f"  {available}")
    config = {k: indicators[k] for k in available}
    groups = _group_indicators(config)
    print(groups)

    if verbose:
        print(f"Units: {len(units)}")
        print(f"Available indicators: {len(available)}")
        print(f"Groups: {len(groups)}")
        for (typ, sub), cols in sorted(groups.items()):
            action = "sub-poset" if len(cols) > 1 else "pass-through"
            print(f"  [{typ}] {sub}: {cols} → {action}")
        print()

    sub_poset_details = {}
    results_by_type: Dict[str, Dict[str, np.ndarray]] = {
        "indicator": {},
        "public_expenditure": {},
    }
    pass_through = {}

    for (typ, sub), cols in sorted(groups.items()):
        if len(cols) == 1:
            values = df[cols[0]].to_numpy().astype(np.float64)
            results_by_type[typ][sub] = values
            pass_through[(typ, sub)] = cols[0]
            if verbose:
                print(f"  [{typ}] {sub}: pass-through ({cols[0]})")
        else:
            if verbose:
                print(f"  [{typ}] {sub}: building sub-poset on {cols}...")

            scores = _build_sub_poset_mrp_ordinal(
                df=df,
                units=units,
                cols=cols,
                mrp_mode=mrp_mode,
                mrp_n_samples=mrp_n_samples,
                mrp_converge_tol=mrp_converge_tol,
                seed=seed,
                verbose=verbose,
            )

            results_by_type[typ][sub] = scores["mrp_scores"]
            sub_poset_details[(typ, sub)] = scores

            if verbose:
                print(f"    → {scores['n_extensions']} extensions, "
                      f"score range: [{scores['mrp_scores'].min():.3f}, "
                      f"{scores['mrp_scores'].max():.3f}]")

    # Build output DataFrames (with min-max normalisation)
    id_columns = _extract_id_columns(df, col1, col2)
    output = {}
    for typ in ("indicator", "public_expenditure"):
        if not results_by_type[typ]:
            continue
        data = dict(id_columns)
        for sub, values in sorted(results_by_type[typ].items()):
            arr = np.asarray(values, dtype=np.float64)
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                vmin, vmax = valid.min(), valid.max()
                if vmax > vmin:
                    arr = (arr - vmin) / (vmax - vmin)
                else:
                    arr = np.where(np.isnan(arr), np.nan, 0.5)
            data[sub] = arr
        output[typ] = pl.DataFrame(data)

    if verbose:
        print()
        id_set = set(id_columns.keys())
        for typ in ("indicator", "public_expenditure"):
            if typ in output and isinstance(output[typ], pl.DataFrame):
                dims = [c for c in output[typ].columns if c not in id_set]
                print(f"[{typ}] Final dimensions: {len(dims)} → {dims}")

    return {**output, "sub_poset_details": sub_poset_details, "pass_through": pass_through}


def cascade_level2(
    df: pl.DataFrame,
    groups: Optional[Dict[str, List[str]]] = None,
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    unit_sep: str = "_",
    n_levels: int = 4,
    discretize_method: str = "quantile",
    mrp_mode: str = "approximate",
    mrp_n_samples: int = 100_000,
    mrp_converge_tol: Optional[float] = 0.005,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Level 2 cascade: group subdimension MRP scores (continuous [0,1])
    into macro-dimensions.

    Steps per group:
      1. Discretise the MRP scores into ordinal levels
      2. Build sub-poset
      3. Compute MRP score

    Parameters
    ----------
    df : pl.DataFrame
        Output of ``cascade_aggregate``'s ``'indicator'`` key.
        Columns: col1, col2, and **continuous** subdimension MRP scores.
    groups : dict, optional
        ``{macro_name: [subdimension_col, ...]}``
        Default: ``LEVEL2_INDICATOR_GROUPS``.
    col1, col2 : str, optional
    n_levels : int
        Discretisation levels for the MRP scores (default: 4).
    discretize_method : str
    mrp_mode, mrp_n_samples, mrp_converge_tol, seed, verbose : ...

    Returns
    -------
    dict with keys:
        'aggregated' : pl.DataFrame (col1, col2, macro-dimension columns)
        'details'    : dict
    """
    if groups is None:
        groups = LEVEL2_INDICATOR_GROUPS

    df = _ensure_id_col(df, col1, col2, unit_sep)
    units = df["__unit_id__"].to_list()

    if verbose:
        print(f"Level 2 cascade: {len(units)} units")
        print(f"Macro-dimensions: {len(groups)}")
        for name, cols in groups.items():
            present = [c for c in cols if c in df.columns]
            missing = [c for c in cols if c not in df.columns]
            action = "sub-poset" if len(present) > 1 else "pass-through"
            msg = f"  {name}: {present} → {action}"
            if missing:
                msg += f"  (missing: {missing})"
            print(msg)
        print()

    details = {}
    macro_scores: Dict[str, np.ndarray] = {}

    for name, cols in groups.items():
        present = [c for c in cols if c in df.columns]

        if not present:
            if verbose:
                print(f"  {name}: SKIPPED (no columns available)")
            continue

        if len(present) == 1:
            values = df[present[0]].to_numpy().astype(np.float64)
            macro_scores[name] = values
            if verbose:
                print(f"  {name}: pass-through ({present[0]})")
        else:
            if verbose:
                print(f"  {name}: discretising + sub-poset on {present}...")

            df_disc = _discretize_columns(df, present, n_levels, discretize_method)
            ord_cols = [f"{c}_ord" for c in present]

            df_ord = df_disc.select(
                [pl.col("__unit_id__")]
                + [pl.col(oc).cast(pl.Int32).alias(c) for oc, c in zip(ord_cols, present)]
            )

            scores = _build_sub_poset_mrp_ordinal(
                df=df_ord,
                units=units,
                cols=present,
                mrp_mode=mrp_mode,
                mrp_n_samples=mrp_n_samples,
                mrp_converge_tol=mrp_converge_tol,
                seed=seed,
                verbose=verbose,
            )

            macro_scores[name] = scores["mrp_scores"]
            details[name] = scores

            if verbose:
                print(f"    → {scores['n_extensions']} extensions, "
                      f"score range: [{scores['mrp_scores'].min():.3f}, "
                      f"{scores['mrp_scores'].max():.3f}]")

    id_columns = _extract_id_columns(df, col1, col2)
    data = dict(id_columns)
    for name, values in macro_scores.items():
        data[name] = values

    df_out = pl.DataFrame(data)

    if verbose:
        id_set = set(id_columns.keys())
        dims = [c for c in df_out.columns if c not in id_set]
        print(f"\nLevel 2 output: {len(dims)} macro-dimensions → {dims}")

    return {
        "aggregated": df_out,
        "details": details,
    }


# ===================================================================
# Internal helpers
# ===================================================================

def _ensure_id_col(df: pl.DataFrame, col1: Optional[str], col2: Optional[str], unit_sep: str) -> pl.DataFrame:
    """Build __unit_id__ from col1 + sep + col2 (internal use only)."""
    if "__unit_id__" in df.columns:
        return df
    if col1 is not None and col2 is not None:
        if col1 in df.columns and col2 in df.columns:
            return df.with_columns(
                (pl.col(col1).cast(pl.Utf8)
                 + unit_sep
                 + pl.col(col2).cast(pl.Utf8)).alias("__unit_id__")
            )
    if col1 is not None and col1 in df.columns:
        return df.with_columns(
            pl.col(col1).cast(pl.Utf8).alias("__unit_id__")
        )
    raise ValueError("Cannot build unit ID. Provide col1 (and optionally col2).")


def _extract_id_columns(
    df: pl.DataFrame,
    col1: Optional[str],
    col2: Optional[str],
) -> Dict[str, list]:
    """Extract col1/col2 values for output DataFrames."""
    out = {}
    if col1 is not None and col1 in df.columns:
        out[col1] = df[col1].to_list()
    if col2 is not None and col2 in df.columns:
        out[col2] = df[col2].to_list()
    return out


def _group_indicators(
    config: Dict[str, dict],
) -> Dict[Tuple[str, str], List[str]]:
    """Group indicator codes by (type, subdimension)."""
    groups: Dict[Tuple[str, str], List[str]] = {}
    for code, meta in config.items():
        typ = meta.get("type", "indicator")
        sub = meta["subdimension"]
        key = (typ, sub)
        groups.setdefault(key, []).append(code)
    return groups


def _discretize_columns(
    df: pl.DataFrame,
    cols: List[str],
    n_levels: int,
    method: str,
) -> pl.DataFrame:
    """Discretise continuous columns into ordinal levels {1, ..., n_levels}."""
    exprs = []
    for col_name in cols:
        values = df[col_name].drop_nulls()
        if method == "quantile":
            breakpoints = [
                float(values.quantile(q))
                for q in [i / n_levels for i in range(1, n_levels)]
            ]
        elif method == "equal_width":
            vmin, vmax = float(values.min()), float(values.max())
            step = (vmax - vmin) / n_levels if vmax > vmin else 1.0
            breakpoints = [vmin + step * i for i in range(1, n_levels)]
        else:
            raise ValueError(f"Unknown method: {method}")

        col = pl.col(col_name)
        expr = pl.when(col.is_null()).then(None)
        for i, bp in enumerate(breakpoints):
            expr = expr.when(col <= bp).then(i + 1)
        expr = expr.otherwise(n_levels)
        exprs.append(expr.alias(f"{col_name}_ord"))

    return df.with_columns(exprs)


def _build_sub_poset_mrp_ordinal(
    df: pl.DataFrame,
    units: List[str],
    cols: List[str],
    mrp_mode: str,
    mrp_n_samples: int,
    mrp_converge_tol: Optional[float],
    seed: int,
    verbose: bool,
) -> Dict:
    """
    Build a sub-poset from **already ordinal** columns and return MRP scores.

    No discretisation is performed — columns are expected to contain
    integer ordinal values.
    """
    sub_df = df.select(
        [pl.col("__unit_id__")]
        + [pl.col(c).cast(pl.Int32) for c in cols]
    )

    result = poset_from_polars(
        sub_df,
        id_col="__unit_id__",
        indicator_cols=cols,
        higher_is_better=True,
        dominance_mode="certain_or_possible",
    )

    poset = result["poset_certain"]

    if mrp_mode == "exact":
        try:
            mrp_result = ExactMRP(poset)
            method_used = "exact"
        except Exception as e:
            if verbose:
                print(f"    Exact MRP failed ({e}), falling back to approximate...")
            gen = BubleyDyerMRPGenerator(poset, seed=seed)
            if mrp_converge_tol is not None:
                mrp_result = gen.update(converge_tol=mrp_converge_tol)
            else:
                mrp_result = gen.update(n=mrp_n_samples)
            method_used = "approximate"
    else:
        gen = BubleyDyerMRPGenerator(poset, seed=seed)
        if mrp_converge_tol is not None:
            mrp_result = gen.update(converge_tol=mrp_converge_tol)
        else:
            mrp_result = gen.update(n=mrp_n_samples)
        method_used = "approximate"

    mrp_matrix = mrp_result["MRP"]
    scores_by_element = mrp_matrix.mean(axis=0)
    elements = mrp_result["elements"]

    elem_to_score = {e: float(s) for e, s in zip(elements, scores_by_element)}
    aligned_scores = np.array([elem_to_score.get(u, np.nan) for u in units])

    return {
        "mrp_scores": aligned_scores,
        "mrp_matrix": mrp_matrix,
        "elements": elements,
        "n_extensions": mrp_result["n_extensions"],
        "poset": poset,
        "poset_result": result,
        "method": method_used,
        "indicators": cols,
    }
