
import polars as pl

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
        Configuration dictionary. Default: global INDICATORS.
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

        # Avoid division by zero if range is zero
        if col_max == col_min:
            exprs.append(pl.lit(0.5).alias(f"{col_name}{suffix}"))
            continue

        denom = col_max - col_min

        if meta["direction"] == "negative":
            # Inverse: (max - x) / range
            normalized = (pl.lit(col_max) - col) / denom
        else:
            # Direct: (x - min) / range
            normalized = (col - pl.lit(col_min)) / denom

        # Clamp to [0, 1] (useful if robust quantiles are used)
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
        "quantile" → cut-point based on the quantiles of the distribution
        "equal_width" → cut-point equidistanti su [0, 1]
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

        # Building when/then expression for assigning levels
        col = pl.col(col_name)
        expr = pl.when(col.is_null()).then(None)

        for i, bp in enumerate(breakpoints):
            expr = expr.when(col <= bp).then(i + 1)

        expr = expr.otherwise(n_levels)

        out_name = col_name.replace("_norm", "") + suffix if "_norm" in col_name else col_name + suffix
        exprs.append(expr.alias(out_name))

    return df.with_columns(exprs)
