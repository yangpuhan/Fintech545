"""Turning a price series into a return series.

Prices are non-stationary, so nothing from the first three weeks applies to
them directly.  Returns are the transform that makes them apply.

Covers functional tests 6.1 and 6.2.
"""

import numpy as np
import pandas as pd


def return_calculate(prices, method="DISCRETE", date_column="Date"):
    """Arithmetic ("DISCRETE") or log ("LOG") returns from a price DataFrame.

    Returns a DataFrame with the same columns and one fewer row: the first
    price has no predecessor, so the returned frame is dated from the *second*
    price onward.

    Which method to use depends on which direction the sum runs.  Arithmetic
    returns are additive across assets within a period, so a portfolio return
    is exactly a weighted sum of them.  Log returns are additive across time,
    so a multi-day return is exactly the sum of the daily ones.  No system is
    additive in both directions.
    """
    method = method.upper()
    if method not in ("DISCRETE", "LOG"):
        raise ValueError(f'method: {method} must be in ("LOG","DISCRETE")')
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame: {list(prices.columns)}")

    value_columns = [c for c in prices.columns if c != date_column]
    p = prices[value_columns].to_numpy(dtype=float)
    ratio = p[1:, :] / p[:-1, :]

    out = ratio - 1.0 if method == "DISCRETE" else np.log(ratio)

    # Built in one pass: inserting a hundred columns one at a time fragments
    # the frame and pandas warns about it.
    return pd.concat(
        [pd.DataFrame({date_column: prices[date_column].to_numpy()[1:]}),
         pd.DataFrame(out, columns=value_columns)],
        axis=1)
