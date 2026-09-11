"""(d) Modified exponential with a FIXED ceiling (not estimated -- an
unidentified saturation ceiling is exactly what the n~15-19 constraint rules
out for logistic/Gompertz). Model: log(ceiling - p) = a + b*year, fit by OLS
on the transformed series; ceiling is a parameter, not a free parameter of
the fit.

Default ceiling = DEFAULT_CEILING_MULTIPLIER * max(observed population) when
the caller doesn't supply one. This is an explicit, documented assumption
(not a literature-derived value) -- flagged here and in RESULTS.md.
"""

import numpy as np

from models.common import future_years

DEFAULT_CEILING_MULTIPLIER = 3.0


def fit(series, ceiling=None):
    years, values = series
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    if ceiling is None:
        ceiling = DEFAULT_CEILING_MULTIPLIER * values.max()

    gap = ceiling - values
    if np.any(gap <= 0):
        # Observed population already at/above the assumed ceiling -- widen it
        # rather than take log of a non-positive number.
        ceiling = 1.5 * values.max() + ceiling
        gap = ceiling - values

    b, a = np.polyfit(years, np.log(gap), 1)
    return {"a": float(a), "b": float(b), "ceiling": float(ceiling), "last_year": int(years[-1])}


def predict(fitted, h):
    ys = future_years(fitted["last_year"], h)
    return fitted["ceiling"] - np.exp(fitted["a"] + fitted["b"] * ys)
