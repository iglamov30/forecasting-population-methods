import os
import sys
import numpy as np

if __package__ in (None, ""):  # direct run: put the source root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import future_years

# Explicit assumption, not a literature-derived value (flagged in RESULTS.md):
# the saturation ceiling is fixed, never estimated from the data.
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
