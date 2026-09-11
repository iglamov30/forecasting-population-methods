"""(b) Linear trend: OLS of population on calendar year (raw scale, not
log). Regressing on actual year rather than a sequential index handles the
2020 gap for free."""

import numpy as np

from models.common import future_years


def fit(series):
    years, values = series
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    b, a = np.polyfit(years, values, 1)  # values ~ a + b*year
    return {"a": float(a), "b": float(b), "last_year": int(years[-1])}


def predict(fitted, h):
    ys = future_years(fitted["last_year"], h)
    return fitted["a"] + fitted["b"] * ys
