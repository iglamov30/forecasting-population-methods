"""(c) Exponential / geometric growth: OLS of log(population) on calendar
year -> constant annualized growth rate. Regressing on actual year (not a
sequential index) handles the 2020 gap for free."""

import numpy as np

from models.common import future_years


def fit(series):
    years, values = series
    years = np.asarray(years, dtype=float)
    log_v = np.log(np.asarray(values, dtype=float))
    b, a = np.polyfit(years, log_v, 1)  # log(value) ~ a + b*year
    return {"a": float(a), "b": float(b), "last_year": int(years[-1])}


def predict(fitted, h):
    ys = future_years(fitted["last_year"], h)
    return np.exp(fitted["a"] + fitted["b"] * ys)
