import os
import sys
import numpy as np

if __package__ in (None, ""):  # direct run: put the source root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
