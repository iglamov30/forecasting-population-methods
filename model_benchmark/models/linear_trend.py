import os
import sys
import numpy as np

if __package__ in (None, ""):  # direct run: put the source root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
