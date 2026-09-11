"""(a) Naive: last observed value carried forward. The benchmark every other
model must beat."""

import numpy as np


def fit(series):
    years, values = series
    return {"last_value": float(values[-1]), "last_year": int(years[-1])}


def predict(fitted, h):
    return np.full(h, fitted["last_value"])
