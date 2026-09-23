import os
import sys

if __package__ in (None, ""):  # direct run: put the source root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import arima, ets, exponential, linear_trend, modified_exponential, naive
from models.common import MIN_OBS

MODELS = {
    "naive": naive,
    "linear_trend": linear_trend,
    "exponential": exponential,
    "modified_exponential": modified_exponential,
    "ets_damped": ets,
    "arima_111": arima,
}


def fit_predict(name, series, h):
    years, values = series
    if len(values) < MIN_OBS[name]:
        return None
    try:
        fitted = MODELS[name].fit(series)
        return MODELS[name].predict(fitted, h)
    except Exception:
        return None
