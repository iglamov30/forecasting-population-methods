import os
import sys
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

if __package__ in (None, ""):  # direct run: put the source root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import annual_index_with_gaps


def fit(series):
    years, values = series
    log_series = np.log(annual_index_with_gaps(years, values))
    model = ARIMA(log_series, order=(1, 1, 1))
    results = model.fit()
    return {"results": results}


def predict(fitted, h):
    log_fc = fitted["results"].forecast(h)
    return np.exp(np.asarray(log_fc))
