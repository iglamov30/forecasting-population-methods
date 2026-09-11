"""(f) ARIMA(1,1,1), fit in log(population).

Uses statsmodels' state-space ARIMA implementation on a full annual
PeriodIndex with 2020 set to NaN -- same rationale as ets.py: the Kalman
filter skips the update for the missing observation but still advances the
state through the elapsed calendar year, which correctly reflects a 2-year
gap rather than a 1-year step.
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

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
