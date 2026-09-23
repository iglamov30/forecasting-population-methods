import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Unlike arima.py, ETS is fit on the observed values in sequential order with
# 2020 simply omitted: this statsmodels version's ETSModel turns the state
# permanently NaN after a missing observation. That treats 2019->2021 as a
# one-year step, so ETS is least reliable around that transition. See the
# "What is NOT established" section of RESULTS.md.


def fit(series):
    years, values = series
    log_series = pd.Series(np.log(np.asarray(values, dtype=float)))
    model = ETSModel(log_series, error="add", trend="add", damped_trend=True, seasonal=None)
    results = model.fit(disp=False)
    return {"results": results}


def predict(fitted, h):
    log_fc = fitted["results"].forecast(h)
    return np.exp(np.asarray(log_fc))
