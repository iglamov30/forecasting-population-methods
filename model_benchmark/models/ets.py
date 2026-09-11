"""(e) ETS with an additive damped trend (damped Holt), fit in log(population).

GAP-HANDLING NOTE (deviates from arima.py -- read this before trusting ETS
near 2020): statsmodels' state-space ETSModel was tried first with the same
NaN-at-2020 approach used for ARIMA (see models/common.annual_index_with_gaps
and the docstring in arima.py). It does NOT work: empirically, once the
recursion hits the missing observation the level/trend state becomes NaN and
never recovers for the rest of the series (verified directly -- fitted
values and forecasts are all NaN from 2020 onward, regardless of
damped_trend True/False). This is a real limitation of this statsmodels
version's ETSModel, not a design choice.

Fallback used instead: fit on the observed values in sequential order with
2020 simply omitted (no NaN row), i.e. treat 2019->2021 as if it were a
single one-year step. This is a real approximation: that one step's
per-period growth is the true 2-year compounded growth, which the damped
trend then treats as a same-size one-year change, so the local trend
estimate right around the gap is biased toward overstating that single
step's annual rate. With one gap year in a ~15-19 year series this affects
one transition out of ~15+; treat ETS-based results as slightly less
reliable specifically for cities where the 2019-2021 transition is unusual,
and prefer ARIMA's forecast when the two disagree sharply.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


def fit(series):
    years, values = series
    log_series = pd.Series(np.log(np.asarray(values, dtype=float)))
    model = ETSModel(log_series, error="add", trend="add", damped_trend=True, seasonal=None)
    results = model.fit(disp=False)
    return {"results": results}


def predict(fitted, h):
    log_fc = fitted["results"].forecast(h)
    return np.exp(np.asarray(log_fc))
