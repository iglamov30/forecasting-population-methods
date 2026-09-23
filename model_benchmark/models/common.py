import numpy as np
import pandas as pd

# Uneven spacing (no 2020 ACS1 vintage): OLS-in-year models regress on the
# actual calendar year, so the gap needs no handling. State-space models
# (ETS, ARIMA) get a full annual index with 2020 as NaN via
# annual_index_with_gaps, so the state advances through the missing year
# instead of compressing 2019->2021 into one step.
MIN_OBS = {
    "naive": 1,
    "linear_trend": 2,
    "exponential": 2,
    "modified_exponential": 3,
    "ets_damped": 5,
    "arima_111": 6,
}


def annual_index_with_gaps(years, values):
    years = np.asarray(years, dtype=int)
    values = np.asarray(values, dtype=float)
    idx = pd.period_range(start=str(years.min()), end=str(years.max()), freq="Y-DEC")
    s = pd.Series(np.nan, index=idx)
    s.loc[pd.PeriodIndex([str(y) for y in years], freq="Y-DEC")] = values
    return s


def future_years(last_year, h):
    return np.arange(last_year + 1, last_year + h + 1)
