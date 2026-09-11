"""
Shared helpers for the model zoo.

UNEVEN SPACING (2020 gap): ACS1 has no 2020 vintage, so a city's observed
years are e.g. [...,2018,2019,2021,2022,...]. Two different strategies are
used depending on the model family:

  - OLS-in-year models (linear trend, exponential, modified exponential) use
    the actual calendar `year` as the regressor, so the 2019->2021 gap needs
    no special handling: the fitted line/curve is a function of calendar
    time and is evaluated at whatever future years are requested.

  - State-space models (ETS, ARIMA) are given a full annual index from
    min(year) to max(year) with 2020 inserted as NaN. Both statsmodels
    ETSModel and the state-space ARIMA implementation treat missing
    observations by advancing the state through the missing period without
    an update, which is the state-space-correct way to handle a real
    calendar gap (as opposed to silently compressing 2019->2021 into a
    single "period", which would make that one step's implied growth rate
    look ~2x too large and bias the trend/AR estimates).

All six models are fit in log(population) except (a) naive and (b) linear
trend, which the spec keeps on the raw population scale.
"""

import numpy as np
import pandas as pd

MIN_OBS = {
    "naive": 1,
    "linear_trend": 2,
    "exponential": 2,
    "modified_exponential": 3,
    "ets_damped": 5,
    "arima_111": 6,
}


def annual_index_with_gaps(years, values):
    """Build a pandas Series on a full annual PeriodIndex [min(years),
    max(years)], inserting NaN for any calendar year missing from `years`
    (i.e. 2020). Used by the state-space models (e, f)."""
    years = np.asarray(years, dtype=int)
    values = np.asarray(values, dtype=float)
    idx = pd.period_range(start=str(years.min()), end=str(years.max()), freq="Y-DEC")
    s = pd.Series(np.nan, index=idx)
    s.loc[pd.PeriodIndex([str(y) for y in years], freq="Y-DEC")] = values
    return s


def future_years(last_year, h):
    return np.arange(last_year + 1, last_year + h + 1)
