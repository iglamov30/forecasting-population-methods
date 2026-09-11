"""
Expanding-window rolling-origin backtest.

NO-LEAKAGE ENFORCEMENT (the CRITICAL rule from the spec): in
`_run_one_city`, every model is fit on `years[:t+1], values[:t+1]` where `t`
is the origin index -- i.e. only data up to and including the origin year.
Nothing at or after `years[t+1]` is ever passed into `fit()`. This is the
single point in the whole codebase where training data is sliced, and every
later phase (ensembles, pooled/shrinkage model) reuses this same backtest
output rather than re-slicing data itself, so the guarantee holds
everywhere derived from it.

MIN_TRAIN_SIZE = 8: the smallest model (ARIMA(1,1,1)) needs at least 6
observations to have any degrees of freedom; 8 gives a small margin above
that so the first backtest origin isn't right at the failure boundary. A
consequence: cities with fewer than 9 total observations (e.g. Honolulu
CDP, which stops appearing in ACS1 after 2009) never generate a backtest
origin and are silently absent from every backtest-derived table. That's a
real limitation of the population, not a bug -- it's noted in RESULTS.md.

2020 GAP HANDLING: horizon h at origin year Y always means calendar year
Y+h (not "the h-th available future observation"). If Y+h is 2020, or is
beyond the last year present for that city, that (city, model, origin,
horizon) row is simply dropped from scoring. This means horizons that
happen to land on 2020 for a given origin contribute fewer scored
observations than other horizons -- an unavoidable consequence of an
irregularly-spaced series, flagged again in the score table's `n` column.
"""

import numpy as np
import pandas as pd

from models.common import MIN_OBS

HORIZONS = range(1, 6)
MIN_TRAIN_SIZE = 8


def _run_one_city(years, values, place_id, city, models, horizons):
    rows = []
    n = len(years)
    year_to_value = dict(zip(years.tolist(), values.tolist()))

    for t in range(MIN_TRAIN_SIZE - 1, n - 1):
        train_years = years[: t + 1]
        train_values = values[: t + 1]
        origin_year = int(train_years[-1])

        for name, mod in models.items():
            if len(train_values) < MIN_OBS[name]:
                continue
            try:
                fitted = mod.fit((train_years, train_values))
                preds = mod.predict(fitted, max(horizons))
            except Exception:
                continue

            for h in horizons:
                target_year = origin_year + h
                if target_year not in year_to_value:
                    continue  # 2020 gap or beyond this city's observed range
                rows.append({
                    "place_id": place_id,
                    "city": city,
                    "model": name,
                    "origin_year": origin_year,
                    "horizon": h,
                    "target_year": target_year,
                    "actual": year_to_value[target_year],
                    "predicted": float(preds[h - 1]),
                })
    return rows


def run_backtest(panel, models, horizons=HORIZONS):
    """panel: DataFrame [year, place_id, city, population]."""
    all_rows = []
    for pid, g in panel.groupby("place_id"):
        g = g.sort_values("year")
        years = g["year"].to_numpy()
        values = g["population"].to_numpy()
        city = g["city"].iloc[0]
        all_rows.extend(_run_one_city(years, values, pid, city, models, horizons))
    return pd.DataFrame(all_rows)


def add_error_columns(results_df):
    df = results_df.copy()
    df["error"] = df["predicted"] - df["actual"]  # + = overprediction
    df["abs_error"] = df["error"].abs()
    df["ape"] = df["abs_error"] / df["actual"]
    return df


def score_by_model_horizon(results_df):
    df = add_error_columns(results_df)
    table = (
        df.groupby(["model", "horizon"])
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda e: np.sqrt(np.mean(e ** 2))),
            mape=("ape", "mean"),
            median_ape=("ape", "median"),
            bias=("error", "mean"),
            n=("error", "size"),
        )
        .reset_index()
        .sort_values(["horizon", "mae"])
    )
    return table
