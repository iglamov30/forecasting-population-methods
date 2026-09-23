import os
import sys
import numpy as np
import pandas as pd

if __package__ in (None, ""):  # direct run: put the source root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import MIN_OBS

HORIZONS = range(1, 6)
# 8 keeps the first origin clear of ARIMA(1,1,1)'s 6-observation floor. Cities
# with fewer than 9 observations never produce an origin and are absent from
# every backtest-derived table (see RESULTS.md).
MIN_TRAIN_SIZE = 8


def _run_one_city(years, values, place_id, city, models, horizons):
    rows = []
    n = len(years)
    year_to_value = dict(zip(years.tolist(), values.tolist()))

    for t in range(MIN_TRAIN_SIZE - 1, n - 1):
        # NO LEAKAGE: models only ever see data up to and including the origin
        # year. This is the only place training data is sliced; later stages
        # reuse this output rather than re-slicing, so the guarantee carries.
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
