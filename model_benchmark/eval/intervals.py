"""
Phase 4: empirical prediction intervals (Rayer, Smith & Tayman 2009 style)
vs. the model-internal intervals ARIMA and ETS produce on their own.

SCOPE / LEAKAGE CAVEAT (be upfront about this): the empirical intervals
below are built from the pooled Phase-2 backtest error sample and then
evaluated for coverage against that SAME sample. That is an in-sample
calibration check, not a genuine held-out prospective test -- with only
~2-4 backtest origins per city there isn't enough data left to also hold
out a separate coverage-test fold without the percentile estimates
themselves becoming too noisy to trust. Treat the coverage numbers here as
"is the shape of these intervals roughly right," not as proof a NEW,
future forecast would hit 90% coverage. This is exactly the kind of
qualification RESULTS.md is asked to be explicit about.

Empirical intervals use the RATIO actual/predicted (not the raw signed
error) so interval width scales proportionally with a city's population
level, which is what lets one pooled or per-decile ratio distribution
apply sensibly across cities of very different sizes.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from eval.backtest import MIN_TRAIN_SIZE
from models.common import MIN_OBS

ALPHA = 0.10  # nominal 90% interval
LOWER_Q, UPPER_Q = ALPHA / 2, 1 - ALPHA / 2


def _add_ratio(df):
    df = df.copy()
    df["ratio"] = df["actual"] / df["predicted"]
    return df


def empirical_pi_pooled(backtest_df, model="arima_111"):
    df = _add_ratio(backtest_df[backtest_df["model"] == model])
    by_h = df.groupby("horizon")["ratio"].quantile([LOWER_Q, UPPER_Q]).unstack()
    by_h.columns = ["ratio_lo", "ratio_hi"]
    return by_h.reset_index()


def assign_population_decile(panel):
    """Decile by each city's mean population across the whole panel (not
    just its latest value, so a city's decile doesn't drift year to year)."""
    mean_pop = panel.groupby("place_id")["population"].mean()
    decile = pd.qcut(mean_pop, 10, labels=False, duplicates="drop") + 1
    return decile.rename("pop_decile")


def empirical_pi_by_decile(backtest_df, panel, model="arima_111"):
    decile = assign_population_decile(panel)
    df = _add_ratio(backtest_df[backtest_df["model"] == model]).merge(
        decile, left_on="place_id", right_index=True
    )
    by_cell = (
        df.groupby(["pop_decile", "horizon"])["ratio"]
        .quantile([LOWER_Q, UPPER_Q])
        .unstack()
    )
    by_cell.columns = ["ratio_lo", "ratio_hi"]
    return by_cell.reset_index(), df


def mape_by_decile(backtest_df, panel, model="arima_111"):
    """The literature (Rayer, Smith & Tayman) says error scales inversely
    with area size -- check that against our own backtest before trusting
    the decile-cut intervals built on top of it."""
    decile = assign_population_decile(panel)
    df = backtest_df[backtest_df["model"] == model].merge(
        decile, left_on="place_id", right_index=True
    )
    df["ape"] = (df["predicted"] - df["actual"]).abs() / df["actual"]
    return df.groupby("pop_decile")["ape"].agg(["mean", "median", "size"]).reset_index()


def _model_based_one_city(years, values, place_id, city, horizons=range(1, 6)):
    rows = []
    n = len(years)
    year_to_value = dict(zip(years.tolist(), values.tolist()))

    for t in range(MIN_TRAIN_SIZE - 1, n - 1):
        train_years, train_values = years[: t + 1], values[: t + 1]
        origin_year = int(train_years[-1])
        log_v = np.log(train_values.astype(float))

        # ARIMA analytic interval (log scale -> exponentiate bounds)
        if len(train_values) >= MIN_OBS["arima_111"]:
            try:
                res = ARIMA(pd.Series(log_v), order=(1, 1, 1)).fit()
                fc = res.get_forecast(max(horizons))
                ci = fc.conf_int(alpha=ALPHA)
                mean = fc.predicted_mean
                for h in horizons:
                    target_year = origin_year + h
                    if target_year not in year_to_value:
                        continue
                    lo, hi = np.exp(ci.iloc[h - 1, 0]), np.exp(ci.iloc[h - 1, 1])
                    rows.append({
                        "place_id": place_id, "city": city, "approach": "arima_analytic",
                        "origin_year": origin_year, "horizon": h, "target_year": target_year,
                        "actual": year_to_value[target_year],
                        "predicted": float(np.exp(mean.iloc[h - 1])),
                        "lower": float(lo), "upper": float(hi),
                    })
            except Exception:
                pass

        # ETS simulated interval (same gap-handling caveat as models/ets.py:
        # fit sequentially on observed values, 2020 simply omitted)
        if len(train_values) >= MIN_OBS["ets_damped"]:
            try:
                res = ETSModel(
                    pd.Series(log_v), error="add", trend="add", damped_trend=True, seasonal=None
                ).fit(disp=False)
                sims = np.asarray(res.simulate(nsimulations=max(horizons), repetitions=500, anchor="end"))
                lo_arr = np.exp(np.percentile(sims, LOWER_Q * 100, axis=1))
                hi_arr = np.exp(np.percentile(sims, UPPER_Q * 100, axis=1))
                mean_arr = np.exp(sims.mean(axis=1))
                for h in horizons:
                    target_year = origin_year + h
                    if target_year not in year_to_value:
                        continue
                    rows.append({
                        "place_id": place_id, "city": city, "approach": "ets_simulated",
                        "origin_year": origin_year, "horizon": h, "target_year": target_year,
                        "actual": year_to_value[target_year],
                        "predicted": float(mean_arr[h - 1]),
                        "lower": float(lo_arr[h - 1]), "upper": float(hi_arr[h - 1]),
                    })
            except Exception:
                pass

    return rows


def model_based_intervals(panel, horizons=range(1, 6)):
    all_rows = []
    for pid, g in panel.groupby("place_id"):
        g = g.sort_values("year")
        years, values = g["year"].to_numpy(), g["population"].to_numpy()
        city = g["city"].iloc[0]
        all_rows.extend(_model_based_one_city(years, values, pid, city, horizons))
    return pd.DataFrame(all_rows)


def empirical_intervals_for_backtest(backtest_df, pooled_pi, model="arima_111"):
    """Attach [lower, upper] to every backtest row for `model` using the
    POOLED (horizon-only) empirical ratio percentiles."""
    df = backtest_df[backtest_df["model"] == model].merge(pooled_pi, on="horizon")
    df["lower"] = df["predicted"] * df["ratio_lo"]
    df["upper"] = df["predicted"] * df["ratio_hi"]
    return df


def empirical_intervals_by_decile_for_backtest(backtest_df, panel, decile_pi, model="arima_111"):
    decile = assign_population_decile(panel)
    df = backtest_df[backtest_df["model"] == model].merge(
        decile, left_on="place_id", right_index=True
    ).merge(decile_pi, on=["pop_decile", "horizon"])
    df["lower"] = df["predicted"] * df["ratio_lo"]
    df["upper"] = df["predicted"] * df["ratio_hi"]
    return df


def coverage_by_horizon(interval_df, label):
    df = interval_df.copy()
    df["covered"] = (df["actual"] >= df["lower"]) & (df["actual"] <= df["upper"])
    out = df.groupby("horizon")["covered"].agg(["mean", "size"]).reset_index()
    out.columns = ["horizon", "coverage", "n"]
    out.insert(0, "approach", label)
    return out
