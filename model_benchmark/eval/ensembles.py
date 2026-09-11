"""
Phase 3: ensemble/selection rules evaluated on the same backtest windows as
Phase 2.

NO-LEAKAGE ENFORCEMENT: the two ensembles (simple mean, trimmed mean) only
combine forecasts that were already produced at a given origin in
eval/backtest.py -- they add no new information, so they inherit that
module's leakage guarantee for free.

`select_best_model_per_origin` is the one place in this phase that
estimates something from the backtest itself (which model is "best"), so it
needs its own guarantee: at each origin, the selected model is whichever
had the lowest cumulative MAE using ONLY that city's STRICTLY EARLIER
origins (see the `cum_sum`/`cum_n` update order below -- the current
origin's own errors are folded into the running totals only AFTER the
selection for that origin has been made). A city's very first backtest
origin therefore has no prior history to select from and is dropped from
all three approaches, so the comparison in Phase 3 is apples-to-apples.
"""

import numpy as np
import pandas as pd

from eval.backtest import add_error_columns


def _pivot_wide(results_df):
    return results_df.pivot_table(
        index=["place_id", "city", "origin_year", "horizon", "target_year", "actual"],
        columns="model",
        values="predicted",
    )


def simple_mean_ensemble(results_df):
    wide = _pivot_wide(results_df)
    pred = wide.mean(axis=1, skipna=True)
    return pred.rename("predicted").reset_index()


def _trimmed_mean_row(row):
    vals = row.dropna().to_numpy()
    if len(vals) <= 2:
        return vals.mean() if len(vals) else np.nan
    vals = np.sort(vals)
    return vals[1:-1].mean()


def trimmed_mean_ensemble(results_df):
    wide = _pivot_wide(results_df)
    pred = wide.apply(_trimmed_mean_row, axis=1)
    return pred.rename("predicted").reset_index()


def select_best_model_per_origin(results_df):
    """Leakage-free 'production rule': at each origin, use whichever model
    had the lowest cumulative MAE over that city's strictly earlier
    origins. Returns rows only for origins where a prior-origin history
    exists (i.e. every city's first backtest origin is dropped)."""
    df = add_error_columns(results_df)
    out_rows = []

    for pid, g in df.groupby("place_id"):
        origins = sorted(g["origin_year"].unique())
        cum_sum, cum_n = {}, {}

        for origin in origins:
            if cum_n:
                avg = {m: cum_sum[m] / cum_n[m] for m in cum_n if cum_n[m] > 0}
                if avg:
                    best_model = min(avg, key=avg.get)
                    sel = g[(g["origin_year"] == origin) & (g["model"] == best_model)]
                    for _, r in sel.iterrows():
                        out_rows.append({
                            "place_id": pid,
                            "city": r["city"],
                            "origin_year": origin,
                            "horizon": r["horizon"],
                            "target_year": r["target_year"],
                            "actual": r["actual"],
                            "predicted": r["predicted"],
                            "selected_model": best_model,
                        })

            cur = g[g["origin_year"] == origin]
            for m, sub in cur.groupby("model"):
                cum_sum[m] = cum_sum.get(m, 0.0) + sub["abs_error"].sum()
                cum_n[m] = cum_n.get(m, 0) + sub["abs_error"].count()

    return pd.DataFrame(out_rows)


def score_approach(pred_df, label):
    df = pred_df.copy()
    df["error"] = df["predicted"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["ape"] = df["abs_error"] / df["actual"]
    table = (
        df.groupby("horizon")
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda e: np.sqrt(np.mean(e ** 2))),
            mape=("ape", "mean"),
            median_ape=("ape", "median"),
            bias=("error", "mean"),
            n=("error", "size"),
        )
        .reset_index()
    )
    table.insert(0, "approach", label)
    return table


def compare_approaches(results_df):
    """Restricts all three approaches to the (city, origin) pairs where a
    best-model selection is possible, then scores each on that common set."""
    best = select_best_model_per_origin(results_df)
    common_keys = best[["place_id", "origin_year"]].drop_duplicates()

    restricted = results_df.merge(common_keys, on=["place_id", "origin_year"])
    mean_ens = simple_mean_ensemble(restricted)
    trimmed_ens = trimmed_mean_ensemble(restricted)

    # ARIMA-only baseline (the single fixed model that wins Phase 2) on the
    # same restricted rows, for reference alongside the per-city selector.
    arima_only = restricted[restricted["model"] == "arima_111"][
        ["place_id", "city", "origin_year", "horizon", "target_year", "actual", "predicted"]
    ]

    tables = [
        score_approach(mean_ens, "simple_mean_ensemble"),
        score_approach(trimmed_ens, "trimmed_mean_ensemble"),
        score_approach(best, "best_model_by_backtest_mae"),
        score_approach(arima_only, "arima_111_always"),
    ]
    return pd.concat(tables, ignore_index=True).sort_values(["horizon", "mae"])
