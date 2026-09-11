"""
Phase 5: pooled / global AR model on annualized log growth, with
per-city shrinkage toward the pooled coefficient.

NO-LEAKAGE ENFORCEMENT: the outer loop in `run_pooled_backtest` is over a
shared calendar ORIGIN_YEAR (not per-city index, unlike eval/backtest.py --
pooling across cities only makes sense on a common time axis). At each
origin_year, `_growth_panel_through` rebuilds the pooled regression, every
city's local regression, and every city's own last growth observations
using ONLY g[i,t] rows whose target year t is <= origin_year. Nothing at or
after origin_year+1 enters any estimation step. Year fixed effects are
estimated only from data <= origin_year and are then set to 0 when
forecasting forward (see `_forecast_city`), since a future year's shock
(the next 2008 or 2020) is by definition unknown at forecast time.

GROWTH DEFINITION: g[i,t] = (log p[i,t] - log p[i,t-1]) / (year[t] -
year[t-1]) -- dividing by the elapsed calendar gap annualizes the one
2019->2021 transition per city so it is comparable to every other,
single-year transition, instead of implicitly doubling that one step's
apparent growth.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

LAM_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
MIN_G_FOR_LOCAL_AR1 = 4
MIN_G_FOR_LOCAL_AR2 = 5


def build_growth_panel(panel):
    """Returns [place_id, city, year, g] with g = annualized log growth
    ending at `year`, i.e. g[i,t] uses only p[i,t] and p[i,t-1]."""
    rows = []
    for pid, g in panel.groupby("place_id"):
        g = g.sort_values("year")
        years = g["year"].to_numpy()
        logp = np.log(g["population"].to_numpy())
        city = g["city"].iloc[0]
        for k in range(1, len(years)):
            dt = years[k] - years[k - 1]
            growth = (logp[k] - logp[k - 1]) / dt
            rows.append({"place_id": pid, "city": city, "year": int(years[k]), "g": growth})
    return pd.DataFrame(rows)


def _add_lags(growth_panel, n_lags=2):
    df = growth_panel.sort_values(["place_id", "year"]).copy()
    for lag in range(1, n_lags + 1):
        df[f"g_lag{lag}"] = df.groupby("place_id")["g"].shift(lag)
    return df


def fit_pooled_ar(g_df, order=1):
    """Pooled AR(order) with city + year fixed effects. Returns (rho, dict
    of city fixed effects, dict of year fixed effects, base intercept)."""
    lag_cols = [f"g_lag{k}" for k in range(1, order + 1)]
    sub = g_df.dropna(subset=lag_cols + ["g"]).copy()

    city_dum = pd.get_dummies(sub["place_id"], prefix="city", drop_first=True)
    year_dum = pd.get_dummies(sub["year"], prefix="year", drop_first=True)
    X = pd.concat([sub[lag_cols], city_dum, year_dum], axis=1)
    X = sm.add_constant(X)
    X = X.astype(float)
    y = sub["g"].astype(float)

    model = sm.OLS(y, X).fit()

    rho = {k: model.params[col] for k, col in zip(range(1, order + 1), lag_cols)}
    base_const = model.params["const"]
    city_fe = {c: model.params.get(f"city_{c}", 0.0) for c in sub["place_id"].unique()}
    year_fe = {yv: model.params.get(f"year_{yv}", 0.0) for yv in sub["year"].unique()}
    return {"rho": rho, "const": base_const, "city_fe": city_fe, "year_fe": year_fe, "order": order,
            "n_obs": int(model.nobs)}


def fit_local_ar(city_g, order=1):
    """Per-city AR(order), no pooling. Returns None if too little data."""
    lag_cols = [f"g_lag{k}" for k in range(1, order + 1)]
    sub = city_g.dropna(subset=lag_cols + ["g"])
    min_needed = MIN_G_FOR_LOCAL_AR1 if order == 1 else MIN_G_FOR_LOCAL_AR2
    if len(sub) < min_needed:
        return None
    X = sm.add_constant(sub[lag_cols].astype(float))
    y = sub["g"].astype(float)
    model = sm.OLS(y, X).fit()
    rho = {k: model.params[col] for k, col in zip(range(1, order + 1), lag_cols)}
    return {"rho": rho, "const": model.params["const"], "order": order, "n_obs": int(model.nobs)}


def shrink_rho(pooled_rho, local_rho, lam, order):
    """theta_city = lam*pooled + (1-lam)*local, component-wise. Falls back
    to pure pooled (lam=1 behavior) if no local estimate exists for a
    city -- there isn't enough history to do anything else."""
    if local_rho is None:
        return dict(pooled_rho)
    return {k: lam * pooled_rho[k] + (1 - lam) * local_rho[k] for k in pooled_rho}


def _forecast_city(last_log_pop, last_year, last_g_values, rho, const, order, h_max):
    """Recursively forecast g forward using the (possibly shrunk) AR
    coefficients, with year FE fixed at 0 (see module docstring), then
    integrate g back to the population level."""
    g_hist = list(last_g_values)  # [..., g_{T-1}, g_T], most recent last
    g_future = []
    for _ in range(h_max):
        g_new = const
        for lag in range(1, order + 1):
            g_new += rho[lag] * g_hist[-lag]
        g_future.append(g_new)
        g_hist.append(g_new)

    log_pop_path = last_log_pop + np.cumsum(g_future)
    return np.exp(log_pop_path)


def run_pooled_backtest(panel, order=1, lam_grid=LAM_GRID, origin_years=None, horizons=range(1, 6)):
    """Expanding-window backtest of the pooled/shrunk AR(order) model.
    Returns a long DataFrame [place_id, city, origin_year, horizon,
    target_year, actual, lam, predicted]."""
    g_all = _add_lags(build_growth_panel(panel), n_lags=order)
    year_to_value = {
        pid: dict(zip(g["year"], g["population"]))
        for pid, g in panel.groupby("place_id")
    }
    log_pop_lookup = {
        pid: {y: np.log(v) for y, v in yv.items()} for pid, yv in year_to_value.items()
    }

    if origin_years is None:
        all_years = sorted(g_all["year"].unique())
        min_needed = 8 if order == 1 else 9
        origin_years = all_years[min_needed:-1]

    out_rows = []
    for origin_year in origin_years:
        train = g_all[g_all["year"] <= origin_year]
        pooled = fit_pooled_ar(train, order=order)

        local_by_city = {
            pid: fit_local_ar(city_g, order=order)
            for pid, city_g in train.groupby("place_id")
        }

        for pid, city_g in train.groupby("place_id"):
            city_g = city_g.sort_values("year")
            if city_g["year"].max() != city_g["year"].max():  # no-op guard
                continue
            # last `order` observed g values strictly through origin_year
            recent = city_g.dropna(subset=["g"])
            if len(recent) < order:
                continue
            last_g_values = recent["g"].to_numpy()[-order:]

            lp = log_pop_lookup[pid]
            if origin_year not in lp:
                continue
            last_log_pop = lp[origin_year]
            city = city_g["city"].iloc[0]

            local_rho = local_by_city[pid]["rho"] if local_by_city[pid] else None
            const = pooled["const"] + pooled["city_fe"].get(pid, 0.0)

            for lam in lam_grid:
                rho_city = shrink_rho(pooled["rho"], local_rho, lam, order)
                preds = _forecast_city(
                    last_log_pop, origin_year, last_g_values, rho_city, const, order, max(horizons)
                )
                for h in horizons:
                    target_year = origin_year + h
                    actual = year_to_value[pid].get(target_year)
                    if actual is None:
                        continue
                    out_rows.append({
                        "place_id": pid, "city": city, "origin_year": origin_year,
                        "horizon": h, "target_year": target_year, "actual": actual,
                        "lam": lam, "predicted": float(preds[h - 1]),
                        "has_local": local_rho is not None,
                    })

    return pd.DataFrame(out_rows), pooled


def score_by_lam(pooled_bt_df):
    df = pooled_bt_df.copy()
    df["abs_error"] = (df["predicted"] - df["actual"]).abs()
    df["ape"] = df["abs_error"] / df["actual"]
    return (
        df.groupby("lam")
        .agg(mae=("abs_error", "mean"), mape=("ape", "mean"), n=("abs_error", "size"))
        .reset_index()
    )


def score_by_lam_and_horizon(pooled_bt_df):
    df = pooled_bt_df.copy()
    df["abs_error"] = (df["predicted"] - df["actual"]).abs()
    return (
        df.groupby(["lam", "horizon"])["abs_error"]
        .mean()
        .reset_index()
        .rename(columns={"abs_error": "mae"})
    )


def nickell_bias_estimate(pooled_rho1, avg_T):
    """Leading-order Nickell (1981) bias for a dynamic panel AR(1) with
    fixed effects: bias ~= -(1+rho)/T. This is the well-known first-order
    approximation, not a full small-sample correction (e.g. Kiviet 1995 or
    Arellano-Bond GMM) -- those are out of scope here; this only reports
    the plausible MAGNITUDE of the downward bias in rho_pooled."""
    return -(1 + pooled_rho1) / avg_T
