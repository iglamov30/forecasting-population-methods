import numpy as np
import pandas as pd
import statsmodels.api as sm

LAM_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
MIN_G_FOR_LOCAL_AR1 = 4
MIN_G_FOR_LOCAL_AR2 = 5


def build_growth_panel(panel):
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
    if local_rho is None:
        return dict(pooled_rho)
    return {k: lam * pooled_rho[k] + (1 - lam) * local_rho[k] for k in pooled_rho}


def _forecast_city(last_log_pop, last_year, last_g_values, rho, const, order, h_max):
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
        # NO LEAKAGE: the pooled regression, every city's local regression and
        # each city's last growth observations are all rebuilt here from rows
        # with year <= origin_year. Year fixed effects are estimated only from
        # that window and zeroed when forecasting forward, since a future year's
        # shock is unknown at forecast time (see _forecast_city).
        train = g_all[g_all["year"] <= origin_year]
        pooled = fit_pooled_ar(train, order=order)

        local_by_city = {
            pid: fit_local_ar(city_g, order=order)
            for pid, city_g in train.groupby("place_id")
        }

        for pid, city_g in train.groupby("place_id"):
            city_g = city_g.sort_values("year")
            # The city must have a growth observation at the origin itself --
            # a stale series (one that stops before origin_year) carries no
            # usable starting point. Currently subsumed by the log_pop_lookup
            # check below, but stated here so the requirement is explicit.
            if city_g["year"].max() != origin_year:
                continue
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
    return -(1 + pooled_rho1) / avg_T
