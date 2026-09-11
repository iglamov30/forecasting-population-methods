"""
Florida 5-city composite investment score for Blackstone-style hotel and
multifamily screening.

Combines four independently-forecasted signals per city:
  - Population growth        (Florida EDR municipal estimates, via edr_population.py)
  - Employment growth        (BLS LAUS, via bls_employment.py)
  - Housing permit growth    (Census BPS, via census_permits.py)
  - Tourism demand growth    (TDT collections + airport enplanements,
                               via tourism_signals.py)

Each signal is forecast independently (ETS vs ARIMA, best model chosen by
backtest MAE) over a 5-year horizon, then risk-adjusted by dividing the
forecast growth rate by that model's own backtest RMSE (as a % of the
latest value) — same logic as forecast_citywide_adj.py, just applied per
metric instead of just to population.

Two separate composite scores are produced because hotels and multifamily
respond to different drivers:
  HOTEL score        = population + employment + tourism (heavier weight)
  MULTIFAMILY score   = population + employment + permits (heavier weight)

Run this after you've populated tdt_manual.csv / enplanements_manual.csv /
permits_manual.csv (see the *_template.csv files and each module's
docstring for where to get that data) — population and employment will
work out of the box against the live Census/BLS APIs if you have keys set.
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fl_cities import FL_CITIES, short_name
from edr_population import build_panel, city_series
from charlotte_5y import fit, forecast_horizon, rolling_backtest, MODELS

HORIZON = 5
ALPHA = 0.10
EPS = 1e-6
MIN_OBS = HORIZON + 9

# Composite weights -- adjust to taste. Each set is renormalized at runtime
# over whatever signals actually have enough data.
HOTEL_WEIGHTS = {"population": 0.25, "employment": 0.25, "permits": 0.10, "tourism": 0.40}
MULTIFAMILY_WEIGHTS = {"population": 0.30, "employment": 0.30, "permits": 0.35, "tourism": 0.05}


def forecast_metric(yrs, series, label):
    """
    Run both models via the existing charlotte_5y backtest/forecast
    machinery, pick the better one by backtest MAE, and return a
    risk-adjusted growth score plus the raw numbers for reporting.
    """
    if len(series) < MIN_OBS:
        return None

    best = None
    for m in MODELS:
        try:
            mae, rmse, preds, lower, upper, start = rolling_backtest(series, m, alpha=ALPHA)
            if len(preds) == 0:
                continue
            if best is None or mae < best["mae"]:
                fitted = fit(series, m)
                mean, fc_lower, fc_upper = forecast_horizon(fitted, m, alpha=ALPHA)
                best = {"model": m, "mae": mae, "rmse": rmse, "mean": mean}
        except Exception:
            continue

    if best is None:
        return None

    last_val = series[-1]
    end_forecast = best["mean"][-1]
    pct_growth = (end_forecast - last_val) / last_val * 100
    rmse_pct = best["rmse"] / last_val * 100
    risk_adj = pct_growth / (rmse_pct + EPS)

    return {
        "label": label,
        "model": best["model"],
        "pct_growth": pct_growth,
        "rmse_pct": rmse_pct,
        "risk_adj": risk_adj,
        "forecast_path": best["mean"],
    }


def collect_signals():
    """Pull/load all four panels. Returns dict of place_id -> {metric: result}."""
    current_year = datetime.now().year
    pull_years = [y for y in range(2010, current_year + 1) if y != 2020]

    print("Loading population panel (Florida EDR municipal estimates)...")
    pop_panel = build_panel(pull_years)

    print("Fetching employment panel (BLS LAUS)...")
    try:
        from bls_employment import build_employment_panel, county_series
        emp_panel = build_employment_panel(FL_CITIES, 2010, current_year)
    except Exception as exc:
        print(f"  employment pull failed ({exc}) — employment will be excluded from scoring.")
        emp_panel, county_series = None, None

    print("Loading permits panel...")
    permits_panel = None
    try:
        from census_permits import try_fetch_cbsa_permits, permits_series
        permits_panel = try_fetch_cbsa_permits(FL_CITIES, 2010, current_year)
    except Exception as exc:
        print(f"  automated permits pull failed ({exc}), trying manual file...")
        try:
            from census_permits import load_permits_manual, permits_series
            permits_panel = load_permits_manual("permits_manual.csv")
        except Exception:
            print("  no permits_manual.csv found — permits excluded from scoring. "
                  "See permits_template.csv.")
            permits_series = None

    print("Loading tourism panels (TDT + enplanements)...")
    from tourism_signals import tourism_series
    tdt_panel = enplane_panel = None
    try:
        from tourism_signals import load_tdt_manual
        tdt_panel = load_tdt_manual("tdt_manual.csv")
    except Exception:
        print("  no tdt_manual.csv found — TDT excluded from scoring. See tdt_template.csv.")
    try:
        from tourism_signals import load_enplanements_manual
        enplane_panel = load_enplanements_manual("enplanements_manual.csv")
    except Exception:
        print("  no enplanements_manual.csv found — enplanements excluded from scoring. "
              "See enplanements_template.csv.")

    results = {}
    for pid, meta in FL_CITIES.items():
        city_results = {}

        yrs, pop = city_series(pop_panel, pid)
        r = forecast_metric(yrs, pop, "population")
        if r:
            city_results["population"] = r

        if emp_panel is not None:
            yrs, emp = county_series(emp_panel, pid)
            r = forecast_metric(yrs, emp, "employment")
            if r:
                city_results["employment"] = r

        if permits_panel is not None:
            yrs, permits = permits_series(permits_panel, pid)
            r = forecast_metric(yrs, permits, "permits")
            if r:
                city_results["permits"] = r

        tourism_parts = []
        if tdt_panel is not None:
            yrs, tdt = tourism_series(tdt_panel, pid, "tdt_collections")
            r = forecast_metric(yrs, tdt, "tdt")
            if r:
                tourism_parts.append(r["risk_adj"])
        if enplane_panel is not None:
            yrs, enp = tourism_series(enplane_panel, pid, "enplanements")
            r = forecast_metric(yrs, enp, "enplanements")
            if r:
                tourism_parts.append(r["risk_adj"])
        if tourism_parts:
            city_results["tourism"] = {"risk_adj": float(np.mean(tourism_parts))}

        results[pid] = city_results

    return results


def composite_score(city_results, weights):
    available = {k: w for k, w in weights.items() if k in city_results}
    if not available:
        return None
    total_w = sum(available.values())
    score = sum(
        city_results[k]["risk_adj"] * (w / total_w) for k, w in available.items()
    )
    return score, list(available.keys())


def main():
    results = collect_signals()

    rows = []
    for pid, meta in FL_CITIES.items():
        cr = results[pid]
        hotel = composite_score(cr, HOTEL_WEIGHTS)
        multi = composite_score(cr, MULTIFAMILY_WEIGHTS)
        rows.append({
            "city": short_name(meta["city"]),
            "place_id": pid,
            "signals_used": ", ".join(sorted(cr.keys())) or "none",
            "hotel_score": hotel[0] if hotel else np.nan,
            "multifamily_score": multi[0] if multi else np.nan,
            "pop_growth_5y_pct": cr.get("population", {}).get("pct_growth", np.nan),
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("FLORIDA 5-CITY COMPOSITE INVESTMENT SCORES")
    print("=" * 78)
    print(df.sort_values("hotel_score", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # Simple side-by-side bar chart comparing the two asset-class scores
    plot_df = df.dropna(subset=["hotel_score", "multifamily_score"], how="all")
    if not plot_df.empty:
        x = np.arange(len(plot_df))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(x - width / 2, plot_df["hotel_score"].fillna(0), width,
               label="Hotel score", color="#2196F3")
        ax.bar(x + width / 2, plot_df["multifamily_score"].fillna(0), width,
               label="Multifamily score", color="#FF9800")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["city"], rotation=15)
        ax.set_ylabel("Risk-adjusted composite score")
        ax.set_title("Florida 5-City Composite Score by Asset Class")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return df


if __name__ == "__main__":
    main()
