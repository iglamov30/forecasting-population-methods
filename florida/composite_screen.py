import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sources.cities import FL_CITIES, short_name
from sources.edr_population import build_panel, city_series

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "charlotte"))
from charlotte_5y import fit, forecast_horizon, rolling_backtest, MODELS, HORIZON

START_YEAR = 2010
ALPHA = 0.10
EPS = 1e-6
MIN_OBS = HORIZON + 9
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "composite_scores.csv")

HOTEL_WEIGHTS = {"population": 0.25, "employment": 0.25, "permits": 0.10, "tourism": 0.40}
MULTIFAMILY_WEIGHTS = {"population": 0.30, "employment": 0.30, "permits": 0.35, "tourism": 0.05}


def forecast_metric(yrs, series, label):
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
    current_year = datetime.now().year

    print("Loading population (EDR)...")
    pop_panel = build_panel([START_YEAR, current_year])

    print("Fetching employment (BLS LAUS)...")
    try:
        from sources.bls_employment import build_employment_panel, county_series
        emp_panel = build_employment_panel(FL_CITIES, START_YEAR, current_year)
    except Exception as exc:
        print(f"  employment pull failed ({exc}); employment excluded.")
        emp_panel, county_series = None, None

    print("Fetching permits (Census BPS)...")
    from sources.census_permits import fetch_place_permits, load_permits_manual, permits_series
    permits_panel = None
    try:
        permits_panel = fetch_place_permits(FL_CITIES, START_YEAR, current_year)
    except Exception as exc:
        print(f"  download failed ({exc}); trying data/permits_manual.csv...")
        try:
            permits_panel = load_permits_manual()
        except FileNotFoundError:
            print("  no data/permits_manual.csv; permits excluded.")

    print("Loading tourism (hand-entered)...")
    from sources.tourism import load_tdt_manual, load_enplanements_manual, tourism_series
    tdt_panel = enplane_panel = None
    try:
        tdt_panel = load_tdt_manual()
    except FileNotFoundError:
        print("  no data/tdt_manual.csv; TDT excluded.")
    try:
        enplane_panel = load_enplanements_manual()
    except FileNotFoundError:
        print("  no data/enplanements_manual.csv; enplanements excluded.")

    results = {}
    for pid in FL_CITIES:
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
    return sum(city_results[k]["risk_adj"] * (w / total_w) for k, w in available.items())


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
            "hotel_score": np.nan if hotel is None else hotel,
            "multifamily_score": np.nan if multi is None else multi,
            "pop_growth_5y_pct": cr.get("population", {}).get("pct_growth", np.nan),
        })

    df = pd.DataFrame(rows).sort_values("hotel_score", ascending=False)

    print("\nComposite scores (higher is better)")
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH}")

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
        ax.set_title("Florida composite score by asset class")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return df


if __name__ == "__main__":
    main()
