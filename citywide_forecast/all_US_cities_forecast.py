import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "charlotte"))

from census_acs1 import build_panel, city_series, list_cities
from charlotte_5y import rolling_backtest, fit, forecast_horizon, MODELS, HORIZON

START_YEAR = 2005
MIN_OBS = HORIZON + 9   

ALPHA = 0.10           

def run_city(yrs, pop):
    if len(pop) < MIN_OBS:
        return None
    results = {}
    for m in MODELS:
        try:
            mae, rmse, preds, lower, upper, start = rolling_backtest(pop, m, alpha=ALPHA)
            if len(preds) == 0:
                continue
            fitted = fit(pop, m)
            mean, fc_lower, fc_upper = forecast_horizon(fitted, m, alpha=ALPHA)
            results[m] = {
                "mae": mae, "rmse": rmse,
                "mean": mean, "lower": fc_lower, "upper": fc_upper,
            }
        except Exception:
            pass

    return results if results else None

def run_all(start_year=START_YEAR):
    pull_years = [y for y in range(start_year, datetime.now().year + 1) if y != 2020]

    print(f"Fetching Census ACS1 panel ({pull_years[0]}–{pull_years[-1]}, skipping 2020)...")
    panel = build_panel(pull_years)

    last_year = int(panel["year"].max())
    future_years = np.arange(last_year + 1, last_year + HORIZON + 1)

    cities = list_cities(panel)
    total = len(cities)
    print(f"Running {HORIZON}-year forecasts for {total} cities...\n")

    rows = []
    skipped = 0

    for i, (place_id, meta) in enumerate(cities.iterrows(), 1):
        yrs, pop = city_series(panel, place_id)
        city_results = run_city(yrs, pop)

        if city_results is None:
            skipped += 1
            continue

        best_m = min(city_results, key=lambda m: city_results[m]["mae"])

        record = {
            "place_id":        place_id,
            "city":            meta["city"],
            "n_obs":           len(pop),
            "last_year":       int(yrs[-1]),
            "last_population": int(pop[-1]),
            "best_model":      best_m,
        }

        for m in MODELS:
            if m not in city_results:
                continue
            r = city_results[m]
            record[f"mae_{m}"]  = round(r["mae"])
            record[f"rmse_{m}"] = round(r["rmse"])
            for j, fy in enumerate(future_years):
                record[f"{fy}_{m}"] = int(round(r["mean"][j]))

        bm = city_results[best_m]
        for j, fy in enumerate(future_years):
            record[f"{fy}_forecast"] = int(round(bm["mean"][j]))
            record[f"{fy}_lower"]    = int(round(bm["lower"][j]))
            record[f"{fy}_upper"]    = int(round(bm["upper"][j]))

        end_year = future_years[-1]
        record["pct_change_5y"] = round(
            (record[f"{end_year}_forecast"] - int(pop[-1])) / int(pop[-1]) * 100, 2
        )

        rows.append(record)

        if i % 100 == 0 or i == total:
            print(f"  {i}/{total} processed  ({skipped} skipped — too few observations)")

    df = pd.DataFrame(rows).sort_values("last_population", ascending=False).reset_index(drop=True)

    out = Path(__file__).resolve().parent / "all_US_cities_forecast.csv"
    df.to_csv(out, index=False)
    print(f"\nFinished. {len(df)} cities saved to '{out}'  ({skipped} skipped).")
    return df

if __name__ == "__main__":
    run_all()
