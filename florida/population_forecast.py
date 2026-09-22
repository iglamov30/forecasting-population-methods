import argparse
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sources.cities import CITY_NAMES
from sources.edr_population import build_panel, city_series, list_cities, filter_florida, clean_city_name

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "charlotte"))
from charlotte_5y import fit, rolling_backtest, forecast_horizon, plot_backtest, MODELS, HORIZON

MIN_OBS = HORIZON + 9   # enough history for at least one backtest window
TARGET_CITIES = tuple(CITY_NAMES.values())
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_forecast_city(years, series, forecast_years, mean, lower, upper,
                        model_name, city_label, conf=0.95, horizon=HORIZON):
    plt.figure(figsize=(12, 7))
    plt.plot(years, series, marker="o", ms=3, label="Actual")
    if len(mean):
        fy = np.r_[years[-1], forecast_years]
        plt.plot(
            fy, np.r_[series[-1], mean], marker="s", ms=4, linestyle="--",
            label=f"Forecast ({model_name.upper()})",
        )
        plt.fill_between(
            fy, np.r_[series[-1], lower], np.r_[series[-1], upper],
            alpha=0.2, color="C1", label=f"{int(conf * 100)}% interval",
        )
    plt.title(f"{city_label} Population {horizon}-Year Forecast ({model_name.upper()})")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.gca().yaxis.set_major_locator(ticker.AutoLocator())
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.legend()
    plt.grid(which="major", alpha=0.4)
    plt.grid(which="minor", alpha=0.15)
    plt.tight_layout()


def run_city(city_label, yrs, pop, alpha, show):
    last_year = int(yrs[-1])
    future_years = np.arange(last_year + 1, last_year + HORIZON + 1)

    bt_results = {}
    for m in MODELS:
        mae, rmse, preds, lower, upper, start = rolling_backtest(pop, m, alpha=alpha)
        bt_results[m] = {
            "mae": mae, "rmse": rmse,
            "preds": preds, "lower": lower, "upper": upper,
            "start": start,
        }

    print(f"\n{city_label}")
    print(f"  Backtest comparison ({HORIZON}-year-ahead, {int((1 - alpha) * 100)}% CI)")
    print(f"  {'Model':<8}{'MAE':>16}{'RMSE':>18}")
    for m in MODELS:
        print(f"  {m:<8}{bt_results[m]['mae']:>16,.1f}{bt_results[m]['rmse']:>18,.1f}")

    best_model = min(bt_results, key=lambda m: bt_results[m]["mae"])

    forecasts = {}
    for m in MODELS:
        fitted = fit(pop, m)
        mean, lower, upper = forecast_horizon(fitted, m, alpha=alpha)
        forecasts[m] = {"mean": mean, "lower": lower, "upper": upper}

    best_fc = forecasts[best_model]
    end_forecast = best_fc["mean"][-1]
    last_pop = pop[-1]
    pct_change = (end_forecast - last_pop) / last_pop * 100

    print(f"  Best model: {best_model.upper()}  "
          f"({future_years[-1]} forecast: {end_forecast:,.0f}, "
          f"{pct_change:+.1f}% vs {last_year})")

    if show:
        plot_backtest(yrs, pop, bt_results, horizon=HORIZON, alpha=alpha)
        plt.suptitle(city_label, y=1.02, fontsize=13, fontweight="bold")
        for m in MODELS:
            fc = forecasts[m]
            plot_forecast_city(
                yrs, pop, future_years,
                fc["mean"], fc["lower"], fc["upper"],
                m, city_label, conf=1 - alpha, horizon=HORIZON,
            )

    return {
        "city": city_label,
        "last_year": last_year,
        "last_population": int(last_pop),
        "best_model": best_model,
        "mae_ets": round(bt_results["ets"]["mae"]),
        "rmse_ets": round(bt_results["ets"]["rmse"]),
        "mae_arima": round(bt_results["arima"]["mae"]),
        "rmse_arima": round(bt_results["arima"]["rmse"]),
        f"{future_years[-1]}_forecast": round(end_forecast),
        f"{future_years[-1]}_lower": round(best_fc["lower"][-1]),
        f"{future_years[-1]}_upper": round(best_fc["upper"][-1]),
        "pct_change_5y": round(pct_change, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ETS/ARIMA 5-year population forecast for Florida's five largest cities."
    )
    parser.add_argument("--start-year", type=int, default=1979,
                         help="First year of EDR history to use (workbook starts at 1979).")
    parser.add_argument("--alpha", type=float, default=0.05,
                         help="Significance level for confidence intervals (default 0.05 = 95%% CI).")
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "outputs", "population_forecast.csv"),
                         help="Output CSV path for the summary table.")
    parser.add_argument("--no-show", action="store_true",
                         help="Skip plots and only print the summary.")
    args = parser.parse_args()
    show = not args.no_show

    years = list(range(args.start_year, datetime.now().year + 1))
    print(f"Loading EDR municipal panel for {years[0]}-{years[-1]}...")
    fl_panel = filter_florida(build_panel(years))

    all_cities = list_cities(fl_panel)
    targets = all_cities[all_cities["city"].isin(TARGET_CITIES)]
    missing = sorted(set(TARGET_CITIES) - set(targets["city"]))
    if missing:
        print(f"Warning: no EDR series found for: {', '.join(missing)}")
    if targets.empty:
        print("None of the target cities were found in the panel.")
        return

    print(f"\nForecasting {len(targets)} Florida cities by latest population:")
    print(targets.assign(population=targets["population"].map("{:,.0f}".format)).to_string())

    rows = []
    for place_id, meta in targets.iterrows():
        yrs, pop = city_series(fl_panel, place_id)
        if len(pop) < MIN_OBS:
            print(f"\n{meta['city']}: skipped, only {len(pop)} years of data "
                  f"(need {MIN_OBS} for a rolling backtest).")
            continue
        label = clean_city_name(meta["city"])
        rows.append(run_city(label, yrs, pop, alpha=args.alpha, show=show))

    if not rows:
        print("\nNo cities had enough history to forecast.")
        return

    summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"\nSaved {args.output}")

    if show:
        plt.show()


if __name__ == "__main__":
    main()
