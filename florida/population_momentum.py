import argparse
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sources.cities import CITY_NAMES
from sources.edr_population import build_panel, city_series, list_cities, filter_florida, clean_city_name

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "charlotte"))
from charlotte_5y import fit, rolling_backtest, forecast_horizon, MODELS, HORIZON

MIN_OBS = HORIZON + 9   # enough history for at least one backtest window
TARGET_CITIES = tuple(CITY_NAMES.values())
EPS = 1e-6

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


def forecast_city(yrs, pop, alpha):
    bt = {}
    for m in MODELS:
        mae, rmse, preds, lower, upper, start = rolling_backtest(pop, m, alpha=alpha)
        bt[m] = {"mae": mae, "rmse": rmse}

    best_model = min(bt, key=lambda m: bt[m]["mae"])

    fitted = fit(pop, best_model)
    mean, lower, upper = forecast_horizon(fitted, best_model, horizon=HORIZON, alpha=alpha)

    last_year = int(yrs[-1])
    future_years = np.arange(last_year + 1, last_year + HORIZON + 1)
    new_year = int(future_years[-1]) + 1

    growth_ratio = mean[-1] / mean[-2] if mean[-2] != 0 else 1.0
    extra_fc = mean[-1] * growth_ratio

    width_prev = upper[-2] - lower[-2]
    width_last = upper[-1] - lower[-1]
    widen_ratio = width_last / width_prev if width_prev != 0 else 1.0
    half_width_extra = (width_last * widen_ratio) / 2

    extra_lo = extra_fc - half_width_extra
    extra_hi = extra_fc + half_width_extra

    last_pop = pop[-1]
    pct_change = (extra_fc - last_pop) / last_pop * 100

    mae_pct = bt[best_model]["mae"] / last_pop * 100
    rmse_pct = bt[best_model]["rmse"] / last_pop * 100
    risk_adjusted_score = pct_change / (rmse_pct + EPS)

    return {
        "best_model": best_model,
        "future_years": future_years,
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "new_year": new_year,
        "extra_fc": extra_fc,
        "extra_lo": extra_lo,
        "extra_hi": extra_hi,
        "pct_change": pct_change,
        "mae_pct": mae_pct,
        "rmse_pct": rmse_pct,
        "risk_adjusted_score": risk_adjusted_score,
    }


def plot_city(label, yrs, pop, result, color):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    x_hist = yrs.tolist()
    y_hist = pop.tolist()

    x_fc = list(result["future_years"]) + [result["new_year"]]
    y_fc = list(result["mean"]) + [result["extra_fc"]]
    y_lo = list(result["lower"]) + [result["extra_lo"]]
    y_hi = list(result["upper"]) + [result["extra_hi"]]

    ax.plot(x_hist + x_fc, y_hist + y_fc, color=color, linewidth=2.2,
             marker="o", markersize=4)
    ax.fill_between(x_fc, y_lo, y_hi, color=color, alpha=0.18)

    ax.scatter([x_hist[-1]], [y_hist[-1]], color="white", edgecolors=color,
               zorder=5, linewidth=2, s=60)
    ax.scatter([result["new_year"]], [result["extra_fc"]], color=color,
               edgecolors="black", zorder=6, linewidth=1.2, s=70, marker="D")

    ax.set_title(
        f"{label}\n{result['pct_change']:+.1f}% by {result['new_year']}",
        fontsize=13, fontweight="bold",
    )
    ax.text(
        0.04, 0.97,
        f"Model: {result['best_model'].upper()}\n"
        f"MAE: {result['mae_pct']:.1f}% of pop.\n"
        f"RMSE: {result['rmse_pct']:.1f}% of pop.\n"
        f"Score: {result['risk_adjusted_score']:.2f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=color, alpha=0.85),
    )
    ax.set_xlim(x_hist[0] - 0.5, result["new_year"] + 0.8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda v, _: f"{v/1_000:.0f}K" if v < 1_000_000 else f"{v/1_000_000:.2f}M"
        )
    )
    ax.set_xlabel("Year", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(
        description="Rank Florida's five largest cities by risk-adjusted population growth."
    )
    parser.add_argument("--start-year", type=int, default=1979,
                         help="First year of EDR history to use (workbook starts at 1979).")
    parser.add_argument("--alpha", type=float, default=0.10,
                         help="Significance level for CIs (default 0.10 = 90%% CI, "
                         "matching citywide_forecast/all_US_cities_forecast.py).")
    parser.add_argument("--output", default=None,
                         help="Output CSV path (default outputs/population_momentum.csv).")
    parser.add_argument("--no-show", action="store_true", help="Skip rendering plots.")
    args = parser.parse_args()
    show = not args.no_show
    output_path = args.output if args.output else os.path.join(SCRIPT_DIR, "outputs", "population_momentum.csv")

    years = list(range(args.start_year, datetime.now().year + 1))
    print(f"Loading EDR municipal panel for {years[0]}-{years[-1]}...")
    fl_panel = filter_florida(build_panel(years))

    all_cities = list_cities(fl_panel)
    targets = all_cities[all_cities["city"].isin(TARGET_CITIES)]
    print(f"\n{len(targets)} Florida cities by latest population:")
    print(targets.assign(population=targets["population"].map("{:,.0f}".format)).to_string())

    rows = []
    for place_id, meta in targets.iterrows():
        yrs, pop = city_series(fl_panel, place_id)
        if len(pop) < MIN_OBS:
            print(f"\n{meta['city']}: skipped, only {len(pop)} years of data "
                  f"(need {MIN_OBS}).")
            continue
        label = clean_city_name(meta["city"])
        result = forecast_city(yrs, pop, alpha=args.alpha)
        result["label"] = label
        result["place_id"] = place_id
        result["last_population"] = int(pop[-1])
        result["yrs"] = yrs
        result["pop"] = pop
        rows.append(result)

    if not rows:
        print("\nNo cities had enough history to forecast.")
        return

    rows.sort(key=lambda r: r["risk_adjusted_score"], reverse=True)

    print(f"\nRanked by risk-adjusted growth through {rows[0]['new_year']}:")
    print(f"{'City':<20}{'Pop.':>12}{'% Growth':>10}{'Model':>8}{'MAE%':>8}{'RMSE%':>8}{'Score':>9}")
    for r in rows:
        print(f"{r['label']:<20}{r['last_population']:>12,}{r['pct_change']:>9.2f}%  "
              f"{r['best_model'].upper():>6}{r['mae_pct']:>7.1f}%{r['rmse_pct']:>7.1f}%"
              f"{r['risk_adjusted_score']:>9.2f}")

    summary = pd.DataFrame([{
        "place_id": r["place_id"],
        "city": r["label"],
        "last_population": r["last_population"],
        "best_model": r["best_model"],
        "mae_pct": round(r["mae_pct"], 2),
        "rmse_pct": round(r["rmse_pct"], 2),
        f"{r['new_year']}_forecast": round(r["extra_fc"]),
        f"{r['new_year']}_lower": round(r["extra_lo"]),
        f"{r['new_year']}_upper": round(r["extra_hi"]),
        "pct_change": round(r["pct_change"], 2),
        "risk_adjusted_score": round(r["risk_adjusted_score"], 3),
    } for r in rows])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"\nSaved {os.path.abspath(output_path)}")

    if show:
        for r, color in zip(rows, PALETTE):
            plot_city(r["label"], r["yrs"], r["pop"], r, color)
        plt.show()


if __name__ == "__main__":
    main()
