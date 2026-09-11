"""
florida_top5_momentum.py

Same "momentum" methodology as forecast_citywide_adj.py (growth forecast
risk-adjusted by the model's own backtest error), applied to the 5 largest
Florida cities -- with NO population floor, since the candidate set is
already exactly those 5 cities. The floor in forecast_citywide_adj.py
existed to narrow a nationwide list down to a comparable set of large
cities; here that narrowing already happened via city_series/list_cities'
population sort, so a floor would just be redundant filtering on top of
filtering.

Unlike forecast_citywide_adj.py (which reads a pre-built CSV), this script
computes the ETS/ARIMA backtest + forecast directly, the same way
florida_top5_forecast.py does, then adds the momentum scoring on top:

  1. Forecast HORIZON years ahead (ETS vs ARIMA, best model by backtest MAE)
  2. Extrapolate one extra year past the horizon using the last two
     forecast years' growth ratio (same trick as forecast_citywide_adj.py's
     2029 -> 2030 step), and widen the CI by the same ratio
  3. risk_adjusted_score = pct_change_to_extrapolated_year / (rmse_pct + eps)
  4. Rank all 5 cities by that score (not by raw growth, and not by size --
     size was already fixed by construction)

USAGE
-----
    python florida_top5_momentum.py
    python florida_top5_momentum.py --alpha 0.10 --start-year 2005
    python florida_top5_momentum.py --no-show

SETUP
-----
Same as the rest of the project: put CENSUS_API_KEY in secret.py or the
environment. See census_devs.py / test_census.py.
"""

import argparse
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from edr_population import build_panel, city_series, list_cities, filter_florida, clean_city_name
from charlotte_5y import fit, rolling_backtest, forecast_horizon, MODELS, HORIZON

MIN_OBS = HORIZON + 9   # enough history for at least one rolling-backtest window
N_CITIES = 5
EPS = 1e-6

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


def forecast_city(yrs, pop, alpha):
    """Rolling backtest both models, pick best by MAE, forecast HORIZON years
    ahead, then extrapolate one extra year using the last-two-years growth
    ratio -- same as forecast_citywide_adj.py's 2029->2030 step, generalized
    to whatever the last forecast year actually is."""
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

    # Extrapolate one more year using the final growth ratio
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
        f"{label}\n+{result['pct_change']:.1f}% by {result['new_year']}",
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
        description="Rank the 5 largest FL cities by risk-adjusted growth momentum."
    )
    parser.add_argument("--start-year", type=int, default=1979,
                         help="First year of EDR history to use (workbook starts at 1979).")
    parser.add_argument("--alpha", type=float, default=0.10,
                         help="Significance level for CIs (default 0.10 = 90%% CI, "
                         "matching the national all_cities_forecast.py run).")
    parser.add_argument("--output", default=None,
                         help="Output CSV path. Defaults to 'florida_top5_momentum.csv' "
                         "next to this script.")
    parser.add_argument("--no-show", action="store_true", help="Skip rendering plots.")
    args = parser.parse_args()
    show = not args.no_show
    output_path = args.output if args.output else os.path.join(SCRIPT_DIR, "florida_top5_momentum.csv")

    years = list(range(args.start_year, datetime.now().year + 1))
    print(f"Loading EDR municipal panel for {years[0]}-{years[-1]}...")
    panel = build_panel(years)

    fl_panel = filter_florida(panel)
    if fl_panel.empty:
        print("No Florida places found — check your years/key.")
        return

    top5 = list_cities(fl_panel).head(N_CITIES)
    print(f"\nTop {N_CITIES} Florida cities by population (candidate set, no size floor applied):")
    print(top5.assign(population=top5["population"].map("{:,.0f}".format)).to_string())

    rows = []
    for place_id, meta in top5.iterrows():
        yrs, pop = city_series(fl_panel, place_id)
        if len(pop) < MIN_OBS:
            print(f"\n{meta['city']}: skipped — only {len(pop)} years of data "
                  f"(need >= {MIN_OBS}).")
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

    print(f"\nFlorida top {N_CITIES} cities ranked by risk-adjusted growth momentum "
          f"(through {rows[0]['new_year']}):")
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
    summary.to_csv(output_path, index=False)
    print(f"\nSaved summary to: {os.path.abspath(output_path)}")

    if show:
        for r, color in zip(rows, PALETTE):
            plot_city(r["label"], r["yrs"], r["pop"], r, color)
        plt.show()


if __name__ == "__main__":
    main()
