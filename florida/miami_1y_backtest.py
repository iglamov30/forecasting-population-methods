"""
miami_1y_backtest.py

Miami-only companion to florida_top5_forecast.py that runs the SAME
ETS/ARIMA rolling backtest, but one year ahead instead of five.

Why a separate file: Miami's EDR series has a one-off level shift between
2020 and 2021 (497,924 -> 449,747, a Census-2020 benchmark revision).
The 5-year-ahead backtest in florida_top5_forecast.py smears that break
across five overlapping windows, which makes both models look badly
biased. Re-running with a 1-year horizon isolates the damage to the
single window that predicts 2021 (ETS is off -12% there, within +/-3.5%
everywhere else) and shows how tight the models actually are in every
normal year.

Everything else is reused from florida_top5_forecast, so the methodology
is identical to the main script -- only HORIZON changes:

    import florida_top5_forecast as ftf
    ftf.build_panel / ftf.city_series / ftf.list_cities / ftf.filter_florida
    ftf.rolling_backtest / ftf.fit / ftf.forecast_horizon / ftf.MODELS
    ftf.plot_backtest / ftf.plot_forecast_city

USAGE
-----
    python miami_1y_backtest.py
    python miami_1y_backtest.py --horizon 1 --city Miami --alpha 0.05
    python miami_1y_backtest.py --no-show
"""

import argparse
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

import florida_top5_forecast as ftf


def backtest_city(city_label, yrs, pop, horizon, alpha, show):
    """Same shape as florida_top5_forecast.run_city, but horizon is a
    parameter instead of the hardcoded HORIZON, and it prints every
    rolling window so the 2020 break is visible."""
    last_year = int(yrs[-1])
    future_years = np.arange(last_year + 1, last_year + horizon + 1)

    results = {}
    for m in ftf.MODELS:
        mae, rmse, preds, lower, upper, start = ftf.rolling_backtest(
            pop, m, horizon=horizon, alpha=alpha
        )
        results[m] = {
            "mae": mae, "rmse": rmse,
            "preds": preds, "lower": lower, "upper": upper, "start": start,
        }

    print(f"\n{city_label} - rolling {horizon}-year-ahead backtest "
          f"({int((1 - alpha) * 100)}% CI)")
    print(f"  {'Model':<8}{'MAE':>14}{'RMSE':>14}")
    for m in ftf.MODELS:
        print(f"  {m:<8}{results[m]['mae']:>14,.1f}{results[m]['rmse']:>14,.1f}")

    # Per-window detail: target year -> actual / prediction / error.
    for m in ftf.MODELS:
        r = results[m]
        start = r["start"]
        sl = slice(start + horizon - 1, start + len(r["preds"]) + horizon - 1)
        tgt_years = yrs[sl]
        actual = pop[sl]
        err = actual - r["preds"]
        worst = np.max(np.abs(err))
        print(f"\n  [{m.upper()}] per-window  (target: actual / pred / error)")
        for ty, a, p, e in zip(tgt_years, actual, r["preds"], err):
            flag = "   <-- worst miss" if abs(e) == worst else ""
            print(f"    {int(ty)}: {int(a):>9,} / {int(p):>9,} / "
                  f"{int(e):>+8,} ({e / a * 100:+5.1f}%){flag}")
        ex_worst = err[np.abs(err) < worst]
        if len(ex_worst):
            print(f"    MAE excluding the worst window: "
                  f"{np.mean(np.abs(ex_worst)):,.1f}")

    best_model = min(results, key=lambda m: results[m]["mae"])

    # A real horizon-ahead forecast off the full series, for context.
    forecasts = {}
    for m in ftf.MODELS:
        fitted = ftf.fit(pop, m)
        mean, lower, upper = ftf.forecast_horizon(
            fitted, m, horizon=horizon, alpha=alpha
        )
        forecasts[m] = {"mean": mean, "lower": lower, "upper": upper}

    best_fc = forecasts[best_model]
    end_forecast = best_fc["mean"][-1]
    last_pop = pop[-1]
    pct = (end_forecast - last_pop) / last_pop * 100
    print(f"\n  Best model by MAE: {best_model.upper()}  "
          f"({future_years[-1]} forecast: {end_forecast:,.0f}, "
          f"{pct:+.1f}% vs {last_year})")

    if show:
        ftf.plot_backtest(yrs, pop, results, horizon=horizon, alpha=alpha)
        plt.suptitle(f"{city_label} - {horizon}-year-ahead", y=1.02,
                     fontsize=13, fontweight="bold")
        for m in ftf.MODELS:
            fc = forecasts[m]
            ftf.plot_forecast_city(
                yrs, pop, future_years, fc["mean"], fc["lower"], fc["upper"],
                m, city_label, conf=1 - alpha, horizon=horizon,
            )

    return results, forecasts


def main():
    parser = argparse.ArgumentParser(
        description="Miami-only 1-year-ahead rolling backtest, reusing "
                    "florida_top5_forecast's ETS/ARIMA methodology."
    )
    parser.add_argument("--city", default="Miami",
                        help="EDR municipality name to test (default: Miami).")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Forecast horizon in years (default: 1).")
    parser.add_argument("--start-year", type=int, default=1979,
                        help="First year of EDR history to use.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="CI significance level (default 0.05 = 95%% CI).")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip plots, just print the tables.")
    args = parser.parse_args()
    show = not args.no_show

    years = list(range(args.start_year, datetime.now().year + 1))
    print(f"Loading EDR municipal panel for {years[0]}-{years[-1]}...")
    panel = ftf.filter_florida(ftf.build_panel(years))

    cities = ftf.list_cities(panel)
    match = cities[cities["city"] == args.city]
    if match.empty:
        print(f"City {args.city!r} not found. Top rows available:\n"
              f"{cities.head(10)}")
        return
    place_id = match.index[0]
    label = ftf.clean_city_name(args.city)

    yrs, pop = ftf.city_series(panel, place_id)
    min_obs = args.horizon + 9
    if len(pop) < min_obs:
        print(f"{label}: only {len(pop)} years of data (need >= {min_obs}).")
        return
    print(f"{label}: {len(pop)} years, {int(yrs[0])}-{int(yrs[-1])}, "
          f"latest {int(pop[-1]):,}")

    backtest_city(label, yrs, pop, horizon=args.horizon, alpha=args.alpha,
                  show=show)

    if show:
        plt.show()


if __name__ == "__main__":
    main()
