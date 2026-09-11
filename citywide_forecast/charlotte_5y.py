import warnings
warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

MODELS = ("ets", "arima")
HORIZON = 5  # 5-year real-estate investment horizon

def load_data(path="charlotte_population_updated.csv"):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"year": "Year", "population": "Population"})
    df = df[["Year", "Population"]].dropna()
    df["Year"] = df["Year"].astype(int)
    df["Population"] = df["Population"].astype(float)
    df = df.sort_values("Year").reset_index(drop=True)
    return df["Year"].to_numpy(), df["Population"].to_numpy()

def fit(train, model_type):
    if model_type == "ets":
        return ExponentialSmoothing(train, trend="add", damped_trend=True).fit()
    elif model_type == "arima":
        return ARIMA(train, order=(1, 1, 1)).fit()
    raise ValueError(f"Unknown model_type: {model_type!r}")

def forecast(fitted_model, steps):
    return np.asarray(fitted_model.forecast(int(steps))).ravel()

def backtest_ci(fitted_model, model_type, steps, alpha=0.05, n_sims=2000, seed=0):
    steps = int(steps)
    if steps <= 0:
        empty = np.array([])
        return empty, empty, empty

    if model_type == "arima":
        fc = fitted_model.get_forecast(steps)
        mean = np.asarray(fc.predicted_mean).ravel()
        ci = np.asarray(fc.conf_int(alpha=alpha))
        lower, upper = ci[:, 0], ci[:, 1]
    elif model_type == "ets":
        mean = np.asarray(fitted_model.forecast(steps)).ravel()
        sims = np.asarray(
            fitted_model.simulate(steps, repetitions=n_sims, anchor="end", random_state=seed)
        )
        lower = np.percentile(sims, 100 * alpha / 2, axis=1)
        upper = np.percentile(sims, 100 * (1 - alpha / 2), axis=1)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return mean, lower, upper

def rolling_backtest(series, model_type="ets", start=None, horizon=HORIZON, alpha=0.05):
    """Rolling backtest predicting `horizon` years ahead at each window."""
    n = len(series)
    if start is None:
        start = max(8, n // 2)
    start = max(2, min(start, n - horizon - 1))

    predictions, lower_list, upper_list = [], [], []
    for i in range(start, n - horizon + 1):
        train = series[:i]
        fitted = fit(train, model_type)
        fc_mean, fc_lower, fc_upper = backtest_ci(fitted, model_type, horizon, alpha=alpha)
        # Record only the horizon-th step (the 5-year-ahead prediction)
        predictions.append(fc_mean[-1])
        lower_list.append(fc_lower[-1])
        upper_list.append(fc_upper[-1])

    predictions = np.asarray(predictions)
    lower_arr = np.asarray(lower_list)
    upper_arr = np.asarray(upper_list)
    # Actual values at the horizon-th step for each window
    actual = series[start + horizon - 1: start + len(predictions) + horizon - 1]
    errors = actual - predictions
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse, predictions, lower_arr, upper_arr, start

def forecast_horizon(fitted_model, model_type, horizon=HORIZON, alpha=0.05):
    """Produce a `horizon`-step-ahead forecast with a 95% confidence interval."""
    return backtest_ci(fitted_model, model_type, horizon, alpha=alpha)

def plot_backtest(years, series, results, horizon=HORIZON, alpha=0.05):
    plt.figure(figsize=(12, 7))
    all_starts = [results[m]["start"] for m in results]
    global_start = min(all_starts)

    # Show all actuals from global_start for historical context
    plt.plot(years[global_start:], series[global_start:], marker="o", ms=3, label="Actual")

    for model_name, r in results.items():
        preds = r["preds"]
        lower = r["lower"]
        upper = r["upper"]
        start = r["start"]
        # Each prediction at window i targets years[i + horizon - 1]
        model_years = years[start + horizon - 1: start + len(preds) + horizon - 1]
        plt.plot(
            model_years, preds, marker="x", ms=4, linestyle="--",
            label=f"{horizon}-year-ahead backtest ({model_name.upper()})",
        )
        plt.fill_between(
            model_years, lower, upper, alpha=0.15,
            label=f"{int((1 - alpha) * 100)}% CI ({model_name.upper()})",
        )

    plt.title(f"Rolling {horizon}-Year-Ahead Backtest — Both Models")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.gca().yaxis.set_major_locator(ticker.AutoLocator())
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.legend()
    plt.grid(which="major", alpha=0.4)
    plt.grid(which="minor", alpha=0.15)
    plt.tight_layout()

def plot_forecast(years, series, forecast_years, mean, lower, upper,
                  model_name, conf=0.95, horizon=HORIZON):
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
    plt.title(f"Charlotte Population {horizon}-Year Forecast ({model_name.upper()})")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.gca().yaxis.set_major_locator(ticker.AutoLocator())
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.legend()
    plt.grid(which="major", alpha=0.4)
    plt.grid(which="minor", alpha=0.15)
    plt.tight_layout()

def run(path, show=True):
    years, series = load_data(path)
    last_year = int(years[-1])
    alpha = 0.01
    future_years = np.arange(last_year + 1, last_year + HORIZON + 1)

    bt_results = {}
    for m in MODELS:
        mae, rmse, preds, lower, upper, start = rolling_backtest(series, m, alpha=alpha)
        bt_results[m] = {
            "mae": mae, "rmse": rmse,
            "preds": preds, "lower": lower, "upper": upper,
            "start": start,
        }
    print(f"Backtest comparison ({HORIZON}-year-ahead, 95% confidence intervals)")
    print(f"{'Model':<8}{'Mean Absolute Error':>25}{'Root Mean Square Error':>25}")
    for m in MODELS:
        print(f"{m:<8}{bt_results[m]['mae']:>14,.1f}{bt_results[m]['rmse']:>25,.1f}")

    forecasts = {}
    for m in MODELS:
        fitted = fit(series, m)
        mean, lower, upper = forecast_horizon(fitted, m, alpha=alpha)
        forecasts[m] = {"mean": mean, "lower": lower, "upper": upper}

    if show:
        plot_backtest(years, series, bt_results, horizon=HORIZON, alpha=alpha)

        for m in MODELS:
            fc = forecasts[m]
            plot_forecast(
                years, series, future_years,
                fc["mean"], fc["lower"], fc["upper"],
                m, conf=1 - alpha, horizon=HORIZON,
            )

        plt.show()
    return bt_results, forecasts

if __name__ == "__main__":
    run("charlotte_population_updated.csv")
