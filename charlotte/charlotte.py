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
        return ExponentialSmoothing(train, trend="add").fit()
    elif model_type == "arima":
        return ARIMA(train, order=(1, 1, 1)).fit()
    raise ValueError(f"Unknown model_type: {model_type!r}")

def forecast(fit, steps): # Fix warnings
    return np.asarray(fit.forecast(int(steps))).ravel()

def rolling_backtest(series, model_type="ets", start=None):
    n = len(series)
    if start is None:
        start = max(8, n // 2)
    start = max(2, min(start, n - 3))

    predictions = []
    for i in range(start, n):
        train = series[:i]
        fitted = fit(train, model_type)
        predictions.append(forecast(fitted, 1)[0])

    predictions = np.asarray(predictions)
    actual = series[start:start + len(predictions)]
    errors = actual - predictions
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse, predictions, start

def backtest_ci(fit, model_type, steps, alpha=0.05, n_sims=2000, seed=0):
    steps = int(steps)
    if steps <= 0:
        empty = np.array([])
        return empty, empty, empty

    if model_type == "arima":
        fc = fit.get_forecast(steps)
        mean = np.asarray(fc.predicted_mean).ravel()
        ci = np.asarray(fc.conf_int(alpha=alpha))
        lower, upper = ci[:, 0], ci[:, 1]
    elif model_type == "ets":
        mean = np.asarray(fit.forecast(steps)).ravel()
        sims = np.asarray(fit.simulate(steps, repetitions=n_sims, anchor="end", random_state=seed))
        lower = np.percentile(sims, 100 * alpha / 2, axis=1)
        upper = np.percentile(sims, 100 * (1 - alpha / 2), axis=1)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return mean, lower, upper

def forecast_2030(fit, last_year, model_type, target_year=2030, alpha=0.05):
    steps = int(target_year) - int(last_year)
    return backtest_ci(fit, model_type, steps, alpha=alpha)

def plot_backtest(years, series, results):
    plt.figure(figsize=(12, 7))
    all_starts = [results[m]["start"] for m in results]
    global_start = min(all_starts)
    bt_years = years[global_start:]
    actual = series[global_start:]
    plt.plot(bt_years, actual, marker="o", ms=3, label="Actual")

    for model_name, r in results.items():
        preds = r["preds"]
        start = r["start"]
        model_years = years[start:start + len(preds)]
        plt.plot(model_years, preds, marker="x", ms=4, linestyle="--", label=f"One-step backtest ({model_name.upper()})")

    plt.title("Rolling One-Step-Ahead Backtest — Both Models")
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
                  model_name, conf=0.95):
    plt.figure(figsize=(12, 7))
    plt.plot(years, series, marker="o", ms=3, label="Actual")
    if len(mean):
        fy = np.r_[years[-1], forecast_years]
        plt.plot(fy, np.r_[series[-1], mean], marker="s", ms=4, linestyle="--", label=f"Forecast ({model_name.upper()})")
        plt.fill_between(fy, np.r_[series[-1], lower], np.r_[series[-1], upper], alpha=0.2, color="C1", label=f"{int(conf * 100)}% interval")
    plt.title("Charlotte Population Forecast to 2030")
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
    alpha = 0.05
    future_years = np.arange(last_year + 1, 2031)

    bt_results = {}
    for m in MODELS:
        mae, rmse, preds, start = rolling_backtest(series, m)
        bt_results[m] = {"mae": mae, "rmse": rmse, "preds": preds, "start": start}
    print("Backtest comparison (one-year-ahead)")
    print(f"{'Model':<8}{'Mean Absolute Error':>25}{'Root Mean Square Error':>25}")
    for m in MODELS:
        print(f"{m:<8}{bt_results[m]['mae']:>14,.1f}{bt_results[m]['rmse']:>25,.1f}")

    forecasts = {}
    for m in MODELS:
        fitted = fit(series, m)
        mean, lower, upper = forecast_2030(fitted, last_year, m, alpha=alpha)
        forecasts[m] = {"mean": mean, "lower": lower, "upper": upper}

    if show:
        plot_backtest(years, series, bt_results)

        for m in MODELS:
            fc = forecasts[m]
            plot_forecast(years, series, future_years, fc["mean"], fc["lower"], fc["upper"], m, conf=1 - alpha)
        
        plt.show()
    return bt_results, forecasts

if __name__ == "__main__":
    run("charlotte_population_updated.csv")

