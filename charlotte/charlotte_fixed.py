"""
Charlotte population forecasting to 2030.

Pipeline:
  1. Load the (Year, Population) series.
  2. Rolling one-step-ahead backtest for each candidate model
     (Exponential Smoothing and ARIMA), scored with MAE and RMSE.
  3. Pick the model with the lowest backtest RMSE.
  4. Refit it on the full history and forecast through 2030.
  5. Plot the backtest fit and the forecast.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

MODELS = ("ets", "arima")


# ---------------------------
# LOAD DATA
# ---------------------------
def load_data(path="charlotte_population_updated.csv"):
    """Read a (Year, Population) table from CSV or Excel and return clean arrays."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Be forgiving about column capitalisation / whitespace.
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"year": "Year", "population": "Population"})

    df = df[["Year", "Population"]].dropna()
    df["Year"] = df["Year"].astype(int)
    df["Population"] = df["Population"].astype(float)
    df = df.sort_values("Year").reset_index(drop=True)

    return df["Year"].to_numpy(), df["Population"].to_numpy()


# ---------------------------
# MODEL HELPERS
# ---------------------------
def _fit(train, model_type):
    """Fit a single model on a training array. Warnings are silenced here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_type == "ets":
            return ExponentialSmoothing(train, trend="add").fit()
        elif model_type == "arima":
            return ARIMA(train, order=(1, 1, 1)).fit()
    raise ValueError(f"Unknown model_type: {model_type!r}")


def _forecast(fit, steps):
    """One- or multi-step forecast returned as a plain 1-D numpy array.

    Casting `steps` to a built-in int matters: statsmodels' ARIMA rejects a
    numpy integer here, and wrapping the result with np.asarray avoids the
    pandas index-alignment pitfall when the output is later put in a DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(fit.forecast(int(steps))).ravel()


# ---------------------------
# ROLLING BACKTEST (with preds)
# ---------------------------
def rolling_backtest(series, model_type="ets", start=None):
    """One-step-ahead rolling backtest.

    Returns mae, rmse, predictions, and the start index actually used so the
    plotting code can stay perfectly aligned with these predictions.
    """
    n = len(series)
    if start is None:
        start = max(8, n // 2)          # adaptive: enough train, enough test
    start = max(2, min(start, n - 3))    # always leave a few points to test on

    preds = []
    for i in range(start, n):
        train = series[:i]
        fit = _fit(train, model_type)
        preds.append(_forecast(fit, 1)[0])

    preds = np.asarray(preds)
    actual = series[start:start + len(preds)]
    errors = actual - preds

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse, preds, start


# ---------------------------
# MODEL SELECTION
# ---------------------------
def compare_models(series):
    """Backtest every candidate model once and pick the best by RMSE."""
    results = {}
    for m in MODELS:
        mae, rmse, preds, start = rolling_backtest(series, m)
        results[m] = {"mae": mae, "rmse": rmse, "preds": preds, "start": start}

    best = min(results, key=lambda m: results[m]["rmse"])
    return best, results


# ---------------------------
# FINAL MODEL FIT + FORECAST
# ---------------------------
def fit_model(series, model_type):
    return _fit(series, model_type)


def forecast_to_2030(fit, last_year, target_year=2030):
    steps = int(target_year) - int(last_year)
    if steps <= 0:
        return np.array([])          # data already reaches/passes the target
    return _forecast(fit, steps)


# ---------------------------
# PLOTS
# ---------------------------
def plot_backtest(years, series, preds, start, model_name):
    bt_years = years[start:start + len(preds)]
    actual = series[start:start + len(preds)]

    plt.figure(figsize=(10, 5))
    plt.plot(bt_years, actual, marker="o", ms=3, label="Actual")
    plt.plot(bt_years, preds, marker="x", ms=4, linestyle="--",
             label=f"One-step backtest ({model_name.upper()})")
    plt.title("Rolling One-Step-Ahead Backtest")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forecast(years, series, forecast_years, forecast, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(years, series, marker="o", ms=3, label="Actual")
    if len(forecast):
        # connect the last actual point to the first forecast point
        plt.plot(np.r_[years[-1], forecast_years],
                 np.r_[series[-1], forecast],
                 marker="s", ms=4, linestyle="--",
                 label=f"Forecast ({model_name.upper()})")
    plt.title("Charlotte Population Forecast to 2030")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------
# FULL PIPELINE
# ---------------------------
def run_pipeline(path, show=True):
    years, series = load_data(path)
    last_year = int(years[-1])

    # Step 1+2: backtest every model once, choose the best.
    best_model, results = compare_models(series)

    print("Backtest comparison (one-step-ahead):")
    print(f"  {'model':<8}{'MAE':>14}{'RMSE':>14}")
    for m in MODELS:
        print(f"  {m:<8}{results[m]['mae']:>14,.1f}{results[m]['rmse']:>14,.1f}")
    print(f"  -> selected: {best_model.upper()}\n")

    bt = results[best_model]

    # Step 3+4: refit on full history and forecast.
    fit = fit_model(series, best_model)
    forecast = forecast_to_2030(fit, last_year)
    future_years = np.arange(last_year + 1, 2031)

    # Step 5: plots (displayed inline, not saved).
    if show:
        plot_backtest(years, series, bt["preds"], bt["start"], best_model)
        plot_forecast(years, series, future_years, forecast, best_model)

    # Step 6: output table.
    result = pd.DataFrame({
        "year": future_years,
        "predicted_population": np.round(forecast).astype(int),
    })
    return result, best_model, results


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    result, model, _ = run_pipeline("charlotte_population_updated.csv")
    print(f"Best model: {model.upper()}")
    print(result.to_string(index=False))
