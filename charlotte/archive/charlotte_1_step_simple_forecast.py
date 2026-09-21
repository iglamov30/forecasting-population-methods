import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for this script. Install it with: pip install statsmodels"
    ) from exc

MODELS = ("ets", "arima")

def _resolve_data_path(path):
    base_dir = Path(__file__).resolve().parent
    candidate = Path(str(path)).expanduser()

    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()

    if not candidate.exists():
        alt = (base_dir.parent / "data" / candidate.name).resolve()
        if alt.exists():
            candidate = alt

    if not candidate.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Tried: {candidate} and {base_dir.parent / 'data' / Path(path).name}"
        )

    return str(candidate)


def load_data(path="../data/charlotte_population_updated.csv"):
    resolved_path = _resolve_data_path(path)

    if str(resolved_path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(resolved_path)
    else:
        df = pd.read_csv(resolved_path)

    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"year": "Year", "population": "Population"})
    df = df[["Year", "Population"]].dropna()
    df["Year"] = df["Year"].astype(int)
    df["Population"] = df["Population"].astype(float)
    df = df.sort_values("Year").reset_index(drop=True)

    return df["Year"].to_numpy(), df["Population"].to_numpy()

def _fit(train, model_type):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_type == "ets":
            return ExponentialSmoothing(train, trend="add").fit()
        elif model_type == "arima":
            return ARIMA(train, order=(1, 1, 1)).fit()
    raise ValueError(f"Unknown model_type: {model_type!r}")


def _forecast(fit, steps):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(fit.forecast(int(steps))).ravel()

def rolling_backtest(series, model_type="ets", start=None):
    n = len(series)
    if start is None:
        start = max(8, n // 2)         
    start = max(2, min(start, n - 3))    

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


def compare_models(series):
    results = {}
    for m in MODELS:
        mae, rmse, preds, start = rolling_backtest(series, m)
        results[m] = {"mae": mae, "rmse": rmse, "preds": preds, "start": start}

    best = min(results, key=lambda m: results[m]["rmse"])
    return best, results

def fit_model(series, model_type):
    return _fit(series, model_type)

def forecast_to_2030(fit, last_year, target_year=2030):
    steps = int(target_year) - int(last_year)
    if steps <= 0:
        return np.array([])         
    return _forecast(fit, steps)

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

def run_pipeline(path, show=True):
    years, series = load_data(path)
    last_year = int(years[-1])

    best_model, results = compare_models(series)

    print("Backtest comparison (one-step-ahead):")
    print(f"  {'model':<8}{'MAE':>14}{'RMSE':>14}")
    for m in MODELS:
        print(f"  {m:<8}{results[m]['mae']:>14,.1f}{results[m]['rmse']:>14,.1f}")
    print(f"  -> selected: {best_model.upper()}\n")

    bt = results[best_model]

    fit = fit_model(series, best_model)
    forecast = forecast_to_2030(fit, last_year)
    future_years = np.arange(last_year + 1, 2031)

    if show:
        plot_backtest(years, series, bt["preds"], bt["start"], best_model)
        plot_forecast(years, series, future_years, forecast, best_model)

    result = pd.DataFrame({
        "year": future_years,
        "predicted_population": np.round(forecast).astype(int),
    })
    return result, best_model, results


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "charlotte_population_updated.csv")
    result, model, _ = run_pipeline(data_path)
    print(f"Best model: {model.upper()}")
    print(result.to_string(index=False))
