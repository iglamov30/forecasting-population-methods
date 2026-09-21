# Initial pipeline

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "charlotte_population_updated.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.sort_values("Year")
    return df["Year"].values, df["Population"].values

def rolling_backtest(series, model_type="ets", start=40):
    errors = []
    preds = []

    for i in range(start, len(series)):
        train = series[:i]
        test = series[i]

        if model_type == "ets":
            model = ExponentialSmoothing(train, trend="add")
            fit = model.fit()
            pred = fit.forecast(1)[0]

        elif model_type == "arima":
            model = ARIMA(train, order=(1,1,1))
            fit = model.fit()
            pred = fit.forecast(1)[0]

        preds.append(pred)
        errors.append(test - pred)

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))

    return mae, rmse, preds

def select_model(series):
    _, ets_rmse, _ = rolling_backtest(series, "ets")
    _, arima_rmse, _ = rolling_backtest(series, "arima")

    if ets_rmse < arima_rmse:
        return "ets"
    else:
        return "arima"

def fit_model(series, model_type):
    if model_type == "ets":
        model = ExponentialSmoothing(series, trend="add")
        return model.fit()

    elif model_type == "arima":
        model = ARIMA(series, order=(1,1,1))
        return model.fit()

def forecast_to_2030(fit, last_year):
    steps = 2030 - last_year
    return fit.forecast(steps)

def plot_forecast(years, series, forecast_years, forecast):
    plt.figure()
    plt.plot(years, series, label="Actual")
    plt.plot(forecast_years, forecast, linestyle="--", label="Forecast")
    plt.title("Population Forecast to 2030")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.show()

def plot_backtest(years, series, preds, start=40):
    plt.figure()
    actual = series[start:start+len(preds)]
    years_bt = years[start:start+len(preds)]

    plt.plot(years_bt, actual, label="Actual")
    plt.plot(years_bt, preds, linestyle="--", label="Backtest")
    plt.title("Rolling Backtest Fit")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.show()

def run_pipeline(path):
    years, series = load_data(path)

    last_year = years[-1]

    best_model = select_model(series)
    print("Best model:", best_model)
    _, _, preds = rolling_backtest(series, best_model)

    fit = fit_model(series, best_model)

    forecast = forecast_to_2030(fit, last_year)

    future_years = np.arange(last_year + 1, 2031)

    plot_backtest(years, series, preds)
    plot_forecast(years, series, future_years, forecast)

    result = pd.DataFrame({
        "year": future_years,
        "predicted_population": forecast
    })

    return result, best_model

if __name__ == "__main__":
    result, model = run_pipeline(DATA_PATH)
    print("Best model:", model)
    print(result)