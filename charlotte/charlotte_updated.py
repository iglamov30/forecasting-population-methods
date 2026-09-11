import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


# ---------------------------
# LOAD DATA
# ---------------------------
def load_data(path="charlotte_population_updated.csv"):
    df = pd.read_csv(path)
    df = df.sort_values("Year")
    return df["Year"].values, df["Population"].values


# ---------------------------
# ROLLING BACKTEST (with preds)
# ---------------------------
def rolling_backtest(series, model_type="ets", start=40):
    errors = []
    preds = []

    for i in range(start, len(series) - 1):
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


# ---------------------------
# MODEL SELECTION
# ---------------------------
def select_model(series):
    _, ets_rmse, _ = rolling_backtest(series, "ets")
    _, arima_rmse, _ = rolling_backtest(series, "arima")

    if ets_rmse < arima_rmse:
        return "ets"
    else:
        return "arima"


# ---------------------------
# FINAL MODEL FIT
# ---------------------------
def fit_model(series, model_type):
    if model_type == "ets":
        model = ExponentialSmoothing(series, trend="add")
        return model.fit()

    elif model_type == "arima":
        model = ARIMA(series, order=(1,1,1))
        return model.fit()


# ---------------------------
# FORECAST TO 2030
# ---------------------------
def forecast_to_2030(fit, last_year):
    steps = 2030 - last_year
    return fit.forecast(steps)


# ---------------------------
# PLOTS
# ---------------------------
def plot_forecast(years, series, forecast_years, forecast):
    plt.figure()
    plt.plot(years, series, label="Actual")
    plt.plot(forecast_years, forecast, linestyle="--", label="Forecast")
    plt.title("Population Forecast to 2030")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.show()


def plot_backtest(series, preds, start=40):
    plt.figure()
    actual = series[start:start+len(preds)]
    years_bt = np.arange(start, start+len(preds))

    plt.plot(years_bt, actual, label="Actual")
    plt.plot(years_bt, preds, linestyle="--", label="Backtest")
    plt.title("Rolling Backtest Fit")
    plt.xlabel("Time Index")
    plt.ylabel("Population")
    plt.legend()
    plt.show()


# ---------------------------
# FULL PIPELINE
# ---------------------------
def run_pipeline(path):
    years, series = load_data(path)

    last_year = years[-1]

    # Step 1: select model
    best_model = select_model(series)
    print("Best model:", best_model)

    # Step 2: backtest (for plotting + evaluation)
    _, _, preds = rolling_backtest(series, best_model)

    # Step 3: fit final model
    fit = fit_model(series, best_model)

    # Step 4: forecast
    forecast = forecast_to_2030(fit, last_year)

    future_years = np.arange(last_year + 1, 2031)

    # Step 5: plots
    plot_backtest(series, preds)
    plot_forecast(years, series, future_years, forecast)

    # Step 6: output table
    result = pd.DataFrame({
        "year": future_years,
        "predicted_population": forecast
    })

    return result, best_model


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    result, model = run_pipeline("charlotte_population_updated.csv")
    print("Best model:", model)
    print(result)