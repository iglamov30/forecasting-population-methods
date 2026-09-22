# Citywide Forecast (all US cities)

Runs the same ETS/ARIMA backtest-and-forecast methodology as `charlotte/`
across every US city the Census ACS1 API tracks (population >= 65,000),
instead of one city at a time.

## Files

| File | What it does |
|---|---|
| `all_US_cities_forecast.py` | Fetches the full ACS1 panel, backtests ETS + ARIMA for every city, and saves a 5-year forecast per city to `all_US_cities_forecast.csv` |
| `forecast_citywide.py` | Reads `all_US_cities_forecast.csv` and charts the top 5 cities by projected 5-year growth |
| `forecast_citywide_adj.py` | A risk-adjusted variant (growth scaled by backtest error) |
| `all_US_cities_forecast.csv` | Saved output of the last full run |
| `census_acs1.py` | Fetches and reshapes the ACS1 panel |

The forecasting functions (`fit`, `rolling_backtest`, `forecast_horizon`) are imported from `../charlotte/charlotte_5y.py`; `all_US_cities_forecast.py` adds that folder to `sys.path`, so there is no local copy to keep in sync.

## Setup

```bash
pip install pandas numpy matplotlib statsmodels requests openpyxl
```

`all_US_cities_forecast.py` needs a Census API key (see the repo root README
for how to get one) as `CENSUS_API_KEY`, env var or local `secret.py`.
`forecast_citywide.py` only reads the saved CSV, so it needs no key.
`forecast_citywide_adj.py` also re-pulls the ACS1 history for its charts,
so it needs the key too.

## Run it

```bash
cd citywide_forecast
python all_US_cities_forecast.py     # re-fetches everything, ~550 places, takes a while
python forecast_citywide.py          # chart from the saved CSV, no network needed
python forecast_citywide_adj.py      # risk-adjusted top 5; re-pulls history, so needs the API key
```
