# Citywide forecast (all US cities)

Runs the same ETS/ARIMA backtest-and-forecast methodology as `charlotte/`
across every US city the Census ACS1 API tracks (population >= 65,000),
instead of one city at a time.

## Files

| File | What it does |
|---|---|
| `all_cities_forecast.py` | Fetches the full ACS1 panel, backtests ETS + ARIMA for every city, and saves a 5-year forecast per city to `all_cities_forecast.csv` |
| `forecast_citywide.py` | Reads `all_cities_forecast.csv` and charts the top 5 cities by projected 5-year growth |
| `forecast_citywide_adj.py` | A risk-adjusted variant (growth scaled by backtest error) |
| `all_cities_forecast.csv` | Saved output of the last full run |
| `census_devs.py`, `charlotte_5y.py` | Local copies of the `population_matrix/` and `charlotte/` modules this script depends on, kept here so this folder runs standalone |

## Setup

```bash
pip install pandas numpy matplotlib statsmodels requests
```

`all_cities_forecast.py` needs a Census API key (see the repo root README
for how to get one) as `CENSUS_API_KEY`, env var or local `secret.py`.
`forecast_citywide.py` and `forecast_citywide_adj.py` only read the saved
CSV, so they need no key.

## Run it

```bash
cd citywide_forecast
python all_cities_forecast.py     # re-fetches everything, ~96+ cities, takes a while
python forecast_citywide.py       # chart from the saved CSV, no network needed
```
