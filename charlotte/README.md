# Charlotte population forecast

Forecasts the Charlotte-Concord-Gastonia, NC-SC metro area's population 5
years out, using two models (damped-trend ETS and ARIMA(1,1,1)) picked by
rolling backtest rather than by assumption. This was the first city this
repo's forecasting approach was built around — `florida/` and
`citywide_forecast/` reuse the same modeling code (`charlotte_5y.py`) for
other cities.

**Start here:** `charlotte_walkthrough.ipynb` runs the whole pipeline with
inline plots and commentary. `python charlotte_5y.py` does the same thing
from the command line.

## Files

| File | What it is |
|---|---|
| `charlotte_5y.py` | The current version — ETS/ARIMA backtest + 5-year forecast, with confidence intervals. Everything else in this repo that reuses "the Charlotte methodology" imports from this file. |
| `charlotte_walkthrough.ipynb` | Interactive walkthrough of `charlotte_5y.py` |
| `charlotte.py`, `charlotte_fixed.py`, `charlotte_forecast.py`, `charlotte_simple_forecast.py`, `charlotte_updated.py` | Earlier iterations, kept for reference (see `SOURCES.md` and inline comments for what each explored) |
| `charlotte_bls_employment.py` | Pulls Charlotte MSA employment history directly from the BLS API |
| `SOURCES.md` | Where the population/employment/permits series come from, and the metro-boundary-change caveat in the 2009-2010 data |

## Data

`charlotte_population.csv` / `charlotte_population_updated.csv` (annual
population), `charlotte_employment_monthly.csv` /
`charlotte_employment_all_years.csv` (BLS employment), and
`charlotte_permits_monthly.csv` (Census building permits) are all included
so the forecasts are reproducible without re-fetching anything.
`backtest_housing.png`, `decomposition.png`, `fan_chart.png`,
`model_comparison.png`, `backtest_mape.csv`, `forecast_2030.csv`, and
`housing_balance.csv` are saved outputs from earlier runs, kept for
reference.

## Setup

```bash
pip install pandas numpy matplotlib statsmodels requests
```

`charlotte_bls_employment.py` is the only script here that calls a live
API; it works without a key at reduced rate limits, or set `BLS_API_KEY`
(env var, or a local `secret.py` in this folder — see the repo root
README for how to get one).

## Run it

```bash
cd charlotte
python charlotte_5y.py
```
