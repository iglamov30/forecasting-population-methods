# Florida

Population forecasts and an investment screen for Florida's five largest
cities: Jacksonville, Miami, Tampa, Orlando and St. Petersburg.

The forecasting method is the one built in [`../charlotte/`](../charlotte/):
damped-trend ETS and ARIMA(1,1,1) are each backtested 5 years ahead from a
rolling origin, and the model with the lower MAE makes the forecast. The
modeling functions are imported from `../charlotte/charlotte_5y.py`, so keep
`charlotte/` next to `florida/`.

**Start here:** [`florida_walkthrough.ipynb`](florida_walkthrough.ipynb) runs
the population forecast step by step. It needs no API key or network.

## Layout

```
florida/
  population_forecast.py     5-year population forecast per city
  population_momentum.py     cities ranked by risk-adjusted growth
  miami_1y_backtest.py       short-horizon backtest for one city
  composite_screen.py        hotel and multifamily scores from four signals
  florida_walkthrough.ipynb  notebook walkthrough of population_forecast.py

  sources/                   data loaders, one per source
    cities.py                the five cities, their counties and airports
    edr_population.py        population, Florida EDR workbook
    bls_employment.py        county employment, BLS LAUS API
    census_permits.py        housing permits, Census BPS files
    tourism.py               bed tax and airport passengers, hand-entered

  data/                      inputs and cached downloads
    templates/               blank CSVs for the hand-entered series
  outputs/                   results written by the scripts above
```

The top level has only things you run. Code that fetches or reads data is in
`sources/`, so each script is short and each data source is in one place.
`data/` holds inputs, including cached copies of downloads; `outputs/` holds
results. Everything in `outputs/` can be regenerated.

## Population forecast

| Script | What it does | Writes |
|---|---|---|
| `population_forecast.py` | Backtests ETS and ARIMA 5 years ahead for each city, then forecasts 5 years with the better model's confidence interval. | `outputs/population_forecast.csv` |
| `population_momentum.py` | Divides each city's forecast growth by its backtest error (RMSE as % of population) and ranks by that. A fast-growing city the models predict poorly can rank below a slower, steadier one. Growth is measured to a sixth year: the 5-year forecast is extended one more year by repeating the last year's growth ratio, with the interval widened by the same ratio, so this file ends a year later than `population_forecast.csv`. Same scoring as `../citywide_forecast/forecast_citywide_adj.py`. | `outputs/population_momentum.csv` |
| `miami_1y_backtest.py` | Backtest at a 1-year horizon, printing every window's error. Miami's EDR series drops from 497,924 (2020) to 449,747 (2021) because of the Census 2020 benchmark revision. At 5 years that break affects five backtest windows; at 1 year it affects one, which shows how the models do on clean data. | prints only |

Population comes from `data/FLmupops.xlsx`: the BEBR municipal estimates
(University of Florida), published by the Office of Economic & Demographic
Research (EDR). Every incorporated municipality, 1979 to date. The workbook
has no sheet for the decennial years 1980, 1990, 2000 and 2010, so those four
values are linearly interpolated on load (`_fill_census_years` in
`sources/edr_population.py`); the other 43 are published estimates. Census
ACS1 was used before, but it starts in 2005 and only covers places above
65,000 people.

```bash
python population_forecast.py
python population_forecast.py --start-year 2010 --alpha 0.10 --no-show
python population_momentum.py
python miami_1y_backtest.py --city Tampa --horizon 2
```

## Composite investment screen

`composite_screen.py` forecasts four signals per city from 2010, scores each
as forecast growth divided by backtest error, and combines them with two
weight sets:

| Signal | Source | Geography | Hotel weight | Multifamily weight |
|---|---|---|---|---|
| Population | EDR municipal estimates | City | 0.25 | 0.30 |
| Employment | BLS LAUS, annual average of monthly data | County | 0.25 | 0.30 |
| Housing permits | Census Building Permits Survey place files, 2007+ | City | 0.10 | 0.35 |
| Tourism | Tourist Development Tax collections and airport enplanements | County / airport | 0.40 | 0.05 |

The weights are starting points, set at the top of the script. If a city has
no data for a signal, that signal is dropped and the other weights are
rescaled to sum to 1. The results are written to
`outputs/composite_scores.csv`.

Population, employment and permits load automatically. Tourism has to be
entered by hand, because each county publishes bed tax collections in its own
format. EDR estimates realized collections for all 67 counties in one
workbook, which is the fastest way to fill the file in; links to it and to
each county's own numbers are at the top of `sources/tourism.py`.

```bash
cp data/templates/tdt_template.csv data/tdt_manual.csv
cp data/templates/enplanements_template.csv data/enplanements_manual.csv
# then fill them in
```

Until those files exist, the screen scores without tourism and says so. If
the permits download fails, it falls back to `data/permits_manual.csv` (same
template approach). Any loader can be replaced, for example with STR, CoStar
or AirDNA data, as long as it returns rows of `year, place_id, value`.

## Refreshing the data

Each loader can be run from inside `florida/` to refresh its cached CSV.
Employment and permits download; population reads the local workbook, and
tourism only checks the hand-entered files:

```bash
python -m sources.edr_population   # data/population_top5.csv
python -m sources.bls_employment   # data/employment_annual.csv
python -m sources.census_permits   # data/permits_annual.csv
python -m sources.tourism          # checks the hand-entered files
```

To update population, download a new `FLmupops.xlsx` from
[EDR](https://edr.state.fl.us/Content/population-demographics/data/FLmupops.xlsx)
into `data/` first.

Only the BLS API uses a key, and it also works without one at a lower rate
limit. Set `BLS_API_KEY` in the environment or in a gitignored `secret.py` in
this folder ([free registration](https://data.bls.gov/registrationEngine/)).

## Limitations

- The composite screen has not been checked against outside data. Treat its
  scores as a first pass.
- Employment is by county, and population and permits are by city.
  Jacksonville is consolidated with Duval County, so they match there. Miami
  is about a sixth of Miami-Dade, so its employment signal mostly reflects
  the rest of the county.
- St. Petersburg is matched to Tampa's airport (TPA). St. Pete-Clearwater
  (PIE) is not tracked.
- Not included: short-term rental supply (no free source) and insurance cost
  or hurricane exposure.
