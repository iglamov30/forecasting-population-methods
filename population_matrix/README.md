# Population matrix toolkit

A small, reusable toolkit for pulling annual population data for any US
city (or all of them) from the Census ACS1 API and shaping it into a
year x city matrix. This is the piece to reuse if you just want the data,
not the forecasting -- `charlotte/` and `citywide_forecast/` build
forecasts on top of the same `census_devs.py` functions.

## Files

| File | What it does |
|---|---|
| `census_devs.py` | `build_panel(years)` fetches Census ACS1 place-level population for every place in the US for the given years; `city_series`/`list_cities` slice the result |
| `matrix_test.py` | Turns a panel into a `year x city` matrix (`population_matrix()`), sorted by city size |
| `test_census.py` | Smoke test — confirms your API key works and a panel builds correctly |
| `city_population_matrix.csv` | Example output: a matrix for every US place ACS1 publishes, 2005-present |

## Setup

```bash
pip install pandas numpy requests
```

You need a free Census API key: sign up at
https://api.census.gov/data/key_signup.html, then either

```bash
export CENSUS_API_KEY=your_key_here      # bash
$env:CENSUS_API_KEY = "your_key_here"    # PowerShell
```

or drop it in a local `secret.py` in this folder:

```python
CENSUS_API_KEY = "your_key_here"
```

## Use it

```bash
cd population_matrix
python test_census.py       # confirms your key works
python matrix_test.py       # builds the full matrix, saves city_population_matrix.csv
```

Or from your own script:

```python
from census_devs import build_panel, city_series, list_cities
from matrix_test import population_matrix

# every US place ACS1 tracks, 2005-present, as a year x city matrix
matrix = population_matrix()

# or work with the raw panel directly
panel = build_panel(years=range(2010, 2025))
years, pop = city_series(panel, place_id="3712000")  # Charlotte, NC
```

Note: ACS1 only publishes for places with population >= 65,000, and has
no 2020 data (the survey was suspended that year).
