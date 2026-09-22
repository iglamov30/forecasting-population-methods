# Population & City Forecasting

Forecasting US city population (and related signals — employment, housing
permits) with backtested time-series models, applied at several scales:
a single city, a handful of cities, and a ~96-city cross-sectional
benchmark. Everything here is meant to be reused: the data is included,
the code is plain enough to point at a different city, and two notebooks
walk through the methodology interactively.

## Where to start

- **New to this repo?** Open [`charlotte/charlotte_walkthrough.ipynb`](charlotte/charlotte_walkthrough.ipynb) — an interactive run-through of the core method (backtest ETS vs. ARIMA, forecast with the winner) on a single city.
- **Want the data-fetching code, not the forecasting?** [`population_matrix/`](population_matrix/) is a standalone toolkit for pulling any US city's population history from the Census API into a year x city matrix.
- **Want to see the method validated at scale?** [`model_benchmark/RESULTS.md`](model_benchmark/RESULTS.md) backtests 6 models across ~96 cities and reports what actually wins, with the caveats spelled out.

## Layout

| Folder | What's in it |
|---|---|
| [`charlotte/`](charlotte/) | The original single-city forecast (Charlotte, NC metro) — this is where the modeling approach was built, and everywhere else reuses it |
| [`florida/`](florida/) | The same approach applied to Florida's largest cities, plus an employment/permits/tourism composite investment screen |
| [`population_matrix/`](population_matrix/) | Reusable toolkit: pull Census ACS1 population data for any US city into a year x city matrix |
| [`citywide_forecast/`](citywide_forecast/) | The Charlotte methodology run across every US city ACS1 tracks (~550 places with population >= 65,000) |
| [`model_benchmark/`](model_benchmark/) | A rigorous backtest benchmark: 6 forecasting models, ensembles, prediction intervals, and a pooled cross-city model, ranked head-to-head across ~96 cities. Start with `RESULTS.md`. |
| [`real_estate_investment_thesis/`](real_estate_investment_thesis/) | Placeholder for turning these forecasts into an actual investment thesis |
| [`legacy/`](legacy/) | Older outputs kept for reference, not reproducible from current code |

## Setup

```bash
pip install -r requirements.txt
```

Most scripts need a free Census API key; a few also use a free BLS key.

- Census key: https://api.census.gov/data/key_signup.html
- BLS key: https://data.bls.gov/registrationEngine/

The simplest way to provide them: drop a `secret.py` file with your keys
into whichever folder you're running code from (every script checks for
one automatically):

```python
CENSUS_API_KEY = "your_key_here"
BLS_API_KEY = "your_key_here"       # optional
```

`secret.py` is gitignored — never committed. Each folder that needs one
manages its own local copy (this is a deliberate choice: every folder here
runs standalone, so you can copy just the one you need out of this repo
and it'll still work).

Alternatively, export the keys as real environment variables
(`CENSUS_API_KEY`, `BLS_API_KEY`) before running anything — every script
checks the environment first, before falling back to `secret.py`.
`.env.example` documents the variable names for this path, but note that
nothing here loads `.env` files automatically; you'd need to `export` them
yourself (or add a tool like `python-dotenv` if you want that automated).

Each subfolder's own README has the specifics for what it needs and how
to run it.

## License

MIT — see [`LICENSE`](LICENSE).
