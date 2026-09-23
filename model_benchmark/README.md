# Model Benchmark

Self-contained module, independent of the other forecasting scripts in this
repo. Builds a cross-city *ranking* of population forecasts for
~96 US cities with population >= 250,000, using Census ACS1 place-level data
(2005-present, 2020 missing).

## Setup

Requires `CENSUS_API_KEY` in the environment (this code never imports
`secret.py`):

```
export CENSUS_API_KEY=your_key_here      # bash
$env:CENSUS_API_KEY = "your_key_here"    # PowerShell
```

Dependencies: pandas, numpy, statsmodels, matplotlib, requests, pyarrow
(pyarrow is needed for the parquet cache and is the one dependency this
module adds beyond the rest of the repo).

## Run order

Each script reads from `model_benchmark/data/` and can be re-run independently
once its inputs exist; nothing needs to be re-fetched from the network on
a second run.

```
python model_benchmark/scripts/build_cache.py       # fetch + cache population panel
python model_benchmark/scripts/run_backtest.py      # expanding-window backtest, all 6 models
python model_benchmark/scripts/run_ensembles.py     # ensembles vs. best-model selection
python model_benchmark/scripts/run_intervals.py     # prediction intervals (slow: ~4 min, ETS simulation)
python model_benchmark/scripts/run_pooled_model.py  # pooled/shrunk AR growth model
```

`build_cache.py` and `run_backtest.py` must run before the others.
`run_intervals.py` reads `run_backtest.py`'s output; `run_pooled_model.py`
only needs the cached population panel.

## Layout

```
model_benchmark/
  core/census.py     generalized ACS1 fetcher (list of variables, not one hardcoded var)
  core/cache.py       per-year parquet cache
  models/             the 6-model zoo, common fit(series)/predict(fitted,h) interface
  eval/backtest.py     expanding-window harness (the no-leakage rule lives here)
  eval/ensembles.py    ensembles + leakage-free best-model selector
  eval/intervals.py    empirical + model-based prediction intervals
  eval/pooled_model.py pooled/shrunk AR growth model
  data/                parquet cache + all CSV result tables
  scripts/             one runnable script per step, in the order above
RESULTS.md             what won at each horizon, and the limitations behind it
```

## Where the no-leakage rule is enforced

Every model fit anywhere in this codebase only ever sees data up to and
including its origin year — see the `NO LEAKAGE` comments in
`eval/backtest.py` (the base case: the one place training data is sliced),
`eval/ensembles.py` (the best-model selector uses only strictly-earlier
origins), and `eval/pooled_model.py`
(the pooled regression and each city's local regression are both re-fit at
every origin using only data through that origin, and year fixed effects
are zeroed out when forecasting forward since a future year's shock is
unknown at forecast time).
