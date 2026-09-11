# Florida 5-City Composite Investment Screen

Extends the existing Charlotte/nationwide forecasting pipeline to the 5
largest Florida cities (Jacksonville, Miami, Tampa, Orlando, St. Petersburg),
adding employment, housing permits, and tourism as signals, and scoring
each city separately for hotel and multifamily relevance.

## Files

| File | What it does | Status |
|---|---|---|
| `fl_cities.py` | Registry of the 5 cities + county/airport crosswalk | Ready, no dependencies |
| `census_devs.py`, `charlotte_5y.py` | Unchanged copies of your existing modules (population fetch, ETS/ARIMA fit+backtest) | Ready |
| `bls_employment.py` | County employment via BLS LAUS API | Ready — needs a free BLS key (works without one at low rate limits) |
| `census_permits.py` | Housing permits via Census EITS/BPS API, with manual fallback | Best-effort automated + manual fallback (see below) |
| `tourism_signals.py` | TDT (bed tax) + airport enplanements loaders | **Manual data entry required** (see below) |
| `florida_composite_forecast.py` | Main script: forecasts all 4 signals per city, builds hotel & multifamily composite scores, prints table + bar chart | Ready to run once inputs are in place |

Population-only forecasting for the same cities (plus Jacksonville) has
since moved to a second, self-contained pipeline sourced from the state's
own EDR municipal estimates instead of Census ACS1 (40+ years of history vs.
ACS1's ~19): `edr_population.py` (data access), `florida_top5_forecast.py`
(backtest + forecast, reuses `charlotte_5y.py`'s modeling code),
`florida_top5_momentum.py` (a momentum-based variant), and
`florida_walkthrough.ipynb` (an interactive walkthrough of
`florida_top5_forecast.py` — start here if you just want to see it run).

## Honest limitations — read this before trusting the output

This hasn't been run end-to-end against live Census/BLS APIs yet, so run
it in your own environment first and sanity-check the printed numbers
before using them in anything client-facing.

Two of the four signals are **not** cleanly available through a free,
standardized API the way ACS1 population is:

- **Housing permits**: `census_permits.py` first tries the Census
  EITS/BPS API at the CBSA (metro) level, which is real and reliable, but
  is metro-wide, not city-proper — that's actually probably the *better*
  geography for permits anyway (see conversation above), but be aware
  it's not apples-to-apples with the city-level population/employment
  numbers. If that call fails or you want city/place-level precision
  instead, fill in `permits_template.csv` by hand from
  https://www.census.gov/construction/bps/ and save it as
  `permits_manual.csv`.

- **Tourism (TDT + airport enplanements)**: there is no unified API for
  county Tourist Development Tax collections — every FL county
  publishes these differently. Fill in `tdt_template.csv` and
  `enplanements_template.csv` by hand (source links are in each file and
  in `tourism_signals.py`'s docstring) and save them as `tdt_manual.csv`
  / `enplanements_manual.csv`. The script runs fine without these —
  it'll just print a warning and exclude tourism from the composite score
  until you add the data.

If your firm already licenses STR (hotel occupancy/RevPAR), CoStar, or
AirDNA data, swap those in — `tourism_signals.py` and `census_permits.py`
are written so any loader can be replaced as long as it returns the same
`[year, place_id, value]` shape; the scoring logic downstream doesn't care
where the numbers came from.

## Setup

```bash
pip install pandas numpy matplotlib requests statsmodels

# Census key (you likely already have this)
export CENSUS_API_KEY=your_key

# BLS key (free, 2 min to register: https://data.bls.gov/registrationEngine/)
export BLS_API_KEY=your_key
```

Or add both to a `secret.py` file in this folder:
```python
CENSUS_API_KEY = "..."
BLS_API_KEY = "..."
```

## Run it

```bash
# sanity check the city registry (no network needed)
python fl_cities.py

# test each data source independently
python bls_employment.py
python census_permits.py
python tourism_signals.py    # will just print "no manual file found" until you fill in the CSVs

# full pipeline
python florida_composite_forecast.py
```

## How the scoring works

Each of the 4 signals (population, employment, permits, tourism) is
forecast independently 5 years out using the same ETS-vs-ARIMA
backtest-and-pick-best logic as your existing `all_cities_forecast.py`,
then risk-adjusted as `pct_growth / backtest_RMSE_as_pct_of_latest_value` —
same idea as `forecast_citywide_adj.py`'s `risk_adjusted_score`, just
computed per metric instead of only for population.

Two weighted composites combine these per city:

```python
HOTEL_WEIGHTS        = {"population": 0.25, "employment": 0.25, "permits": 0.10, "tourism": 0.40}
MULTIFAMILY_WEIGHTS  = {"population": 0.30, "employment": 0.30, "permits": 0.35, "tourism": 0.05}
```

These are starting points, not gospel — adjust them at the top of
`florida_composite_forecast.py`. If a signal is missing for a city (e.g.
you haven't filled in TDT data yet), the weights are renormalized over
whatever's actually available rather than silently treating the missing
signal as zero.

## Not yet built (worth flagging, didn't want to fake it)

- **Short-term-rental (Airbnb/VRBO) penetration** as a multifamily supply
  risk penalty — no reliable free source (AirDNA is paid); left out
  rather than approximated badly.
- **Insurance cost / hurricane risk overlay** — genuinely FL-specific risk
  that isn't in any of these series; worth adding as a qualitative flag
  per city before this goes in front of an investment committee.
- **City-proper vs. metro geography mismatch** for permits (see above) —
  fine for a first pass, worth reconciling before this is decision-grade.
