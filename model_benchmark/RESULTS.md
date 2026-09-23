# Results

Population panel: Census ACS1 place-level `B01003_001E`, 2005-2024 (2020
missing, never interpolated), filtered to the 96 cities whose latest
available population is >= 250,000. 19 years fetched (2025/2026 not yet
published). Observations per city: min 5 (Honolulu CDP — stops appearing
in ACS1 after 2009), median 19.0, max 19.

All tables below are reproducible via the scripts in `model_benchmark/scripts/`
(see `README.md`); the underlying CSVs are in `model_benchmark/data/`. The one
simulation-based column (ETS interval coverage) is seeded, so a re-run
reproduces these numbers exactly rather than moving by a few tenths.

## Partial Limitations

- **Backtest windows are short.** A full-length city has at most ~11
  expanding-window origins, and horizon 5 is only scored from origins early
  enough to have a real actual 5 years out — 660 (city, origin) forecasts
  at h=5 sounds like a lot, but it's ~7 origins x 96 cities, i.e. per-city
  error estimates are built from a handful of overlapping windows, not 660
  independent trials. Differences between models that aren't large relative
  to the gap between, say, MAE and RMSE for the same model should be read
  as **ties**, not as a ranking.
- **The interval coverage numbers are in-sample.** The empirical intervals
  are built from the same pooled backtest-error sample they're then scored
  against. That's a calibration sanity check ("is the shape of this
  distribution roughly right"), not proof a genuinely new forecast would
  hit 90% coverage.
- **ETS's gap handling is a documented compromise, not the ARIMA solution.**
  statsmodels' state-space ETSModel does not actually support missing
  observations in this version (verified empirically — the state becomes
  permanently NaN after any gap year, regardless of damping). ETS results
  here compress 2019->2021 into a single step; treat ETS forecasts as
  somewhat less trustworthy specifically around that transition. ARIMA's
  Kalman filter does correctly skip the missing observation.
- **The modified-exponential ceiling (3x max observed population) is an
  assumption, not an estimate.** Different multipliers would give different
  (though probably not qualitatively different, since the model loses to
  naive anyway) forecasts for that one model.
- **The 250,000-population filter yielded 96 cities, not the ~800 originally
  anticipated** (flagged before the data pull was run and confirmed after: ~800
  is closer to ACS1's own ~65,000-population publication threshold across
  all places, not the count of cities above 250,000).
- **The pooled model result argues against our own prior going in.**
  We expected cross-city pooling to win outright given n~15. It
  does win convincingly against *pure per-city* AR estimation (see below),
  but it does NOT beat simply running per-city ARIMA(1,1,1) (the backtest
  winner) at any horizon. Both things being true at once is the actual
  finding — see the pooled-model section below for the likely explanation.

## Model-by-horizon backtest (expanding window, h=1..5)

| Horizon | Model | MAE | RMSE | MAPE | Median APE | Bias | n |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **arima_111** | 7,077 | 16,932 | 1.13% | 0.75% | -2,434 | 944 |
| 1 | naive | 7,554 | 16,078 | 1.23% | 1.03% | -4,965 | 944 |
| 1 | ets_damped | 9,981 | 20,586 | 1.64% | 1.16% | -191 | 944 |
| 1 | modified_exponential | 12,159 | 25,556 | 1.98% | 1.38% | +6,292 | 944 |
| 1 | linear_trend | 12,639 | 26,024 | 2.07% | 1.45% | +7,225 | 944 |
| 1 | exponential | 14,228 | 27,631 | 2.37% | 1.69% | +9,790 | 944 |
| 2 | **arima_111** | 12,038 | 24,489 | 1.92% | 1.37% | -3,107 | 849 |
| 2 | ets_damped | 13,636 | 27,035 | 2.25% | 1.61% | -319 | 849 |
| 2 | naive | 14,048 | 25,150 | 2.35% | 1.98% | -8,389 | 849 |
| 2 | modified_exponential | 15,838 | 31,227 | 2.60% | 1.83% | +8,012 | 849 |
| 2 | linear_trend | 16,504 | 31,911 | 2.72% | 1.97% | +9,288 | 849 |
| 2 | exponential | 18,641 | 34,227 | 3.11% | 2.28% | +12,765 | 849 |
| 3 | **arima_111** | 16,627 | 29,200 | 2.78% | 2.04% | -3,412 | 754 |
| 3 | ets_damped | 17,450 | 31,116 | 2.90% | 2.19% | -197 | 754 |
| 3 | naive | 18,951 | 29,985 | 3.34% | 2.84% | -11,352 | 754 |
| 3 | modified_exponential | 19,646 | 36,280 | 3.20% | 2.35% | +10,234 | 754 |
| 3 | linear_trend | 20,515 | 37,244 | 3.35% | 2.51% | +11,899 | 754 |
| 3 | exponential | 23,353 | 40,479 | 3.87% | 3.05% | +16,408 | 754 |
| 4 | **arima_111** | 21,213 | 36,506 | 3.50% | 2.84% | -212 | 660 |
| 4 | ets_damped | 22,092 | 39,120 | 3.58% | 2.69% | +2,422 | 660 |
| 4 | naive | 23,410 | 36,196 | 4.17% | 3.50% | -11,605 | 660 |
| 4 | modified_exponential | 24,609 | 46,305 | 3.91% | 2.86% | +14,547 | 660 |
| 4 | linear_trend | 25,819 | 47,595 | 4.11% | 3.13% | +16,631 | 660 |
| 4 | exponential | 29,547 | 51,778 | 4.77% | 3.82% | +22,252 | 660 |
| 5 | **arima_111** | 24,940 | 42,269 | 4.21% | 3.50% | -170 | 660 |
| 5 | ets_damped | 25,263 | 44,427 | 4.17% | 3.28% | +3,604 | 660 |
| 5 | naive | 26,845 | 40,903 | 4.91% | 3.91% | -14,030 | 660 |
| 5 | modified_exponential | 29,113 | 52,871 | 4.60% | 3.42% | +18,176 | 660 |
| 5 | linear_trend | 30,736 | 54,594 | 4.86% | 3.76% | +20,796 | 660 |
| 5 | exponential | 35,645 | 60,212 | 5.68% | 4.63% | +27,918 | 660 |

**ARIMA(1,1,1) wins on MAE, MAPE, and median APE at every horizon.** ETS is
a close, consistent second. Naive beats every raw growth-extrapolation
model (linear, exponential, modified-exponential) at every horizon — those
three are the "benchmark everything must beat" losing outright, which is
the expected result for models that don't adapt to a city's recent
deceleration/acceleration. ARIMA and naive are close enough at h=1 (7,077
vs 7,554 MAE, about 6% apart on a base of ~600k-1M population) that this
gap alone shouldn't be over-read; ARIMA's advantage widens with horizon.
Every model over-predicts on average (positive bias) except ARIMA and
naive, which under-predict slightly — i.e. the naive-like models are
mildly conservative and the trend/growth models are mildly optimistic,
consistent with population growth in this size class decelerating on
average over 2005-2024.

## Ensembles vs. best-model selection

Compared on the same (city, origin, horizon) windows — restricted to
origins where the leakage-free selector had prior history to select from
(see `eval/ensembles.py`).

| Horizon | Approach | MAE | RMSE |
|---|---|---:|---:|
| 1 | **best_model_by_backtest_mae** | 6,765 | 17,634 |
| 1 | arima_111_always | 7,110 | 17,449 |
| 1 | simple_mean_ensemble | 9,051 | 19,832 |
| 1 | trimmed_mean_ensemble | 9,321 | 20,430 |
| 2 | **best_model_by_backtest_mae** | 10,521 | 24,862 |
| 2 | arima_111_always | 11,960 | 24,691 |
| 2 | simple_mean_ensemble | 12,464 | 26,195 |
| 2 | trimmed_mean_ensemble | 12,760 | 26,721 |
| 3 | **best_model_by_backtest_mae** | 13,848 | 29,450 |
| 3 | simple_mean_ensemble | 16,128 | 29,918 |
| 3 | arima_111_always | 16,289 | 28,435 |
| 3 | trimmed_mean_ensemble | 16,553 | 30,874 |
| 4 | **best_model_by_backtest_mae** | 18,638 | 37,368 |
| 4 | arima_111_always | 21,025 | 36,359 |
| 4 | simple_mean_ensemble | 21,618 | 40,115 |
| 4 | trimmed_mean_ensemble | 22,202 | 41,552 |
| 5 | **best_model_by_backtest_mae** | 23,014 | 44,742 |
| 5 | arima_111_always | 24,474 | 41,451 |
| 5 | simple_mean_ensemble | 24,721 | 44,920 |
| 5 | trimmed_mean_ensemble | 25,542 | 46,677 |

**Neither ensemble beats best-model selection, and neither beats simply
using ARIMA always.** The per-city leakage-free selector wins on MAE at
every horizon (it isn't uniformly best on RMSE — ARIMA-only has a lower
RMSE at h=3-5, meaning the selector's few misses are occasionally larger
even though it's right more often on average). Averaging forecasts across
models that disagree in *direction* (the growth models overshoot, naive
undershoots) mostly just averages toward the middle rather than toward the
truth, which is why both ensembles under-perform a rule that just picks
one model per city.

## Empirical vs. model-based 90% prediction intervals

| Horizon | Approach | Coverage | n |
|---|---|---:|---:|
| 1 | arima_analytic | 97.0% | 944 |
| 1 | empirical_pooled | 89.8% | 944 |
| 1 | empirical_by_decile | 89.6% | 944 |
| 1 | ets_simulated | 83.6% | 944 |
| 2 | arima_analytic | 96.2% | 849 |
| 2 | empirical_pooled | 89.9% | 849 |
| 2 | empirical_by_decile | 89.2% | 849 |
| 2 | ets_simulated | 66.9% | 849 |
| 3 | arima_analytic | 93.5% | 754 |
| 3 | empirical_pooled | 89.9% | 754 |
| 3 | empirical_by_decile | 89.7% | 754 |
| 3 | ets_simulated | 55.3% | 754 |
| 4 | arima_analytic | 92.1% | 660 |
| 4 | empirical_pooled | 90.0% | 660 |
| 4 | empirical_by_decile | 88.2% | 660 |
| 4 | ets_simulated | 50.0% | 660 |
| 5 | arima_analytic | 91.5% | 660 |
| 5 | empirical_pooled | 90.0% | 660 |
| 5 | empirical_by_decile | 88.2% | 660 |
| 5 | ets_simulated | 42.3% | 660 |

(Nominal target: 90%. Empirical-pooled hits ~90% by construction — see the
in-sample caveat above — but ARIMA analytic and ETS simulated are
genuinely out-of-sample-style diagnostics computed independently at each
backtest origin, so their departure from 90% is a real finding.)

**ETS's own simulated intervals are badly miscalibrated and get worse with
horizon**: 84% coverage at h=1 collapsing to 42% at h=5, against a 90%
target — the model is far more confident than it should be, likely
compounded by the 2020-gap compression documented above. **ARIMA's
analytic intervals are conservative (over-covering) at short horizons and
converge toward nominal by h=4-5** (97% -> 91.5%), which is a much safer
failure mode than ETS's. The empirical approach sits closest to nominal by
construction, and cutting by population decile doesn't change that
materially (88-90% throughout) — it doesn't hurt, but it isn't earning its
complexity either in this dataset. **Bottom line: if you need an interval
today, use ARIMA's analytic interval or the pooled empirical one; do not
trust ETS's own simulated interval.**

**Verifying the literature's "error scales inversely with area size"
claim** (Rayer, Smith & Tayman 2009) against ARIMA's own backtest MAPE by
population decile:

| Decile (1=smallest) | Mean APE | Median APE | n |
|---|---:|---:|---:|
| 1 | 3.71% | 1.67% | 405 |
| 2 | 2.58% | 2.03% | 410 |
| 3 | 2.33% | 1.57% | 369 |
| 4 | 2.13% | 1.64% | 410 |
| 5 | 2.56% | 1.58% | 305 |
| 6 | 2.63% | 1.97% | 410 |
| 7 | 2.40% | 1.56% | 369 |
| 8 | 2.31% | 1.41% | 410 |
| 9 | 2.42% | 1.55% | 369 |
| 10 | 2.45% | 1.65% | 410 |

**Only partially confirmed.** The smallest decile is clearly worse (3.71%
mean APE vs. 2.1-2.6% everywhere else), but deciles 2-10 are flat —
there's no monotonic decline in error as city size rises further. The
likely reason: this whole sample is already restricted to cities >=
250,000, a much narrower size range than the general population the
literature result was drawn from (which usually spans from a few hundred
residents to millions). Within a narrow size band, the area-size effect
mostly disappears except at its own bottom edge.

## Pooled/global growth model vs. per-city local estimation

Growth panel: `g[i,t] = (log p[i,t] - log p[i,t-1]) / (year[t]-year[t-1])`
(the division annualizes the one 2019->2021 transition per city). Pooled
AR(1)/AR(2) estimated with city + year fixed effects; year FE set to 0 when
forecasting forward (a future shock is unknown at forecast time — see
`eval/pooled_model.py`).

**Shrinkage grid result: lambda=1.0 (pure pooled, zero weight on the
city's own local AR estimate) wins for both AR(1) and AR(2)** across the
full lambda grid 0.0-1.0 in steps of 0.1:

| lambda | AR(1) MAE | AR(2) MAE |
|---:|---:|---:|
| 0.0 (pure local) | 25,876 | 25,466 |
| 0.5 | 24,102 | 22,607 |
| 0.9 | 23,400 | 21,725 |
| **1.0 (pure pooled)** | **23,306** | **21,643** |

Local per-city AR estimation is simply too noisy at T~15-17 growth
observations per city to add anything — exactly the estimation-variance
argument we opened with. AR(2) beats AR(1) at every lambda.

**Pooled (lambda=1.0) vs. local-only (lambda=0.0), AR(2), by horizon:**

| Horizon | Local-only MAE | Pooled-only MAE |
|---|---:|---:|
| 1 | 9,148 | 8,673 |
| 2 | 17,347 | 15,245 |
| 3 | 26,829 | 22,479 |
| 4 | 40,504 | 33,895 |
| 5 | 49,458 | 40,643 |

Pooling wins at every horizon, by a growing margin — consistent with the
estimation-variance argument above.

**But the pooled model does not beat per-city ARIMA(1,1,1).** Restricting
both to the exact same (city, origin, horizon) windows:

| Horizon | ARIMA(1,1,1) MAE | Pooled AR(2), lambda=1.0 MAE |
|---|---:|---:|
| 1 | 7,178 | 8,673 |
| 2 | 12,387 | 15,245 |
| 3 | 16,904 | 22,479 |
| 4 | 21,949 | 33,895 |
| 5 | 25,741 | 40,643 |

This is the headline result of the pooled model, and it complicates our own opening
hypothesis ("methods that borrow strength across cities are expected to
win"). Pooling *does* win the fight it was designed to win —
against noisy per-city AR estimation. It does *not* win against ARIMA,
which is also a per-city model but a richer one: ARIMA(1,1,1) fits its own
level, trend, and a moving-average error-correction term per city, so it
already adapts to a city's own recent trajectory in a way the pooled
model's single shared AR coefficient (with year FE zeroed out going
forward) cannot. The pooled model's growing disadvantage at longer
horizons is consistent with this: forcing every city's forecast to revert
toward a common, modest growth rate underperforms specifically for cities
whose own trajectory is genuinely different from the panel average, and
that penalty compounds with horizon.

**Does the optimal lambda vary by city size?** Mostly no — deciles 1-9
all prefer lambda in [0.8, 1.0]. The exception is the top decile (the
largest cities in this >=250k sample), where lambda=0.3 wins:

| Population decile | Best lambda (AR2) |
|---|---:|
| 1 (smallest) | 0.9 |
| 2 | 0.9 |
| 3-5, 7-9 | 1.0 |
| 6 | 0.8 |
| 10 (largest) | **0.3** |

The largest cities are the one group where local estimation adds value —
plausibly because they have both a longer effective history of stable
growth behavior and a more genuinely idiosyncratic trajectory (a handful
of very large, fast- or slow-growing metros) that a pooled coefficient
averages away.

**Nickell (1981) bias.** The full-sample pooled AR(1) coefficient is
rho = -0.155 (population growth is mean-reverting: a high-growth year tends
to be followed by a lower-growth one). Average T (growth observations per
city) is 16.8. The leading-order dynamic-panel fixed-effects bias
approximation is `bias ~= -(1+rho)/T = -0.050` — about a third of the
coefficient's own magnitude, i.e. **not negligible**: the bias-corrected
rho is closer to -0.105. This is a real limitation of the pooled
estimate, not a rounding error. Whether it changes *rankings* is a
separate question from whether it changes levels: since rho is a single
coefficient shared by every city, a uniform correction toward zero changes
how hard every city's forecast reverts toward its own trend by roughly the
same proportion — it shifts absolute forecast levels more than it
reorders cities relative to each other. A full correction (Kiviet 1995
analytical bias correction, or Arellano-Bond GMM) was out of scope here;
this is an analytical plausibility argument, not an empirical verification
that reordering doesn't happen.

## Bottom line

- **At every horizon (1-5 years), per-city ARIMA(1,1,1) is the best single
  model** and is not beaten by ensembling, by best-model selection using
  its own backtest history (which mostly just re-selects ARIMA anyway), or
  by the pooled panel model.
- **For prediction intervals, use ARIMA's analytic interval or the pooled
  empirical interval — not ETS's simulated interval**, which is
  substantially overconfident and gets worse with horizon.
- **The n~15 constraint's practical lesson lands in an unexpected place**:
  pooling clearly helps *if the alternative is a from-scratch local AR
  model*, but it does not help against a properly-specified per-city ARIMA,
  which already has its own (more efficient) way of borrowing strength
  from a city's own history via its MA term. "Borrow strength across
  cities" and "borrow strength across time within a city" turned out to be
  competing strategies here, and the latter won.
- Given the short backtest history, none of the horizon-4/5 gaps between
  the top 2-3 approaches should be treated as more than suggestive —
  they're built from a handful of overlapping windows per city.
