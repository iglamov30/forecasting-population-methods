"""
Charlotte Population Forecast to 2030 — population-only version
================================================================
Requires only charlotte_population.csv (date, pop_thousands).
Uses three models: local-linear-trend Kalman, damped-Holt exponential
smoothing, and demographic accounting (NI schedule + AR(1) migration).
Outputs a fan chart and forecast_2030_simple.csv.

Run:  python charlotte_simple_forecast.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.holtwinters import Holt

RNG   = np.random.default_rng(16740)
H_END = 2030
N_SIMS = 20_000

# -----------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------

def load_population(path="charlotte_population.csv"):
    """
    Expected columns: date (YYYY-MM-DD), pop_thousands
    Returns a pd.Series indexed by integer year (July-1 snapshots).
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["pop"] = df["pop_thousands"] * 1_000.0

    new = df[df.year >= 2010].set_index("year")["pop"]
    old = df[df.year <= 2009].set_index("year")["pop"]

    # Splice the MSA-definition break at 2009->2010
    g_old = old.pct_change().dropna()
    g_bridge = np.mean([g_old.iloc[-1], new.pct_change().iloc[0]])
    levels = {2010: new.iloc[0], 2009: new.iloc[0] / (1 + g_bridge)}
    for y in sorted(g_old.index, reverse=True):
        if y - 1 < old.index.min():
            break
        levels[y - 1] = levels[y] / (1 + g_old.loc[y])

    spliced = pd.Series(levels).sort_index()
    pop = pd.concat([spliced.loc[:2009], new]).rename("pop")
    pop.index.name = "year"
    return pop


# -----------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------

def m1_state_space(pop, t0, t1):
    """Kalman local-level model on log growth rates."""
    g = np.log(pop.loc[:t0]).diff().dropna()
    mod = UnobservedComponents(g, level="local level")
    res = mod.fit_constrained(
        {"sigma2.irregular": 0.35 * g.var()}, disp=False
    )
    g_f = res.forecast(t1 - t0)
    levels = pop.loc[t0] * np.exp(np.cumsum(g_f.values))
    return pd.Series(levels, index=range(t0 + 1, t1 + 1))


def m2_damped_holt(pop, t0, t1):
    """Damped-trend exponential smoothing on log levels."""
    fit = Holt(
        np.log(pop.loc[:t0]),
        damped_trend=True,
        initialization_method="estimated",
    ).fit(optimized=True)
    return pd.Series(
        np.exp(fit.forecast(t1 - t0).values),
        index=range(t0 + 1, t1 + 1),
    )


def ni_rate(year):
    """Natural increase per 1,000 residents (slowly declining schedule)."""
    base = 5.5 - 0.18 * (year - 2011)
    if year in (2021, 2022):
        base -= 0.9
    return max(base, 0.5)


def m4_demographic(pop, t0, t1, mig_adjust=0.0):
    """NI schedule + AR(1) mean-reverting net migration."""
    sub = pop.loc[2011:t0]
    ni = pd.Series(
        {y: ni_rate(y) / 1000 * pop.loc[y - 1] for y in sub.index}
    )
    mig = sub.diff().fillna(sub.iloc[0] - pop.loc[2010]) - ni
    w = 0.85 ** np.arange(len(mig) - 1, -1, -1)
    mu = np.average(mig.values, weights=w)
    x, y_ = mig.values[:-1] - mu, mig.values[1:] - mu
    rho = float(np.clip((x @ y_) / max(x @ x, 1e-9), 0.0, 0.95))
    level, m_prev = pop.loc[t0], mig.iloc[-1]
    out = {}
    for t in range(t0 + 1, t1 + 1):
        m = (mu + mig_adjust) + rho * (m_prev - (mu + mig_adjust))
        level = level + ni_rate(t) / 1000 * level + m
        out[t] = level
        m_prev = m
    return pd.Series(out)


# -----------------------------------------------------------------------
# ENSEMBLE + MONTE CARLO
# -----------------------------------------------------------------------

def ensemble(pop, t0=2025, t1=H_END):
    P = pd.DataFrame({
        "M1_state_space": m1_state_space(pop, t0, t1),
        "M2_damped_holt": m2_damped_holt(pop, t0, t1),
        "M4_demographic": m4_demographic(pop, t0, t1),
    })
    # Equal weights (no backtest data; use 1/3 each)
    point = P.mean(axis=1)
    return point, P


def fan_chart(pop, point, t0=2025, n_sims=N_SIMS):
    g = np.log(pop.loc[t0])
    g_hist = np.log(pop).diff().dropna()
    sigma = g_hist.std()           # unconditional vol of log growth
    h = len(point)
    shocks = RNG.standard_t(df=5, size=(n_sims, h))
    shocks *= sigma / np.sqrt(5 / 3)
    paths = pop.loc[t0] * np.exp(
        np.cumsum(
            point.values[None, :] / pop.loc[t0]  # dummy; replaced below
        )
    )
    # Proper: draw log-growth paths around the ensemble point path
    g_point = np.log(
        pd.concat([pop.loc[[t0]], point])
    ).diff().dropna().values
    paths = pop.loc[t0] * np.exp(
        np.cumsum(g_point[None, :] + shocks, axis=1)
    )
    q = np.percentile(paths, [10, 25, 50, 75, 90], axis=0)
    return pd.DataFrame(
        q.T,
        index=point.index,
        columns=["P10", "P25", "P50", "P75", "P90"],
    ), paths


# -----------------------------------------------------------------------
# SCENARIOS
# -----------------------------------------------------------------------

def scenarios(pop, point, t0=2025):
    base = m4_demographic(pop, t0, H_END)
    frz  = m4_demographic(pop, t0, H_END, mig_adjust=-12_000)
    hi   = m4_demographic(pop, t0, H_END, mig_adjust=+8_000)
    return {
        "immigration_freeze": point * (frz / base),
        "high_growth":        point * (hi  / base),
    }


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    pop = load_population()
    t0 = pop.index.max()        # last observed year (e.g. 2025)

    print(f"Loaded {len(pop)} annual observations  ({pop.index.min()}–{t0})")
    print(f"Latest population: {pop.loc[t0]:,.0f}")

    point, P = ensemble(pop, t0=t0)
    bands, paths = fan_chart(pop, point, t0=t0)
    scen = scenarios(pop, point, t0=t0)

    print("\n--- Model-by-model 2030 ---")
    for col in P.columns:
        print(f"  {col:20s}: {P.loc[H_END, col]:,.0f}")
    print(f"\nEnsemble point forecast 2030 : {point.loc[H_END]:,.0f}")
    print(f"Growth {t0}→2030            : "
          f"{point.loc[H_END]/pop.loc[t0]-1:+.2%}  "
          f"(CAGR {(point.loc[H_END]/pop.loc[t0])**0.2-1:+.2%})")

    print("\n--- Fan chart (Monte Carlo) ---")
    print(bands.round(0).astype(int).to_string())

    print("\n--- Scenarios (2030) ---")
    for k, v in scen.items():
        print(f"  {k:20s}: {v.loc[H_END]:,.0f}  "
              f"({v.loc[H_END]/pop.loc[t0]-1:+.2%} vs {t0})")

    # --- save ---
    out = bands.copy()
    out.insert(0, "point", point.round(0))
    for k, v in scen.items():
        out[k] = v.round(0)
    out.to_csv("forecast_2030_simple.csv")
    print("\nSaved forecast_2030_simple.csv")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    hist_yrs = pop.index[pop.index >= 2010]
    ax.plot(hist_yrs, pop.loc[hist_yrs] / 1e6, "k-", lw=2,
            label="Actual (Census PEP)")
    bridge = pd.concat([pop.loc[[t0]], point])
    yrs_f  = bridge.index
    b10 = pd.concat([pop.loc[[t0]], bands["P10"]])
    b90 = pd.concat([pop.loc[[t0]], bands["P90"]])
    b25 = pd.concat([pop.loc[[t0]], bands["P25"]])
    b75 = pd.concat([pop.loc[[t0]], bands["P75"]])
    ax.fill_between(yrs_f, b10 / 1e6, b90 / 1e6,
                    alpha=.18, color="tab:blue", label="P10–P90")
    ax.fill_between(yrs_f, b25 / 1e6, b75 / 1e6,
                    alpha=.30, color="tab:blue", label="P25–P75")
    ax.plot(yrs_f, bridge / 1e6, "b--o", lw=2, ms=4,
            label="Ensemble point")
    colors = {"immigration_freeze": "tab:red", "high_growth": "tab:green"}
    styles = {"immigration_freeze": ":",     "high_growth": "--"}
    for k, v in scen.items():
        ax.plot(yrs_f, pd.concat([pop.loc[[t0]], v]) / 1e6,
                styles[k], color=colors[k], label=k)
    ax.set_title("Charlotte MSA — population forecast to 2030\n"
                 "(population-only model: M1 Kalman + M2 Holt + M4 demographic)")
    ax.set_ylabel("millions")
    ax.set_xlabel("year")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    plt.savefig("forecast_2030_simple.png", dpi=150)
    print("Saved forecast_2030_simple.png")
    plt.show()


if __name__ == "__main__":
    main()
