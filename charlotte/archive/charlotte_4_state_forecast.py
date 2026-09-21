import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.holtwinters import Holt
from arch import arch_model

RNG = np.random.default_rng(16740)          
BASE = Path(__file__).resolve().parent.parent
DATA, OUT = BASE / "data", BASE / "outputs"
H_END = 2030                                 
N_SIMS = 20_000

def census_year(ts):
    return ts.year if ts.month <= 6 else ts.year + 1

def load_data():
    pop = pd.read_csv(f"{DATA}/charlotte_population.csv", parse_dates=["date"])
    pop["year"] = pop["date"].dt.year
    pop["pop"] = pop["pop_thousands"] * 1000.0
    new = pop[pop.year >= 2010].set_index("year")["pop"]
    old = pop[pop.year <= 2009].set_index("year")["pop"]
    g_old = old.pct_change().dropna()                       # 2001..2009
    g_bridge = np.mean([g_old.loc[2009], new.pct_change().loc[2011]])
    levels = {2010: new.loc[2010], 2009: new.loc[2010] / (1 + g_bridge)}
    for y in range(2009, 2000, -1):                          # chain back
        levels[y - 1] = levels[y] / (1 + g_old.loc[y])
    spliced = pd.Series(levels).sort_index()
    pop_s = pd.concat([spliced.loc[:2009], new]).rename("pop")
    pop_s.index.name = "year"
    emp = pd.read_csv(f"{DATA}/charlotte_employment_monthly.csv",
                      parse_dates=["date"])
    emp["cy"] = emp["date"].map(census_year)
    emp_a = (emp.groupby("cy")["emp_thousands"].mean() * 1000.0)
    emp_a = emp_a[(emp_a.index >= 1991) & (emp_a.index <= 2026)]

    per = pd.read_csv(f"{DATA}/charlotte_permits_monthly.csv",
                      parse_dates=["date"])
    per["cy"] = per["date"].map(census_year)
    per_a = per.groupby("cy")["permits_units"].sum().astype(float)
    per_a = per_a[(per_a.index >= 1989) & (per_a.index <= 2026)]
    per_a.loc[2026] = per_a.loc[2026] * 12 / 10

    df = pd.DataFrame({"pop": pop_s})
    df["g_pop"] = np.log(df["pop"]).diff()
    df["emp"] = emp_a
    df["g_emp"] = np.log(df["emp"]).diff()
    df["permits"] = per_a
    return df, emp_a, per_a

def m1_state_space(pop, t0, t1):
    g = np.log(pop.loc[:t0]).diff().dropna()
    mod = UnobservedComponents(g, level="local level")
    res = mod.fit_constrained({"sigma2.irregular": 0.35 * g.var()},
                              disp=False)
    g_f = res.forecast(t1 - t0)
    levels = pop.loc[t0] * np.exp(np.cumsum(g_f.values))
    return pd.Series(levels, index=range(t0 + 1, t1 + 1))

def m2_damped_holt(pop, t0, t1):
    y = np.log(pop.loc[:t0])
    fit = Holt(y, damped_trend=True, initialization_method="estimated"
               ).fit(optimized=True)
    f = fit.forecast(t1 - t0)
    return pd.Series(np.exp(f.values), index=range(t0 + 1, t1 + 1))


def m3_labor_pull(df, t0, t1):
    d = df.loc[:t0].dropna(subset=["g_pop", "g_emp"]).copy()
    d["g_emp_l1"] = d["g_emp"].shift(1)
    d["g_pop_l1"] = d["g_pop"].shift(1)
    d = d.dropna()
    X = np.column_stack([np.ones(len(d)), d["g_emp_l1"], d["g_pop_l1"]])
    beta, *_ = np.linalg.lstsq(X, d["g_pop"].values, rcond=None)
    emp_hist = np.log(df["emp"].loc[:t0].dropna())
    emp_fit = Holt(emp_hist, damped_trend=True,
                   initialization_method="estimated").fit(optimized=True)
    g_emp_f = pd.Series(emp_fit.forecast(t1 - t0).values,
                        index=range(t0 + 1, t1 + 1)).diff()
    g_emp_f.iloc[0] = emp_fit.forecast(1).values[0] - emp_hist.iloc[-1]

    g_emp_last = df["g_emp"].loc[t0]
    g_pop_last = df["g_pop"].loc[t0]
    level = df["pop"].loc[t0]
    out = {}
    for t in range(t0 + 1, t1 + 1):
        ge = g_emp_last if t == t0 + 1 else g_emp_f.loc[t - 1]
        g = beta[0] + beta[1] * ge + beta[2] * g_pop_last
        level *= np.exp(g)
        out[t] = level
        g_pop_last = g
    return pd.Series(out), beta



def ni_rate(year):
    base = 5.5 - 0.18 * (year - 2011)
    if year in (2021, 2022):
        base -= 0.9                       # excess-mortality notch
    return max(base, 0.5)


def m4_demographic(df, t0, t1, mig_adjust=0.0):
    d = df.loc[2011:t0]
    ni = pd.Series({y: ni_rate(y) / 1000 * df["pop"].loc[y - 1]
                    for y in d.index})
    mig = d["pop"].diff().fillna(d["pop"].loc[2011] - df["pop"].loc[2010]) - ni
    w = 0.85 ** np.arange(len(mig) - 1, -1, -1)
    mu = np.average(mig.values, weights=w)
    x, y_ = mig.values[:-1] - mu, mig.values[1:] - mu
    rho = float(np.clip((x @ y_) / max(x @ x, 1e-9), 0.0, 0.95))
    level, m_prev = df["pop"].loc[t0], mig.iloc[-1]
    out = {}
    for t in range(t0 + 1, t1 + 1):
        m = (mu + mig_adjust) + rho * (m_prev - (mu + mig_adjust))
        ni_t = ni_rate(t) / 1000 * level
        level = level + ni_t + m
        out[t] = level
        m_prev = m
    return pd.Series(out), dict(mu=mu, rho=rho, mig_last=mig.iloc[-1],
                                mig_series=mig, ni_series=ni)

MODELS = {
    "M1_state_space":  lambda df, t0, t1: m1_state_space(df["pop"], t0, t1),
    "M2_damped_holt":  lambda df, t0, t1: m2_damped_holt(df["pop"], t0, t1),
    "M3_labor_pull":   lambda df, t0, t1: m3_labor_pull(df, t0, t1)[0],
    "M4_demographic":  lambda df, t0, t1: m4_demographic(df, t0, t1)[0],
}

def naive(df, t0, t1):
    """Benchmark: last observed growth rate persists (RW with drift)."""
    g = df["g_pop"].loc[t0]
    lvl = df["pop"].loc[t0]
    return pd.Series([lvl * np.exp(g * (h + 1)) for h in range(t1 - t0)],
                     index=range(t0 + 1, t1 + 1))

def backtest(df, origins=range(2012, 2021), h_max=5):
    rows = []
    for t0 in origins:
        t1 = min(t0 + h_max, 2025)
        actual = df["pop"].loc[t0 + 1:t1]
        for name, fn in list(MODELS.items()) + [("naive", naive)]:
            try:
                pred = fn(df, t0, t1)
            except Exception:
                continue
            for h, (yy, p) in enumerate(pred.items(), start=1):
                if yy in actual.index:
                    rows.append(dict(model=name, origin=t0, horizon=h,
                                     year=yy, pred=p, actual=actual.loc[yy],
                                     ape=abs(p / actual.loc[yy] - 1)))
    bt = pd.DataFrame(rows)
    summary = (bt.groupby(["model", "horizon"])["ape"]
                 .mean().unstack().round(5))
    # inverse-MSE weights over all horizons (excl. naive)
    mse = bt[bt.model != "naive"].groupby("model")["ape"].apply(
        lambda s: (s ** 2).mean())
    w = (1 / mse) / (1 / mse).sum()
    return bt, summary, w

def innovation_vol(df):
    g = df["g_pop"].dropna() * 100                 
    x = g.values[:-1] - g.mean()
    y = g.values[1:] - g.mean()
    rho = float(np.clip((x @ y) / (x @ x), 0, 0.95))
    resid = y - rho * x
    ewma_var, lam = resid[0] ** 2, 0.80
    for e in resid[1:]:
        ewma_var = lam * ewma_var + (1 - lam) * e ** 2
    sigma_ewma = np.sqrt(ewma_var)
    try:
        am = arch_model(pd.Series(resid), vol="GARCH", p=1, q=1,
                        mean="Zero", dist="t")
        garch = am.fit(disp="off")
        sigma_garch = float(np.sqrt(
            garch.forecast(horizon=1).variance.values[-1, 0]))
        gparams = dict(garch.params)
    except Exception:
        sigma_garch, gparams = sigma_ewma, {}
    sigma = max(sigma_garch, sigma_ewma) / 100      # back to log units
    return sigma, rho, dict(sigma_garch=sigma_garch / 100,
                            sigma_ewma=sigma_ewma / 100,
                            ar1_rho=rho, garch_params=gparams)



def ensemble_path(df, w, t0=2025, t1=H_END):
    preds = {n: fn(df, t0, t1) for n, fn in MODELS.items()}
    P = pd.DataFrame(preds)
    point = (P * w).sum(axis=1)
    return point, P


def monte_carlo(df, point, sigma, rho, t0=2025):
    yrs = point.index
    g_point = np.log(pd.concat([df["pop"].loc[[t0]], point])).diff().dropna()
    n, h = N_SIMS, len(yrs)
    shocks = RNG.standard_t(df=5, size=(n, h))
    shocks *= sigma / np.sqrt(5 / 3)               # unit-variance t -> sigma
    eps = np.zeros((n, h))
    eps[:, 0] = shocks[:, 0]
    for j in range(1, h):                          # persistent forecast errors
        eps[:, j] = rho * eps[:, j - 1] + shocks[:, j] * np.sqrt(1 - rho**2)
    g = g_point.values[None, :] + eps
    paths = df["pop"].loc[t0] * np.exp(np.cumsum(g, axis=1))
    q = np.percentile(paths, [10, 25, 50, 75, 90], axis=0)
    bands = pd.DataFrame(q.T, index=yrs,
                         columns=["P10", "P25", "P50", "P75", "P90"])
    return bands, paths



def scenarios(df, point, t0=2025, t1=H_END):
    out = {}
    m4_base, _ = m4_demographic(df, t0, t1)
    m4_frz, _ = m4_demographic(df, t0, t1, mig_adjust=-12_000)
    out["immigration_freeze"] = point * (m4_frz / m4_base)
    m4_hi, _ = m4_demographic(df, t0, t1, mig_adjust=+8_000)
    out["high_growth"] = point * (m4_hi / m4_base)
    m3_base, beta = m3_labor_pull(df, t0, t1)
    level = df["pop"].loc[t0]
    g_emp_path = {2026: 0.010, 2027: -0.030, 2028: -0.005,
                  2029: 0.015, 2030: 0.020}
    g_prev, rec = df["g_pop"].loc[t0], {}
    g_emp_prev = df["g_emp"].loc[t0]
    for t in range(t0 + 1, t1 + 1):
        g = beta[0] + beta[1] * g_emp_prev + beta[2] * g_prev
        level *= np.exp(g)
        rec[t] = level
        g_prev, g_emp_prev = g, g_emp_path[t]
    out["recession_2027"] = point * (pd.Series(rec) / m3_base)
    return out



def housing_balance(df, point, per_a, hh_size=2.5, completion_lag=1,
                    loss_rate=0.002):
    pop_all = pd.concat([df["pop"], point])
    d_pop = pop_all.diff()
    stock_proxy = pop_all / hh_size
    demand = d_pop / hh_size + loss_rate * stock_proxy.shift(1)
    permits_f = per_a.reindex(range(1989, H_END + 1))
    for y in range(2027, H_END + 1):
        permits_f.loc[y] = 22_000
    supply = permits_f.shift(completion_lag) * 0.96   # 4% never built
    bal = pd.DataFrame({"demand_units": demand, "supply_units": supply})
    bal["gap"] = bal["demand_units"] - bal["supply_units"]
    return bal.loc[2015:H_END].round(0)


def main():
    df, emp_a, per_a = load_data()

    print("=" * 70)
    print("STEP 1  Backtest (expanding window, origins 2012-2020, h=1..5)")
    bt, summary, w = backtest(df)
    print("\nMean absolute percentage error by horizon (levels):")
    print((summary * 100).round(2).to_string())
    print("\nEnsemble weights (inverse-MSE):")
    print((w * 100).round(1).astype(str) + " %")

    print("\n" + "=" * 70)
    print("STEP 2  Volatility of growth innovations")
    sigma, rho_eps, vol_info = innovation_vol(df)
    print(f"  GARCH(1,1) 1-step sigma : {vol_info['sigma_garch']*100:.3f} pp")
    print(f"  EWMA sigma              : {vol_info['sigma_ewma']*100:.3f} pp")
    print(f"  AR(1) of growth (rho)   : {vol_info['ar1_rho']:.2f}")
    print(f"  Simulation sigma (max)  : {sigma*100:.3f} pp")

    print("\n" + "=" * 70)
    print("STEP 3  Ensemble forecast to 2030")
    point, P = ensemble_path(df, w)
    bands, paths = monte_carlo(df, point, sigma, rho_eps)
    _, m4info = m4_demographic(df, 2025, H_END)
    print("\nModel-by-model 2030 levels:")
    print((P.loc[H_END] / 1e6).round(3).astype(str) + " M")
    print(f"\nPOINT FORECAST 2030: {point.loc[H_END]:,.0f}")
    print(f"  growth 2025->2030  : {point.loc[H_END]/df['pop'].loc[2025]-1:+.2%}"
          f"  (CAGR {(point.loc[H_END]/df['pop'].loc[2025])**0.2-1:+.2%})")
    print("\nFan (Monte Carlo, 20k paths):")
    print(bands.round(0).astype(int).to_string())
    print(f"\nP(pop2030 > 3.2M) = "
          f"{(paths[:, -1] > 3.2e6).mean():.1%}")
    print(f"P(pop2030 > 3.3M) = {(paths[:, -1] > 3.3e6).mean():.1%}")
    print(f"Migration AR(1): mu={m4info['mu']:,.0f}/yr, rho={m4info['rho']:.2f}")

    print("\n" + "=" * 70)
    print("STEP 4  Scenarios (2030 levels)")
    scen = scenarios(df, point)
    for k, v in scen.items():
        print(f"  {k:20s}: {v.loc[H_END]:,.0f} "
              f"({v.loc[H_END]/df['pop'].loc[2025]-1:+.2%} vs 2025)")

    print("\n" + "=" * 70)
    print("STEP 5  Multifamily demand/supply balance")
    bal = housing_balance(df, point, per_a)
    print(bal.loc[2022:].to_string())
    cum_gap = bal.loc[2026:H_END, "gap"].sum()
    print(f"\nCumulative 2026-2030 housing gap: {cum_gap:+,.0f} units "
          f"({'UNDERSUPPLY -> rent pressure' if cum_gap > 0 else 'oversupply'})")

    fc = bands.copy()
    fc.insert(0, "point", point.round(0))
    for k, v in scen.items():
        fc[k] = v.round(0)
    fc.to_csv(f"{OUT}/forecast_2030.csv")
    summary.to_csv(f"{OUT}/backtest_mape.csv")
    bal.to_csv(f"{OUT}/housing_balance.csv")

    plot_all(df, point, bands, P, scen, bal, bt, m4info, emp_a)
    print(f"\nSaved: {OUT}/forecast_2030.csv, backtest_mape.csv, housing_balance.csv")


def plot_all(df, point, bands, P, scen, bal, bt, m4info, emp_a):
    from matplotlib.ticker import MaxNLocator
    yrs_h = df.index[df.index >= 2010]
    last = df["pop"].loc[[2025]]
    pt = pd.concat([last, point])          
    yrs_f = pt.index

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(yrs_h, df["pop"].loc[yrs_h] / 1e6, "k-", lw=2, label="Actual (Census V2025)")
    b10 = pd.concat([last, bands["P10"]]); b90 = pd.concat([last, bands["P90"]])
    b25 = pd.concat([last, bands["P25"]]); b75 = pd.concat([last, bands["P75"]])
    ax.fill_between(yrs_f, b10 / 1e6, b90 / 1e6,
                    alpha=.18, color="tab:blue", label="P10-P90")
    ax.fill_between(yrs_f, b25 / 1e6, b75 / 1e6,
                    alpha=.30, color="tab:blue", label="P25-P75")
    ax.plot(yrs_f, pt / 1e6, "b--o", lw=2, ms=4, label="Ensemble point")
    for k, st in [("immigration_freeze", ":"), ("recession_2027", "-.")]:
        ax.plot(yrs_f, pd.concat([last, scen[k]]) / 1e6, st,
                color="tab:red", label=k)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Charlotte MSA population forecast to 2030")
    ax.set_ylabel("millions"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(yrs_h, df["pop"].loc[yrs_h] / 1e6, "k-", lw=2, label="Actual")
    for c in P.columns:
        ax.plot(P.index, P[c] / 1e6, "--", label=c)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Component models, 2026-2030"); ax.set_ylabel("millions")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    g_emp = df["g_emp"] * 100
    g_pop = df["g_pop"] * 100
    ax[0].plot(g_emp.index, g_emp, "tab:orange", label="employment growth")
    ax[0].plot(g_pop.index, g_pop, "tab:blue", label="population growth")
    ax[0].set_title("People follow jobs (annual % growth)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ms, ns = m4info["mig_series"], m4info["ni_series"]
    ax[1].bar(ms.index, ms / 1000, color="tab:green", label="net migration (residual)")
    ax[1].bar(ns.index, ns / 1000, bottom=ms / 1000, color="tab:gray",
              label="natural increase (schedule)")
    ax[1].set_title("Growth decomposition (thousands/yr)")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    piv = (bt.groupby(["model", "horizon"])["ape"].mean().unstack() * 100)
    for m in piv.index:
        ax[0].plot(piv.columns, piv.loc[m], "-o", label=m)
    ax[0].set_title("Backtest MAPE by horizon (%)"); ax[0].set_xlabel("horizon (yrs)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    b = bal.loc[2018:]
    ax[1].bar(b.index, b["demand_units"] / 1000, alpha=.6, label="HH demand (k units)")
    ax[1].plot(b.index, b["supply_units"] / 1000, "r-o", label="supply (lagged permits)")
    ax[1].axvline(2025.5, color="gray", ls=":")
    ax[1].set_title("Multifamily lens: demand vs supply")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
