import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from eval.intervals import assign_population_decile
from eval.pooled_model import (
    run_pooled_backtest,
    score_by_lam,
    build_growth_panel,
    _add_lags,
    fit_pooled_ar,
    nickell_bias_estimate,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    panel = pd.read_parquet(os.path.join(DATA_DIR, "population_panel.parquet"))

    for order in (1, 2):
        bt, _ = run_pooled_backtest(panel, order=order)
        bt.to_parquet(os.path.join(DATA_DIR, f"pooled_ar{order}_backtest.parquet"), index=False)
        lam_table = score_by_lam(bt)
        lam_table.to_csv(os.path.join(DATA_DIR, f"pooled_ar{order}_lambda_selection.csv"), index=False)
        print(f"AR({order}) lambda selection:")
        print(lam_table.to_string(index=False))
        print()

        decile = assign_population_decile(panel)
        btd = bt.merge(decile, left_on="place_id", right_index=True)
        btd["abs_error"] = (btd["predicted"] - btd["actual"]).abs()
        by_decile_lam = btd.groupby(["pop_decile", "lam"])["abs_error"].mean().reset_index()
        best_lam_by_decile = by_decile_lam.loc[
            by_decile_lam.groupby("pop_decile")["abs_error"].idxmin()
        ][["pop_decile", "lam"]]
        best_lam_by_decile.to_csv(
            os.path.join(DATA_DIR, f"pooled_ar{order}_best_lambda_by_decile.csv"), index=False
        )
        print(f"AR({order}) best lambda by population decile:")
        print(best_lam_by_decile.to_string(index=False))
        print()

    g1 = _add_lags(build_growth_panel(panel), n_lags=1)
    full_fit = fit_pooled_ar(g1, order=1)
    rho1 = full_fit["rho"][1]
    avg_T = g1.dropna(subset=["g_lag1", "g"]).groupby("place_id").size().mean()
    bias = nickell_bias_estimate(rho1, avg_T)
    print(f"Full-sample pooled AR(1) rho: {rho1:.4f}")
    print(f"Average T (growth obs/city): {avg_T:.1f}")
    print(f"Nickell bias estimate: {bias:.4f}  (bias-adjusted rho ~= {rho1 - bias:.4f})")


if __name__ == "__main__":
    main()
