"""Phase 4: empirical prediction intervals (pooled + by population decile)
vs. ARIMA's analytic and ETS's simulated intervals, and their 90% coverage.
Run after scripts/run_backtest.py. This is the slow phase (ETS simulation
with 500 repetitions per backtest origin) -- budget a few minutes.

    python model_benchmark/scripts/run_intervals.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from eval.intervals import (
    empirical_pi_pooled,
    empirical_pi_by_decile,
    mape_by_decile,
    model_based_intervals,
    empirical_intervals_for_backtest,
    empirical_intervals_by_decile_for_backtest,
    coverage_by_horizon,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    results = pd.read_parquet(os.path.join(DATA_DIR, "backtest_results.parquet"))
    panel = pd.read_parquet(os.path.join(DATA_DIR, "population_panel.parquet"))

    pooled_pi = empirical_pi_pooled(results)
    decile_pi, _ = empirical_pi_by_decile(results, panel)
    decile_pi.to_csv(os.path.join(DATA_DIR, "phase4_decile_pi.csv"), index=False)

    decile_mape = mape_by_decile(results, panel)
    decile_mape.to_csv(os.path.join(DATA_DIR, "phase4_decile_mape.csv"), index=False)

    mb = model_based_intervals(panel)
    mb.to_parquet(os.path.join(DATA_DIR, "phase4_model_based_intervals.parquet"), index=False)

    emp_pooled_intervals = empirical_intervals_for_backtest(results, pooled_pi)
    emp_decile_intervals = empirical_intervals_by_decile_for_backtest(results, panel, decile_pi)

    cov_tables = [
        coverage_by_horizon(emp_pooled_intervals, "empirical_pooled"),
        coverage_by_horizon(emp_decile_intervals, "empirical_by_decile"),
    ]
    for approach in ("arima_analytic", "ets_simulated"):
        cov_tables.append(coverage_by_horizon(mb[mb["approach"] == approach], approach))

    cov = pd.concat(cov_tables, ignore_index=True).sort_values(["horizon", "approach"])
    cov.to_csv(os.path.join(DATA_DIR, "phase4_coverage.csv"), index=False)

    pd.set_option("display.width", 140)
    print("MAPE by population decile:")
    print(decile_mape.to_string(index=False))
    print("\n90% interval coverage by approach and horizon:")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
