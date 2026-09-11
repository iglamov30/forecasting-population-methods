"""Phase 3: compares simple-mean ensemble, trimmed-mean ensemble, and
leakage-free per-city best-model selection on the same backtest windows.
Run after scripts/run_backtest.py.

    python model_benchmark/scripts/run_ensembles.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from eval.ensembles import compare_approaches

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    results = pd.read_parquet(os.path.join(DATA_DIR, "backtest_results.parquet"))
    table = compare_approaches(results)
    table.to_csv(os.path.join(DATA_DIR, "phase3_ensemble_comparison.csv"), index=False)

    pd.set_option("display.width", 140)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
