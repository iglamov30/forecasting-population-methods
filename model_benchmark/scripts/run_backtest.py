"""Phase 2: runs the expanding-window backtest for all six models and saves
the model-by-horizon error table. Run after scripts/build_cache.py.

    python model_benchmark/scripts/run_backtest.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from eval.backtest import run_backtest, score_by_model_horizon
from models.registry import MODELS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    panel = pd.read_parquet(os.path.join(DATA_DIR, "population_panel.parquet"))
    results = run_backtest(panel, MODELS)
    results.to_parquet(os.path.join(DATA_DIR, "backtest_results.parquet"), index=False)

    table = score_by_model_horizon(results)
    table.to_csv(os.path.join(DATA_DIR, "phase2_score_table.csv"), index=False)

    pd.set_option("display.width", 140)
    print(f"Backtest rows: {len(results)}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
