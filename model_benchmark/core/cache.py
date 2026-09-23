import os
import sys
import pandas as pd

if __package__ in (None, ""):  # direct run: put the source root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.census import fetch_acs1_places

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def _cache_path(year):
    return os.path.join(DATA_DIR, f"acs1_{year}.parquet")


def get_or_fetch_year(year, variables, key=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _cache_path(year)
    variables = list(variables)

    cached = None
    if os.path.exists(path):
        cached = pd.read_parquet(path)
        have = set(cached["variable"].unique())
        if set(variables).issubset(have):
            return cached[cached["variable"].isin(variables)].reset_index(drop=True)

    fetched = fetch_acs1_places(year, variables, key=key)
    if cached is not None:
        merged = pd.concat([cached, fetched], ignore_index=True)
        merged = merged.drop_duplicates(subset=["place_id", "variable"], keep="last")
    else:
        merged = fetched

    merged.to_parquet(path, index=False)
    return merged[merged["variable"].isin(variables)].reset_index(drop=True)
