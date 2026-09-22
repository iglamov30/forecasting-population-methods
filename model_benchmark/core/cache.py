"""
Per-year parquet cache for ACS1 pulls.

One file per year at model_benchmark/data/raw/acs1_{year}.parquet. If a cached
file already contains every requested variable, no API call is made. If it
exists but is missing a newly-requested variable, that year is re-fetched
(the Census API returns every requested variable in one call regardless) and
the cache file is overwritten with the union of variables.
"""

import os

import pandas as pd

try:
    from .census import fetch_acs1_places
except ImportError:  # pragma: no cover
    from core.census import fetch_acs1_places

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def _cache_path(year):
    return os.path.join(DATA_DIR, f"acs1_{year}.parquet")


def get_or_fetch_year(year, variables, key=None):
    """Returns a long-format DataFrame [year, place_id, city, variable, value]
    for the requested variables in one year, using the parquet cache."""
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
