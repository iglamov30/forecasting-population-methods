"""
edr_population.py

Drop-in replacement for census_devs.py's panel interface, sourced from the
Florida Office of Economic & Demographic Research (EDR) municipal population
workbook (FLmupops.xlsx) instead of the Census ACS1 API.

Why switch sources: ACS1 (1-year estimates) only publishes for places with
population >= 65,000 and only goes back to 2005 -- about 19 usable years, and
only for the very largest cities. The EDR "BEBR" municipal estimates cover
every incorporated Florida municipality every year from 1979, which gives
40+ years of history for all five target cities and hundreds of smaller ones.

This module exposes the same names the Florida forecast scripts import, so
those scripts run unchanged once their import line is switched to point here:

    build_panel(years=None)      -> DataFrame[year, place_id, city, county, population]
    city_series(panel, place_id) -> (years ndarray, population ndarray)
    list_cities(panel)           -> DataFrame indexed by place_id, cols [city, population]
    filter_florida(panel)        -> panel unchanged (it is already Florida-only)
    clean_city_name(name)        -> name (EDR names are already clean)

Source file (kept alongside this script):
    FLmupops.xlsx
    https://edr.state.fl.us/Content/population-demographics/data/FLmupops.xlsx

The workbook has one sheet per estimate year ("2025 BEBR", "2024 BEBR", ...).
The decennial-census years 1980/1990/2000/2010 have no EDR estimate sheet; by
default those four gap years are filled per-city by linear interpolation so
the series handed to ETS/ARIMA is a clean annual grid. Pass
fill_census_years=False to keep the raw gaps instead.
"""

import hashlib
import os
import re

import numpy as np
import pandas as pd

XLSX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "FLmupops.xlsx"
)

# Real Census place FIPS for the five target cities, so downstream output keeps
# the same place_id values the old ACS1 path produced (see fl_cities.py).
KNOWN_PLACE_IDS = {
    "Jacksonville": "1235000",
    "Miami": "1245000",
    "Tampa": "1271000",
    "Orlando": "1253000",
    "St. Petersburg": "1263000",
}

# Non-municipality rows that appear in the "Municipality" column of some sheets.
_SUMMARY_ROWS = (
    "total population",
    "incorporated population",
    "unincorporated population",
)

_cache = {}


def _synth_place_id(city, county):
    """Deterministic synthetic id for municipalities without a known FIPS."""
    h = int(hashlib.md5(f"{city}|{county}".encode("utf-8")).hexdigest(), 16)
    return "12" + f"{h % 100000:05d}"


def _place_id(city, county):
    return KNOWN_PLACE_IDS.get(city, _synth_place_id(city, county))


def _parse_sheet(raw):
    """Turn one raw EDR year sheet into [city, county, population] rows."""
    header_row = None
    for i in range(min(15, len(raw))):
        cells = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        if "municipality" in cells and "population" in cells:
            header_row = i
            break
    if header_row is None:
        return None

    cols = [str(x).strip() for x in raw.iloc[header_row].tolist()]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = cols
    df = df.rename(
        columns={"Municipality": "city", "County": "county", "Population": "population"}
    )
    df = df[[c for c in ("city", "county", "population") if c in df.columns]]

    df["city"] = df["city"].astype(str).str.strip()
    # The "Revised 2020" sheet tags some rows with a trailing footnote asterisk
    # ("Miami *"); strip it so those merge into the main city series.
    df["city"] = df["city"].str.replace(r"\s*\*+$", "", regex=True).str.strip()
    df["county"] = df["county"].astype(str).str.strip()
    df["population"] = pd.to_numeric(df["population"], errors="coerce")

    df = df[df["population"].notna() & (df["population"] > 0)]
    df = df[~df["city"].str.lower().isin(_SUMMARY_ROWS)]
    df = df[df["county"].str.lower().ne("nan") & (df["county"] != "")]
    return df


def _load_raw_panel():
    if "raw" in _cache:
        return _cache["raw"]

    if not os.path.exists(XLSX_PATH):
        raise FileNotFoundError(
            f"EDR workbook not found at {XLSX_PATH}. Download it from "
            "https://edr.state.fl.us/Content/population-demographics/data/FLmupops.xlsx"
        )

    xls = pd.ExcelFile(XLSX_PATH)
    frames = []
    for sheet in xls.sheet_names:
        m = re.search(r"(19|20)\d{2}", sheet)
        if not m:
            continue
        parsed = _parse_sheet(pd.read_excel(XLSX_PATH, sheet_name=sheet, header=None))
        if parsed is None or parsed.empty:
            continue
        parsed["year"] = int(m.group(0))
        frames.append(parsed)

    if not frames:
        raise RuntimeError("No usable year sheets parsed from the EDR workbook.")

    panel = pd.concat(frames, ignore_index=True)
    # Collapse any accidental duplicate (city, county, year).
    panel = panel.groupby(["city", "county", "year"], as_index=False)["population"].mean()
    panel["place_id"] = [
        _place_id(c, k) for c, k in zip(panel["city"], panel["county"])
    ]
    panel = panel.sort_values(["place_id", "year"]).reset_index(drop=True)

    _cache["raw"] = panel
    return panel


def _fill_census_years(panel):
    """Linearly interpolate the missing 1980/1990/2000/2010 rows per city."""
    out = []
    for pid, g in panel.groupby("place_id", sort=False):
        g = g.sort_values("year")
        grid = np.arange(int(g["year"].min()), int(g["year"].max()) + 1)
        pop = (
            g.set_index("year")["population"]
            .reindex(grid)
            .interpolate(method="linear")
            .round()
        )
        out.append(
            pd.DataFrame(
                {
                    "city": g["city"].iloc[0],
                    "county": g["county"].iloc[0],
                    "year": grid,
                    "population": pop.to_numpy(),
                    "place_id": pid,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def build_panel(years=None, fill_census_years=True):
    """EDR analogue of census_devs.build_panel.

    `years` is treated as an inclusive [min, max] range filter (matching how
    the forecast scripts pass a contiguous list); pass None for all history.
    """
    panel = _load_raw_panel().copy()
    if fill_census_years:
        panel = _fill_census_years(panel)
    if years is not None and len(years):
        lo, hi = min(years), max(years)
        panel = panel[(panel["year"] >= lo) & (panel["year"] <= hi)]
    return panel.sort_values(["place_id", "year"]).reset_index(drop=True)


def filter_florida(panel):
    """No-op: the EDR panel is already Florida-only. Kept for import parity."""
    return panel.reset_index(drop=True)


def clean_city_name(name):
    """EDR names are already plain ('Miami', 'St. Petersburg'); trim only."""
    return str(name).replace(", Florida", "").strip()


def city_series(panel, place_id):
    sub = (
        panel[panel["place_id"] == place_id]
        .dropna(subset=["population"])
        .sort_values("year")
    )
    return sub["year"].to_numpy(), sub["population"].to_numpy()


def list_cities(panel):
    latest = panel.sort_values("year").groupby("place_id").last()
    return latest.sort_values("population", ascending=False)[["city", "population"]]


if __name__ == "__main__":
    p = build_panel()
    print(
        f"EDR panel: {p['place_id'].nunique()} municipalities, "
        f"{int(p['year'].min())}-{int(p['year'].max())} "
        f"({p['year'].nunique()} yrs)\n"
    )

    print("Top 10 Florida municipalities by latest population:")
    top = list_cities(p).head(10)
    print(top.assign(population=top["population"].map("{:,.0f}".format)).to_string())

    print("\nTarget-city coverage:")
    for city, pid in KNOWN_PLACE_IDS.items():
        y, v = city_series(p, pid)
        print(
            f"  {city:<16} {len(y):>2} yrs  {int(y[0])}-{int(y[-1])}  "
            f"latest {int(v[-1]):,}"
        )

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "edr_top5_population.csv"
    )
    wide = (
        p[p["place_id"].isin(KNOWN_PLACE_IDS.values())]
        .pivot(index="year", columns="city", values="population")
    )
    wide.to_csv(out)
    print(f"\nSaved top-5 wide series -> {out}")
