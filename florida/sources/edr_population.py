import hashlib
import os
import re
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.cities import CITY_NAMES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
XLSX_PATH = os.path.join(DATA_DIR, "FLmupops.xlsx")

KNOWN_PLACE_IDS = {name: pid for pid, name in CITY_NAMES.items()}

_SUMMARY_ROWS = (
    "total population",
    "incorporated population",
    "unincorporated population",
)

_cache = {}

def _synth_place_id(city, county):
    h = int(hashlib.md5(f"{city}|{county}".encode("utf-8")).hexdigest(), 16)
    return "12" + f"{h % 100000:05d}"


def _place_id(city, county):
    return KNOWN_PLACE_IDS.get(city, _synth_place_id(city, county))


def _parse_sheet(raw):
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
    panel = _load_raw_panel().copy()
    if fill_census_years:
        panel = _fill_census_years(panel)
    if years is not None and len(years):
        lo, hi = min(years), max(years)
        panel = panel[(panel["year"] >= lo) & (panel["year"] <= hi)]
    return panel.sort_values(["place_id", "year"]).reset_index(drop=True)


def filter_florida(panel):
    return panel.reset_index(drop=True)


def clean_city_name(name):
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

    out = os.path.join(DATA_DIR, "population_top5.csv")
    wide = (
        p[p["place_id"].isin(KNOWN_PLACE_IDS.values())]
        .pivot(index="year", columns="city", values="population")
    )
    wide.to_csv(out)
    print(f"\nSaved {out}")
