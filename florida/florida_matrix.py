"""
Same idea as matrix_test.py, but Florida-only.

Difference from just filtering matrix_test.py's output: census_devs.py's
fetch_acs1_places() always queries `in=state:*` (all 50 states) and relies
on you to filter afterward. For a FL-only matrix that means pulling and
discarding ~49 states' worth of places every run. fetch_acs1_places_state()
below issues the same ACS1 call but scoped to state:12 up front, so it's
the same data, just a much smaller/faster pull.

Produces a year x city matrix for ALL Florida places (not just the top 5)
sorted by latest population, same shape/behavior as matrix_test.py's
population_matrix().
"""

from datetime import datetime

import pandas as pd
import requests

from census_devs import _resolve_key, POP_VAR

START_YEAR = 2005
FL_STATE_FIPS = "12"


def fetch_acs1_places_state(year, state_fips=FL_STATE_FIPS, key=None):
    key = _resolve_key(key)
    if not key:
        raise RuntimeError("No Census API key found (secret.py or CENSUS_API_KEY env var).")

    url = f"https://api.census.gov/data/{year}/acs/acs1"
    params = {
        "get": f"NAME,{POP_VAR}",
        "for": "place:*",
        "in": f"state:{state_fips}",
        "key": key,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    rows = resp.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.rename(columns={"NAME": "city", POP_VAR: "population"})
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["year"] = int(year)
    df["place_id"] = df["state"].str.zfill(2) + df["place"].str.zfill(5)
    return df[["year", "place_id", "city", "population"]]


def build_panel_fl(years, state_fips=FL_STATE_FIPS):
    frames = []
    for y in years:
        try:
            frames.append(fetch_acs1_places_state(y, state_fips=state_fips))
        except requests.HTTPError as exc:
            print(f"  skip {y}: {exc}")
    if not frames:
        raise RuntimeError("No data returned for any requested year.")
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["place_id", "year"]).reset_index(drop=True)


def to_matrix(panel, start_year=START_YEAR, sort_by_size=True):
    panel = panel.dropna(subset=["population"])
    panel = panel[panel["population"] > 0]

    wide = panel.pivot(index="year", columns="place_id", values="population")
    names = panel.drop_duplicates("place_id").set_index("place_id")["city"]
    wide.columns = [names[pid] for pid in wide.columns]
    end_year = int(panel["year"].max())
    wide = wide.reindex(range(start_year, end_year + 1))
    wide.index.name = "year"

    if sort_by_size:
        latest = wide.apply(
            lambda col: col.dropna().iloc[-1] if col.notna().any() else float("nan")
        )
        wide = wide[latest.sort_values(ascending=False).index]

    return wide


def florida_population_matrix(years=None, start_year=START_YEAR, sort_by_size=True):
    if years is None:
        years = [y for y in range(start_year, datetime.now().year + 1) if y != 2020]
    panel = build_panel_fl(years)
    return to_matrix(panel, start_year=start_year, sort_by_size=sort_by_size)


if __name__ == "__main__":
    mat = florida_population_matrix()
    print(f"matrix shape: {mat.shape[0]} years x {mat.shape[1]} Florida cities\n")
    print(mat.iloc[:, :5].to_string())
    out = "florida_city_population_matrix.csv"
    mat.to_csv(out)
    print(f"\nSaved -> {out}")
