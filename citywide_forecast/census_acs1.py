import os
import requests
import numpy as np
import pandas as pd

POP_VAR = "B01003_001E"        

def _resolve_key(key=None):
    if key:
        return key
    env = os.environ.get("CENSUS_API_KEY")
    if env:
        return env
    try:
        from secret import CENSUS_API_KEY
        return CENSUS_API_KEY
    except Exception:
        return None

def fetch_acs1_places(year, key=None):
    key = _resolve_key(key)
    if not key:
        raise RuntimeError(
            "No Census API key found. Set CENSUS_API_KEY as an environment "
            "variable or put it in a local secret.py."
        )

    url = f"https://api.census.gov/data/{year}/acs/acs1"
    params = {
        "get": f"NAME,{POP_VAR}",
        "for": "place:*",
        "in": "state:*",
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

def build_panel(years):
    frames = []
    for y in years:
        try:
            frames.append(fetch_acs1_places(y))
        except requests.HTTPError as exc:
            # str(exc) contains the request URL, which includes the API key
            print(f"  skip {y}: HTTP {exc.response.status_code}")
    if not frames:
        raise RuntimeError("No data returned for any requested year.")
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["place_id", "year"]).reset_index(drop=True)

def city_series(panel, place_id):
    sub = panel[panel["place_id"] == place_id].dropna(subset=["population"])
    sub = sub.sort_values("year")
    return sub["year"].to_numpy(), sub["population"].to_numpy()

def list_cities(panel):
    latest = panel.sort_values("year").groupby("place_id").last()
    return latest.sort_values("population", ascending=False)[["city", "population"]]

if __name__ == "__main__":
    years = [y for y in range(2010, 2024) if y != 2020]
    panel = build_panel(years)
    print(f"Pulled {panel['place_id'].nunique()} cities across {panel['year'].nunique()} years.")
    print(list_cities(panel).head(10).to_string())

    yrs, pop = city_series(panel, "3712000")
    print("\nCharlotte series:", list(zip(yrs.tolist(), pop.astype(int).tolist())))
