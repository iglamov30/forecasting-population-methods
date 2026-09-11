import os
import time
import requests
import pandas as pd
import numpy as np

BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
MEASURE_EMPLOYMENT = "05"


def _resolve_key(key=None):
    if key:
        return key
    env = os.environ.get("BLS_API_KEY")
    if env:
        return env
    try:
        from secret import BLS_API_KEY
        return BLS_API_KEY
    except Exception:
        return None  # BLS API works without a key at reduced rate limits


def laus_series_id(county_fips, measure=MEASURE_EMPLOYMENT):
    """county_fips must be 5 digits, e.g. '12031' for Duval County, FL."""
    county_fips = str(county_fips).zfill(5)
    state, county = county_fips[:2], county_fips[2:]
    return f"LAUCN{state}{county}00000000{measure}"


def fetch_laus(series_ids, start_year, end_year, key=None):
    """
    Fetch one or more LAUS series from the BLS API in batches of <=50 series
    and <=20 years per request (public API v2 limits), returns a long-format
    DataFrame: series_id, year, period, value.
    """
    key = _resolve_key(key)
    all_rows = []

    years = list(range(start_year, end_year + 1))
    year_chunks = [years[i:i + 19] for i in range(0, len(years), 19)]

    for chunk in year_chunks:
        payload = {
            "seriesid": series_ids,
            "startyear": str(chunk[0]),
            "endyear": str(chunk[-1]),
        }
        if key:
            payload["registrationkey"] = key

        resp = requests.post(BLS_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "REQUEST_SUCCEEDED":
            msgs = data.get("message", [])
            raise RuntimeError(f"BLS API error: {msgs}")

        for series in data["Results"]["series"]:
            sid = series["seriesID"]
            for item in series["data"]:
                all_rows.append({
                    "series_id": sid,
                    "year": int(item["year"]),
                    "period": item["period"],  # M01..M12
                    "value": float(item["value"]) if item["value"] not in ("-", "") else None,
                })

        time.sleep(0.5)  # be polite to the API

    return pd.DataFrame(all_rows)


def build_employment_panel(fl_cities, start_year, end_year, key=None):
    """
    fl_cities: dict shaped like fl_cities.FL_CITIES
    Returns annual-average employment per county as a panel matching the
    shape of census_devs.build_panel(): columns [year, place_id, city, employment]
    """
    county_to_place = {v["county_fips"]: (pid, v["city"]) for pid, v in fl_cities.items()}
    series_ids = [laus_series_id(fips) for fips in county_to_place]
    sid_to_fips = {laus_series_id(fips): fips for fips in county_to_place}

    raw = fetch_laus(series_ids, start_year, end_year, key=key)
    if raw.empty:
        raise RuntimeError("BLS API returned no data — check your key and date range.")

    raw = raw.dropna(subset=["value"])
    raw = raw[raw["period"].str.match(r"M\d{2}")]  # drop annual-average M13 rows, keep monthly
    raw = raw[raw["period"] != "M13"]

    annual = (
        raw.groupby(["series_id", "year"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "employment"})
    )

    annual["county_fips"] = annual["series_id"].map(sid_to_fips)
    annual["place_id"] = annual["county_fips"].map(lambda f: county_to_place[f][0])
    annual["city"] = annual["county_fips"].map(lambda f: county_to_place[f][1])

    panel = annual[["year", "place_id", "city", "employment"]].sort_values(
        ["place_id", "year"]
    ).reset_index(drop=True)
    return panel


def county_series(panel, place_id):
    """Same interface as census_devs.city_series for drop-in reuse of forecast code."""
    sub = panel[panel["place_id"] == place_id].dropna(subset=["employment"])
    sub = sub.sort_values("year")
    return sub["year"].to_numpy(), sub["employment"].to_numpy()


if __name__ == "__main__":
    import datetime

    from fl_cities import FL_CITIES, short_name

    # BLS LAUS county-level series begin in 1990; pull through the current
    # year to capture all history the API has, whatever that turns out to be.
    START_YEAR = 1990
    END_YEAR = datetime.date.today().year

    panel = build_employment_panel(FL_CITIES, START_YEAR, END_YEAR)
    print(f"Pulled employment for {panel['place_id'].nunique()} FL counties, "
          f"{panel['year'].nunique()} years "
          f"({panel['year'].min()}-{panel['year'].max()}).\n")
    for pid, meta in FL_CITIES.items():
        yrs, emp = county_series(panel, pid)
        if len(emp):
            print(f"{short_name(meta['city']):<16} {yrs[0]}-{yrs[-1]}  "
                  f"latest employment: {emp[-1]:,.0f}")

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "florida_employment_all_years.csv"
    )
    wide = panel.pivot(index="year", columns="city", values="employment")
    wide.to_csv(out)
    print(f"\nSaved wide employment series ({len(wide)} years) -> {out}")