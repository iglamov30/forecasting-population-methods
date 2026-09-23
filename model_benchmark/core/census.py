import os
import pandas as pd
import requests

POP_VAR = "B01003_001E"

ACS1_BASE_URL = "https://api.census.gov/data/{year}/acs/acs1"


def _resolve_key(key=None):
    if key:
        return key
    env = os.environ.get("CENSUS_API_KEY")
    if env:
        return env
    raise RuntimeError(
        "No Census API key found. Pass key=... or set the CENSUS_API_KEY "
        "environment variable."
    )


def fetch_acs1_places(year, variables, key=None):
    key = _resolve_key(key)
    variables = list(variables)

    url = ACS1_BASE_URL.format(year=year)
    params = {
        "get": f"NAME,{','.join(variables)}",
        "for": "place:*",
        "in": "state:*",
        "key": key,
    }
    resp = requests.get(url, params=params, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        # Error messages embed the full request URL, key included. Never let
        # that reach a print()/log -- sanitize before it leaves this function.
        sanitized = str(exc).replace(key, "***")
        raise requests.HTTPError(sanitized) from None

    rows = resp.json()
    wide = pd.DataFrame(rows[1:], columns=rows[0])

    wide = wide.rename(columns={"NAME": "city"})
    wide["year"] = int(year)
    wide["place_id"] = wide["state"].str.zfill(2) + wide["place"].str.zfill(5)

    long_df = wide.melt(
        id_vars=["year", "place_id", "city"],
        value_vars=variables,
        var_name="variable",
        value_name="value",
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    return long_df[["year", "place_id", "city", "variable", "value"]]


def build_panel(years, variables, key=None):
    frames = []
    fetched_years = []
    for y in years:
        try:
            frames.append(fetch_acs1_places(y, variables, key=key))
            fetched_years.append(y)
        except requests.HTTPError as exc:
            print(f"  skip {y}: {exc}")
    if not frames:
        raise RuntimeError("No data returned for any requested year.")
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["place_id", "variable", "year"]).reset_index(drop=True), fetched_years


def city_series(panel, place_id, variable=POP_VAR):
    sub = panel[(panel["place_id"] == place_id) & (panel["variable"] == variable)]
    sub = sub.dropna(subset=["value"]).sort_values("year")
    return sub["year"].to_numpy(), sub["value"].to_numpy()


def list_cities(panel, variable=POP_VAR):
    sub = panel[panel["variable"] == variable]
    latest = sub.sort_values("year").groupby("place_id").last()
    return latest.sort_values("value", ascending=False)[["city", "value"]]
