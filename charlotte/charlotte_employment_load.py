# Pull Charlotte metro area employment data directly from the BLS API, output to csv

import datetime
import os
import time
import requests

BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
CHARLOTTE_MSA_SERIES_ID = "LAUMT371674000000005"

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
        return None  


def fetch_laus(series_id, start_year, end_year, key=None):
    key = _resolve_key(key)
    all_rows = []
    years = list(range(start_year, end_year + 1))
    year_chunks = [years[i:i + 19] for i in range(0, len(years), 19)]

    for chunk in year_chunks:
        payload = {
            "seriesid": [series_id],
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
            for item in series["data"]:
                all_rows.append({
                    "year": int(item["year"]),
                    "period": item["period"],  # M01..M12
                    "value": float(item["value"]) if item["value"] not in ("-", "") else None,
                })

        time.sleep(0.5)  # be polite to the API
    return all_rows


def build_annual_employment(start_year, end_year, key=None):
    rows = fetch_laus(CHARLOTTE_MSA_SERIES_ID, start_year, end_year, key=key)
    if not rows:
        raise RuntimeError("BLS API returned no data — check your key and date range.")
    by_year = {}
    for row in rows:
        if row["value"] is None or row["period"] == "M13":
            continue
        by_year.setdefault(row["year"], []).append(row["value"])

    return {year: sum(vals) / len(vals) for year, vals in sorted(by_year.items())}


if __name__ == "__main__":
    START_YEAR = 1990
    END_YEAR = datetime.date.today().year

    annual = build_annual_employment(START_YEAR, END_YEAR)
    years = sorted(annual)
    print(f"Pulled Charlotte MSA employment for {len(years)} years "
          f"({years[0]}-{years[-1]}).\n"
          f"Latest ({years[-1]}) employment: {annual[years[-1]]:,.0f}")

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "charlotte_employment_all_years.csv"
    )
    with open(out, "w") as f:
        f.write("year,employment\n")
        for year in years:
            f.write(f"{year},{annual[year]:.4f}\n")
    print(f"\nSaved annual employment series ({len(years)} years) -> {out}")
