"""
Housing permit data (Census Building Permits Survey, "BPS").

IMPORTANT CAVEAT (read before trusting this blindly):
Unlike ACS1 population or BLS LAUS employment, the Building Permits Survey
does NOT have a clean, reliable, key-based JSON API at the place level going
back many years. Census exposes a "timeseries/eits/bps" API that mostly
covers national/regional/state aggregates and CBSA-level monthly permits —
county- or place-level annual history is distributed as flat survey files
(https://www2.census.gov/econ/bps/) that change format across years and are
genuinely painful to parse generically and reliably.

Given that, this module does two things:
  1. try_fetch_cbsa_permits() — a best-effort pull from the EITS API at CBSA
     level (Census Bureau-defined metro areas), which IS reliably available
     year over year and is arguably the more correct geography for permits
     anyway (permits get pulled at the metro/county level, not city-proper).
  2. load_permits_manual() — a loader for a manual CSV you fill in yourself
     from the Census BPS place-level annual tables
     (https://www.census.gov/construction/bps/), which is the more accurate
     but manual route. A template is provided: permits_template.csv

Recommendation: use try_fetch_cbsa_permits() to get moving fast, but if this
matters for a real investment decision, cross-check / replace with the
manual place-level pull before relying on it.
"""

import requests
import pandas as pd

EITS_URL = "https://api.census.gov/data/timeseries/eits/bps"

# Census CBSA codes for the metros containing our 5 FL cities
FL_CBSAS = {
    "1235000": {"cbsa": "27260", "name": "Jacksonville, FL"},
    "1245000": {"cbsa": "33100", "name": "Miami-Fort Lauderdale-Pompano Beach, FL"},
    "1271000": {"cbsa": "45300", "name": "Tampa-St. Petersburg-Clearwater, FL"},
    "1253000": {"cbsa": "36740", "name": "Orlando-Kissimmee-Sanford, FL"},
    "1263000": {"cbsa": "45300", "name": "Tampa-St. Petersburg-Clearwater, FL"},
}


def try_fetch_cbsa_permits(fl_cities, start_year, end_year, key=None):
    """
    Best-effort pull of annual total authorized housing units by CBSA from
    the Census EITS/BPS API. Returns a panel shaped like the population
    panel: [year, place_id, city, permits_total].

    This can fail or return partial data depending on Census API uptime and
    year coverage — wrap calls to this in a try/except and fall back to
    load_permits_manual() if it does.
    """
    from census_devs import _resolve_key
    key = _resolve_key(key)

    rows = []
    seen_cbsas = set(v["cbsa"] for v in FL_CBSAS.values())

    for cbsa in seen_cbsas:
        params = {
            "get": "cell_value,time_slot_id",
            "for": f"metropolitan statistical area/micropolitan statistical area:{cbsa}",
            "time": f"from {start_year} to {end_year}",
            "category_code": "TOTAL",
            "data_type_code": "TOTAL",
        }
        if key:
            params["key"] = key
        try:
            resp = requests.get(EITS_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  [permits] CBSA {cbsa} fetch failed: {exc}")
            continue

        if not data or len(data) < 2:
            continue
        cols = data[0]
        for r in data[1:]:
            rec = dict(zip(cols, r))
            rows.append({"cbsa": cbsa, **rec})

    if not rows:
        raise RuntimeError(
            "No permit data returned from EITS API — fall back to "
            "load_permits_manual() with a hand-downloaded BPS file."
        )

    raw = pd.DataFrame(rows)
    # EITS "time" field is usually like "2023-12" (monthly) — aggregate to annual
    raw["year"] = raw["time"].str[:4].astype(int) if "time" in raw.columns else None
    raw["cell_value"] = pd.to_numeric(raw["cell_value"], errors="coerce")
    annual = raw.groupby(["cbsa", "year"], as_index=False)["cell_value"].sum()
    annual = annual.rename(columns={"cell_value": "permits_total"})

    out_rows = []
    for pid, meta in fl_cities.items():
        cbsa = FL_CBSAS[pid]["cbsa"]
        sub = annual[annual["cbsa"] == cbsa]
        for _, r in sub.iterrows():
            out_rows.append({
                "year": int(r["year"]),
                "place_id": pid,
                "city": meta["city"],
                "permits_total": r["permits_total"],
            })

    return pd.DataFrame(out_rows).sort_values(["place_id", "year"]).reset_index(drop=True)


def load_permits_manual(path="permits_manual.csv"):
    """
    Load hand-entered permit data. Expected columns:
        year, place_id, city, permits_total, permits_sf, permits_mf
    (permits_sf = single-family authorized units, permits_mf = 5+ unit
    structures — the multifamily-relevant split.)
    See permits_template.csv for the exact format and a data source note.
    """
    df = pd.read_csv(path, dtype={"place_id": str})
    return df.sort_values(["place_id", "year"]).reset_index(drop=True)


def permits_series(panel, place_id, column="permits_total"):
    sub = panel[panel["place_id"] == place_id].dropna(subset=[column])
    sub = sub.sort_values("year")
    return sub["year"].to_numpy(), sub[column].to_numpy()


if __name__ == "__main__":
    from fl_cities import FL_CITIES, short_name

    try:
        panel = try_fetch_cbsa_permits(FL_CITIES, 2010, 2024)
        print("Pulled CBSA-level permits via EITS API.\n")
    except Exception as exc:
        print(f"EITS pull failed ({exc}), loading manual template instead.\n")
        panel = load_permits_manual("permits_template.csv")

    for pid, meta in FL_CITIES.items():
        yrs, vals = permits_series(panel, pid)
        if len(vals):
            print(f"{short_name(meta['city']):<16} {yrs[0]}-{yrs[-1]}  "
                  f"latest permits: {vals[-1]:,.0f}")
