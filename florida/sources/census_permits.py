import io
import os
import pandas as pd
import requests

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BPS_PLACE_URL = "https://www2.census.gov/econ/bps/Place/South%20Region/so{year}a.txt"
FIRST_YEAR = 2007

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PERMITS_MANUAL_PATH = os.path.join(DATA_DIR, "permits_manual.csv")
PERMITS_TEMPLATE_PATH = os.path.join(DATA_DIR, "templates", "permits_template.csv")

COL_STATE, COL_PLACE = 1, 5
COL_UNITS_1, COL_UNITS_2, COL_UNITS_34, COL_UNITS_5PLUS = 18, 21, 24, 27


def _fetch_year(year):
    resp = requests.get(BPS_PLACE_URL.format(year=year), timeout=60)
    resp.raise_for_status()
    return pd.read_csv(
        io.StringIO(resp.text), header=None, skiprows=3, dtype=str, on_bad_lines="skip"
    )

def fetch_place_permits(fl_cities, start_year=FIRST_YEAR, end_year=None):
    if end_year is None:
        end_year = pd.Timestamp.today().year
    place_fips = {pid: pid[2:] for pid in fl_cities}   
    rows = []
    for year in range(max(start_year, FIRST_YEAR), end_year + 1):
        try:
            raw = _fetch_year(year)
        except requests.RequestException as exc:
            status = getattr(exc.response, "status_code", type(exc).__name__)
            print(f"  [permits] {year}: not available ({status})")
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            print(f"  [permits] {year}: unreadable ({type(exc).__name__}); skipped")
            continue

        fl = raw[raw[COL_STATE].str.strip() == "12"].copy()
        fl["fips"] = fl[COL_PLACE].str.strip().str.zfill(5)
        for c in (COL_UNITS_1, COL_UNITS_2, COL_UNITS_34, COL_UNITS_5PLUS):
            fl[c] = pd.to_numeric(fl[c], errors="coerce").fillna(0)

        for pid, fips in place_fips.items():
            sub = fl[fl["fips"] == fips]
            if sub.empty:
                continue
            rows.append({
                "year": year,
                "place_id": pid,
                "city": fl_cities[pid]["city"],
                "permits_total": sub[[COL_UNITS_1, COL_UNITS_2, COL_UNITS_34, COL_UNITS_5PLUS]].to_numpy().sum(),
                "permits_sf": sub[COL_UNITS_1].sum(),
                "permits_mf": sub[COL_UNITS_5PLUS].sum(),
            })

    if not rows:
        raise RuntimeError(
            "no BPS place rows matched the five cities; check the Census "
            "place files at " + BPS_PLACE_URL.format(year="<year>")
        )
    return pd.DataFrame(rows).sort_values(["place_id", "year"]).reset_index(drop=True)


def load_permits_manual(path=PERMITS_MANUAL_PATH):
    df = pd.read_csv(path, dtype={"place_id": str}, comment="#")
    return df.sort_values(["place_id", "year"]).reset_index(drop=True)


def permits_series(panel, place_id, column="permits_total"):
    sub = panel[panel["place_id"] == place_id].dropna(subset=[column])
    sub = sub.sort_values("year")
    return sub["year"].to_numpy(), sub[column].to_numpy()


if __name__ == "__main__":
    from sources.cities import FL_CITIES, short_name

    try:
        panel = fetch_place_permits(FL_CITIES)
        print("Pulled place-level permits from the Census BPS files.\n")
    except Exception as exc:
        print(f"BPS download failed ({exc}), loading the empty manual template instead.\n")
        panel = load_permits_manual(PERMITS_TEMPLATE_PATH)

    for pid, meta in FL_CITIES.items():
        yrs, vals = permits_series(panel, pid)
        if len(vals):
            print(f"{short_name(meta['city']):<16} {yrs[0]}-{yrs[-1]}  "
                  f"latest permits: {vals[-1]:,.0f}")

    if not panel.empty:
        out = os.path.join(DATA_DIR, "permits_annual.csv")
        panel.to_csv(out, index=False)
        print(f"\nSaved {out}")
