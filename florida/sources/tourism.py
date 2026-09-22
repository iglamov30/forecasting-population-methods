"""Tourism signals: county bed tax collections and airport passengers.

Both series are typed in by hand; the loaders below only read the files.

Tourist Development Tax ("bed tax"), by county:

  All counties in one file, the quickest way to fill in tdt_manual.csv.
  EDR, Local Option Tourist Taxes, "Realized and Unrealized Revenues by
  County" - one sheet per state fiscal year, estimated realized revenue:
    https://edr.state.fl.us/Content/local-government/data/data-a-to-z/realizedtourism.xlsx
    index: https://edr.state.fl.us/Content/local-government/data/data-a-to-z/g-l.cfm

  Actual collections, from the office that collects them:
    Duval / Jacksonville:
      https://www.coj.net/departments/tax-collector/convention-tourist-development-tax
    Miami-Dade / Miami:
      https://www.miamidade.gov/global/service.page?Mduid_service=ser1499797928395868
    Hillsborough / Tampa:
      https://www.hillstax.org/taxes/tourist-development-tax/
    Orange / Orlando:
      https://www.occompt.com/finance/tourist-development-tax/
    Pinellas / St. Petersburg:
      https://pinellastaxcollector.gov/property-tax/tourist-development-taxes/

Airport enplanements:
    BTS T-100 Market: https://www.transtats.bts.gov
    FAA passenger boarding data:
      https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger
"""

import os
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TDT_MANUAL_PATH = os.path.join(DATA_DIR, "tdt_manual.csv")
ENPLANEMENTS_MANUAL_PATH = os.path.join(DATA_DIR, "enplanements_manual.csv")


def load_tdt_manual(path=TDT_MANUAL_PATH):
    df = pd.read_csv(path, dtype={"place_id": str}, comment="#")
    return df.sort_values(["place_id", "year"]).reset_index(drop=True)

def load_enplanements_manual(path=ENPLANEMENTS_MANUAL_PATH):
    df = pd.read_csv(path, dtype={"place_id": str}, comment="#")
    return df.sort_values(["place_id", "year"]).reset_index(drop=True)

def tourism_series(panel, place_id, column):
    sub = panel[panel["place_id"] == place_id].dropna(subset=[column])
    sub = sub.sort_values("year")
    return sub["year"].to_numpy(), sub[column].to_numpy()


if __name__ == "__main__":
    from sources.cities import FL_CITIES, short_name
    for loader, path, col, label in [
        (load_tdt_manual, TDT_MANUAL_PATH, "tdt_collections", "TDT"),
        (load_enplanements_manual, ENPLANEMENTS_MANUAL_PATH, "enplanements", "Enplanements"),
    ]:
        if not os.path.exists(path):
            print(f"[{label}] {os.path.basename(path)} not found; copy it from data/templates/.")
            continue
        panel = loader(path)
        if panel.empty:
            print(f"[{label}] {os.path.basename(path)} has no rows yet.")
            continue
        for pid, meta in FL_CITIES.items():
            yrs, vals = tourism_series(panel, pid, col)
            if len(vals):
                print(f"[{label}] {short_name(meta['city']):<16} "
                      f"{yrs[0]}-{yrs[-1]}  latest: {vals[-1]:,.0f}")
