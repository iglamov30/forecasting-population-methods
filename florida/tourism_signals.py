"""
Tourism demand signals, split by relevance to hotels vs. multifamily.

DATA REALITY CHECK:
- Tourist Development Tax (TDT / "bed tax") collections are the best public
  proxy for hotel demand in Florida, but every county publishes them
  differently (PDF reports, ad hoc Excel files, some via open-data portals)
  and there is no unified statewide API. This module loads a manual CSV you
  fill in from each county's tourism authority / tax collector site:
    - Duval:      https://www.coj.net  (search "tourist development tax")
    - Miami-Dade: https://www.miamidade.gov/global/economy/tourism.page
    - Hillsborough/Pinellas: https://www.visittampabay.com,
      https://www.visitstpeteclearwater.com (both publish TDT collection reports)
    - Orange:     https://www.occompt.com  (Orange County Comptroller, TDT reports)

- Airport enplanements (BTS T-100 data) are a good secondary hotel-demand
  proxy and are somewhat more standardized, but still require a manual pull
  from https://www.transtats.bts.gov (Air Carriers: T-100 Domestic Market)
  since there's no simple key-based REST endpoint for it either.

If your firm has an STR/CoStar/AirDNA license, replace these manual loaders
with real API pulls — this module is structured so any of these functions
can be swapped for a live fetch without touching the composite-score logic
downstream, as long as the returned panel has the same [year, place_id,
value] shape.
"""

import pandas as pd


def load_tdt_manual(path="tdt_manual.csv"):
    """
    Expected columns: year, place_id, city, tdt_collections
    (annual TDT / bed-tax collections in nominal USD)
    """
    df = pd.read_csv(path, dtype={"place_id": str})
    return df.sort_values(["place_id", "year"]).reset_index(drop=True)


def load_enplanements_manual(path="enplanements_manual.csv"):
    """
    Expected columns: year, place_id, city, airport, enplanements
    (annual total enplaned passengers at the city's primary airport)
    """
    df = pd.read_csv(path, dtype={"place_id": str})
    return df.sort_values(["place_id", "year"]).reset_index(drop=True)


def tourism_series(panel, place_id, column):
    sub = panel[panel["place_id"] == place_id].dropna(subset=[column])
    sub = sub.sort_values("year")
    return sub["year"].to_numpy(), sub[column].to_numpy()


if __name__ == "__main__":
    from fl_cities import FL_CITIES, short_name

    for loader, col, label in [
        (load_tdt_manual, "tdt_collections", "TDT"),
        (load_enplanements_manual, "enplanements", "Enplanements"),
    ]:
        try:
            panel = loader(f"{label.lower()}_template.csv")
        except FileNotFoundError:
            print(f"[{label}] no manual file found yet — see docstring for source links.")
            continue
        for pid, meta in FL_CITIES.items():
            yrs, vals = tourism_series(panel, pid, col)
            if len(vals):
                print(f"[{label}] {short_name(meta['city']):<16} "
                      f"{yrs[0]}-{yrs[-1]}  latest: {vals[-1]:,.0f}")
