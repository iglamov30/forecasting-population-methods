from datetime import datetime

import pandas as pd
from census_devs import build_panel

START_YEAR = 2005          

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


def population_matrix(years=None, start_year=START_YEAR, sort_by_size=True):
    if years is None:
        years = list(range(start_year, datetime.now().year + 1))
    panel = build_panel(years)        
    return to_matrix(panel, start_year=start_year, sort_by_size=sort_by_size)


if __name__ == "__main__":
    mat = population_matrix()
    print(f"matrix shape: {mat.shape[0]} years x {mat.shape[1]} cities\n")
    print(mat.iloc[:, 60:65].to_string())
    out = "city_population_matrix.csv"
    mat.to_csv(out)
