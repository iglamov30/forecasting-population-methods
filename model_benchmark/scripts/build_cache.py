import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cache import DATA_DIR, get_or_fetch_year
from core.census import POP_VAR

import pandas as pd

POP_THRESHOLD = 250_000
PANEL_OUT = os.path.join(os.path.dirname(DATA_DIR), "population_panel.parquet")


def target_years():
    latest_possible = datetime.date.today().year - 1
    return [y for y in range(2005, latest_possible + 1) if y != 2020]


def main():
    years = target_years()
    frames = []
    fetched_years = []
    for y in years:
        try:
            frames.append(get_or_fetch_year(y, [POP_VAR]))
            fetched_years.append(y)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  skip {y}: {exc}")

    if not frames:
        raise RuntimeError("No years fetched -- check CENSUS_API_KEY.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["value"]).rename(columns={"value": "population"})
    panel = panel.drop(columns=["variable"])

    latest = panel.sort_values("year").groupby("place_id").last()["population"]
    keep_ids = latest[latest >= POP_THRESHOLD].index
    filtered = panel[panel["place_id"].isin(keep_ids)].sort_values(
        ["place_id", "year"]
    ).reset_index(drop=True)

    filtered.to_parquet(PANEL_OUT, index=False)

    counts = filtered.groupby("place_id").size()
    print(f"Years fetched ({len(fetched_years)}): {fetched_years}")
    print(f"Cities retained (population >= {POP_THRESHOLD:,} in latest year): {len(keep_ids)}")
    print(
        "Observations per city -- "
        f"min: {counts.min()}, median: {counts.median():.1f}, max: {counts.max()}"
    )
    print(f"Saved panel -> {PANEL_OUT}")


if __name__ == "__main__":
    main()
