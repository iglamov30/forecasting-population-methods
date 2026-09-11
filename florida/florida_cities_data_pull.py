"""
florida_cities_data_pull.py

Pulls population data for ALL Florida places (cities, towns, villages, CDPs)
from the Census ACS1 API, reusing the existing census_devs.py pipeline
(same key resolution, same place_id scheme, same panel shape) rather than
hitting the API a different way.

Florida is isolated by filtering on place_id's state-FIPS prefix ("12"),
since census_devs.fetch_acs1_places() pulls every place nationwide
(for=place:*, in=state:*) in a single call per year.

USAGE
-----
    python florida_cities_data_pull.py
    python florida_cities_data_pull.py --years 2018 2019 2021 2022 2023
    python florida_cities_data_pull.py --start-year 2010 --output fl_all.csv

Note: ACS1 (1-year estimates) only publishes for places with population
>= 65,000, and there is no 2020 vintage (Census suspended ACS1 that year),
matching the skip-2020 pattern used elsewhere in this project.

SETUP
-----
Same as the rest of the project: put CENSUS_API_KEY in secret.py or the
environment. See census_devs.py / test_census.py.
"""

import argparse
from datetime import datetime

import pandas as pd

from census_devs import build_panel, list_cities

FLORIDA_STATE_FIPS = "12"


def filter_florida(panel: pd.DataFrame) -> pd.DataFrame:
    """Restrict a nationwide census_devs panel to Florida places only."""
    return panel[panel["place_id"].str.startswith(FLORIDA_STATE_FIPS)].reset_index(drop=True)


def clean_city_name(name: str) -> str:
    """Strip Census suffixes like ', Florida' and city/town/CDP labels."""
    name = name.replace(", Florida", "")
    for suffix in [" city", " town", " CDP", " village", " municipality"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Pull population data for all Florida places via census_devs.py."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of years to pull (e.g. --years 2019 2021 2022 2023). "
        "Overrides --start-year if given.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="First year to pull through the current year (2020 skipped automatically). "
        "Ignored if --years is given.",
    )
    parser.add_argument(
        "--output",
        default="florida_cities_population.csv",
        help="Output CSV file path.",
    )
    args = parser.parse_args()

    if args.years:
        years = sorted(args.years)
    else:
        years = [y for y in range(args.start_year, datetime.now().year + 1) if y != 2020]

    print(f"Fetching nationwide ACS1 panel for {years[0]}-{years[-1]} "
          f"({len(years)} year(s), skipping 2020 if present)...")
    panel = build_panel(years)

    fl_panel = filter_florida(panel)
    fl_panel["city_clean"] = fl_panel["city"].apply(clean_city_name)

    if fl_panel.empty:
        print("No Florida places found in the pulled panel — check your years/key.")
        return

    fl_panel = fl_panel.sort_values(["place_id", "year"]).reset_index(drop=True)
    fl_panel.to_csv(args.output, index=False)

    n_places = fl_panel["place_id"].nunique()
    n_years = fl_panel["year"].nunique()
    print(f"\nDone. {n_places} Florida places across {n_years} year(s).")
    print(f"Saved to: {args.output}\n")

    print("Top 10 Florida places by latest population:")
    top10 = list_cities(fl_panel).head(10)
    print(top10.assign(population=top10["population"].map("{:,.0f}".format)).to_string())


if __name__ == "__main__":
    main()
