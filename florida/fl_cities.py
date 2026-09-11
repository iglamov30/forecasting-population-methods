"""
Registry of the 5 largest Florida cities (by city-proper population) and the
crosswalk data needed to join Census ACS1 place-level data with county-level
BLS employment data and airport-level tourism data.

place_id format matches census_devs.py: state FIPS (2) + place FIPS (5).
county_fips format matches BLS LAUS convention: state FIPS (2) + county FIPS (3).
"""

FL_CITIES = {
    "1235000": {
        "city": "Jacksonville city, Florida",
        "place_id": "1235000",
        "county_fips": "12031",
        "county_name": "Duval County, FL",
        "primary_airport": "JAX",
    },
    "1245000": {
        "city": "Miami city, Florida",
        "place_id": "1245000",
        "county_fips": "12086",
        "county_name": "Miami-Dade County, FL",
        "primary_airport": "MIA",
    },
    "1271000": {
        "city": "Tampa city, Florida",
        "place_id": "1271000",
        "county_fips": "12057",
        "county_name": "Hillsborough County, FL",
        "primary_airport": "TPA",
    },
    "1253000": {
        "city": "Orlando city, Florida",
        "place_id": "1253000",
        "county_fips": "12095",
        "county_name": "Orange County, FL",
        "primary_airport": "MCO",
    },
    "1263000": {
        "city": "St. Petersburg city, Florida",
        "place_id": "1263000",
        "county_fips": "12103",
        "county_name": "Pinellas County, FL",
        # St. Pete doesn't have a major airport of its own; it shares the
        # Tampa Bay air catchment (TPA) plus the smaller St Pete-Clearwater (PIE).
        "primary_airport": "TPA",
    },
}

# Quick lookup helpers
PLACE_IDS = list(FL_CITIES.keys())
COUNTY_FIPS_TO_PLACE = {v["county_fips"]: k for k, v in FL_CITIES.items()}


def short_name(full):
    return full.split(" city,")[0].split(" (")[0].strip()


if __name__ == "__main__":
    for pid, meta in FL_CITIES.items():
        print(f"{pid}  {short_name(meta['city']):<20} county={meta['county_fips']} "
              f"({meta['county_name']:<22}) airport={meta['primary_airport']}")
