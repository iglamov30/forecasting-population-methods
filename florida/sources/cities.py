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
        "primary_airport": "TPA",
    },
}

PLACE_IDS = list(FL_CITIES.keys())
COUNTY_FIPS_TO_PLACE = {v["county_fips"]: k for k, v in FL_CITIES.items()}

def short_name(full):
    return full.split(" city,")[0].split(" (")[0].strip()

CITY_NAMES = {pid: short_name(meta["city"]) for pid, meta in FL_CITIES.items()}


if __name__ == "__main__":
    for pid, meta in FL_CITIES.items():
        print(f"{pid}  {short_name(meta['city']):<20} county={meta['county_fips']} "
              f"({meta['county_name']:<22}) airport={meta['primary_airport']}")
