"""
Smoke test for the Census API connection.

Run this AFTER putting your key in secret.py:
    python test_census.py

It checks four things in order and prints PASS/FAIL for each:
  1. the key loads from secret.py
  2. a single-city request returns data (Charlotte, NC)
  3. the returned population is a plausible number
  4. a full-year pull returns many cities and ranks them
"""

import sys
from census_devs import _resolve_key, fetch_acs1_places, build_panel, list_cities

CHARLOTTE_ID = "3712000"     # state 37 (NC) + place 12000
TEST_YEAR = 2023             # a year ACS 1-year is available


def main():
    ok = True

    # 1) key present
    key = _resolve_key()
    if key:
        print(f"[PASS] key loaded (ends in ...{key[-4:]})")
    else:
        print("[FAIL] no key found — add it to secret.py")
        return 1

    # 2) single-city request
    try:
        df = fetch_acs1_places(TEST_YEAR)
        print(f"[PASS] API responded — {len(df)} places for {TEST_YEAR}")
    except Exception as exc:
        print(f"[FAIL] request failed: {exc}")
        return 1

    # 3) sanity-check Charlotte's value
    row = df[df["place_id"] == CHARLOTTE_ID]
    if len(row) and 700_000 < row["population"].iloc[0] < 1_200_000:
        pop = int(row["population"].iloc[0])
        print(f"[PASS] Charlotte {TEST_YEAR} population = {pop:,} (plausible)")
    else:
        print("[FAIL] Charlotte value missing or out of expected range")
        ok = False

    # 4) multi-year panel + ranking
    try:
        panel = build_panel([2022, 2023])
        top = list_cities(panel).head(5)
        print(f"[PASS] panel built — {panel['place_id'].nunique()} cities")
        print("\nTop 5 cities by latest population:")
        print(top.assign(population=top["population"].map("{:,.0f}".format)).to_string())
    except Exception as exc:
        print(f"[FAIL] panel build failed: {exc}")
        ok = False

    print("\nAll good ✅" if ok else "\nSome checks failed ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
