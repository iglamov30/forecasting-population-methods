"""
Pull ACS1 population (2005-present) for the 5 largest Florida cities.
Run this in your own environment (needs api.census.gov access + your
Census key in secret.py or CENSUS_API_KEY env var).
"""

from historical_population import acs1_frame
from fl_cities import FL_CITIES, short_name

df = acs1_frame(start_year=2005)

out = "fl_top5_acs1_2005_present.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows -> {out}\n")

for pid, meta in FL_CITIES.items():
    sub = df[df["place_id"] == pid].sort_values("year")
    if sub.empty:
        print(f"{short_name(meta['city']):<16} no data returned")
        continue
    print(f"{short_name(meta['city']):<16} {int(sub['year'].min())}-{int(sub['year'].max())} "
          f"({len(sub)} yrs)  latest: {int(sub['population'].iloc[-1]):,}")
