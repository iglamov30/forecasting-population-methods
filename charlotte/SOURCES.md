# Data sources — Charlotte population growth model
Retrieved: 2026-06-11

## Time series (saved in this folder)

1. charlotte_population.csv
   - Resident Population, Charlotte-Concord-Gastonia NC-SC MSA (CBSA 16740)
   - Source: U.S. Census Bureau, Population Estimates Program (PEP), Vintage 2025
   - Retrieved via FRED series CGRPOP (updated 2026-03-27)
   - Annual, July 1 estimates, in thousands
   - NOTE: 2000-2009 values use the OLD (pre-2013) MSA delineation;
     2010-2025 use the current delineation. The 2009->2010 level jump
     (~+505k) is a BOUNDARY CHANGE, not real growth. The pipeline splices
     the two segments via growth-rate chaining.

2. charlotte_employment_monthly.csv
   - All Employees: Total Nonfarm, Charlotte MSA, seasonally adjusted, thousands
   - Source: BLS State & Metro Employment via FRED series CHAR737NA
     (1990-01 .. 2026-04, updated 2026-05-26)

3. charlotte_permits_monthly.csv
   - New Private Housing Units Authorized by Building Permits, Charlotte MSA,
     all structure types, NSA, units
   - Source: Census Building Permits Survey via FRED series CHAR737BPPRIV
     (1988-01 .. 2026-04, updated 2026-06-01)

## External anchor facts (used to calibrate the demographic-accounting model
## and scenarios; cited in README)

- Vintage 2025 (2024->2025): Charlotte metro was a top-5 US metro for numeric
  growth (after Houston, DFW, Atlanta, Phoenix). National metro growth slowed
  from ~1.1% (2024) to ~0.6% (2025), driven by a collapse in net international
  migration (~2.8M -> ~1.3M nationally, roughly -55%). Charlotte kept growing
  against that trend. (Census Bureau press materials; Axios Charlotte 2026-03-27;
  CBS News 2026-03-26)
- Mecklenburg County 2024->2025: net migration +18,278, of which +13,347
  international, +4,931 domestic. (Charlotte Business Journal, 2026-03)
- 2023->2024: Charlotte ranked 11th in numeric growth, +61k (Vintage 2024;
  revised to +66.4k in Vintage 2025).
- US crude birth rate 2025 ~10.6/1000, crude death rate ~9.1/1000. Charlotte's
  population is younger than the US average, so its natural-increase rate is
  materially higher than the national ~1.5/1000 (Mecklenburg and suburban
  counties still post natural increases; several rural metro counties have
  flipped to natural decrease).
- Average household size, Charlotte MSA: ~2.5 persons (ACS), used to translate
  population growth into household formation for the multifamily overlay.
