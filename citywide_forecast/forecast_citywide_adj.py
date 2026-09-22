from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from census_acs1 import build_panel, city_series

CSV = Path(__file__).resolve().parent / "all_US_cities_forecast.csv"
YEARS = [2025, 2026, 2027, 2028, 2029]
NEW_YEAR = 2030
ALL_YEARS = YEARS + [NEW_YEAR]

HIST_START_YEAR = 2010
MIN_POPULATION = 250_000

# Load data
df = pd.read_csv(CSV, dtype={"place_id": str})

# Large cities
n_before = len(df)
df = df[df["last_population"] >= MIN_POPULATION].copy()
n_after = len(df)

print(
    f"Floor(>= {MIN_POPULATION:,} population): "
    f"{n_after}/{n_before} cities remain eligible.\n"
)

# 2030 numbers
growth_ratio = df["2029_forecast"] / df["2028_forecast"].replace(0, np.nan)
df["2030_fc"] = df["2029_forecast"] * growth_ratio.fillna(1.0)

width_2028 = df["2028_upper"] - df["2028_lower"]
width_2029 = df["2029_upper"] - df["2029_lower"]
widen_ratio = width_2029 / width_2028.replace(0, np.nan)
half_width_2030 = (width_2029 * widen_ratio.fillna(1.0)) / 2

df["2030_lo"] = df["2030_fc"] - half_width_2030
df["2030_hi"] = df["2030_fc"] + half_width_2030

# 2030 growth
df["pct_change_to_2030"] = (
    (df["2030_fc"] - df["last_population"])
    / df["last_population"]
    * 100
)

# Select correct model errors without apply()
ets_mask = df["best_model"].eq("ets")
df["best_mae"] = np.where(
    ets_mask,
    df["mae_ets"],
    df["mae_arima"],
)
df["best_rmse"] = np.where(
    ets_mask,
    df["rmse_ets"],
    df["rmse_arima"],
)

df["mae_pct"] = df["best_mae"] / df["last_population"] * 100
df["rmse_pct"] = df["best_rmse"] / df["last_population"] * 100

EPS = 1e-6
df["risk_adjusted_score"] = (
    df["pct_change_to_2030"] /
    (df["rmse_pct"] + EPS)
)

top5 = df.nlargest(5, "risk_adjusted_score").reset_index(drop=True)


def short_name(full):
    return full.split(" city,")[0].split(" metro")[0].split(" (")[0].strip()


top5["label"] = top5["city"].apply(short_name)

current_year = datetime.now().year
hist_years_to_pull = [
    y for y in range(HIST_START_YEAR, current_year + 1)
    if y != 2020
]


# Build
hist_panel = build_panel(hist_years_to_pull)

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

for (_, row), color in zip(top5.iterrows(), PALETTE):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    hist_yrs, hist_pop = city_series(hist_panel, row["place_id"])
    x_hist = hist_yrs.tolist()
    y_hist = hist_pop.tolist()

    x_fc = ALL_YEARS
    y_fc = [
        row[f"{y}_forecast"] if y != NEW_YEAR else row["2030_fc"]
        for y in ALL_YEARS
    ]
    y_lo = [
        row[f"{y}_lower"] if y != NEW_YEAR else row["2030_lo"]
        for y in ALL_YEARS
    ]
    y_hi = [
        row[f"{y}_upper"] if y != NEW_YEAR else row["2030_hi"]
        for y in ALL_YEARS
    ]

    ax.plot(
        x_hist + x_fc,
        y_hist + y_fc,
        color=color,
        linewidth=2.2,
        marker="o",
        markersize=4,
    )

    ax.fill_between(
        x_fc,
        y_lo,
        y_hi,
        color=color,
        alpha=0.18,
    )

    ax.scatter(
        [x_hist[-1]],
        [y_hist[-1]],
        color="white",
        edgecolors=color,
        zorder=5,
        linewidth=2,
        s=60,
    )

    ax.scatter(
        [NEW_YEAR],
        [row["2030_fc"]],
        color=color,
        edgecolors="black",
        zorder=6,
        linewidth=1.2,
        s=70,
        marker="D",
    )

    ax.set_title(
        f"{row['label']}\n+{row['pct_change_to_2030']:.1f}% by {NEW_YEAR}",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(
        0.04,
        0.97,
        f"Model: {row['best_model'].upper()}\n"
        f"MAE: {row['mae_pct']:.1f}% of pop.\n"
        f"RMSE: {row['rmse_pct']:.1f}% of pop.\n"
        f"Score: {row['risk_adjusted_score']:.2f}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=color,
            alpha=0.85,
        ),
    )

    ax.set_xlim(x_hist[0] - 0.5, NEW_YEAR + 0.8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda v, _: f"{v/1_000:.0f}K"
            if v < 1_000_000
            else f"{v/1_000_000:.2f}M"
        )
    )
    ax.set_xlabel("Year", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

print(
    f"\nTop 5 cities by risk-adjusted growth through {NEW_YEAR} "
    f"(population >= {MIN_POPULATION:,}):"
)

for _, row in top5.iterrows():
    print(
        f"{row['label']:<40} "
        f"{row['pct_change_to_2030']:>7.2f}%  "
        f"{row['best_model'].upper():>6}  "
        f"{row['mae_pct']:>6.1f}%  "
        f"{row['rmse_pct']:>6.1f}%  "
        f"{row['risk_adjusted_score']:>7.2f}"
    )

plt.show()
