import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

CSV = Path(__file__).resolve().parent / "all_US_cities_forecast.csv"
YEARS = [2025, 2026, 2027, 2028, 2029]

df = pd.read_csv(CSV)

for y in YEARS:
    df[f"{y}_fc"]    = df[f"{y}_forecast"]
    df[f"{y}_lo"]    = df[f"{y}_lower"]
    df[f"{y}_hi"]    = df[f"{y}_upper"]

def best_mae(row):
    return row["mae_ets"] if row["best_model"] == "ets" else row["mae_arima"]

def best_rmse(row):
    return row["rmse_ets"] if row["best_model"] == "ets" else row["rmse_arima"]

df["best_mae"]  = df.apply(best_mae,  axis=1)
df["best_rmse"] = df.apply(best_rmse, axis=1)

top5 = df.nlargest(5, "pct_change_5y").reset_index(drop=True)

def short_name(full):
    return full.split(" city,")[0].split(" metro")[0].split(" (")[0].strip()

top5["label"] = top5["city"].apply(short_name)

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey=False)
fig.suptitle(
    "Top 5 Cities by 5-Year Projected Population Growth\n"
    "Shaded bands = 90% prediction interval from best forecasting model",
    fontsize=13, fontweight="bold", y=0.985
)

for ax, (_, row), color in zip(axes, top5.iterrows(), PALETTE):
    x_hist  = [row["last_year"]]
    y_hist  = [row["last_population"]]

    x_fc    = YEARS
    y_fc    = [row[f"{y}_fc"] for y in YEARS]
    y_lo    = [row[f"{y}_lo"] for y in YEARS]
    y_hi    = [row[f"{y}_hi"] for y in YEARS]

    x_all = x_hist + x_fc
    y_all = y_hist + y_fc

    ax.plot(x_all, y_all, color=color, linewidth=2.2, marker="o", markersize=4)
    ax.fill_between(x_fc, y_lo, y_hi, color=color, alpha=0.18, label="_nolegend_")

    ax.scatter(x_hist, y_hist, color="white", edgecolors=color,
               zorder=5, linewidth=2, s=60)

    pct = row["pct_change_5y"]
    ax.set_title(f"{row['label']}\n+{pct:.1f}% by 2029", fontsize=10, fontweight="bold")

    mae  = row["best_mae"]
    rmse = row["best_rmse"]
    mdl  = row["best_model"].upper()
    ax.text(
        0.04, 0.97,
        f"Model: {mdl}\nMAE: {mae:,.0f}\nRMSE: {rmse:,.0f}",
        transform=ax.transAxes,
        fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=color, alpha=0.85)
    )

    ax.set_xlim(2023.5, 2029.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1_000:.0f}K" if v < 1_000_000 else f"{v/1_000_000:.2f}M"
    ))
    ax.set_xlabel("Year", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

fig.text(
    0.5, 0.015,
    "Narrower bands mean a more accurate model (lower MAE/RMSE). "
    "Actual values will fall outside the bands about 10% of the time.",
    ha="center", fontsize=8.5, color="#555555", style="italic"
)

plt.tight_layout(rect=[0, 0.05, 1, 1])

print("\nTop 5 cities by projected 5-year growth:")
print(f"{'City':<40} {'Growth':>8}  {'Model':>6}  {'MAE':>10}  {'RMSE':>10}")
print("-" * 80)
for _, row in top5.iterrows():
    print(
        f"{row['label']:<40} {row['pct_change_5y']:>7.2f}%"
        f"  {row['best_model'].upper():>6}"
        f"  {row['best_mae']:>10,.0f}"
        f"  {row['best_rmse']:>10,.0f}"
    )

plt.show()