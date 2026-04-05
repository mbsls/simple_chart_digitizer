"""
plot_timeseries.py
------------------
Plots all series from timeseries.csv on one chart and saves as PNG.

Usage:
    python plot_timeseries.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "timeseries.csv"
OUT_PATH = SCRIPT_DIR / "timeseries_plot.png"

df = pd.read_csv(CSV_PATH, parse_dates=["date"])

fig, ax = plt.subplots(figsize=(12, 5))
for col in df.columns[1:]:
    ax.plot(df["date"], df[col], label=col)

ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Saved: {OUT_PATH}")
