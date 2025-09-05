import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# === Setup and Configuration === #
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Path to regressor dataset
# Folder where model grid search result is stored
Training_File = "Airfoils_for_Regressor_E_Dataset_16.csv"
DATA_PATH = project_root / Training_File
# Make sure your CSV has a column called "R2" or change this accordingly
df = pd.read_csv(DATA_PATH)

# Extract the target column
y = df["E"].values

percentile_bins = [0, 35, 75, 90, 100]
cutoffs = np.percentile(y, percentile_bins)

ranges = []
labels_pct = []
for i in range(len(percentile_bins) - 1):
    low, high = cutoffs[i], cutoffs[i+1]
    mask = (y >= low) & (y < high) if i < len(percentile_bins)-2 else (y >= low) & (y <= high)
    if mask.sum() > 0:
        min_val, max_val = y[mask].min(), y[mask].max()
        ranges.append((min_val, max_val))
        labels_pct.append(f"{percentile_bins[i]}–{percentile_bins[i+1]}%")

# Plot percentile min–max bars
plt.figure(figsize=(8,4))
for i, (low, high) in enumerate(ranges):
    plt.barh(i, high-low, left=low, color="skyblue", edgecolor="k")
plt.yticks(range(len(labels_pct)), labels_pct)
plt.xlabel("E")
plt.title(f"Min–Max Values per Percentile Range of E")
plt.grid(axis='x', linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Compute percentile cutoffs
value_bins = [23, 32, 38, 40, 70]  # your cutoffs
cutoffs = np.percentile(y, percentile_bins)

# Count samples in each bin
counts = []
labels = []
for i in range(len(value_bins) - 1):
    low, high = value_bins[i], value_bins[i+1]
    mask = (y >= low) & (y < high) if i < len(value_bins)-2 else (y >= low) & (y <= high)
    counts.append(mask.sum())
    labels.append(f"{low}–{high}")

# Pie chart
scaled_counts = np.sqrt(counts)

plt.figure(figsize=(7,7))
wedges, texts, autotexts = plt.pie(
    scaled_counts,
    labels=labels,
    autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100*sum(counts)))} pts)",
    startangle=90,
    counterclock=False
)
plt.title(f"Distribution of E by Value Ranges (scaled view)")
plt.show()