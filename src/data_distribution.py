import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy.stats as stats

'''
Allows to visualize data to assess if a logarithmic transformation of the output variable (y) is viable.
'''

# Load CSV file
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
Training_File = "regressor_dataset_16.csv"
DATA_PATH = project_root / "data" / "datasets" / "dataset_16" / Training_File
df = pd.read_csv(DATA_PATH)

# Extract last column
y = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna()

# Take log of the data (shift slightly if any zeros or negatives exist)
y_shifted = y[y > 0]  # log requires strictly positive values
log_y = np.log(y_shifted)

# --- Summary statistics ---
print("Raw y statistics:")
print(y.describe())
print("\nLog(y) statistics:")
print(log_y.describe())

# Skewness check
print(f"\nSkewness of y: {stats.skew(y)}")
print(f"Skewness of log(y): {stats.skew(log_y)}")

# --- Visualization ---
plt.figure(figsize=(14, 8))

# Histogram of y
plt.subplot(2, 2, 1)
sns.histplot(y, bins=50, kde=True)
plt.title("Histogram of y (raw data)")

# Histogram of log(y)
plt.subplot(2, 2, 2)
sns.histplot(log_y, bins=50, kde=True, color="orange")
plt.title("Histogram of log(y)")

# Q-Q plot of log(y)
plt.subplot(2, 2, 3)
stats.probplot(log_y, dist="norm", plot=plt)
plt.title("Q-Q Plot of log(y) vs Normal")

# Boxplot to show skewness
plt.subplot(2, 2, 4)
sns.boxplot(x=y)
plt.title("Boxplot of y")

plt.tight_layout()
plt.show()
