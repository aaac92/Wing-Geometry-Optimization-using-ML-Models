import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load CSV file
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
Training_File = "Airfoils_for_Regressor_E_Dataset_16.csv"
DATA_PATH = project_root / Training_File
df = pd.read_csv(DATA_PATH)

# Extract last column
y = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna()
threshold = np.percentile(y, 99.6)
df_top = df[y >= threshold]
rep_factor = 0
if rep_factor == 0:
    df = df
else:
    df = pd.concat([df, pd.concat([df_top]*rep_factor, ignore_index=True)], ignore_index=True)
y = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna()

# Estimate lambda (mean of the data)
lambda_est = y.mean()

print(f"Estimated λ (lambda): {lambda_est:.4f}")
print(f"Sample variance: {y.var():.4f}")

# Plot histogram with Poisson PMF overlay
plt.figure(figsize=(8, 6))
counts, bins, _ = plt.hist(y, bins=30, edgecolor='black', alpha=0.7, density=True)

# Overlay Poisson distribution using estimated λ
x = np.arange(int(y.min()), int(y.max())+1)
poisson_pmf = (np.exp(-lambda_est) * (lambda_est**x) / np.array([math.factorial(i) for i in x]))

plt.plot(x, poisson_pmf, 'r-', marker='o', label=f"Poisson PMF (λ={lambda_est:.2f})")

plt.title("Distribution of Last Column with Poisson Fit")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()