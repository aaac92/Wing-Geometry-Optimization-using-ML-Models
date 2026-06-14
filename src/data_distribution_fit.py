import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde

'''
This script allows to fit the dataset's data to a gaussian mixture, enabling a compound model training if desired.
'''

# Load CSV file
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
Training_File = "regressor_dataset_16.csv"
DATA_PATH = project_root / "data" / "datasets" / "dataset_16" / Training_File
df = pd.read_csv(DATA_PATH)

# Extract last column
y = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna()
y = y[y > 0]  # GMM requires positive data in your context

# Reshape for sklearn (expects 2D array)
y_reshaped = y.values.reshape(-1, 1)

# Compute key quantiles
quantiles = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
q_values = y.quantile(quantiles)

print("Key quantiles of y:")
for q, val in zip(quantiles, q_values):
    print(f"{q:.2f} quantile: {val:.2f}")

# Estimate density with KDE
kde = gaussian_kde(y)
x_grid = np.linspace(y.min(), y.max(), 1000)
density = kde(x_grid)

# Find local maxima (peaks)
from scipy.signal import find_peaks
peaks, _ = find_peaks(density)

print("Approximate peak values in y:")
for p in peaks:
    print(f"Peak at ~ {x_grid[p]:.2f}")

# Fit Gaussian Mixture
k = 4
gmm = GaussianMixture(n_components=k, random_state=42)
gmm.fit(y_reshaped)

# Predict cluster labels
labels = gmm.predict(y_reshaped)

# Add cluster info to dataframe
df_clusters = pd.DataFrame({"y": y, "Cluster": labels})

# Plot histogram with cluster colors
plt.figure(figsize=(12, 6))
sns.histplot(data=df_clusters, x="y", hue="Cluster", bins=80, kde=True, palette="Set1", element="step")
plt.title(f"Clustering of y using Gaussian Mixture Model ({k} clusters)")
plt.show()

# Print cluster means
for i in range(k):
    mean = gmm.means_[i][0]
    weight = gmm.weights_[i]
    print(f"Cluster {i}: mean = {mean:.2f}, weight = {weight:.2%}")

# Try different numbers of clusters
n_components = range(1, 10)
aic = []
bic = []

for k in n_components:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(y_reshaped)
    aic.append(gmm.aic(y_reshaped))
    bic.append(gmm.bic(y_reshaped))

# Plot AIC/BIC
plt.figure(figsize=(8, 5))
plt.plot(n_components, aic, label="AIC", marker="o")
plt.plot(n_components, bic, label="BIC", marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Score (lower is better)")
plt.title("Model selection with AIC and BIC")
plt.legend()
plt.show()
