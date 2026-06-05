import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# == Setup and Configuration === #
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, project_root)
# Path to regressor dataset
Training_File = "Airfoils_for_Regressor_E_Dataset_16_autocorrected.csv"
DATA_PATH = project_root / Training_File

# Example y (continuous target values)
#y = torch.tensor([0.1, 0.15, 0.2, 0.5, 1.2, 1.25, 1.3, 2.5, 2.6, 3.0], dtype=torch.float32).to(device)
df = pd.read_csv(DATA_PATH)
y = df.iloc[:, -1].values.astype(np.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

y = y.flatten()

# Number of bins
num_bins = 20  

# Find min and max
y_min, y_max = y.min().item(), y.max().item()

# bin edges (num_bins+1 edges)
edges = torch.linspace(y_min, y_max, steps=num_bins + 1, device=device)

# bucketize returns indices in [0, len(edges)] so subtract 1 to get [ -1 .. num_bins-1 ]
bin_idx = torch.bucketize(y, edges, right=False) - 1
bin_idx = torch.clamp(bin_idx, min=0, max=num_bins-1).to(torch.long)

# Count samples per bin
bin_counts = torch.bincount(bin_idx, minlength=num_bins).float()

# Inverse frequency weighting (higher weight for bins with fewer samples)
inv_freq = 1.0 / (bin_counts + 1e-8)   # avoid div by zero

# Map each sample to its bin weight
weights = inv_freq[bin_idx].float()

# Normalize so weights have mean = 1 (optional)
weights = weights * (len(y) / weights.sum())

print("y:", y)
print("bin_indices:", bin_idx)
print("bin_counts:", bin_counts)
print("weights:", weights)
