import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# === Setup and Configuration === #
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Path to regressor dataset
# Folder where model grid search result is stored
dataset_num = 16
model_folder = 'optimized_LGBM_regressor_model'
results = project_root / "AI_Models" / f"Dataset_{dataset_num}" / model_folder / "lgbm_grid_search_results.csv"
# Make sure your CSV has a column called "R2" or change this accordingly
df = pd.read_csv(results)

# Split features and target
df["rep_factor_effective"] = df.apply(
    lambda row: row["rep_factor"] if row["os_mode"] != "None" else 0,
    axis=1
)
X = df.drop(columns=["R2_Score", "MAE", "rep_factor"])
X = pd.get_dummies(X, drop_first=False)
y = df["R2_Score"]

print("Shape after encoding:", X.shape)
print("Example columns:", list(X.columns)[:10])

# === Fit model to features === #
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Surrogate Model
model = RandomForestRegressor(
    n_estimators=300, 
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Surrogate R² on test set:", r2_score(y_test, y_pred))

# === Feature Importance === #
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp, y=feat_imp.index, palette="viridis")
plt.title("Hyperparameter Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Hyperparameter")
plt.tight_layout()
plt.show()

# === Partial Dependence-Like Plots === #
cols = list(X.columns)
idx = 0

fig, ax = plt.subplots(figsize=(6,4))

def plot_feature(i):
    ax.clear()
    col = cols[i]
    sns.scatterplot(x=X[col], y=y, alpha=0.6, label="Observed", ax=ax)
    sns.scatterplot(x=X[col], y=model.predict(X), alpha=0.6, label="Predicted", ax=ax)
    ax.set_title(f"Effect of {col} on R²")
    ax.set_xlabel(col)
    ax.set_ylabel("R²")
    ax.legend()
    fig.canvas.draw()

def on_key(event):
    global idx
    if event.key == "right":
        idx = (idx + 1) % len(cols)
        plot_feature(idx)
    elif event.key == "left":
        idx = (idx - 1) % len(cols)
        plot_feature(idx)

fig.canvas.mpl_connect("key_press_event", on_key)

plot_feature(idx)
plt.show()

# Define search space (adjust ranges to your hyperparams)
search_space = {
    "learning_rate": (0.001, 0.3),
    "max_depth": (3, 20),
    "n_estimators": (2000, 3000),
    "weight_type": ["instance", "frequency", "linear", "exponential", "quantile", "polynomial", None],
    "num_leaves": [20, 80],
    "rep_factor": [0, 25],
    "os_type": ["None", "hybrid", "top"]
}

# Generate random candidates
n_candidates = 2000
candidates = []
for _ in range(n_candidates):
    os_choice = np.random.choice(search_space["os_type"])
    if os_choice == "None":
        rep_factor = 0
    else:
        rep_factor = np.random.randint(*search_space["rep_factor"])
    cand = {
        "learning_rate": np.random.uniform(*search_space["learning_rate"]),
        "max_depth": np.random.randint(*search_space["max_depth"]),
        "n_estimators": np.random.randint(*search_space["n_estimators"]),
        "weight_type": np.random.choice(search_space["weight_type"]),
        "num_leaves": np.random.randint(*search_space["num_leaves"]),
        "rep_factor": rep_factor,
        "os_type": os_choice
    }
    candidates.append(cand)

candidates_df = pd.DataFrame(candidates)
candidates_df["rep_factor_effective"] = candidates_df.apply(
    lambda row: row["rep_factor"] if row["os_type"] != "None" else 0,
    axis=1
)
candidates_encoded = candidates_df.drop(columns=["rep_factor"])

for col in candidates_encoded.select_dtypes(include=["object", "category"]).columns:
    candidates_encoded[col] = candidates_encoded[col].astype(str)

# One-hot encode (must match training encoding)
candidates_encoded = pd.get_dummies(candidates_encoded, drop_first=False)
# Align with training features
candidates_encoded = candidates_encoded.reindex(columns=X.columns, fill_value=0)

# Predict R² with surrogate
candidates_df["predicted_R2"] = model.predict(candidates_encoded)

# Show best suggestions
best_suggestions = candidates_df.sort_values("predicted_R2", ascending=False).head(10)
print(best_suggestions)
