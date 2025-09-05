# hyperparam_search.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import ParameterSampler
from ai_support import TrainingManager
import optuna

# === Setup and Configuration === #
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, project_root)
Training_File = "Airfoils_for_Regressor_E_Dataset_16.csv"
DATA_PATH = project_root / Training_File

# === Randomized Search === #
def random_search(data_path, param_grid, n_iter=10, output_csv="random_search_results.csv"):
    results = []
    sampler = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

    for i, params in enumerate(sampler, 1):
        print(f"\n=== Random Search Trial {i}/{n_iter} ===")
        manager = TrainingManager(**params)
        metrics_df = manager.train(data_path)
        mean_metrics = metrics_df.mean().to_dict()
        row = {**params, **mean_metrics}
        results.append(row)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print("Saved results to", output_csv)
    return df_results


# === Optuna Search === #
def objective(trial, data_path):
    params = {
        "model_type": trial.suggest_categorical("model_type", ["mlp", "lgbm"]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh", "leakyrelu"]),
        "loss_fn": trial.suggest_categorical("loss_fn", ["mse", "mae"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "patience": trial.suggest_int("patience", 10, 30),
        "weighting": trial.suggest_categorical("weighting", ["none", "proportional", "quantile"]),
        "alpha": trial.suggest_float("alpha", 0.5, 2.0),
    }

    manager = TrainingManager(**params)
    metrics_df = manager.train(data_path)
    mean_tail_r2 = metrics_df["tail_r2"].mean()
    return mean_tail_r2


def optuna_search(data_path, n_trials=20, output_csv="optuna_results.csv"):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data_path), n_trials=n_trials)

    df_results = pd.DataFrame([t.params | {"value": t.value} for t in study.trials])
    df_results.to_csv(output_csv, index=False)
    print("Saved Optuna results to", output_csv)
    return study, df_results


if __name__ == "__main__":
    data_path = "your_dataset.csv"

    # Example Random Search
    param_grid = {
        "model_type": ["mlp", "lgbm"],
        "hidden_size": [64, 128],
        "dropout_rate": [0.1, 0.3],
        "activation": ["relu", "tanh"],
        "loss_fn": ["mse", "mae"],
        "learning_rate": [1e-3, 5e-4],
        "patience": [15],
        "weighting": ["none", "proportional"],
        "alpha": [1.0]
    }
    #random_search(DATA_PATH, param_grid, n_iter=5)

    # Example Optuna Search
    optuna_search(DATA_PATH, n_trials=10)
