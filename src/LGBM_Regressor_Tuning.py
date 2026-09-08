import sys
import os
import optuna
import json
import random
import platform
import lightgbm as lgb
from pathlib import Path
from ai_support import LGBM_Regressor
from ux_support import recovery_json

# === LightGBM Logger Deduplication === #
# Two LightGBM Info-level messages are printed on every fit() call / every boosting
# iteration and flood the terminal across many folds/trials. Neither indicates a
# real problem, so this logger prints each one once per process run instead of
# repeatedly:
#   1. "constrain tree depth ... without explicitly setting num_leaves" - the
#      lr_decay callback (lgb.reset_parameter) re-triggers LightGBM's internal
#      param-conflict check on every boosting iteration, which reports num_leaves
#      as "not explicitly set" even though it is. Cosmetic only; the num_leaves
#      value actually used for training is unaffected.
#   2. "Auto-choosing col-wise multi-threading" - printed once per fit() call
#      (i.e. once per fold) while LightGBM times whether column-wise or row-wise
#      histogram building is faster. The timing test costs microseconds; purely
#      informational, no functional impact.
class _DedupLogger:
    _seen = set()
    def info(self, msg):
        if "constrain tree depth" in msg:
            key = "num_leaves_maxdepth"
        elif "Auto-choosing" in msg and "multi-threading" in msg:
            key = "auto_choosing_multithreading"
        else:
            print(msg)
            return
        if key not in self._seen:
            self._seen.add(key)
            print(msg + " (further repeats of this message suppressed)")
    def warning(self, msg):
        print(msg)

lgb.register_logger(_DedupLogger())

# === Recovery Setup === #
recovery_dir = Path("C:/lgbm_reg_rec") if platform.system() == "Windows" else Path.home() / ".lgbm_reg_rec"
recovery_dir.mkdir(parents=True, exist_ok=True)
rec_json = recovery_json(directory=recovery_dir)
status = rec_json.get_recovery_status()

# === Directory configuration === #
# Get the path to the src directory.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
datasets_dir = project_root / "data" / "datasets"
# Path to dataset
dataset_num = 17
dataset_name = f"regressor_dataset_{dataset_num}"
dataset_dir = datasets_dir / f"dataset_{dataset_num}"
training_file_dir = dataset_dir / f"{dataset_name}.csv"
model_save_dir = project_root / "models" / f"dataset_{dataset_num}" / "lgbm_regressor"
os.makedirs(model_save_dir, exist_ok=True)
# Verify that dataset exists
if not os.path.exists(training_file_dir):
    print(f"[ERROR] Dataset not found at '{training_file_dir}'. Please check the file path.")
    sys.exit()

# === Optuna Objective Function === #
N_TRIALS = 60
def objective(trial):
    """
    Takes a trial object from Optuna, suggests hyperparameters for LightGBM, 
    trains the regressor, and evaluates the performance.
    """
    print("\n" + "="*50)
    print(f"Starting Trial {trial.number + 1}/{N_TRIALS}")

    # Dynamically build hyperparameter space
    n_estimators = trial.suggest_int('n_estimators', 1000, 4000, step=500)
    lr = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    lr_decay = trial.suggest_float('lr_decay', 0.90, 0.99)

    # max_depth is sampled first; num_leaves is then bounded above by 2^max_depth
    # (LightGBM silently clamps num_leaves to this cap anyway for leaf-wise growth,
    # so sampling above it wastes trials on combos that collapse to the same model).
    # PRACTICAL_MAX_LEAVES keeps the ceiling sane at high max_depth values, where
    # 2^max_depth would otherwise reach the hundreds of thousands.
    PRACTICAL_MAX_LEAVES = 100
    max_depth = trial.suggest_int('max_depth', 5, 20, step=5)
    num_leaves_upper = min(2 ** max_depth, PRACTICAL_MAX_LEAVES)
    num_leaves = trial.suggest_int('num_leaves', 2, num_leaves_upper)
    objective_type = trial.suggest_categorical('objective', ['regression_l2', 'combo'])
    weight = trial.suggest_categorical('weight', ['None', 'instance', 'frequency', 'linear', 'exponential', 'quantile', 'polynomial'])

    print(f"PARAMETERS:")
    print(f"Estimators = {n_estimators}, LR = {lr:.4f}, LR Decay = {lr_decay:.4f}")
    print(f"Leaves = {num_leaves}, Max Depth = {max_depth}")
    print(f"Objective = {objective_type}, Weight = {weight}")
    print("="*50)

    # Instantiate and train the regressor
    regressor = LGBM_Regressor(
        n_estimators=n_estimators,
        learning_rate=lr,
        lr_decay=lr_decay,
        num_leaves=num_leaves,
        max_depth=max_depth,
        objective=objective_type,
        weight=weight
    )

    final_metrics = regressor.train(data_path=training_file_dir)

    # Optimize for MAE
    mae = final_metrics.get('MAE', float('inf'))
    
    trial.set_user_attr('R2-Score', final_metrics.get('R2_Score', float('-inf')))

    return mae

# === Optuna Study Execution === #
storage = f"sqlite:///{recovery_dir / 'optuna_study.db'}"
study = optuna.create_study(study_name=dataset_name, storage=storage, direction='minimize', load_if_exists=True)

if not status.get('optuna_complete', False):
    completed_trials = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    remaining_trials = N_TRIALS - completed_trials

    if remaining_trials > 0:
        print(f"\nStarting Optuna Optimization for {remaining_trials} Trials...")
        study.optimize(objective, n_trials=remaining_trials)

    status['optuna_complete'] = True
    rec_json.save_recovery_status(status)


# === Final Report and Model Saving === #
print("\n" + "#"*60)
print("Optimization Complete. Analyzing Results...")
print("#"*60 + "\n")

print(f"Best MAE: {study.best_value:.4f}")
print("Best parameters found:")
best_params = study.best_params
for key, value in best_params.items():
    print(f"  - {key}: {value}")

names = [ "Marbas", "Orobas", "Agares", "Foras", "Phobos", "Nemesis", "Kuutar"]
if 'best_mae_name' not in status:
    status['best_mae_name'] = random.choice(names)
    rec_json.save_recovery_status(status)
best_mae_name = status['best_mae_name']

if not status.get('finalized', False):
    # Retrain the final best model
    print("\n--- Retraining and Saving the Best Model ---")
    final_model = LGBM_Regressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        lr_decay=best_params['lr_decay'],
        num_leaves=best_params['num_leaves'],
        max_depth=best_params['max_depth'],
        objective=best_params['objective'],
        weight=best_params['weight']
    )
    final_model.train(data_path=training_file_dir)

    # Save parameters for best model
    json_name = f"{best_mae_name}_regressor_model_parameters.json"
    param_file = os.path.join(model_save_dir, json_name)
    with open(param_file, 'w') as json_file:
        json.dump(best_params, json_file, indent=4)

    # Save model
    final_model.save_model(model_save_dir, best_mae_name)

    # Save all results
    results_df = study.trials_dataframe()
    results_csv_path = model_save_dir / f"{best_mae_name}_optuna_search_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nFull optimization results saved to '{results_csv_path}'")

    status['finalized'] = True
    rec_json.save_recovery_status(status)

# === Recovery Cleanup === #
rec_json.recover_cleanup()
db_path = recovery_dir / 'optuna_study.db'
if db_path.exists():
    db_path.unlink()