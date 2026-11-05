"""
Trains a single model with fixed hyperparameters, or retrains (updates) existing models on a dataset.

The tuning scripts (src/*_Tuning.py) search hyperparameters with Optuna, which can take days. This tool
trains one configuration directly: the training itself is done by the ai_support model classes, this
script only resolves the configuration, asks for confirmation and saves the result in the same file
layout the tuning scripts and inference_and_testing.py use.

Model types
-----------
mlp          MLP regressor        (models/dataset_<N>/mlp_regressor[_<features>])
lgbm         LightGBM regressor   (models/dataset_<N>/lgbm_regressor[_<features>])
classifier   MLP classifier       (models/dataset_<N>/classifier)
Regressors accept --features raw | station | sum_station (CSTFeatureEngineer); 'raw' is no feature
engineering. The classifier always uses the raw CST columns.

Modes
-----
train          Trains a new model. Hyperparameters come from the command line, optionally on top of a
               parameters JSON (--params), on top of the model class defaults.
update         Scans a folder for saved models, reads their architecture from their parameters and
               metadata JSON files, asks which ones to retrain and retrains them on the given dataset.
               The new model gets the next minor version (Murata -> Murata_1.1, Murata_1.9 ->
               Murata_1.10) in the same folder; the old model's files move to '<folder>/Old models'.
major-update   Same as update, but the new model gets the next major version (Murata_1.4 -> Murata_2.0).
retrain        Same as update, but the model keeps its name and its files are overwritten (no copy is kept).
               Meant for a model trained on bad data, e.g. retraining Murata_1.1 after its dataset was fixed.

update and major-update also take:
  --version X.Y  the exact new version instead of the next one (Murata_1.1 -> Murata_1.5).
  --dataset N    when N is not the dataset of the --target folder, the new models are saved in
                 models/dataset_<N>/<same type folder> and the old ones stay where they are (they still belong
                 to their own dataset). Example: dataset 17.5 rebuilds 17, its models start from the 17 ones.

Every trained model also gets '<name>_training_info.json' (dataset, configuration, test metrics and,
for updates, the model it replaced).

Training protocol: one final fit on the training split (10% of it early-stops the model), scored on the test
split. --folds 0 (default) skips cross-validation; --folds 5 adds it (CV_* metrics), like the tuning scripts.

Examples
--------
    python src/utilities/ml_gym.py train --model mlp --features sum_station --dataset 17 --layers 364 424 242 474 395 --log-trans
    python src/utilities/ml_gym.py train --model lgbm --dataset 17 --params models/dataset_17/lgbm_regressor_sum_station/Agares_regressor_model_parameters.json --name Foras
    python src/utilities/ml_gym.py update --target models/dataset_17 --dataset 17 --csv data/datasets/dataset_17/regressor_dataset_17_rectified.csv
    python src/utilities/ml_gym.py major-update --target models/dataset_17/mlp_regressor --dataset 17 --select Odysseus
    python src/utilities/ml_gym.py retrain --target models/dataset_17 --dataset 17 --csv data/datasets/dataset_17/regressor_dataset_17_rectified.csv --select Murata_1.1
    python src/utilities/ml_gym.py update --target models/dataset_17 --dataset 17.5 --version 1.5 --select Agares_1.1 Odysseus_1.1 Murata_1.1 Calipso
"""
import argparse
import json
import platform
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402
from ai_support import MLP_Regressor, LGBM_Regressor, Classifier, CSTFeatureEngineer  # noqa: E402
from file_handling_support.dataset_names import dataset_csv, dataset_number_arg, models_folder, parent_dataset  # noqa: E402

# =============================================================================================== #
# === Model types === #
# =============================================================================================== #

MODEL_TYPES = ("mlp", "lgbm", "classifier")
REGRESSOR_TYPES = ("mlp", "lgbm")
TORCH_TYPES = ("mlp", "classifier")

# Same name pools as the tuning scripts
NAME_POOLS = {
    "mlp": ["Murata", "Beelzebul", "Buer", "Odysseus", "Paimon", "Ronove", "Columbina"],
    "lgbm": ["Marbas", "Orobas", "Agares", "Foras", "Phobos", "Nemesis", "Kuutar"],
    "classifier": ["Barbatos", "Focalors", "Istaroth", "Calipso", "Asmoday", "Lauma"],
}

MLP_WEIGHTS = ("None", "instance", "frequency", "exponential", "linear", "combo")
LGBM_WEIGHTS = ("None", "instance", "frequency", "linear", "exponential", "quantile", "polynomial")
LGBM_OBJECTIVES = ("regression_l2", "regression_l1", "combo")

# Hyperparameters per type and their class defaults. Keys follow the tuning scripts' parameters JSON.
DEFAULTS = {
    "mlp": {"layers": [128, 128, 128, 128, 64], "dropout_rate": 0.05, "learning_rate": 1e-3, "batch_size": 1024,
            "epochs": 500, "patience": 16, "weight": "None", "log_trans": False},
    "lgbm": {"n_estimators": 1000, "learning_rate": 0.1, "num_leaves": 20, "max_depth": 10, "objective": "regression_l2",
             "weight": "None", "lr_decay": None, "stopping_rounds": 100},
    "classifier": {"layers": [128], "learning_rate": 0.075, "epochs": 400, "patience": 40},
}

OLD_MODELS_DIR = "Old models"
# Every file a model owns is '<name>_<suffix>'; matching whole suffixes keeps 'Murata' from claiming 'Murata_1.1'.
MODEL_FILE_SUFFIXES = (
    "model.pth", "model.joblib", "scaler.joblib", "poly.joblib", "meta.json", "regressor_model_parameters.json",
    "parameters.json", "optuna_search_results.csv", "training_info.json",
)
VERSION_PATTERN = re.compile(r"^(?P<base>.+)_(?P<major>\d+)\.(?P<minor>\d+)$")


def folder_name(model_type: str, feature_mode: str) -> str:
    if model_type == "classifier":
        return "classifier"
    return f"{model_type}_regressor" + ("" if feature_mode == "raw" else f"_{feature_mode}")


def parameters_file_name(model_type: str, name: str) -> str:
    return f"{name}_parameters.json" if model_type == "classifier" else f"{name}_regressor_model_parameters.json"


# =============================================================================================== #
# === Versions and model files === #
# =============================================================================================== #

def split_version(name: str):
    """(base, major, minor); an unversioned name counts as version 1.0."""
    match = VERSION_PATTERN.match(name)
    if not match:
        return name, 1, 0
    return match["base"], int(match["major"]), int(match["minor"])


def bump_version(name: str, major: bool) -> str:
    base, mj, mn = split_version(name)
    return f"{base}_{mj + 1}.0" if major else f"{base}_{mj}.{mn + 1}"


def model_files(folder: Path, name: str) -> list:
    return [folder / f"{name}_{suffix}" for suffix in MODEL_FILE_SUFFIXES if (folder / f"{name}_{suffix}").exists()]


def name_taken(folder: Path, name: str) -> bool:
    return bool(model_files(folder, name) or model_files(folder / OLD_MODELS_DIR, name))


def next_free_version(folder: Path, name: str, major: bool) -> str:
    """Next version of 'name'; if it is already used in the folder or its old models, keeps counting."""
    candidate = bump_version(name, major)
    while name_taken(folder, candidate):
        candidate = bump_version(candidate, major)
    return candidate


def retire(folder: Path, name: str) -> list:
    """Moves every file of model 'name' into '<folder>/Old models'. Returns the new paths."""
    old_dir = folder / OLD_MODELS_DIR
    old_dir.mkdir(exist_ok=True)
    moved = []
    for path in model_files(folder, name):
        target = old_dir / path.name
        if target.exists():
            raise FileExistsError(f"'{target}' already exists; refusing to overwrite an old model.")
        shutil.move(str(path), str(target))
        moved.append(target)
    return moved


def random_name(model_type: str) -> str:
    """A pool name not used by any model under models/ (current or old)."""
    models_root = PROJECT_ROOT / "models"
    used = {split_version(p.name[:-len("_meta.json")])[0] for p in models_root.rglob("*_meta.json")} if models_root.exists() else set()
    free = [n for n in NAME_POOLS[model_type] if n not in used]
    if not free:
        raise SystemExit(f"[ERROR] Every {model_type} pool name is already used; choose one with --name.")
    return random.choice(free)


# =============================================================================================== #
# === Discovery of saved models === #
# =============================================================================================== #

def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def detect_model(meta_path: Path):
    """Describes the model behind a '<name>_meta.json', or returns (None, reason) if it cannot be retrained."""
    folder, name = meta_path.parent, meta_path.name[:-len("_meta.json")]
    meta = read_json(meta_path)
    if (folder / f"{name}_model.joblib").exists():
        model_type = "lgbm"
    elif (folder / f"{name}_model.pth").exists():
        # The folder name decides first: older MLP metadata files may lack 'y_min'
        if folder.name.startswith("classifier"):
            model_type = "classifier"
        elif folder.name.startswith("mlp"):
            model_type = "mlp"
        else:
            model_type = "mlp" if "y_min" in meta else "classifier"
    else:
        return None, "no model weights file"
    feature_mode = meta.get("feature_engineer", {}).get("mode", "raw")
    params_path = next((p for p in (folder / parameters_file_name(model_type, name), folder / f"{name}_parameters.json")
                        if p.exists()), None)
    if params_path is None:
        return None, f"no parameters JSON ({parameters_file_name(model_type, name)})"
    try:
        hyper = hyperparameters_from_params(model_type, read_json(params_path))
    except KeyError as e:
        return None, f"parameters JSON lacks {e}"
    return {"name": name, "folder": folder, "type": model_type, "features": feature_mode,
            "hyper": hyper, "params_path": params_path}, None


def scan_models(target: Path):
    """Every retrainable model under target (Old models folders excluded), plus the ones that are not."""
    found, skipped = [], []
    for meta_path in sorted(target.rglob("*_meta.json")):
        if OLD_MODELS_DIR in meta_path.relative_to(target).parts:
            continue
        model, reason = detect_model(meta_path)
        if model:
            found.append(model)
        else:
            skipped.append((meta_path, reason))
    return found, skipped


# =============================================================================================== #
# === Hyperparameters === #
# =============================================================================================== #

def hyperparameters_from_params(model_type: str, params: dict) -> dict:
    """Hyperparameters from a tuning-style parameters JSON; missing optional keys take the class defaults."""
    hyper = dict(DEFAULTS[model_type])
    if model_type in TORCH_TYPES:
        hyper["layers"] = [int(params[f"n_units_l{i}"]) for i in range(int(params["n_layers"]))]
    for key in hyper:
        if key in params and key != "layers":
            hyper[key] = params[key]
    return hyper


def params_to_json(model_type: str, hyper: dict) -> dict:
    """Tuning-style parameters JSON (read by inference_and_testing.py and by update)."""
    out = {}
    if model_type in TORCH_TYPES:
        out["n_layers"] = len(hyper["layers"])
        out.update({f"n_units_l{i}": int(n) for i, n in enumerate(hyper["layers"])})
    out.update({k: v for k, v in hyper.items() if k != "layers"})
    return out


def hyperparameters_from_cli(args, model_type: str) -> tuple:
    """Command line > --params file > class defaults. Returns (hyperparameters, source of each value)."""
    hyper, sources = dict(DEFAULTS[model_type]), {k: "default" for k in DEFAULTS[model_type]}
    if args.params:
        params = read_json(Path(args.params))
        from_file = hyperparameters_from_params(model_type, params)
        file_keys = set(params) | ({"layers"} if "n_layers" in params else set())
        for key in hyper:
            if key in file_keys:
                hyper[key], sources[key] = from_file[key], "params file"
    cli = {"layers": args.layers, "dropout_rate": args.dropout, "learning_rate": args.lr, "batch_size": args.batch_size,
           "epochs": args.epochs, "patience": args.patience, "weight": args.weight, "log_trans": args.log_trans,
           "n_estimators": args.n_estimators, "num_leaves": args.num_leaves, "max_depth": args.max_depth,
           "objective": args.objective, "lr_decay": args.lr_decay, "stopping_rounds": args.stopping_rounds}
    for key, value in cli.items():
        if value is not None and key in hyper:
            hyper[key], sources[key] = value, "command line"
        elif value is not None:
            print(f"[NOTE] --{key.replace('_', '-')} does not apply to the {model_type} model and is ignored.")
    if model_type == "mlp" and hyper["weight"] not in MLP_WEIGHTS:
        raise SystemExit(f"[ERROR] MLP weight must be one of {MLP_WEIGHTS}.")
    if model_type == "lgbm" and hyper["weight"] not in LGBM_WEIGHTS:
        raise SystemExit(f"[ERROR] LGBM weight must be one of {LGBM_WEIGHTS}.")
    return hyper, sources


# =============================================================================================== #
# === Training === #
# =============================================================================================== #

def build_model(model_type: str, feature_mode: str, hyper: dict, protocol: dict):
    if model_type == "mlp":
        return MLP_Regressor(hidden_layer_sizes=list(hyper["layers"]), dropout_rate=hyper["dropout_rate"],
                             learning_rate=hyper["learning_rate"], batch_size=hyper["batch_size"], epochs=hyper["epochs"],
                             patience=hyper["patience"], weight=hyper["weight"], log_trans=bool(hyper["log_trans"]),
                             k=protocol["folds"], test_split=protocol["test_split"], random_state=protocol["seed"],
                             feature_mode=feature_mode)
    if model_type == "lgbm":
        return LGBM_Regressor(n_estimators=hyper["n_estimators"], learning_rate=hyper["learning_rate"],
                              lr_decay=hyper["lr_decay"], num_leaves=hyper["num_leaves"], max_depth=hyper["max_depth"],
                              stopping_rounds=hyper["stopping_rounds"], objective=hyper["objective"], weight=hyper["weight"],
                              k=protocol["folds"], test_size=protocol["test_split"], random_state=protocol["seed"],
                              feature_mode=feature_mode)
    return Classifier(hidden_layer_sizes=list(hyper["layers"]), epochs=hyper["epochs"], lr=hyper["learning_rate"],
                      patience=hyper["patience"], k=protocol["folds"], test_size=protocol["test_split"],
                      random_state=protocol["seed"])


def train_and_save(job: dict, csv_path: Path, protocol: dict, mode: str) -> dict:
    """Trains one job ({type, features, hyper, name, folder, previous}) and writes its files. Returns the metrics."""
    print("\n" + "#" * 100)
    print(f"Training {job['name']} ({job['type']}, features: {job['features']}) on {csv_path.name}")
    print("#" * 100)
    started = datetime.now()
    t0 = time.time()
    model = build_model(job["type"], job["features"], job["hyper"], protocol)
    metrics = {k: float(v) for k, v in model.train(data_path=csv_path).items()}

    folder = job["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    model.save_model(folder, job["name"])
    with open(folder / parameters_file_name(job["type"], job["name"]), "w") as f:
        json.dump(params_to_json(job["type"], job["hyper"]), f, indent=4)
    base, major, minor = split_version(job["name"])
    info = {
        "name": job["name"], "base_name": base, "version": f"{major}.{minor}", "mode": mode,
        "previous_model": job.get("previous") or job.get("source_model"), "model_type": job["type"], "feature_mode": job["features"],
        "dataset_csv": str(csv_path), "hyperparameters": job["hyper"], "protocol": protocol, "test_metrics": metrics,
        "started": started.strftime("%Y-%m-%d %H:%M:%S"), "duration_s": round(time.time() - t0, 1),
        "platform": platform.platform(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "command": " ".join(sys.argv),
    }
    with open(folder / f"{job['name']}_training_info.json", "w") as f:
        json.dump(info, f, indent=4)
    # The old model is retired only once the new one is fully saved
    if job.get("previous"):
        moved = retire(folder, job["previous"])
        print(f"Moved {len(moved)} file(s) of {job['previous']} to '{folder / OLD_MODELS_DIR}'.")
    print(f"Saved {job['name']} in '{folder}' | test metrics: " + ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items()))
    return metrics


# =============================================================================================== #
# === Dataset and environment checks === #
# =============================================================================================== #

def resolve_csv(args, model_type: str) -> Path:
    """Training CSV of a model type. In update modes --csv covers the regressors and --classifier-csv the classifier."""
    explicit = args.csv
    if args.mode != "train":
        explicit = args.classifier_csv if model_type == "classifier" else args.csv
    if explicit:
        path = Path(explicit)
    elif args.dataset is not None:
        kind = "classifier" if model_type == "classifier" else "regressor"
        path = dataset_csv(args.dataset, PROJECT_ROOT, kind)  # expansions (17.5.3) sit inside their parent's folder
    else:
        raise SystemExit("[ERROR] Give --dataset, --csv or both.")
    path = path if path.is_absolute() else (Path.cwd() / path)
    if not path.exists():
        raise SystemExit(f"[ERROR] Dataset not found: {path}")
    return path.resolve()


def dataset_number(args, csv_path: Path):
    if args.dataset is not None:
        return args.dataset
    match = re.search(r"dataset_(\d+(?:\.\d+){0,2})", csv_path.name)
    return match.group(1) if match else None


def folder_dataset(folder: Path):
    """Dataset number of a model folder (the 'dataset_<N>' part of its path), or None."""
    for part in reversed(folder.parts):
        match = re.fullmatch(r"dataset_(\d+(?:\.\d+){0,2})", part)
        if match:
            return match.group(1)
    return None


def destination_folder(model_folder: Path, dataset):
    """Where an updated model goes: its own folder, or the same type folder of another dataset (--dataset)."""
    source = folder_dataset(model_folder)
    if dataset is None or source is None or source == dataset:
        return model_folder
    root = next(p for p in [model_folder, *model_folder.parents] if p.name == f"dataset_{source}")
    if parent_dataset(dataset) or parent_dataset(source):  # expansions live inside their parent dataset's folder
        return models_folder(dataset, PROJECT_ROOT) / model_folder.relative_to(root)
    return root.parent / f"dataset_{dataset}" / model_folder.relative_to(root)


def check_device(types, allow_cpu: bool) -> None:
    if any(t in TORCH_TYPES for t in types) and not torch.cuda.is_available():
        if not allow_cpu:
            raise SystemExit("[ERROR] MLP models need a CUDA GPU (training on CPU takes days). Use --allow-cpu to force it.")
        print("[WARNING] No CUDA GPU detected; MLP training will run on the CPU.")
    elif torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")


def confirm(args, prompt: str = "Start training? [y/n]: ") -> None:
    if args.yes:
        return
    if input(prompt).strip().lower() not in ("y", "yes"):
        raise SystemExit("Cancelled.")


def protocol_text(args) -> str:
    cv = f"{args.folds}-fold cross-validation + final fit" if args.folds >= 2 else "final fit only (no cross-validation)"
    return f"{cv}, test split {args.test_split}, seed {args.seed}"


def describe(hyper: dict) -> str:
    return ", ".join(f"{k} = {v}" for k, v in hyper.items())


# =============================================================================================== #
# === Modes === #
# =============================================================================================== #

def run_train(args) -> None:
    model_type = args.model
    feature_mode = "raw" if model_type == "classifier" else args.features
    if model_type == "classifier" and args.features != "raw":
        print("[NOTE] The classifier has no feature engineering; --features is ignored.")
    csv_path = resolve_csv(args, model_type)
    hyper, sources = hyperparameters_from_cli(args, model_type)
    name = args.name or random_name(model_type)
    if not name or any(c.isspace() for c in name):
        raise SystemExit(f"[ERROR] Invalid model name '{name}': spaces are not allowed.")
    if args.output_dir:
        folder = Path(args.output_dir)
    else:
        number = dataset_number(args, csv_path)
        if number is None:
            raise SystemExit("[ERROR] Cannot infer the dataset number from the CSV name; give --dataset or --output-dir.")
        folder = models_folder(number, PROJECT_ROOT) / folder_name(model_type, feature_mode)
    folder = folder.resolve()
    if name_taken(folder, name):
        raise SystemExit(f"[ERROR] A model named '{name}' already exists in '{folder}' (or its old models). "
                         f"Use another --name, or the update mode to retrain it.")
    protocol = {"folds": args.folds, "test_split": args.test_split, "seed": args.seed}

    print("\n" + "=" * 100)
    print(f"Train | model: {model_type} | features: {feature_mode} | name: {name}")
    print("=" * 100)
    print(f"  Dataset CSV    {csv_path}")
    print(f"  Output folder  {folder}")
    print(f"  Protocol       {protocol_text(args)}")
    for key, value in hyper.items():
        print(f"  {key:<15}{str(value):<40} [{sources[key]}]")
    print("=" * 100)
    check_device([model_type], args.allow_cpu)
    confirm(args)
    train_and_save({"type": model_type, "features": feature_mode, "hyper": hyper, "name": name, "folder": folder},
                   csv_path, protocol, "train")


def parse_selection(text: str, count: int) -> list:
    text = text.strip().lower()
    if text in ("all", "a"):
        return list(range(count))
    picks = []
    for token in re.split(r"[,\s]+", text):
        if not token:
            continue
        if "-" in token:
            lo, hi = (int(v) for v in token.split("-", 1))
            picks += range(lo - 1, hi)
        else:
            picks.append(int(token) - 1)
    if any(p < 0 or p >= count for p in picks):
        raise ValueError("selection out of range")
    return sorted(set(picks))


def run_update(args, mode: str) -> None:
    """update / major-update: new version, old files retired. retrain: same name, files overwritten."""
    in_place = mode == "retrain"
    target = Path(args.target).resolve()
    if not target.is_dir():
        raise SystemExit(f"[ERROR] Target folder not found: {target}")
    found, skipped = scan_models(target)
    if not found:
        raise SystemExit(f"[ERROR] No retrainable models under '{target}'.")

    if args.version and in_place:
        raise SystemExit("[ERROR] --version does not apply to retrain (the model keeps its name).")
    rows = []
    for model in found:
        destination = model["folder"] if in_place else destination_folder(model["folder"], args.dataset)
        if in_place:
            new_name = model["name"]
        elif args.version:
            new_name = f"{split_version(model['name'])[0]}_{args.version}"
        else:
            new_name = next_free_version(destination, model["name"], mode == "major-update")
        rows.append({**model, "new_name": new_name, "destination": destination})
    print("\n" + "=" * 100)
    print(f"{mode} | models found under {target}")
    print("=" * 100)
    print(f"  {'#':>3}  {'Model':<22} {'New version':<22} {'Type':<11} {'Features':<12} Folder")
    for i, row in enumerate(rows, 1):
        print(f"  {i:>3}  {row['name']:<22} {row['new_name']:<22} {row['type']:<11} {row['features']:<12} "
              f"{row['folder'].relative_to(target) if row['folder'] != target else '.'}")
    moved_dataset = any(row["destination"] != row["folder"] for row in rows)
    if moved_dataset:
        print(f"  New versions are saved under models/dataset_{args.dataset}; the models above stay where they are.")
    for meta_path, reason in skipped:
        print(f"  ---  {meta_path.name[:-len('_meta.json')]:<22} not retrainable: {reason}")
    print("=" * 100)

    if args.select:
        names = {row["name"]: i for i, row in enumerate(rows)}
        unknown = [n for n in args.select if n not in names]
        if unknown:
            raise SystemExit(f"[ERROR] Not found among the models above: {unknown}")
        picks = sorted(names[n] for n in args.select)
    else:
        while True:
            try:
                picks = parse_selection(input("Models to retrain (e.g. '1 3', '1-4', 'all'; empty to cancel): "), len(rows))
                break
            except ValueError as e:
                print(f"  Invalid selection ({e}).")
        if not picks:
            raise SystemExit("Cancelled.")
    jobs = [rows[i] for i in picks]
    if not in_place:
        taken = [f"{job['new_name']} ({job['destination']})" for job in jobs if name_taken(job["destination"], job["new_name"])]
        if taken:
            raise SystemExit(f"[ERROR] These models already exist: {', '.join(taken)}. Choose another --version.")

    protocol = {"folds": args.folds, "test_split": args.test_split, "seed": args.seed}
    csv_paths = {}
    print("\nPlanned retraining:")
    for job in jobs:
        csv_paths[job["name"]] = resolve_csv(args, job["type"])
        change = "retrained in place" if in_place else f"-> {job['new_name']}"
        if job["destination"] != job["folder"]:
            change += f" in '{job['destination']}'"
        print(f"  {job['name']} {change} | {job['type']}, features: {job['features']} | {csv_paths[job['name']].name}")
        print(f"      {describe(job['hyper'])}")
    print(f"  Protocol: {protocol_text(args)}")
    if in_place:
        print("  The current files of these models are overwritten; no copy is kept.")
    elif all(job["destination"] != job["folder"] for job in jobs):
        print("  The current models are kept: the new versions belong to another dataset.")
    else:
        print(f"  Replaced models move to '<model folder>/{OLD_MODELS_DIR}'.")
    check_device([job["type"] for job in jobs], args.allow_cpu)
    confirm(args)

    results = []
    for job in jobs:
        try:
            # A model moved to another dataset is not retired: it still belongs to its own dataset
            retire_previous = not in_place and job["destination"] == job["folder"]
            metrics = train_and_save({"type": job["type"], "features": job["features"], "hyper": job["hyper"],
                                      "name": job["new_name"], "folder": job["destination"],
                                      "previous": job["name"] if retire_previous else None,
                                      "source_model": None if retire_previous or in_place else str(job["folder"] / job["name"])},
                                     csv_paths[job["name"]], protocol, mode)
            results.append((job["name"], job["new_name"], metrics, None))
        except Exception as e:
            print(f"[ERROR] {job['name']} failed: {type(e).__name__}: {e}. The old model was left in place.")
            results.append((job["name"], job["new_name"], None, f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 100)
    print(f"{mode} summary")
    print("=" * 100)
    for old, new, metrics, error in results:
        outcome = ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items()) if metrics else f"FAILED ({error})"
        print(f"  {old} {'(retrained in place)' if old == new else '-> ' + new}: {outcome}")


# =============================================================================================== #
# === Command line === #
# =============================================================================================== #

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    data = common.add_argument_group("dataset")
    data.add_argument("--dataset", type=dataset_number_arg,
                      help="Dataset number (e.g. 17, 17.5 or an inference expansion such as 17.5.3); selects "
                           "data/datasets/dataset_<N>/<regressor|classifier>_dataset_<N>.csv (expansions: inside dataset_<parent>).")
    data.add_argument("--csv", help="Explicit training CSV (e.g. a rectified dataset); overrides the file chosen by --dataset.")
    proto = common.add_argument_group("training protocol")
    proto.add_argument("--folds", type=int, default=0,
                       help="Cross-validation folds before the final fit; 0 trains only the final model (fast). "
                            "Use 5 to also report cross-validation scores, as the tuning scripts do.")
    proto.add_argument("--test-split", type=float, default=0.2, help="Held-out test fraction.")
    proto.add_argument("--seed", type=int, default=42, help="Random state of the split and folds.")
    run = common.add_argument_group("execution")
    run.add_argument("--allow-cpu", action="store_true", help="Train MLP models without a CUDA GPU.")
    run.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run '<mode> --help' for the options of each mode.")
    modes = parser.add_subparsers(dest="mode", required=True)

    train = modes.add_parser("train", parents=[common], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                             help="Train a new model with the given hyperparameters.",
                             description="Trains one model. Hyperparameters: command line > --params file > class defaults. "
                                         "Unset options keep the class defaults shown in the summary before training.")
    what = train.add_argument_group("model")
    what.add_argument("--model", required=True, choices=MODEL_TYPES, help="Model type.")
    what.add_argument("--features", default="raw", choices=CSTFeatureEngineer.MODES,
                      help="Feature engineering for regressors ('raw' = none).")
    what.add_argument("--name", help="Model name (default: a free name from the tuning scripts' pool). No spaces.")
    what.add_argument("--output-dir", help="Folder for the model files (default: models/dataset_<N>/<type folder>).")
    what.add_argument("--params", help="Tuning-style parameters JSON to start from (e.g. an existing model's).")
    hp = train.add_argument_group("hyperparameters (unset = --params file, then class default)")
    hp.add_argument("--layers", type=int, nargs="+", metavar="UNITS", help="[mlp, classifier] Hidden layer sizes.")
    hp.add_argument("--dropout", type=float, help="[mlp] Dropout rate.")
    hp.add_argument("--lr", type=float, help="[all] Learning rate.")
    hp.add_argument("--batch-size", type=int, help="[mlp] Mini-batch size.")
    hp.add_argument("--epochs", type=int, help="[mlp, classifier] Maximum epochs.")
    hp.add_argument("--patience", type=int, help="[mlp, classifier] Early-stopping / LR-scheduler patience.")
    hp.add_argument("--weight", help=f"[mlp, lgbm] Sample weighting. mlp: {', '.join(MLP_WEIGHTS)}; lgbm: {', '.join(LGBM_WEIGHTS)}.")
    log = hp.add_mutually_exclusive_group()
    log.add_argument("--log-trans", dest="log_trans", action="store_true", default=None, help="[mlp] Train on log1p(E - E_min).")
    log.add_argument("--no-log-trans", dest="log_trans", action="store_false", help="[mlp] Train on E directly.")
    hp.add_argument("--n-estimators", type=int, help="[lgbm] Boosting rounds (early stopping may end sooner).")
    hp.add_argument("--num-leaves", type=int, help="[lgbm] Leaves per tree.")
    hp.add_argument("--max-depth", type=int, help="[lgbm] Maximum tree depth.")
    hp.add_argument("--objective", choices=LGBM_OBJECTIVES, help="[lgbm] Training objective.")
    hp.add_argument("--lr-decay", type=float, help="[lgbm] Per-round learning rate decay factor (off by default).")
    hp.add_argument("--stopping-rounds", type=int, help="[lgbm] Early-stopping rounds.")

    variants = {
        "update": ("minor version bump", "New version: minor (Murata -> Murata_1.1, Murata_1.9 -> Murata_1.10). The replaced "
                                         f"model's files move to '<model folder>/{OLD_MODELS_DIR}'."),
        "major-update": ("major version bump", "New version: major (Murata_1.4 -> Murata_2.0). The replaced model's files move "
                                               f"to '<model folder>/{OLD_MODELS_DIR}'."),
        "retrain": ("same name, files overwritten", "The model keeps its name and its files are overwritten; no copy is kept."),
    }
    for name, (short, detail) in variants.items():
        sub = modes.add_parser(name, parents=[common], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                               help=f"Retrain saved models with their own architecture; {short}.",
                               description=f"Scans --target for saved models, reads their architecture from their parameters "
                                           f"and metadata JSON files, asks which ones to retrain and retrains them on the given "
                                           f"dataset. {detail}")
        sub.add_argument("--target", required=True, help="Folder to scan (e.g. models/dataset_17 or one model type folder).")
        sub.add_argument("--select", nargs="+", metavar="NAME", help="Models to retrain, skipping the interactive selection.")
        sub.add_argument("--classifier-csv", help="Explicit training CSV for classifiers (--csv applies to the regressors here).")
        if name != "retrain":
            sub.add_argument("--version", type=_version_arg, metavar="X.Y",
                             help="Exact new version instead of the next one (e.g. 1.5: Murata_1.1 -> Murata_1.5).")
    return parser


def _version_arg(text: str) -> str:
    if not re.fullmatch(r"\d+\.\d+", text):
        raise argparse.ArgumentTypeError(f"'{text}' is not a version (format X.Y, e.g. 1.5).")
    major, minor = text.split(".")
    return f"{int(major)}.{int(minor)}"


def main() -> None:
    args = build_parser().parse_args()
    if not hasattr(args, "version"):
        args.version = None
    if args.mode == "train":
        run_train(args)
    else:
        run_update(args, args.mode)


if __name__ == "__main__":
    main()
