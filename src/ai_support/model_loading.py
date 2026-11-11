"""
Finding and loading saved models by name, shared by inference_and_testing.py and utilities/model_score.py.

A model is a set of files '<Name>_<suffix>' inside models/dataset_<N>/<type folder> (see
file_handling_support.model_names). The type folder gives the model type:
    mlp_regressor[_<features>]   MLP regressor      (ai_support.MLP_Regressor)
    lgbm_regressor[_<features>]  LightGBM regressor (ai_support.LGBM_Regressor)
    sknn_regressor[...]          polynomial regressor (found, but not loadable here)
    classifier                   MLP classifier     (ai_support.Classifier)
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import torch

from .models import Classifier, LGBM_Regressor, MLP_Regressor

REGRESSOR_FOLDER_TYPES = {'mlp_regressor': 'mlp', 'lgbm_regressor': 'lgbm', 'sknn_regressor': 'sknn'}
TYPE_LABELS = {'mlp': 'MLP', 'lgbm': 'LGBM', 'sknn': 'SKNN', 'classifier': 'MLP classifier'}
CLASSIFIER_FOLDER = "classifier"


def infer_hidden_layers(pth_path: Path) -> list:
    """Hidden layer sizes from a saved state dict (every 2-D '.weight' tensor is a Linear layer; the last is the output)."""
    state = torch.load(pth_path, map_location='cpu', weights_only=True)
    linears = [v for k, v in state.items() if k.endswith('.weight') and v.ndim == 2]
    if len(linears) < 2 or linears[-1].shape[0] != 1:
        raise ValueError(f"Unexpected network layout in '{pth_path.name}'; cannot infer hidden layers.")
    return [int(w.shape[0]) for w in linears[:-1]]


class ModelIncompatibleError(Exception):
    """The model was saved without the files this scheme needs (it is skipped, not fatal)."""


def read_parameters(folder: Path, name: str) -> dict:
    path = folder / f"{name}_regressor_model_parameters.json"
    if not path.exists():
        raise ModelIncompatibleError(f"Parameters file not found: {path}")
    return json.loads(path.read_text())


def list_regressor_names(models_dir: Path) -> list:
    names = []
    for sub in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if any(sub.name.startswith(prefix) for prefix in REGRESSOR_FOLDER_TYPES):
            names += [f.name[:-len('_meta.json')] for f in sub.glob('*_meta.json')]
    return sorted(names)


def resolve_regressor(name: str, models_dir: Path):
    """(folder, type) of regressor 'name'."""
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models folder does not exist: {models_dir}")
    matches = []
    for sub in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        for prefix, mtype in REGRESSOR_FOLDER_TYPES.items():
            if sub.name.startswith(prefix):
                owns = any(sub.glob(f"{name}_*")) if mtype == 'sknn' else (sub / f"{name}_meta.json").exists()
                if owns:
                    matches.append((sub, mtype))
    if not matches:
        raise FileNotFoundError(f"No regressor named '{name}' under '{models_dir}'. Available: {list_regressor_names(models_dir)}")
    if len(matches) > 1:
        raise ValueError(f"Regressor name '{name}' is ambiguous, found in: {[str(m[0]) for m in matches]}")
    return matches[0]


def load_regressor(name: str, folder: Path, mtype: str):
    """The loaded regressor with its training hyperparameters (fine-tuning needs them), or None for SKNN. The model
    classes' own loading messages are silenced (every lane loads its own copies)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return _load_regressor(name, folder, mtype)


def _load_regressor(name: str, folder: Path, mtype: str):
    if mtype == 'sknn':
        return None
    params = read_parameters(folder, name)
    if mtype == 'mlp':
        hidden = infer_hidden_layers(folder / f"{name}_model.pth")
        if 'log_trans' not in params:
            raise ModelIncompatibleError(f"'log_trans' missing in {name}_regressor_model_parameters.json")
        if 'n_layers' in params:
            expected = [int(params[f'n_units_l{i}']) for i in range(int(params['n_layers']))]
            if expected != hidden:
                raise ValueError(f"Layer mismatch for '{name}': weights {hidden} vs parameters file {expected}.")
        model = MLP_Regressor.load_model(folder, name, hidden_layer_sizes=hidden,
                                         dropout_rate=float(params.get('dropout_rate', 0.0)), log_trans=bool(params['log_trans']),
                                         learning_rate=float(params.get('learning_rate', 1e-3)),
                                         batch_size=int(params.get('batch_size', 1024)), weight=params.get('weight', 'None'),
                                         patience=int(params.get('patience', 16)))
    elif mtype == 'lgbm':
        model = LGBM_Regressor.load_model(folder, name)
        for key in ('n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'objective', 'weight', 'lr_decay', 'stopping_rounds'):
            if key in params:
                setattr(model, key, params[key])
    else:
        raise ValueError(f"Unsupported model type '{mtype}'.")
    model.parameters_json = params
    return model


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def check_lfs_pointers(models: list, project_root: Path):
    """
    Stops with a clear message when a model file is a Git LFS pointer (a ~130-byte text file that stands in for the
    real file until 'git lfs pull' downloads it). Unpickling a pointer otherwise fails with a cryptic
    'KeyError: 118' (118 is the 'v' of 'version').
    """
    pointers = []
    for folder, name in models:
        for path in sorted(Path(folder).glob(f"{name}_*")):
            if path.is_file() and path.stat().st_size < 1024:
                with open(path, "rb") as f:
                    if f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                        pointers.append(path)
    if pointers:
        print("\n" + "!" * 90)
        print(f"[ERROR] {len(pointers)} model file(s) are Git LFS pointers, not the real files (the LFS content was "
              f"never downloaded):")
        for path in pointers:
            print(f"        {path.relative_to(project_root) if path.is_relative_to(project_root) else path}")
        print("        Download them from the project folder, then run this script again:")
        print("            git lfs install")
        print("            git lfs pull")
        print("!" * 90)
        sys.exit(1)


def load_classifier(name: str, folder: Path):
    """The loaded classifier (its layer sizes are read from the saved weights)."""
    pth = folder / f"{name}_model.pth"
    if not pth.exists():
        raise FileNotFoundError(f"Classifier '{name}' not found: {pth}")
    with contextlib.redirect_stdout(io.StringIO()):   # the class's own loading messages are silenced
        return Classifier.load_model(folder, name, hidden_layer_sizes=infer_hidden_layers(pth))


def list_classifier_names(models_dir: Path) -> list:
    folder = Path(models_dir) / CLASSIFIER_FOLDER
    return sorted(f.name[:-len('_meta.json')] for f in folder.glob('*_meta.json')) if folder.is_dir() else []


def find_model(name: str, models_dir: Path) -> tuple:
    """(folder, kind, type) of any saved model: kind 'regressor' (type 'mlp', 'lgbm' or 'sknn') or 'classifier'."""
    models_dir = Path(models_dir)
    if (models_dir / CLASSIFIER_FOLDER / f"{name}_meta.json").exists():
        return models_dir / CLASSIFIER_FOLDER, "classifier", "classifier"
    try:
        folder, mtype = resolve_regressor(name, models_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"No model named '{name}' under '{models_dir}'. Regressors: {list_regressor_names(models_dir)}; "
                                f"classifiers: {list_classifier_names(models_dir)}") from None
    return folder, "regressor", mtype
