"""
Model names. Every saved model owns files named '<name>_<suffix>' (e.g. Murata_1.5_model.pth), where the name
is a pool name plus an optional version (Murata, Murata_1.5, or Murata_1.5.3 for a model fine-tuned by a
3-iteration inference run). Inference finds models by name inside models/dataset_<N>, so two models of one
dataset must not share a pool name.
"""
import random
import re
from pathlib import Path

MODEL_FILE_SUFFIXES = ("model.pth", "model.joblib", "meta.json", "parameters.json", "regressor_model_parameters.json")
_VERSION = re.compile(r"_\d+\.\d+(\.\d+)?$")


def model_version(name: str):
    """Murata_1.5 -> '1.5', Murata_1.5.3 -> '1.5.3', Murata -> None."""
    match = _VERSION.search(name)
    return match.group(0)[1:] if match else None


def expanded_model_name(name: str, iterations: int) -> str:
    """Name of a model fine-tuned by an inference run of `iterations` iterations: Murata_1.5 -> Murata_1.5.3
    (an unversioned model counts as 1.0: Murata -> Murata_1.0.3)."""
    version = model_version(name)
    if version and version.count(".") == 2:
        raise ValueError(f"Model {name} is already fine-tuned by an inference run; start from its parent version.")
    return f"{base_name(name)}_{version or '1.0'}.{int(iterations)}"


def saved_models(models_root: Path) -> dict:
    """{model name: folder relative to models_root} for every model saved under models_root (old models included)."""
    found = {}
    if not Path(models_root).exists():
        return found
    for path in Path(models_root).rglob("*"):
        for suffix in MODEL_FILE_SUFFIXES:
            if path.name.endswith(f"_{suffix}"):
                name = path.name[:-len(suffix) - 1]
                if name.endswith("_regressor_model"):  # '<name>_regressor_model_parameters.json' also ends in '_parameters.json'
                    name = name[:-len("_regressor_model")]
                found.setdefault(name, str(path.parent.relative_to(models_root)))
                break
    return found


def base_name(name: str) -> str:
    """Murata_1.5 -> Murata."""
    return _VERSION.sub("", name)


def pick_model_name(pool, models_root: Path, kind: str = "model") -> str:
    """
    A random pool name that no model under models_root uses yet (any version, any model type, old models
    included). Prints the models already there. Stops with a clear message when the whole pool is taken.
    """
    existing = saved_models(models_root)
    if existing:
        print(f"Models already saved in '{models_root}':")
        for name, folder in sorted(existing.items()):
            print(f"  {name:<24} {folder}")
    taken = {base_name(n) for n in existing}
    free = [n for n in pool if n not in taken]
    if not free:
        raise SystemExit(f"[ERROR] Every {kind} pool name ({', '.join(pool)}) is already used in '{models_root}'. "
                         f"Add a name to the pool in this script.")
    name = random.choice(free)
    skipped = [n for n in pool if n in taken]
    print(f"New {kind} name: {name}" + (f" (skipping names in use: {', '.join(skipped)})" if skipped else ""))
    return name
