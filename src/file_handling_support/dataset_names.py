"""
Dataset numbers. A dataset is a whole number (17) or, for a rebuild of an existing dataset with a corrected
pipeline, a decimal one (17.5 is the rebuild of 17). A third part marks an expansion of a dataset by the
iterative inference: 17.5.3 is dataset 17.5 plus the simulations of a 3-iteration run (models 1.5 -> 1.5.3).
The number is kept as text so it appears in folder and file names exactly as written.

Folders: whole and one-decimal datasets sit at the top level, expansions inside their parent dataset:
    data/datasets/dataset_17.5/regressor_dataset_17.5.csv
    data/datasets/dataset_17.5/dataset_17.5.3/regressor_dataset_17.5.3.csv
    models/dataset_17.5/<type folders>          models/dataset_17.5/dataset_17.5.3/<type folders>
"""
import argparse
import re
from pathlib import Path

_PATTERN = re.compile(r"^\d+(\.\d+){0,2}$")


def dataset_number(text) -> str:
    """'17' -> '17', '17.5' -> '17.5', '17.50' -> '17.5', '17.0' -> '17', '17.5.3' -> '17.5.3'. Raises ValueError otherwise.
    Trailing zeros are only dropped from the decimal of a two-part number (17.50 -> 17.5); an expansion keeps its
    iteration count as written (17.5.10 stays 17.5.10)."""
    text = str(text).strip()
    if not _PATTERN.match(text):
        raise ValueError(f"'{text}' is not a dataset number (examples: 17, 17.5, 17.5.3).")
    parts = text.split(".")
    whole = str(int(parts[0]))
    if len(parts) == 3:
        return f"{whole}.{parts[1]}.{int(parts[2])}"
    fraction = parts[1].rstrip("0") if len(parts) == 2 else ""
    return f"{whole}.{fraction}" if fraction else whole


def dataset_number_arg(text) -> str:
    """argparse type for --dataset."""
    try:
        return dataset_number(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def parent_dataset(number: str):
    """'17.5.3' -> '17.5' (the dataset it expands); None for whole and one-decimal datasets."""
    parts = dataset_number(number).split(".")
    return ".".join(parts[:2]) if len(parts) == 3 else None


def dataset_folder(number: str, project_root) -> Path:
    """data/datasets/dataset_<N>, or data/datasets/dataset_<parent>/dataset_<N> for an expansion."""
    number = dataset_number(number)
    parent = parent_dataset(number)
    root = Path(project_root) / "data" / "datasets"
    return (root / f"dataset_{parent}" / f"dataset_{number}") if parent else (root / f"dataset_{number}")


def dataset_csv(number: str, project_root, kind: str = "regressor") -> Path:
    """<dataset folder>/<kind>_dataset_<N>.csv (kind: 'regressor' or 'classifier')."""
    number = dataset_number(number)
    return dataset_folder(number, project_root) / f"{kind}_dataset_{number}.csv"


def models_folder(number: str, project_root) -> Path:
    """models/dataset_<N>, or models/dataset_<parent>/dataset_<N> for an expansion."""
    number = dataset_number(number)
    parent = parent_dataset(number)
    root = Path(project_root) / "models"
    return (root / f"dataset_{parent}" / f"dataset_{number}") if parent else (root / f"dataset_{number}")


def expansion_number(number: str, iterations: int) -> str:
    """Number of dataset <number> expanded by a run of `iterations` iterations: ('17.5', 3) -> '17.5.3'."""
    number = dataset_number(number)
    if parent_dataset(number):
        raise ValueError(f"Dataset {number} is already an expansion; expand its parent {parent_dataset(number)} instead.")
    if "." not in number:
        number = f"{number}.0"
    return f"{number}.{int(iterations)}"
