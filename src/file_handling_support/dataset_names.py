"""
Dataset numbers. A dataset is a whole number (17) or, for a rebuild of an existing dataset with a corrected
pipeline, a decimal one (17.5 is the rebuild of 17). The number is kept as text so it appears in folder and
file names exactly as written: data/datasets/dataset_17.5/regressor_dataset_17.5.csv, models/dataset_17.5.
"""
import argparse
import re

_PATTERN = re.compile(r"^\d+(\.\d+)?$")


def dataset_number(text) -> str:
    """'17' -> '17', '17.5' -> '17.5', '17.50' -> '17.5', '17.0' -> '17'. Raises ValueError for anything else."""
    text = str(text).strip()
    if not _PATTERN.match(text):
        raise ValueError(f"'{text}' is not a dataset number (examples: 17, 17.5).")
    whole, _, fraction = text.partition(".")
    whole = str(int(whole))
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def dataset_number_arg(text) -> str:
    """argparse type for --dataset."""
    try:
        return dataset_number(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
