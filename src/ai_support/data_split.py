"""
Data splits by airfoil shape, shared by every model class (ai_support.models), the model scoring
(ai_support.scoring) and the iterative inference.

WHY BY SHAPE
    A regressor dataset has one row per airfoil shape and wing configuration: the same 10 CST coefficients
    appear three times (rec, flat, 60DEG) with three E values that are almost proportional to each other. A split
    by row puts, for most test rows, the same shape with another wing configuration in the training rows, so the
    test error measures how well the model copies a shape it has seen, not how well it predicts a new shape.
    Until 2026-11-10 every split (test split, cross-validation folds, early-stopping slice) was by row. Now a
    shape is never on both sides of any of them.

HOW
    shape_groups() numbers the shapes: rows with the same values in the CST columns (cst_l_<i>, cst_u_<i>) are one
    shape. The number of wing configurations is not needed and not assumed: it follows from how often each CST
    combination repeats (describe_shapes reports it), and shapes that miss a configuration are handled the same.
    The shapes, not the rows, are then shuffled and divided, so the split depends only on the set of shapes and
    the seed, not on the order of the rows in the CSV.
    A table without CST columns falls back to one "shape" per row (a plain row split).
"""
import re

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

CST_COLUMN = re.compile(r"cst_[lu]_\d+")
CST_DECIMALS = 6   # coefficients are compared after rounding to this many decimals


def cst_columns(columns) -> list:
    """Names of the CST coefficient columns among `columns`, in their order."""
    return [c for c in columns if CST_COLUMN.fullmatch(str(c))]


def shape_groups(table) -> np.ndarray:
    """
    Shape number of every row of a pandas DataFrame with the dataset's columns: rows with the same CST
    coefficients get the same number (0 .. shapes - 1, numbered in sorted CST order).
    """
    columns = cst_columns(table.columns)
    if not columns:
        return np.arange(len(table))
    values = np.round(table[columns].to_numpy(np.float64), CST_DECIMALS) + 0.0   # + 0.0: -0.0 and 0.0 are one value
    _, groups = np.unique(values, axis=0, return_inverse=True)
    return np.asarray(groups).reshape(-1)


def describe_shapes(groups) -> dict:
    """Rows, shapes and rows per shape (the number of wing configurations) of a table."""
    groups = np.asarray(groups)
    counts = np.bincount(groups) if len(groups) else np.zeros(0, dtype=int)
    counts = counts[counts > 0]
    configurations = int(counts.max()) if len(counts) else 0
    return {"rows": int(len(groups)), "shapes": int(len(counts)), "configurations": configurations,
            "incomplete_shapes": int((counts < configurations).sum())}


def shapes_text(groups) -> str:
    """One line for the console and the reports, e.g. '404,031 rows = 134,677 shapes x 3 configurations'."""
    d = describe_shapes(groups)
    text = f"{d['rows']:,} rows = {d['shapes']:,} shapes x {d['configurations']} configuration(s)"
    return text + (f" ({d['incomplete_shapes']:,} shapes miss a configuration)" if d["incomplete_shapes"] else "")


def _group_labels(groups, stratify):
    """(unique group numbers, the class of each group or None). A group's class is the class of its first row."""
    unique, first = np.unique(groups, return_index=True)
    return unique, (None if stratify is None else np.asarray(stratify).reshape(-1)[first])


def split_by_shape(groups, test_size: float, random_state, stratify=None) -> tuple:
    """
    (train row indices, test row indices): `test_size` of the SHAPES go to the test side with all of their rows.
    stratify: optional class of every row (classifier): both sides keep the class proportions.
    """
    groups = np.asarray(groups)
    unique, labels = _group_labels(groups, stratify)
    train_groups, test_groups = train_test_split(unique, test_size=test_size, random_state=random_state, stratify=labels)
    in_test = np.isin(groups, test_groups)
    return np.flatnonzero(~in_test), np.flatnonzero(in_test)


def kfold_by_shape(groups, k: int, random_state, stratify=None) -> list:
    """
    k cross-validation folds as [(train row indices, validation row indices)]: the SHAPES are divided into k parts,
    so a shape is never in the training and the validation rows of the same fold.
    """
    groups = np.asarray(groups)
    unique, labels = _group_labels(groups, stratify)
    if labels is None:
        parts = KFold(n_splits=k, shuffle=True, random_state=random_state).split(unique)
    else:
        parts = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state).split(unique, labels)
    folds = []
    for _, val_part in parts:
        in_val = np.isin(groups, unique[val_part])
        folds.append((np.flatnonzero(~in_val), np.flatnonzero(in_val)))
    return folds
