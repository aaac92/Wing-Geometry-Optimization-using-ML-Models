"""
Test-split score of a trained model, computed the same way everywhere: by the tuning scripts and ml_gym right
after they train a model (the score is saved next to the model), by utilities/model_score.py for any saved model,
and by the iterative inference for the models it fine-tuned (how much of the original data they still know).

The test split is the split by airfoil shape of ai_support.data_split with the model's split seed (42 unless a
script says otherwise): the model classes use the same functions when they train, so the rows scored here are
exactly the rows a model trained with that seed never saw.

Files written next to a model by save_model_score:
    <Name>_test_score.json   the report (metrics, split, dataset), read back by read_model_score
    <Name>_test_score.txt    the same numbers as text
"""
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, fbeta_score,
                             max_error, mean_absolute_error, precision_score, r2_score, recall_score)

from .data_split import cst_columns, describe_shapes, shape_groups, split_by_shape

DEFAULT_SPLIT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
PREDICT_CHUNK = 65_536          # rows per model call (bounded GPU memory)
CLASSIFIER_TARGET = "valid"
CLASSIFIER_THRESHOLD = 0.5


# =============================================================================================== #
# === Metrics === #
# =============================================================================================== #

def regression_metrics(y_true, y_pred) -> dict:
    """MAE, R2 and the largest absolute error, in the units of the target (E)."""
    y_true, y_pred = np.asarray(y_true, dtype=np.float64).reshape(-1), np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "R2_Score": float(r2_score(y_true, y_pred)),
            "Max_Error": float(max_error(y_true, y_pred))}


def classification_metrics(y_true, y_pred) -> dict:
    """
    Metrics of a valid (1) / invalid (0) classifier from true and predicted labels.
    Accuracy, Precision, Recall, F1 and F2 describe the VALID class, which is about 91% of the shapes: a model that
    calls everything valid already scores 0.91 / 0.91 / 1.0 / 0.95 on them. The numbers that show whether the
    classifier has learned the geometric filters are therefore:
      Invalid_Recall     share of the invalid shapes it recognises (specificity)
      Invalid_Precision  share of its "invalid" answers that are right
      Balanced_Accuracy  mean of the recall of both classes (0.5 for the model that calls everything valid)
    and the confusion matrix itself (Valid_as_Valid, Valid_as_Invalid, Invalid_as_Valid, Invalid_as_Invalid).
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {"Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
            "F2-Score": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
            "Invalid_Recall": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "Invalid_Precision": float(tn / (tn + fn)) if (tn + fn) else 0.0,
            "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
            "Valid_as_Valid": int(tp), "Valid_as_Invalid": int(fn), "Invalid_as_Valid": int(fp), "Invalid_as_Invalid": int(tn)}


# =============================================================================================== #
# === Dataset and split === #
# =============================================================================================== #

def load_table(csv_path, kind: str) -> SimpleNamespace:
    """
    A training CSV as the model classes read it: feature table X (DataFrame), target y and the shape number of
    every row. kind 'regressor': the last column is the target. kind 'classifier': the target is the 'valid'
    column and repeated rows are removed (a classifier dataset has one row per shape; the files written before
    2026-11-10 repeated every valid shape once per wing configuration).
    """
    df = pd.read_csv(csv_path)
    duplicates = 0
    if kind == "classifier":
        if CLASSIFIER_TARGET not in df.columns:
            raise ValueError(f"'{Path(csv_path).name}' has no '{CLASSIFIER_TARGET}' column: it is not a classifier dataset.")
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        duplicates = before - len(df)
        X, y = df.drop(columns=[CLASSIFIER_TARGET]), df[CLASSIFIER_TARGET].to_numpy(np.float32)
    else:
        X, y = df.iloc[:, :-1], df.iloc[:, -1].to_numpy(np.float32)
    return SimpleNamespace(X=X, y=y, groups=shape_groups(X), duplicates=duplicates, target=df.columns[-1] if kind != "classifier" else CLASSIFIER_TARGET)


def load_test_split(csv_path, kind: str, seed: int = DEFAULT_SPLIT_SEED, test_size: float = DEFAULT_TEST_SIZE) -> SimpleNamespace:
    """The table of load_table plus the train and test row indices of the split by shape with that seed."""
    table = load_table(csv_path, kind)
    table.train_idx, table.test_idx = split_by_shape(table.groups, test_size, seed,
                                                     stratify=table.y if kind == "classifier" else None)
    return table


# =============================================================================================== #
# === Scoring === #
# =============================================================================================== #

def predict_rows(model, X: np.ndarray) -> np.ndarray:
    """model.predict over raw feature rows, in chunks, as a flat array."""
    parts = [np.asarray(model.predict(X[i:i + PREDICT_CHUNK])).reshape(-1) for i in range(0, len(X), PREDICT_CHUNK)]
    return np.concatenate(parts) if parts else np.zeros(0)


def score_model(model, kind: str, csv_path, seed: int = DEFAULT_SPLIT_SEED, test_size: float = DEFAULT_TEST_SIZE,
                name: str = None, table: SimpleNamespace = None) -> dict:
    """
    Scores a trained model on the test split (by shape) of a training CSV and returns the report (a JSON-ready
    dict; score_lines turns it into text). kind: 'regressor' or 'classifier'. `table` reuses a load_test_split() result
    (several models of one dataset).
    """
    if table is None:
        table = load_test_split(csv_path, kind, seed, test_size)
    X_test = table.X.to_numpy(np.float32)[table.test_idx]
    y_test = table.y[table.test_idx]
    shapes = describe_shapes(table.groups)
    report = {"model": name, "kind": kind, "dataset_csv": str(csv_path),
              "split": {"by": "shape", "seed": seed, "test_size": test_size, **shapes,
                        "test_rows": int(len(table.test_idx)), "test_shapes": int(len(np.unique(table.groups[table.test_idx]))),
                        "duplicate_rows_removed": int(table.duplicates)},
              "scored": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    predictions = predict_rows(model, X_test)
    if kind == "classifier":
        report["metrics"] = classification_metrics(y_test, predictions > CLASSIFIER_THRESHOLD)
        return report
    report["metrics"] = regression_metrics(y_test, predictions)
    # One-hot columns (wing configurations): the same metrics for each of them
    features = table.X.iloc[table.test_idx]
    cst = set(cst_columns(features.columns))
    per_configuration = {}
    for column in features.columns:
        values = features[column].to_numpy()
        if column in cst or not np.isin(values, (0, 1)).all():
            continue
        rows = values == 1
        if rows.any():
            per_configuration[column] = {"rows": int(rows.sum()), **regression_metrics(y_test[rows], predictions[rows])}
    report["per_configuration"] = per_configuration
    return report


def score_lines(report: dict) -> list:
    """The report of score_model as text lines."""
    split, metrics = report["split"], report["metrics"]
    lines = [f"Model: {report['model']} ({report['kind']})",
             f"Dataset: {report['dataset_csv']}",
             f"Test split: by shape, seed {split['seed']}, {split['test_size']:.0%} of {split['shapes']:,} shapes "
             f"({split['rows']:,} rows, {split['configurations']} configuration(s) per shape) -> "
             f"{split['test_shapes']:,} shapes, {split['test_rows']:,} rows"]
    if split.get("duplicate_rows_removed"):
        lines.append(f"            {split['duplicate_rows_removed']:,} repeated rows of the CSV were removed first")
    if report["kind"] == "classifier":
        lines += [f"Accuracy {metrics['Accuracy']:.5f} | balanced accuracy {metrics['Balanced_Accuracy']:.5f}",
                  f"Valid shapes:   recall {metrics['Recall']:.5f}, precision {metrics['Precision']:.5f}, "
                  f"F1 {metrics['F1-Score']:.5f}, F2 {metrics['F2-Score']:.5f}",
                  f"Invalid shapes: recall {metrics['Invalid_Recall']:.5f}, precision {metrics['Invalid_Precision']:.5f}",
                  "Confusion matrix (rows: true class, columns: predicted class)",
                  f"{'':<16} {'pred. valid':>12} {'pred. invalid':>14}",
                  f"{'true valid':<16} {metrics['Valid_as_Valid']:>12,} {metrics['Valid_as_Invalid']:>14,}",
                  f"{'true invalid':<16} {metrics['Invalid_as_Valid']:>12,} {metrics['Invalid_as_Invalid']:>14,}"]
    else:
        lines.append(f"MAE {metrics['MAE']:.4f} | R2 {metrics['R2_Score']:.5f} | largest error {metrics['Max_Error']:.3f}")
        for column, m in report.get("per_configuration", {}).items():
            lines.append(f"    {column:<8} MAE {m['MAE']:.4f} | R2 {m['R2_Score']:.5f} | largest error {m['Max_Error']:.3f} "
                         f"| {m['rows']:,} rows")
    lines.append(f"Scored: {report['scored']}")
    return lines


def save_model_score(folder, name: str, report: dict) -> tuple:
    """Writes <name>_test_score.json and <name>_test_score.txt in the model's folder. Returns both paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    json_path, text_path = folder / f"{name}_test_score.json", folder / f"{name}_test_score.txt"
    json_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    text_path.write_text("\n".join(score_lines(report)) + "\n", encoding="utf-8")
    return json_path, text_path


def read_model_score(folder, name: str):
    """The saved test score of a model (see save_model_score), or None when it has none."""
    path = Path(folder) / f"{name}_test_score.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


def score_and_save(model, kind: str, csv_path, folder, name: str, seed: int = DEFAULT_SPLIT_SEED,
                   test_size: float = DEFAULT_TEST_SIZE) -> dict:
    """Scores a freshly trained model, saves the score next to it and prints it. Returns the report."""
    report = score_model(model, kind, csv_path, seed=seed, test_size=test_size, name=name)
    _, text_path = save_model_score(folder, name, report)
    print("\n" + "\n".join(score_lines(report)))
    print(f"Test score saved: {text_path}")
    return report
