"""
One-time cleanup of a dataset built before 2026-11-10 (dataset 17.5). Delete this script after running it:
Hossain_DB_Gen.py now applies both rules while it builds a dataset.

1. Regressor CSV: removes the rows of the airfoils whose polar is below the laminar CD limit (listed in
   data/logs/dataset_<N>/polars_below_laminar_limit.json). Every other line of the file stays as it is.
2. Classifier CSV: rebuilt with ONE row per CST combination of the generation grid. valid = 1 when the shape
   passed the geometric filters, 0 when they rejected it, so the classifier is a model of the geometric filters
   only (a shape whose simulation failed later still passed the filters and stays valid). The old file had:
     - one valid row per regressor row (three per shape, one per wing configuration), so the same shape sat in
       the training, validation and test splits;
     - valid rows with 4 decimals (0.0417) and invalid rows with 8 (0.04166667), so the label could be read
       from the rounding of a coefficient. Every row is now written with 4 decimals, like the regressor CSV.

Usage
-----
    python src/dataset_cleanup.py --dataset 17.5             # asks before overwriting the two CSVs
    python src/dataset_cleanup.py --dataset 17.5 --dry-run   # only reports what would change
    python src/dataset_cleanup.py --dataset 17.5 --yes

The original files are in git (git diff --stat shows the change; git checkout -- <file> undoes it). A summary is
saved to data/logs/dataset_<N>/7_dataset_cleanup_summary.txt.

Afterwards: retrain the models of the dataset (the test split changes with the rows) and delete or archive its
expansions (data/datasets/dataset_<N>/dataset_<N>.<i> and models/dataset_<N>/dataset_<N>.<i>), which were built
from the old CSV.
"""
import argparse
import ast
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

from geometric_support import create_modification_combinations  # noqa: E402
from file_handling_support.dataset_names import dataset_number_arg, dataset_folder  # noqa: E402

CST_DECIMALS = 4  # precision of the CST columns in the regressor CSV (the .dat header precision)


def shape_text(cst) -> str:
    """CST coefficients as they are written in the CSVs (4 decimals, shortest form): the key of a shape."""
    return ",".join(str(round(float(v), CST_DECIMALS) + 0.0) for v in cst)


def read_generation_log(dataset_num: str) -> dict:
    """n_cst and the CST ranges of each surface from data/logs/Dataset_<N>_*_data_points.log."""
    logs = sorted((PROJECT_ROOT / "data" / "logs").glob(f"Dataset_{dataset_num}_*_data_points.log"))
    if not logs:
        sys.exit(f"[ERROR] No generation log data/logs/Dataset_{dataset_num}_*_data_points.log found.")
    props, section = {}, None
    for line in logs[-1].read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("CST Order (n_cst):"):
            props["n_cst"] = int(stripped.split(":")[1])
        elif stripped in ("Lower surface:", "Upper surface:"):
            section = "ranges_lower" if stripped.startswith("Lower") else "ranges_upper"
        elif stripped.startswith("{") and section:
            props[section] = ast.literal_eval(stripped)
            section = None
    missing = [k for k in ("n_cst", "ranges_lower", "ranges_upper") if k not in props]
    if missing:
        sys.exit(f"[ERROR] {logs[-1].name} does not give {missing}.")
    props["log"] = logs[-1]
    return props


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time cleanup of a dataset built before 2026-11-10.")
    parser.add_argument("--dataset", type=dataset_number_arg, required=True, help="Dataset number, e.g. 17.5.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing.")
    parser.add_argument("--yes", action="store_true", help="Overwrite the CSVs without asking.")
    args = parser.parse_args()

    folder = dataset_folder(args.dataset, PROJECT_ROOT)
    regressor_csv = folder / f"regressor_dataset_{args.dataset}.csv"
    classifier_csv = folder / f"classifier_dataset_{args.dataset}.csv"
    logs_dir = PROJECT_ROOT / "data" / "logs" / f"dataset_{args.dataset}"
    below_json = logs_dir / "polars_below_laminar_limit.json"
    for path in (regressor_csv, classifier_csv, below_json):
        if not path.exists():
            sys.exit(f"[ERROR] Not found: {path}")

    # --- The generation grid: airfoil Test_<i> is combination i - 1 --- #
    props = read_generation_log(args.dataset)
    n_cst = props["n_cst"]
    with contextlib.redirect_stdout(io.StringIO()):
        grid = create_modification_combinations(n_cst, props["ranges_upper"], props["ranges_lower"])
    grid_keys = [shape_text(row) for row in grid]
    if len(set(grid_keys)) != len(grid_keys):
        sys.exit("[ERROR] Two grid combinations are equal at 4 decimals; this script cannot tell them apart.")

    # --- Regressor CSV: drop the airfoils below the laminar CD limit --- #
    with open(below_json) as f:
        below = json.load(f)
    drop = {}
    for entry in below["polars"]:
        number = int(entry["airfoil"].rsplit("_", 1)[1])
        drop[grid_keys[number - 1]] = entry
    lines = regressor_csv.read_text(encoding="utf-8").splitlines()
    header, rows = lines[0], lines[1:]
    n_cols = 2 * n_cst
    kept, removed = [], []
    for line in rows:
        key = shape_text(line.split(",")[:n_cols])
        (removed if key in drop else kept).append(line)
    removed_shapes = {shape_text(line.split(",")[:n_cols]) for line in removed}
    not_found = [entry["airfoil"] for key, entry in drop.items() if key not in removed_shapes]

    # --- Classifier CSV: one row per grid combination, valid = passed the geometric filters --- #
    old = classifier_csv.read_text(encoding="utf-8").splitlines()
    class_header, old_rows = old[0], old[1:]
    old_valid, old_invalid = set(), set()
    for line in old_rows:
        fields = line.split(",")
        (old_valid if fields[-1].strip() == "1" else old_invalid).add(shape_text(fields[:n_cols]))
    grid_set = set(grid_keys)
    problems = []
    if old_valid & old_invalid:
        problems.append(f"{len(old_valid & old_invalid)} shape(s) are both valid and invalid in the old classifier CSV")
    if not old_invalid <= grid_set:
        problems.append(f"{len(old_invalid - grid_set)} invalid shape(s) of the old classifier CSV are not in the grid")
    if not old_valid <= grid_set:
        problems.append(f"{len(old_valid - grid_set)} valid shape(s) of the old classifier CSV are not in the grid")
    new_rows = [f"{key},{0 if key in old_invalid else 1}" for key in grid_keys]
    n_invalid = sum(1 for key in grid_keys if key in old_invalid)
    n_valid = len(grid_keys) - n_invalid
    filter_valid_without_rows = n_valid - len({shape_text(line.split(",")[:n_cols]) for line in kept})

    # --- Summary --- #
    bar = "=" * 96
    text = [bar, f"Dataset {args.dataset} cleanup - summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})", bar,
            f"Grid: {len(grid_keys):,} CST combinations ({props['log'].name})",
            "",
            f"Regressor CSV: {len(rows):,} -> {len(kept):,} rows ({len(removed):,} removed)",
            f"  airfoils below the laminar CD limit ({below['laminar_cd_limit']:.5f}): {len(drop)}"]
    for key, entry in drop.items():
        n = sum(1 for line in removed if shape_text(line.split(",")[:n_cols]) == key)
        text.append(f"    {entry['airfoil']:<12} lowest CD {entry['CD']:.5f} | {n} row(s) removed | CST [{key}]")
    text += ["",
             f"Classifier CSV: {len(old_rows):,} -> {len(new_rows):,} rows (one per CST combination, 4 decimals)",
             f"  valid   (passed the geometric filters)  {n_valid:>9,}   (old file: {len(old_valid):,} shapes in "
             f"{sum(1 for line in old_rows if line.rsplit(',', 1)[-1].strip() == '1'):,} rows)",
             f"  invalid (rejected by the filters)       {n_invalid:>9,}   (old file: {len(old_invalid):,})",
             f"  valid shapes without a regressor row    {filter_valid_without_rows:>9,}   (simulation failed or polar dropped; "
             f"they passed the filters, so they stay valid)"]
    if not_found:
        problems.append(f"no regressor rows found for {not_found}")
    if problems:
        text += ["", "!" * 96] + [f"[ERROR] {p}" for p in problems] + ["Nothing was written.", "!" * 96]
    text.append(bar)
    print("\n".join(text))
    if problems:
        sys.exit(1)
    if args.dry_run:
        print("Dry run: nothing was written.")
        return
    if not args.yes and input(f"Overwrite {regressor_csv.name} and {classifier_csv.name}? [y/n]: ").strip().lower() != "y":
        print("Nothing was written.")
        return

    regressor_csv.write_text("\n".join([header] + kept) + "\n", encoding="utf-8", newline="\n")
    classifier_csv.write_text("\n".join([class_header] + new_rows) + "\n", encoding="utf-8", newline="\n")
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = logs_dir / "7_dataset_cleanup_summary.txt"
    summary_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Written: {regressor_csv.relative_to(PROJECT_ROOT)}")
    print(f"         {classifier_csv.relative_to(PROJECT_ROOT)}")
    print(f"         {summary_path.relative_to(PROJECT_ROOT)}")
    print("Next: retrain the models of this dataset (see the docstring of this script), then delete this script.")


if __name__ == "__main__":
    main()
