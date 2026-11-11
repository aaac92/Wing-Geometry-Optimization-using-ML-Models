"""
Scores saved models on the test split of their dataset and writes a report to results/dataset_<N>/model_scores.

The test split is the split by airfoil shape that the model classes use when they train (ai_support.data_split):
with the same seed, the scored rows are exactly the rows the model never saw. A model trained before 2026-11-10
used a split by row, so its training data contains most of these shapes and its score here is too good; retrain
it first (utilities/ml_gym.py retrain).

The tuning scripts and ml_gym already save this score next to every model they train (<Name>_test_score.txt).
This script recomputes it for any saved model, e.g. with another seed, or for the models an inference run
fine-tuned: for an expansion (17.5.6) the test split is the one of its parent dataset (17.5), which shows how much
of the original data the fine-tuned models still know.

Usage
-----
    python src/utilities/model_score.py --dataset 17.5                      # every model of dataset 17.5
    python src/utilities/model_score.py --dataset 17.5 --models Marbas Murata_1.5 Calipso_1.5
    python src/utilities/model_score.py --dataset 17.5.6 --seed 42          # fine-tuned models, parent test split
    python src/utilities/model_score.py --dataset 17.5 --csv path/to/regressor.csv --classifier-csv path/to/classifier.csv

Output: results/dataset_<N>/model_scores/model_scores_<date>_<time>_seed<seed>.txt and .json.
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from ai_support.model_loading import (ModelIncompatibleError, check_lfs_pointers, find_model, list_classifier_names,  # noqa: E402
                                      list_regressor_names, load_classifier, load_regressor)
from ai_support.scoring import DEFAULT_SPLIT_SEED, DEFAULT_TEST_SIZE, load_test_split, score_lines, score_model  # noqa: E402
from file_handling_support.dataset_names import dataset_csv, dataset_number_arg, models_folder, parent_dataset  # noqa: E402

# LightGBM models saved with feature names warn on every prediction from a plain array; the columns are in order
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scores saved models on the test split (by airfoil shape) of their dataset.")
    parser.add_argument("--dataset", type=dataset_number_arg, required=True, help="Dataset number, e.g. 17.5 (or an expansion, 17.5.6).")
    parser.add_argument("--models", nargs="+", default=None, help="Model names (regressors and classifiers). Default: every model of the dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED, help="Seed of the test split (the models' training seed).")
    parser.add_argument("--test-split", type=float, default=DEFAULT_TEST_SIZE, help="Test share of the shapes.")
    parser.add_argument("--csv", help="Regressor CSV (default: the dataset's, or its parent's for an expansion).")
    parser.add_argument("--classifier-csv", help="Classifier CSV (default: the dataset's, or its parent's for an expansion).")
    args = parser.parse_args()

    models_dir = models_folder(args.dataset, PROJECT_ROOT)
    if not models_dir.is_dir():
        sys.exit(f"[ERROR] Models folder not found: {models_dir}")
    data_number = parent_dataset(args.dataset) or args.dataset   # an expansion is tested on its parent's split
    csv_of = {"regressor": Path(args.csv) if args.csv else dataset_csv(data_number, PROJECT_ROOT),
              "classifier": Path(args.classifier_csv) if args.classifier_csv else dataset_csv(data_number, PROJECT_ROOT, kind="classifier")}
    names = args.models or (list_regressor_names(models_dir) + list_classifier_names(models_dir))
    if not names:
        sys.exit(f"[ERROR] No models found under {models_dir}.")
    try:
        found = {name: find_model(name, models_dir) for name in names}
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[ERROR] {e}")
    check_lfs_pointers([(folder, name) for name, (folder, _, _) in found.items()], PROJECT_ROOT)
    for kind in {kind for _, kind, _ in found.values()}:
        if not csv_of[kind].exists():
            sys.exit(f"[ERROR] {kind.capitalize()} dataset not found: {csv_of[kind]}")

    print(f"Dataset {args.dataset}" + (f" (test split of its parent, dataset {data_number})" if data_number != args.dataset else "")
          + f" | test split by shape, seed {args.seed}, {args.test_split:.0%} of the shapes | {len(names)} model(s)")
    tables, reports, skipped = {}, [], []
    for name, (folder, kind, mtype) in found.items():
        try:
            model = load_classifier(name, folder) if kind == "classifier" else load_regressor(name, folder, mtype)
        except (ModelIncompatibleError, ValueError, FileNotFoundError) as e:
            skipped.append((name, str(e)))
            continue
        if model is None:
            skipped.append((name, "SKNN models cannot be loaded here"))
            continue
        if kind not in tables:
            tables[kind] = load_test_split(csv_of[kind], kind, args.seed, args.test_split)
        report = score_model(model, kind, csv_of[kind], seed=args.seed, test_size=args.test_split, name=name, table=tables[kind])
        report["model_folder"] = str(folder)
        reports.append(report)
        print("\n" + "\n".join(score_lines(report)))

    bar = "=" * 100
    lines = [bar, f"Model scores, dataset {args.dataset} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
             f"Test split: by airfoil shape, seed {args.seed}, {args.test_split:.0%} of the shapes"
             + (f" of dataset {data_number} (the parent of {args.dataset})" if data_number != args.dataset else ""), bar]
    regressors = [r for r in reports if r["kind"] == "regressor"]
    classifiers = [r for r in reports if r["kind"] == "classifier"]
    if regressors:
        lines += ["", f"{'Regressor':<24} {'MAE':>8} {'R2':>9} {'Largest error':>14}"]
        lines += [f"{r['model']:<24} {r['metrics']['MAE']:>8.4f} {r['metrics']['R2_Score']:>9.5f} {r['metrics']['Max_Error']:>14.3f}"
                  for r in sorted(regressors, key=lambda r: r["metrics"]["MAE"])]
    if classifiers:
        lines += ["", f"{'Classifier':<24} {'Accuracy':>9} {'Balanced acc.':>14} {'Invalid recall':>15} {'Valid recall':>13}"]
        lines += [f"{r['model']:<24} {r['metrics']['Accuracy']:>9.5f} {r['metrics']['Balanced_Accuracy']:>14.5f} "
                  f"{r['metrics']['Invalid_Recall']:>15.5f} {r['metrics']['Recall']:>13.5f}" for r in classifiers]
    for name, reason in skipped:
        lines.append(f"[SKIPPED] {name}: {reason}")
    for report in reports:
        lines += ["", "-" * 100] + score_lines(report)
    lines.append(bar)

    out_dir = PROJECT_ROOT / "results" / f"dataset_{args.dataset}" / "model_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"model_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(json.dumps({"dataset": args.dataset, "split_dataset": data_number, "seed": args.seed,
                                                      "test_size": args.test_split, "reports": reports,
                                                      "skipped": [{"model": n, "reason": r} for n, r in skipped]}, indent=4),
                                          encoding="utf-8")
    summary_end = next(i for i, line in enumerate(lines) if line.startswith("-" * 100)) if reports else len(lines)
    print("\n" + "\n".join(lines[:summary_end]).rstrip())
    print(f"\nReport: {(out_dir / (stem + '.txt')).relative_to(PROJECT_ROOT)}")
    if not reports:
        sys.exit(1)


if __name__ == "__main__":
    main()
