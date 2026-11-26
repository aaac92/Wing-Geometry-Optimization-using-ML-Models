"""
Spanish-labelled result figures of iterative inference runs, for the thesis.

Redraws the two figures that inference_and_testing.py writes for every run (convergence.png and parity.png) with
Spanish labels, the thesis palette (the colors of filter_plotter.py) and shared axis limits, so the runs the thesis
reports side by side can be compared. The plot functions live in geometric_support/custom_graphs.py, next to the
filter plots; this driver loads the run files and selects the data.

The data selection is the one of inference_and_testing.save_plots(), done with the same functions:
- Convergence: simulations_to_reach() counts every shape a lane requested, in the order of the iteration that
  requested it, and keeps the best valid simulated E over the wing configurations; dataset shapes count but never
  raise the best. This is not the trust-region monitor's best_E of iterations.csv (they differ in iteration 0).
- Parity: valid_E() of every pick of every lane against each model's prediction when the shape was picked, one
  point per wing configuration.

Inputs per run: detailed_results.json (lanes, picks, dataset best, target), simulations.jsonl (every simulation,
read with Simulation_Store), the run metadata (dataset number, wing mode, models) and the dataset's regressor CSV
(only its CST columns, for the set of dataset shapes).

For the thesis runs A, B and C the script also checks the target, the dataset best and, per lane, the final best E
and the shapes needed to reach the dataset best against the values the thesis reports.

Usage
-----
    python src/utilities/results_plotter.py                     # thesis runs A, B and C
    python src/utilities/results_plotter.py --show              # also open the figures
    python src/utilities/results_plotter.py --runs results/dataset_17.6/<run folder> --labels D --out <folder>
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

utilities_dir = Path(__file__).resolve().parent
src_dir = utilities_dir.parent
project_root = src_dir.parent
results_dir = project_root / "results" / "dataset_17.6"
sys.path.insert(0, str(src_dir))

from geometric_support.custom_graphs import plot_convergence, plot_parity  # noqa: E402
from file_handling_support.dataset_names import dataset_csv  # noqa: E402
from simulation_support import laminar_cd_limit  # noqa: E402
from inference_and_testing import (RE, Simulation_Store, config_identifiers, parse_dataset_columns, shape_key,  # noqa: E402
                                   simulations_to_reach, valid_E)

# Thesis runs, in the order the thesis labels them
THESIS_RUNS = {
    "A": results_dir / "20261125_m_seede021ce33",
    "B": results_dir / "20261125_m_seed67",
    "C": results_dir / "20261125_m_seed69",
}
DEFAULT_OUT = results_dir / "figures_es"
FILE_NAMES = {"convergence": "convergencia_carriles_{label}.png", "parity": "paridad_predicciones_{label}.png"}
DPI = 300

# Values the thesis reports, per run folder: target, dataset best and {lane: (final best E, shapes to reach 100%)}
EXPECTED = {
    "20261125_m_seede021ce33": {"target": 21.8056, "dataset_best": 21.7872, "lanes": {
        "Marbas_1.6": (23.9446, 8), "Odysseus_1.6": (24.4258, 8), "Murata_1.6": (24.6781, 18),
        "consensus": (23.5689, 8), "random": (24.3731, 33)}},
    "20261125_m_seed67": {"target": 21.8056, "dataset_best": 21.7872, "lanes": {
        "Marbas_1.6": (24.2727, 13), "Odysseus_1.6": (22.8648, 12), "Murata_1.6": (22.4389, 35),
        "consensus": (24.2648, 7), "random": (23.4941, 31)}},
    "20261125_m_seed69": {"target": 21.8056, "dataset_best": 21.7872, "lanes": {
        "Marbas_1.6": (23.7140, 9), "Odysseus_1.6": (23.0808, 11), "Murata_1.6": (22.6790, 47),
        "consensus": (23.8488, 9), "random": (23.5322, 35)}},
}
TOLERANCE = 5e-5   # the thesis values have four decimals

# Legend labels of the lanes (the model lanes keep their model name)
LANE_LABELS = {"consensus": "Consenso", "random": "Aleatorio"}

# Convergence palette: one color per lane, the same in every run
CV_COLORS = {
    'cv_Marbas_1.6': '#3B9574',
    'cv_Odysseus_1.6': '#2A4E68',
    'cv_Murata_1.6': '#D1B8E1',
    'cv_Marbas': '#3B9574',          # the lanes of the 17.5 runs
    'cv_Odysseus_1.5': '#2A4E68',
    'cv_Murata_1.5': '#D1B8E1',
    'cv_consensus': '#81C6AA',
    'cv_random': '#484C68',
    'cv_dataset_best': '#3F6D75',
    'cv_target': '#808080',
}
CV_STYLES = {
    'Murata_1.6': {'linewidth': 3},     # light colors, drawn thicker
    'Murata_1.5': {'linewidth': 3},
    'consensus': {'linewidth': 3},
    'random': {'linestyle': '--'},      # control lane
}

# Parity palette: each model keeps the color of its lane
PP_COLORS = {
    'pp_Marbas_1.6': '#3B9574',
    'pp_Odysseus_1.6': '#2A4E68',
    'pp_Murata_1.6': '#D1B8E1',
    'pp_Marbas': '#3B9574',          # the models of the 17.5 runs
    'pp_Odysseus_1.5': '#2A4E68',
    'pp_Murata_1.5': '#D1B8E1',
    'pp_perfect': '#484C68',
    'pp_dataset_best': '#3F6D75',
}

# The thesis shows the figures at half the page width (about a third of their size), so the text is large here
FONT_SIZES = {'label': 24, 'tick': 20, 'legend': 16}
FONT_FAMILIES = ("Helvetica", "Arial")   # first one installed wins (the thesis font is Helvetica)


def use_thesis_font(families=FONT_FAMILIES):
    """
    Sets the first installed family of `families` as the figure font (text and $math$), registering its files when
    matplotlib's font cache predates them (e.g. fonts installed per user on Windows). Returns the family, or None to
    keep the matplotlib default.
    """
    known = {f.name for f in font_manager.fontManager.ttflist}
    system_fonts = None
    for family in families:
        if family not in known:
            system_fonts = system_fonts if system_fonts is not None else font_manager.findSystemFonts()
            token = family.lower().replace(" ", "")
            for path in system_fonts:
                if token in Path(path).stem.lower().replace(" ", "").replace("-", ""):
                    try:
                        font_manager.fontManager.addfont(path)
                    except Exception:
                        pass
            known = {f.name for f in font_manager.fontManager.ttflist}
        if family in known:
            matplotlib.rcParams.update({'font.family': family, 'mathtext.fontset': 'custom', 'mathtext.rm': family,
                                        'mathtext.it': f'{family}:italic', 'mathtext.bf': f'{family}:bold'})
            return family
    return None


def short(path: Path) -> str:
    """Path relative to the project root when possible."""
    path = Path(path).resolve()
    return str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path)


def resolve_run(arg: str) -> Path:
    """A run folder given as written, or relative to the project root."""
    path = Path(arg)
    if not path.exists() and (project_root / path).exists():
        path = project_root / path
    if not (path / "detailed_results.json").exists():
        raise FileNotFoundError(f"No detailed_results.json in run folder '{arg}'.")
    return path.resolve()


_dataset_keys_cache = {}


def dataset_keys(csv_path: Path, wing_mode: str) -> tuple:
    """(dataset shape keys, configuration identifiers in dataset column order), as inference_and_testing.main builds them."""
    if csv_path not in _dataset_keys_cache:
        _, cst_l_cols, cst_u_cols, config_cols = parse_dataset_columns(csv_path, wing_mode)
        cst_names = cst_l_cols + cst_u_cols
        cst = pd.read_csv(csv_path, usecols=cst_names)[cst_names].to_numpy(np.float64)
        identifier_of = config_identifiers(config_cols)
        _dataset_keys_cache[csv_path] = ({shape_key(c) for c in np.unique(cst, axis=0)},
                                         [identifier_of[c] for c in config_cols])
    return _dataset_keys_cache[csv_path]


def load_run(run_dir: Path) -> SimpleNamespace:
    """The run's lanes and a context with what simulations_to_reach() and valid_E() read from the original one."""
    metadata_files = sorted(run_dir.glob("*_run_metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No run metadata (*_run_metadata.json) in '{short(run_dir)}'.")
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    details = json.loads((run_dir / "detailed_results.json").read_text(encoding="utf-8"))
    csv_path = dataset_csv(metadata["dataset"], project_root)
    keys, identifiers = dataset_keys(csv_path, metadata["wing_mode"])

    ctx = SimpleNamespace(dataset_keys=keys, identifiers=identifiers, cd_limit=laminar_cd_limit(RE),
                          data=SimpleNamespace(e_max=float(details["dataset_best"]["E"])),
                          model_names=[m["name"] for m in metadata["models"]])
    ctx.store = Simulation_Store(run_dir / "simulations.jsonl", ctx)
    lanes = [SimpleNamespace(name=d["name"], label=d["name"], keys=d["keys"], picks=d["picks"]) for d in details["lanes"]]
    return SimpleNamespace(dir=run_dir, ctx=ctx, lanes=lanes, target=float(details["target_E"]),
                           dataset_best=ctx.data.e_max, saved_reach=details.get("simulations_to_reach", {}))


def convergence_data(run) -> dict:
    """{lane: (shape counts, best E so far)} and {lane: reach dict}, as save_plots() draws them."""
    curves, reach = {}, {}
    for lane in run.lanes:
        reach[lane.name] = simulations_to_reach(lane, run.ctx)
        curve = [(n, e) for n, e in reach[lane.name]["curve"] if e is not None]
        if curve:
            curves[lane.name] = ([n for n, _ in curve], [e for _, e in curve])
    return curves, reach


def parity_data(run) -> dict:
    """{model: (predicted E, simulated E)} of every pick of every lane, as save_plots() draws them."""
    points = {}
    for name in run.ctx.model_names:
        xs, ys = [], []
        for lane in run.lanes:
            for p in lane.picks:
                values = p.get("preds", {}).get(name)
                if not values:
                    continue
                sim = run.ctx.store.run(p["key"])
                for ident, pred in zip(run.ctx.identifiers, values):
                    e = valid_E(sim, ident, run.ctx)
                    if e is not None and pred is not None and np.isfinite(pred):
                        xs.append(pred)
                        ys.append(e)
        if xs:
            points[name] = (xs, ys)
    return points


def check(value, expected, places=4) -> tuple:
    """(text, matches) of one sanity value; matches is None without an expected value."""
    if value is None:
        return "-", (None if expected is None else False)
    text = f"{value:.{places}f}" if isinstance(value, float) else str(value)
    if expected is None:
        return text, None
    ok = abs(value - expected) <= TOLERANCE if isinstance(value, float) else value == expected
    return f"{text} {'OK' if ok else f'MISMATCH (thesis {expected})'}", ok


def sanity_report(label: str, run, curves: dict, reach: dict) -> list:
    """Prints the run's sanity checks; returns the list of match results (True/False) of the checked values."""
    expected = EXPECTED.get(run.dir.name, {})
    exp_lanes = expected.get("lanes", {})
    results = []
    target_text, ok = check(run.target, expected.get("target"))
    results.append(ok)
    best_text, ok = check(run.dataset_best, expected.get("dataset_best"))
    results.append(ok)
    print(f"\nRun {label} ({short(run.dir)}): target {target_text} | dataset best {best_text}")
    print(f"    {'Lane':<14} {'Final best E':<30} Shapes to reach the dataset best (100%)")
    for lane in run.lanes:
        final = curves[lane.name][1][-1] if lane.name in curves else None
        shapes = reach[lane.name]["reached"]["100%"]
        exp_final, exp_shapes = exp_lanes.get(lane.name, (None, None))
        final_text, ok_final = check(final, exp_final)
        shapes_text, ok_shapes = check(shapes, exp_shapes)
        results += [ok_final, ok_shapes]
        saved = run.saved_reach.get(lane.name, {}).get("100%", shapes)
        note = "" if saved == shapes else f"  [note: detailed_results.json saved {saved}]"
        print(f"    {lane.name:<14} {final_text:<30} {shapes_text}{note}")
    return [r for r in results if r is not None]


def padded(low: float, high: float, share: float) -> tuple:
    pad = (high - low) * share
    return low - pad, high + pad


def parse_args():
    parser = argparse.ArgumentParser(description="Spanish-labelled convergence and parity figures of inference runs.")
    parser.add_argument('--runs', nargs='+', default=None, metavar='RUN_FOLDER',
                        help="Run folders (default: the thesis runs A, B and C of results/dataset_17.6).")
    parser.add_argument('--labels', nargs='+', default=None,
                        help="One label per run, used in the file names (default: A B C).")
    parser.add_argument('--out', default=str(DEFAULT_OUT), help=f"Output folder (default: {short(DEFAULT_OUT)}).")
    parser.add_argument('--show', action='store_true', help="Also open the figures.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runs_in = args.runs or [str(p) for p in THESIS_RUNS.values()]
    labels = args.labels or (list(THESIS_RUNS) if args.runs is None else [chr(ord("A") + i) for i in range(len(runs_in))])
    if len(labels) != len(runs_in):
        sys.exit(f"[ERROR] {len(runs_in)} run(s) but {len(labels)} label(s); give one label per run.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    family = use_thesis_font()
    print(f"Font: {family or 'matplotlib default (Helvetica and Arial not found)'}")

    # --- Load every run first, so the axis limits can be shared --- #
    runs = {}
    for label, run_arg in zip(labels, runs_in):
        run = load_run(resolve_run(run_arg))
        curves, reach = convergence_data(run)
        runs[label] = SimpleNamespace(run=run, curves=curves, reach=reach, points=parity_data(run))

    checks = []
    for label, r in runs.items():
        checks += sanity_report(label, r.run, r.curves, r.reach)

    cv_values = [e for r in runs.values() for _, best in r.curves.values() for e in best]
    cv_values += [v for r in runs.values() for v in (r.run.dataset_best, r.run.target)]
    pp_x = [v for r in runs.values() for xs, _ in r.points.values() for v in xs]
    pp_y = [v for r in runs.values() for _, ys in r.points.values() for v in ys]
    pp_y += [r.run.dataset_best for r in runs.values()]
    cv_ylim = padded(min(cv_values), max(cv_values), 0.04) if cv_values else None
    pp_xlim = padded(min(pp_x), max(pp_x), 0.03) if pp_x else None
    pp_ylim = padded(min(pp_y), max(pp_y), 0.04) if pp_x else None

    # --- Figures --- #
    written = []
    for label, r in runs.items():
        if r.curves:
            fig = plot_convergence(r.curves, r.run.dataset_best, r.run.target, labels=LANE_LABELS, styles=CV_STYLES,
                                   colors=CV_COLORS, ylim=cv_ylim, font_sizes=FONT_SIZES)
            path = out_dir / FILE_NAMES["convergence"].format(label=label)
            fig.savefig(path, dpi=DPI)
            written.append(path)
            if not args.show:
                plt.close(fig)
        if r.points:
            fig = plot_parity(r.points, r.run.dataset_best, colors=PP_COLORS, xlim=pp_xlim, ylim=pp_ylim,
                              font_sizes=FONT_SIZES)
            path = out_dir / FILE_NAMES["parity"].format(label=label)
            fig.savefig(path, dpi=DPI)
            written.append(path)
            if not args.show:
                plt.close(fig)

    # --- Summary --- #
    expected_figures = 2 * len(runs)
    mismatches = checks.count(False)
    print("\n=== Summary ===")
    print(f"Figures written: {len(written)} of {expected_figures} -> {short(out_dir)}")
    for path in written:
        print(f"    {path.name}")
    if checks:
        print(f"Sanity checks: {len(checks) - mismatches} of {len(checks)} values match the thesis")
    if not written:
        print("\n[ERROR] No figure was written: the runs have no simulated curve or pick to draw.")
        sys.exit(1)
    if mismatches:
        print(f"\n[ERROR] {mismatches} value(s) differ from the thesis; see the lines marked MISMATCH above.")
    if len(written) < expected_figures:
        print(f"\n[ERROR] {expected_figures - len(written)} figure(s) missing: a run had nothing to draw.")
    if args.show:
        plt.show()
    if mismatches or len(written) < expected_figures:
        sys.exit(1)
