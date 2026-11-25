"""
Reference simulation of one airfoil with the dataset simulation chain, to compare designs against it.

By default it simulates the unmodified base airfoil (NACA 0012: every CST increment 0) in the three Hossain wing
configurations, with the dataset chain, and reports CL, CD and E per configuration and alpha-sweep protocol.
The base airfoil is missing from dataset 17.5: it is Test_98305 of the generation grid, one of the valid airfoils
that failed later in the pipeline (failed_airfoils.csv), so the script reports any failure loudly instead of hiding it.

Any airfoil and configuration:
    --cst          CST increments on the base airfoil (2 x n_cst values, lower surface first; default all 0)
    --base-airfoil the base airfoil CSV (default data/support/naca0012_unit.csv)
    --configs      any wing configuration registered in simulation_support.GEO_CREATORS_BY_IDENTIFIER; a configuration
                   added there later is available here without changes (default: the Hossain configurations)

Nothing of the chain is reimplemented: protocol_settings() gives the dataset settings (Re 2.5e5, Mach 0.09281,
-20 to 20 deg every 0.5 deg, key-point seed 42), simulate_shapes() runs XFOIL -> key points -> AVL in short temp
folders (XFOIL reads 64 characters of a path at most), compare_runs() compares each protocol with the history-free
cold run, and check_airfoil() applies check_validity and the dataset geometric filters.

Comparisons:
- E_obj: the target E of the dataset's inference runs (dataset best, or its re-simulation when higher).
- Best design (optional, --no-compare skips it): the best fully validated design ("validated" in a run's
  detailed_results.json: beats the target, the cold-start check agrees and every filter passes) over every inference
  run of the dataset, or of the run chosen with --run. Runs that recorded no validation (older run format) are
  skipped. The percentage uses the design's E in the same configuration.
- Dataset grid: if the airfoil is a point of the dataset's CST grid (read from data/logs/Dataset_<N>_*.log), its
  Test_<N> name, its rows in the dataset and its entries in the dataset's failure logs.

Outputs: a console table (configuration x protocol: CL, CD, E, percentages, status) and a JSON with every run record,
polar included (default results/dataset_<N>/<name>_reference.json, e.g. base_airfoil_reference.json).

XFOIL and AVL builds differ between Linux and Windows: values meant to sit next to a dataset must come from the
machine and binaries that built it (dataset 17.5: Linux). The JSON records the platform and binaries used.

Usage
-----
    python src/utilities/reference_calculator.py                                  # base airfoil, Hossain configurations
    python src/utilities/reference_calculator.py --run 20261111_m_seed69          # best design of one run
    python src/utilities/reference_calculator.py --no-compare --configs rec
    python src/utilities/reference_calculator.py --cst -0.0625 0.0417 0 0 0.125 0.0625 0.0833 0.0417 0 0.125 --name my_foil
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import multiprocessing
import platform
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

utilities_dir = Path(__file__).resolve().parent
src_dir = utilities_dir.parent
project_root = src_dir.parent
results_root = project_root / "results"
logs_root = project_root / "data" / "logs"
support_dir = project_root / "data" / "support"
sys.path.insert(0, str(src_dir))

from geometric_support import Airfoil_Generator, create_modification_combinations  # noqa: E402
from geometric_support.filter_settings import check_airfoil  # noqa: E402
from file_handling_support.dataset_names import CURRENT_DATASET, dataset_csv, dataset_number_arg  # noqa: E402
from simulation_support import (GEO_CREATORS_BY_IDENTIFIER, SWEEP_MODES, check_dataset_kp_file, compare_runs, jsonable,  # noqa: E402
                                laminar_cd_limit, protocol_settings, simulate_shapes)
from simulation_support.sweep_protocol_check import CONFIGS, REFERENCE, default_temp_root  # noqa: E402
from ux_support.recovery import atomic_write_json  # noqa: E402

DEFAULT_PROTOCOLS = ["zero_out", "cold"]
DEFAULT_NAME = "base_airfoil"     # name of the all-zero airfoil (JSON: <name>_reference.json)
GRID_TOLERANCE = 1e-6             # CST values equal to a grid point
DATASET_TOLERANCE = 6e-5          # the dataset CSVs store CST values with 4 decimals (1/24 -> 0.0417)
ZERO_LIFT = 1e-6                  # |CL| below this counts as no lift (AVL prints 5 decimals)


# =============================================================================================== #
# === Airfoil and settings === #
# =============================================================================================== #

def shape_key(cst) -> str:
    """Same key format as inference_and_testing.shape_key."""
    return ",".join(f"{float(v):.6f}" for v in cst)


def airfoil_id(cst) -> str:
    """Short file id (XFOIL reads at most 64 characters of a path): 'base' for the all-zero airfoil."""
    return "base" if not np.any(cst) else "s" + hashlib.sha1(shape_key(cst).encode()).hexdigest()[:10]


def resolve_file(arg: str, folder: Path) -> Path:
    """A file given as written, or inside `folder`."""
    path = Path(arg)
    if not path.exists() and (folder / path).exists():
        path = folder / path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {arg}")
    return path.resolve()


def build_settings(protocols, base_airfoil: Path, n_cst: int) -> dict:
    """{protocol: Simulation_Settings}: the dataset settings of protocol_settings() with this airfoil's base and order."""
    settings = {}
    for protocol in protocols:
        s = protocol_settings(protocol, project_root)
        settings[protocol] = replace(s, n_cst=n_cst, base_airfoil=str(base_airfoil))
    for label, path in (("XFOIL", settings[protocols[0]].xfoil_exe), ("AVL", settings[protocols[0]].avl_exe)):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found at {short(path)}; place the binary in bin/ (see README.md).")
    return settings


def short(path) -> str:
    """Path relative to the project root when possible."""
    path = Path(path).resolve()
    return str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path)


# =============================================================================================== #
# === Comparison targets === #
# =============================================================================================== #

def inference_runs(dataset: str) -> list:
    return sorted(p.parent for p in (results_root / f"dataset_{dataset}").glob("*/detailed_results.json"))


def resolve_run(arg: str, dataset: str) -> Path:
    path = Path(arg)
    if not (path / "detailed_results.json").exists():
        path = results_root / f"dataset_{dataset}" / arg
    if not (path / "detailed_results.json").exists():
        names = ", ".join(p.name for p in inference_runs(dataset)) or "none"
        raise FileNotFoundError(f"No inference run '{arg}' for dataset {dataset} (runs: {names}).")
    return path.resolve()


def run_designs(run_dir: Path) -> tuple:
    """(fully validated designs with their lane and run, target E or None, note) of one inference run."""
    details = json.loads((run_dir / "detailed_results.json").read_text(encoding="utf-8"))
    target = details.get("target_E")
    designs = [dict(d, lane=lane, run=run_dir.name) for lane, items in (details.get("designs") or {}).items() for d in items]
    if not designs:
        return [], target, "no designs"
    if not any("validated" in d for d in designs):
        return [], target, "skipped: no validation recorded (older run format)"
    validated = [d for d in designs if d.get("validated") is True]
    if not validated:
        return [], target, "no fully validated design"
    best = max(validated, key=lambda d: d["E"])
    return validated, target, f"{len(validated)} validated design(s), best {best['id']} E {best['E']:.4f}"


def find_best_design(dataset: str, run: str = None) -> tuple:
    """(best fully validated design or None, target E of its run, printed lines)."""
    runs = [resolve_run(run, dataset)] if run else inference_runs(dataset)
    lines = [f"Best design ({'run ' + runs[0].name if run else f'dataset {dataset}, {len(runs)} inference run(s)'}):"]
    best, best_target = None, None
    for run_dir in runs:
        designs, target, note = run_designs(run_dir)
        lines.append(f"    {run_dir.name:<26} {note}")
        for d in designs:
            if best is None or d["E"] > best["E"]:
                best, best_target = d, target
    if best is None:
        lines.append("    [ERROR] No fully validated design found; use --run with another run or --no-compare.")
    else:
        by_config = " | ".join(f"{c} {e:.4f}" for c, e in best.get("E_all_configs", {}).items() if e is not None)
        cold = f", cold E {best['cold_E']:.4f}" if best.get("cold_E") is not None else ""
        lines.append(f"  -> {best['id']} (run {best['run']}, lane {best['lane']}): {by_config}{cold}")
    return best, best_target, lines


def objective_E(dataset: str, run: str = None) -> tuple:
    """(E_obj, where it comes from): the target of the chosen run, else of the latest run, else the dataset best."""
    runs = [resolve_run(run, dataset)] if run else inference_runs(dataset)[::-1]
    for run_dir in runs:
        target = json.loads((run_dir / "detailed_results.json").read_text(encoding="utf-8")).get("target_E")
        if target:
            return float(target), f"target of run {run_dir.name}"
    csv_path = dataset_csv(dataset, project_root)
    if csv_path.exists():
        target_col = pd.read_csv(csv_path, nrows=0).columns[-1]
        return float(pd.read_csv(csv_path, usecols=[target_col])[target_col].max()), f"best E of {csv_path.name}"
    return None, "no inference run or dataset CSV found"


# =============================================================================================== #
# === Dataset grid and logs === #
# =============================================================================================== #

def dataset_grid(dataset: str) -> tuple:
    """(n_cst, lower ranges, upper ranges) from the dataset's generation log, or None."""
    for log in sorted(logs_root.glob(f"Dataset_{dataset}_*_data_points.log")):
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        n_cst, ranges = None, {}
        for i, line in enumerate(lines):
            if line.startswith("CST Order (n_cst):"):
                n_cst = int(line.split(":")[1])
            for surface in ("Lower", "Upper"):
                if line.strip() == f"{surface} surface:":
                    following = next((l for l in lines[i + 1:] if l.strip()), "")
                    try:
                        ranges[surface] = ast.literal_eval(following.strip())
                    except (ValueError, SyntaxError):
                        pass
        if n_cst and len(ranges) == 2:
            return n_cst, ranges["Lower"], ranges["Upper"]
    return None


def grid_lookup(dataset: str, cst: np.ndarray) -> dict:
    """The airfoil's Test_<N> name in the dataset generation grid and its rows in the dataset's failure logs."""
    grid = dataset_grid(dataset)
    if grid is None:
        return {"status": f"no generation log with the CST grid of dataset {dataset}"}
    n_cst, lower, upper = grid
    if len(cst) != 2 * n_cst:
        return {"status": f"dataset {dataset} uses n_cst = {n_cst}"}
    with contextlib.redirect_stdout(io.StringIO()):
        combos = np.asarray(create_modification_combinations(n_cst, upper, lower, debug=False), dtype=np.float64)
    match = np.where(np.all(np.abs(combos - cst) <= GRID_TOLERANCE, axis=1))[0]
    if not len(match):
        return {"status": "not a point of the dataset generation grid"}
    name = f"Test_{int(match[0]) + 1}"   # Hossain_DB_Gen.genesis_worker numbering
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?!\d)")
    logs = {}
    for csv in sorted((logs_root / f"dataset_{dataset}").glob("*.csv")):
        df = pd.read_csv(csv, dtype=str).fillna("")
        rows = df[df.apply(lambda col: col.map(lambda v: bool(pattern.search(v)))).any(axis=1)]
        if len(rows):
            logs[csv.name] = rows.to_dict("records")
    return {"status": "grid point", "name": name, "logs": logs}


def dataset_rows(dataset: str, cst: np.ndarray) -> dict:
    """{configuration column: E} of the airfoil's rows in the dataset's regressor CSV (empty when it is missing)."""
    csv_path = dataset_csv(dataset, project_root)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    cst_cols = ([c for c in df.columns if re.fullmatch(r"cst_l_\d+", c)] + [c for c in df.columns if re.fullmatch(r"cst_u_\d+", c)])
    if len(cst_cols) != len(cst):
        return None
    rows = df[np.all(np.abs(df[cst_cols].to_numpy(np.float64) - cst) <= DATASET_TOLERANCE, axis=1)]
    config_cols = [c for c in df.columns[:-1] if c not in cst_cols]
    return {(row[config_cols].astype(float).idxmax() if config_cols else "E"): float(row.iloc[-1]) for _, row in rows.iterrows()}


# =============================================================================================== #
# === Report === #
# =============================================================================================== #

def pct(value, reference) -> str:
    return f"{100 * value / reference + 0.0:.1f}%" if value is not None and reference else "-"


def num(value, places: int) -> str:
    return f"{value + 0.0:.{places}f}" if value is not None else "-"   # + 0.0 prints -0.0 as 0.0


def zero_lift(run: dict, configs) -> bool:
    """Every configuration simulated and CL = 0 (e.g. a symmetric section at zero incidence): E = 0 is a result."""
    cl = [run.get("CL", {}).get(c) for c in configs]
    return all(v is not None and abs(v) < ZERO_LIFT for v in cl)


def parse_args():
    parser = argparse.ArgumentParser(description="Reference simulation of one airfoil with the dataset simulation chain.")
    parser.add_argument('--dataset', type=dataset_number_arg, default=CURRENT_DATASET,
                        help=f"Dataset whose runs, grid and logs are compared (default: {CURRENT_DATASET}).")
    parser.add_argument('--cst', type=float, nargs='+', default=None,
                        help="CST increments on the base airfoil, lower surface first (default: all 0, the base airfoil).")
    parser.add_argument('--base-airfoil', default=None, help="Base airfoil CSV (default: data/support/naca0012_unit.csv).")
    parser.add_argument('--name', default=None, help="Name in the report and the JSON file name (default: base_airfoil, "
                                                       "or the airfoil's short id).")
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS), choices=list(GEO_CREATORS_BY_IDENTIFIER),
                        help=f"Wing configurations (default: {' '.join(CONFIGS)}).")
    parser.add_argument('--protocols', nargs='+', default=DEFAULT_PROTOCOLS, choices=list(SWEEP_MODES),
                        help=f"Alpha-sweep protocols (default: {' '.join(DEFAULT_PROTOCOLS)}; '{REFERENCE}' is the reference "
                             f"of the comparison).")
    parser.add_argument('--workers', type=int, default=None, help="Simulation processes (default: one per protocol).")
    parser.add_argument('--run', default=None, help="Inference run (folder name or path) for E_obj and the best design "
                                                    "(default: the best validated design of every run of the dataset).")
    parser.add_argument('--no-compare', action='store_true', help="Skip the comparison with the best design.")
    parser.add_argument('--out', default=None, help="JSON path (default: results/dataset_<N>/<name>_reference.json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_dataset_kp_file(args.dataset, project_root)
    protocols = list(dict.fromkeys(args.protocols))
    configs = list(dict.fromkeys(args.configs))
    default_settings = protocol_settings(protocols[0], project_root)
    n_cst = default_settings.n_cst
    if args.cst is not None:
        if len(args.cst) % 2:
            sys.exit(f"[ERROR] --cst needs an even number of values (lower then upper surface), got {len(args.cst)}.")
        n_cst = len(args.cst) // 2
    cst = np.asarray(args.cst if args.cst is not None else [0.0] * (2 * n_cst), dtype=np.float64)
    base_airfoil = resolve_file(args.base_airfoil, support_dir) if args.base_airfoil else Path(default_settings.base_airfoil)
    ident = airfoil_id(cst)
    name = args.name or (DEFAULT_NAME if not np.any(cst) else ident)
    settings = build_settings(protocols, base_airfoil, n_cst)
    cd_limit = laminar_cd_limit(default_settings.reynolds)
    out_path = Path(args.out) if args.out else results_root / f"dataset_{args.dataset}" / f"{name}_reference.json"

    print(f"Airfoil: {name} (id {ident}) | base {base_airfoil.name} | n_cst {n_cst} | CST {np.round(cst, 4).tolist()}")
    print(f"Chain: Re {default_settings.reynolds:g}, Mach {default_settings.mach}, alpha {default_settings.aseq}, key-point "
          f"seed {default_settings.kp_seed} | configurations {' '.join(configs)} | protocols {' '.join(protocols)}")
    print(f"Binaries: {short(default_settings.xfoil_exe)}, {short(default_settings.avl_exe)} ({platform.system()})")

    # --- Geometry --- #
    x, yu, yl = Airfoil_Generator(base_airfoil_csv=str(base_airfoil), n_cst=n_cst, verbose=False).generate(cst_coeffs=cst)
    geometry = check_airfoil(x, yu, yl, cst, n_cst, name=name)
    print(f"\nGeometry: check_validity {'passed' if geometry['valid_geometry'] else 'FAILED'} | dataset filters "
          f"{'passed' if geometry['passed'] else 'FAILED'}")
    for f in geometry["filters"]:
        print(f"    {f['filter']:<20} {'passed' if f['passed'] else 'failed: ' + str(f['reason'])}")
    if geometry["reason"] and not geometry["filters"]:
        print(f"    {geometry['reason']}")

    # --- Dataset grid, dataset rows, comparison targets --- #
    grid = grid_lookup(args.dataset, cst)
    in_dataset = dataset_rows(args.dataset, cst)
    print(f"\nDataset {args.dataset}: " + (f"{grid['name']} of the generation grid" if grid.get("name") else grid["status"]))
    if in_dataset is not None:
        print("    In the regressor dataset: " + (", ".join(f"{c} E {e:.4f}" for c, e in in_dataset.items()) if in_dataset else "no"))
    for log_name, rows in grid.get("logs", {}).items():
        for row in rows:
            print(f"    {log_name}: " + ", ".join(f"{k}={v}" for k, v in row.items() if v))

    best, best_target = None, None
    if not args.no_compare:
        best, best_target, best_lines = find_best_design(args.dataset, args.run)
        print("\n" + "\n".join(best_lines))
    if best is not None and best_target:
        e_obj, e_obj_source = float(best_target), f"target of run {best['run']}"
    else:
        e_obj, e_obj_source = objective_E(args.dataset, args.run)
    print(f"\nE_obj = {num(e_obj, 4)} ({e_obj_source})")

    # --- Simulations --- #
    workers = args.workers or min(len(protocols), max(1, multiprocessing.cpu_count() - 1))
    results = simulate_shapes({shape_key(cst): {"cst": cst.tolist(), "id": ident}}, protocols, workers, configs=configs,
                              settings=settings, temp_root=default_temp_root(), store_polar=True,
                              title="Reference simulation", header=f"{name}: {len(protocols)} protocol(s) x {len(configs)} configuration(s)")
    runs = {p: results.get((shape_key(cst), p)) or {"status": "error", "error": "no result returned", "E": {}} for p in protocols}

    print("\nRuns:")
    for p, run in runs.items():
        cd_min = run.get("cd_min")
        limit = "-" if cd_min is None else ("OK" if cd_min >= cd_limit else "BELOW THE LAMINAR LIMIT")
        print(f"    {p:<10} status {run.get('status')} | polar points {run.get('polar_points', 0)} | CDmin {num(cd_min, 5)} "
              f"(laminar limit {cd_limit:.5f}: {limit})" + (f" | error: {run['error']}" if run.get("error") else ""))

    comparisons = {}
    if REFERENCE in runs:
        for p in protocols:
            if p != REFERENCE:
                comparisons[p] = compare_runs(runs[p], runs[REFERENCE], configs=configs, cd_limit=cd_limit)
        for p, c in comparisons.items():
            share = "-" if c["pre_stall_same_branch"] is None else f"{100 * c['pre_stall_same_branch']:.1f}%"
            ratios = ", ".join(f"{k} {v:.4f}" for k, v in c["E_ratio"].items() if v is not None) or "-"
            print(f"\n{p} vs {REFERENCE}: pre-stall same branch {share} of {c['pre_stall_angles']} angles | E ratio {ratios} | "
                  f"verdict {c['verdict']}")
            if zero_lift(runs[p], configs) and zero_lift(runs[REFERENCE], configs):
                print("    (both runs give CL = 0, so E = 0 and the E ratio is undefined: compare_runs reports 'reference "
                      "failed'; the pre-stall branch share is the comparison that applies)")
    else:
        print(f"\n(no '{REFERENCE}' run: protocol comparison skipped)")

    # --- Table --- #
    best_by_config = (best or {}).get("E_all_configs", {})
    header = f"{'Config':<8} {'Protocol':<10} {'CL':>8} {'CD':>9} {'E':>9} {'% E_obj':>8}"
    header += f" {'% best':>8}" if best else ""
    print("\n" + header + "  Status")
    print("-" * (len(header) + 10))
    for config in configs:
        for p in protocols:
            run = runs[p]
            e = run.get("E", {}).get(config)
            line = (f"{config:<8} {p:<10} {num(run.get('CL', {}).get(config), 4):>8} {num(run.get('CD', {}).get(config), 5):>9} "
                    f"{num(e, 4):>9} {pct(e, e_obj):>8}")
            line += f" {pct(e, best_by_config.get(config)):>8}" if best else ""
            print(line + f"  {run.get('status')}")
    if best:
        print(f"% best: against design {best['id']} in the same configuration")

    # --- JSON --- #
    report = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                     "python": platform.python_version()},
        "binaries": {"xfoil": default_settings.xfoil_exe, "avl": default_settings.avl_exe},
        "dataset": args.dataset,
        "airfoil": {"name": name, "id": ident, "cst": cst.tolist(), "n_cst": n_cst, "base_airfoil": base_airfoil.name},
        "configs": configs, "protocols": protocols,
        "settings": {p: {"signature": s.signature(), "signature_id": s.signature_id()} for p, s in settings.items()},
        "cd_limit": cd_limit,
        "geometry": geometry,
        "dataset_grid": grid, "dataset_rows": in_dataset,
        "E_obj": {"value": e_obj, "source": e_obj_source},
        "best_design": None if best is None else {k: best.get(k) for k in ("id", "run", "lane", "E", "config", "E_all_configs",
                                                                           "cold_E", "cold_check", "cst")},
        "runs": runs,
        "comparison": comparisons,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, jsonable(report))

    # --- Summary --- #
    with_e = [p for p, run in runs.items() if any(v is not None for v in run.get("E", {}).values())]
    print("\n=== Summary ===")
    print(f"Runs with E: {len(with_e)} of {len(protocols)} | JSON: {short(out_path)}")
    if with_e and all(zero_lift(runs[p], configs) for p in with_e):
        print("[NOTE] CL = 0 in every configuration: the AVL run case flies at a fixed angle of attack, and at zero incidence "
              "a symmetric section gives no lift, so E = 0 is a result, not a failure. (Datasets built before 2026-11-13, "
              "17.5 included, dropped such airfoils as 'force file could not be read'; the builder now keeps them.)")
    for p, run in runs.items():
        missing = [c for c in configs if run.get("E", {}).get(c) is None]
        if missing and p in with_e:
            print(f"[WARNING] {p}: no E for {', '.join(missing)} (status {run.get('status')})")
    if not with_e:
        print("\n[ERROR] No protocol produced an E: " + "; ".join(f"{p} {run.get('status')}"
                                                          + (f" ({run['error']})" if run.get("error") else "")
                                                          for p, run in runs.items()))
        sys.exit(1)
