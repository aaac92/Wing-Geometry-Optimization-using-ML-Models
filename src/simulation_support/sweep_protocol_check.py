"""
Parallel airfoil simulations with any alpha-sweep protocol, and the check of whether XFOIL labels depend on the
alpha-sweep protocol (convergence history).

Reusable functions (exported by simulation_support):
  protocol_settings(protocol)   Simulation_Settings of the dataset pipeline with that sweep protocol
  simulate_shapes(...)          many airfoils x protocols in a process pool (one XFOIL run per airfoil and protocol,
                                one AVL run per wing configuration), records reported as they finish
  polar_comparison(a, b)        per-angle CD of two runs of the same airfoil
  compare_runs(run, reference)  pre-stall branch agreement and E ratios of one run against a reference run

At Re = 2.5e5 a laminar separation bubble can make XFOIL's viscous solution bistable: for the same shape
and angle there may be a low-drag and a higher-drag solution, and a warm start from the previous angle
decides which one XFOIL returns. The legacy 'sequential' sweep (from -20 deg upwards) lands on a low-drag
branch for many airfoils (dataset 17 artifact). The protocols:

  cold        a fresh XFOIL process for every angle: no history at all (the reference)
  zero_out    the current pipeline: two warm-started passes from 0 deg outwards
  sequential  the legacy sweep from the lowest angle upwards

Unit test (the original check; this file's __main__ block)
---------
Simulates a sample of dataset airfoils with the protocols and compares polars and E. If zero_out matches cold,
the dataset labels do not depend on the sweep history. Sample:
  flagged   airfoils the dataset 17 rectification corrected (largest E drop first): known bistable shapes
  top       the highest-E airfoils of the dataset being checked
  control   random airfoils of that dataset
Outputs go to data/datasets/dataset_<N>/sweep_check/: a report (.txt), one row per airfoil and
configuration (_rows.csv), the per-angle CD comparison (_angles.csv) and every run with its polar (.json).

    python src/simulation_support/sweep_protocol_check.py --dataset 17.5
    python src/simulation_support/sweep_protocol_check.py --dataset 17.5 --flagged 40 --top 10 --controls 40 --workers 8
    python src/simulation_support/sweep_protocol_check.py --dataset 17.5 --protocols cold zero_out

Moved here from src/utilities (2026-11-04) so the iterative inference can reuse the simulation pool and the
cold-start comparison.
"""
import argparse
import hashlib
import json
import multiprocessing
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
if __package__ in (None, ""):  # run as a script (unit test): make the src packages importable
    sys.path.insert(0, str(SRC_DIR))

from simulation_support.airfoil_simulation import (  # noqa: E402
    GEO_CREATORS_BY_IDENTIFIER, KEY_POINT_NAMES, Simulation_Settings, WORKER_STATE, jsonable, laminar_cd_limit, simulate_airfoil_once,
    worker_init, worker_progress, worker_status,
)
from simulation_support.alpha_sweep import SWEEP_MODES  # noqa: E402
from simulation_support.geometry_checks import check_wings  # noqa: E402
from simulation_support.support_files import BASE_AIRFOIL, CASE_FILE, check_dataset_kp_file, naca_kp_path  # noqa: E402
from ux_support.progress import run_with_progress  # noqa: E402

PROTOCOLS = ("cold", "zero_out", "sequential")
REFERENCE = "cold"
# Two CD values of the same angle are on different branches when they differ by more than this
CD_TOLERANCE = 0.05
# Two E values agree when they differ by at most this fraction
E_TOLERANCE = 0.05
# Angles up to this |alpha| count as before stall (operating range); beyond it XFOIL solutions are
# history-dependent for every protocol, so only this range decides whether a protocol picks the right branch
PRE_STALL = 8.0
CONFIGS = ["rec", "flat", "60DEG"]


def default_temp_root() -> Path:
    # Short root: XFOIL reads at most 64 characters of a file name (see simulation_support.file_names)
    return Path(r"C:\sweep_check_temp") if sys.platform == "win32" else Path.home() / ".sweep_check_temp"


def protocol_settings(protocol: str, project_root: Path = PROJECT_ROOT) -> Simulation_Settings:
    """Simulation settings of the dataset pipeline (Re 2.5e5, Mach 0.09281, -20..20 deg every 0.5 deg,
    key-point seed 42) with the given alpha-sweep protocol."""
    exe = ".exe" if sys.platform == "win32" else ""
    support = Path(project_root) / "data" / "support"
    return Simulation_Settings(
        n_cst=5, aseq=[-20.0, 20.0, 0.5], reynolds=2.5e5, mach=0.09281,
        base_airfoil=str(support / BASE_AIRFOIL), naca_kp_file=str(naca_kp_path(project_root)),
        case_file=str(support / CASE_FILE), xfoil_exe=str(Path(project_root) / "bin" / f"xfoil{exe}"),
        avl_exe=str(Path(project_root) / "bin" / f"avl{exe}"), sweep_mode=protocol, kp_seed=42,
    )


# =============================================================================================== #
# === Simulation pool === #
# =============================================================================================== #

def run_task(task: dict) -> dict:
    """One airfoil, one protocol, in this worker's own temp folder (complete chain: polar -> key points -> AVL)."""
    settings = Simulation_Settings(**task["settings"])
    run_dir = Path(task["temp_root"]) / f"w{WORKER_STATE['slot']}"
    try:
        record = simulate_airfoil_once(settings, task["airfoil_id"], task["cst"], task["configs"], run_dir,
                                       store_polar=task["store_polar"], store_files=task["store_files"])
    except Exception as e:
        record = {"status": "error", "error": f"{type(e).__name__}: {e}", "E": {}}
    worker_status("idle")
    worker_progress()
    return {"key": task["key"], "protocol": task["protocol"], "run": record}


def simulate_shapes(shapes: dict, protocols, workers: int, configs=None, settings: dict = None, temp_root: Path = None,
                    store_polar: bool = True, store_files: bool = False, on_result=None, title: str = "Airfoil simulations",
                    header: str = None, use_gui=None) -> dict:
    """
    Simulates every airfoil with every protocol in a process pool and returns {(key, protocol): run record}.

    shapes      {key: {"cst": [...], "id": short airfoil id}} (ids name the files: keep them short, XFOIL reads 64
                characters of a path at most)
    settings    {protocol: Simulation_Settings} to override protocol_settings(protocol)
    on_result   called in the main process as on_result(key, protocol, record) the moment a run finishes (e.g. to
                write it to disk at once, so an interruption loses nothing that finished)
    use_gui     None: progress window when a display is available; False: console bar
    """
    configs = list(configs or CONFIGS)
    root = Path(temp_root) if temp_root else default_temp_root()
    tasks = []
    for key, entry in shapes.items():
        for protocol in protocols:
            chosen = (settings or {}).get(protocol) or protocol_settings(protocol)
            tasks.append({"key": key, "protocol": protocol, "cst": [float(v) for v in entry["cst"]], "airfoil_id": entry["id"],
                          "settings": asdict(chosen), "temp_root": str(root), "configs": configs,
                          "store_polar": store_polar, "store_files": store_files})
    results = {}
    if not tasks:
        return results
    for kp_file in {task["settings"]["naca_kp_file"] for task in tasks}:  # stops here if a wing is built wrong
        check_wings({c: GEO_CREATORS_BY_IDENTIFIER[c] for c in configs}, kp_file)
    # Cold polars take longest; starting them first keeps every worker busy until the end
    tasks.sort(key=lambda t: t["protocol"] != "cold")
    workers = max(1, min(workers, len(tasks)))
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        status_list = manager.list(["idle"] * workers)
        slot_queue = manager.Queue()
        for i in range(workers):
            slot_queue.put(i)
        pool = multiprocessing.Pool(processes=workers, initializer=worker_init,
                                    initargs=(slot_queue, progress_queue, status_list))

        def job():
            for result in pool.imap_unordered(run_task, tasks):
                results[(result["key"], result["protocol"])] = result["run"]
                if on_result is not None:
                    on_result(result["key"], result["protocol"], result["run"])

        try:
            run_with_progress(job, len(tasks), progress_queue, title=title,
                              header=header or f"{len(shapes)} airfoils x {len(protocols)} protocol(s) ({', '.join(protocols)})",
                              status_list=status_list, n_workers=workers, console_desc="Simulations", unit="run",
                              use_gui=use_gui)
        finally:
            pool.terminate()
            pool.join()
    return results


# =============================================================================================== #
# === Comparison === #
# =============================================================================================== #

def polar_comparison(run_a: dict, run_b: dict):
    """Per common angle: (alpha, CD_a, CD_b). Empty when either run has no polar."""
    pa, pb = (run_a or {}).get("polar"), (run_b or {}).get("polar")
    if not pa or not pb:
        return []
    cd_b = dict(zip(pb["alpha"], pb["CD"]))
    return [(a, cd, cd_b[a]) for a, cd in zip(pa["alpha"], pa["CD"]) if a in cd_b]


def compare_runs(run: dict, reference: dict, configs=None, cd_limit: float = None) -> dict:
    """
    One airfoil's run against a reference run (normally cold):
      pre_stall_angles / pre_stall_same_branch  angles with |alpha| <= PRE_STALL compared, and the share on the same CD
                                                branch (CD within CD_TOLERANCE)
      E_ratio {config: run E / reference E};  E_agree  every ratio within E_TOLERANCE (None: nothing to compare)
      reference_status  the reference run's status ("below_laminar_limit" when its polar CDmin is under cd_limit)
      verdict  "agrees"            same pre-stall branch at >= 99% of the angles AND E agrees in every configuration
                                   the run has; both must actually be measured
               "differs"           the reference is usable and one of the two conditions fails
               "reference failed"  the reference was simulated but gives no usable E (key points or AVL failed, or
                                   its polar is below the laminar CD limit): the design is NOT confirmed
               "no reference"      there is no reference run at all
    Only "agrees" counts as consistent. Until 2026-11-10 a reference without E counted as agreeing (its missing E
    was read as "no disagreement"), e.g. design s3d6ae272c1 of run 20261106_m_seede021ce33.
    """
    configs = list(configs or CONFIGS)
    pairs = [(a, cd, ref) for a, cd, ref in polar_comparison(run, reference) if abs(a) <= PRE_STALL]
    same = [abs(cd / ref - 1) <= CD_TOLERANCE for _, cd, ref in pairs if ref]
    share = float(np.mean(same)) if same else None
    reference_status = (reference or {}).get("status")
    below_limit = bool(reference and cd_limit is not None and reference.get("cd_min") is not None
                       and reference["cd_min"] < cd_limit)
    if below_limit:
        reference_status = "below_laminar_limit"
    run_E, reference_E = (run or {}).get("E", {}), ({} if below_limit else (reference or {}).get("E", {}))
    ratios = {}
    for config in configs:
        e, e_ref = run_E.get(config), reference_E.get(config)
        ratios[config] = (e / e_ref) if (e is not None and e_ref) else None
    expected = [config for config in configs if run_E.get(config) is not None]
    missing = [config for config in expected if ratios[config] is None]
    known = [r for r in ratios.values() if r is not None]
    e_agree = all(abs(r - 1) <= E_TOLERANCE for r in known) if known else None
    if not reference:
        verdict = "no reference"
    elif not expected:
        verdict = "differs"   # the run itself has no E to confirm
    elif missing or not known:
        verdict = "reference failed"
    else:
        verdict = "agrees" if (share is not None and share >= 0.99 and e_agree) else "differs"
    return {"pre_stall_angles": len(same), "pre_stall_same_branch": share, "E_ratio": ratios, "E_agree": e_agree,
            "reference_status": reference_status, "verdict": verdict}


# =============================================================================================== #
# === Unit test helpers (used only by the __main__ block) === #
# =============================================================================================== #
# CST generation grid of datasets 17 and 17.5 (Hossain_DB_Gen.py). The CSVs store CST values with 4 decimals
# (1/24 -> 0.0417), so every sampled airfoil is snapped back to the exact grid value it was generated with:
# on bistable shapes even that small difference can move XFOIL to the other branch.
_RANGES_LOWER = {'LE': (-0.125, 0, 0.0625), 'inter': (0, 0.125, 1/24), 'TE': (0, 0.125, 0.125)}
_RANGES_UPPER = {'LE': (0, 0.125, 0.0625), 'inter': (0, 0.125, 1/24), 'TE': (0, 0.125, 0.125)}


def _snap_to_grid(cst) -> list:
    """Exact generation-grid values of a CST vector read from a CSV; refuses values that are not on the grid."""
    from file_handling_support.database_correction import grid_levels_from_ranges
    grid = grid_levels_from_ranges(5, _RANGES_LOWER, _RANGES_UPPER)
    snapped = []
    for value, levels in zip(cst, grid):
        level = levels[np.abs(levels - float(value)).argmin()]
        if abs(level - float(value)) > 1e-3:
            raise ValueError(f"CST value {value} is not on the generation grid {levels.tolist()}; "
                             f"update _RANGES_LOWER/_RANGES_UPPER in this script.")
        snapped.append(float(level))
    return snapped


def _cst_columns(df: pd.DataFrame) -> list:
    lower = sorted((c for c in df.columns if c.startswith("cst_l_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    upper = sorted((c for c in df.columns if c.startswith("cst_u_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    return lower + upper


def _airfoil_key(cst) -> str:
    return ",".join(f"{v:.6f}" for v in cst)


def _latest_rectification_rows() -> Path:
    rows = sorted((PROJECT_ROOT / "data" / "datasets" / "dataset_17" / "rectification").glob("rectification_*_rows.csv"))
    return rows[-1] if rows else None


def _choose_airfoils(args) -> tuple:
    """({key: {"cst", "groups", "dataset_E": {config: E}, "id"}}, dataset CSV, rectification rows CSV)."""
    from file_handling_support.dataset_names import dataset_csv
    chosen = {}

    def add(cst, group, dataset_E=None):
        cst = _snap_to_grid(cst)
        key = _airfoil_key(cst)
        entry = chosen.setdefault(key, {"cst": [float(v) for v in cst], "groups": set(), "dataset_E": {},
                                        "id": "a" + hashlib.sha1(key.encode()).hexdigest()[:8]})
        entry["groups"].add(group)
        entry["dataset_E"].update(dataset_E or {})

    csv_path = Path(args.csv) if args.csv else dataset_csv(args.dataset, PROJECT_ROOT)
    df = pd.read_csv(csv_path)
    cols = _cst_columns(df)
    configs = [c for c in CONFIGS if c in df.columns]
    df["config"] = df[configs].idxmax(axis=1)
    df["key"] = [_airfoil_key(row) for row in df[cols].to_numpy()]
    by_airfoil = df.groupby("key")
    best = by_airfoil["E"].max().sort_values(ascending=False)
    lookup = {key: (group[cols].iloc[0].to_numpy(), dict(zip(group["config"], group["E"]))) for key, group in by_airfoil}

    rows_path = Path(args.flagged_from) if args.flagged_from else _latest_rectification_rows()
    if args.flagged and rows_path and rows_path.exists():
        rows = pd.read_csv(rows_path)
        rows = rows[rows["outcome"] == "corrected"].copy()
        rows["drop"] = rows["dataset_E"] - rows["rectified_E"]
        rcols = _cst_columns(rows)
        for _, group in sorted(rows.groupby("airfoil"), key=lambda g: -g[1]["drop"].max())[:args.flagged]:
            cst = group[rcols].iloc[0].to_numpy()
            add(cst, "flagged", lookup.get(_airfoil_key(cst), (None, {}))[1])
    elif args.flagged:
        print("[NOTE] No rectification rows file found; the 'flagged' group is empty.")
    for key in best.index[:args.top]:
        add(lookup[key][0], "top", lookup[key][1])
    rng = np.random.default_rng(args.seed)
    for key in rng.choice(best.index.to_numpy(), size=min(args.controls, len(best)), replace=False):
        add(lookup[key][0], "control", lookup[key][1])
    return chosen, csv_path, rows_path


def _compare(chosen: dict, results: dict, protocols) -> tuple:
    rows, angle_rows = [], []
    others = [p for p in protocols if p != REFERENCE]
    for key, entry in chosen.items():
        ref = results.get((key, REFERENCE), {})
        for config in CONFIGS:
            row = {"airfoil": entry["id"], "groups": "+".join(sorted(entry["groups"])), "config": config,
                   "dataset_E": entry["dataset_E"].get(config)}
            for p in protocols:
                row[f"E_{p}"] = results.get((key, p), {}).get("E", {}).get(config)
            for p in others:
                a, b = row[f"E_{p}"], row[f"E_{REFERENCE}"]
                row[f"E_{p}_vs_{REFERENCE}"] = (a / b) if a is not None and b else None
            rows.append(row)
        for p in others:
            for alpha, cd, cd_ref in polar_comparison(results.get((key, p), {}), ref):
                angle_rows.append({"airfoil": entry["id"], "protocol": p, "alpha": alpha, "CD": cd, f"CD_{REFERENCE}": cd_ref,
                                   "ratio": cd / cd_ref if cd_ref else None})
    return pd.DataFrame(rows), pd.DataFrame(angle_rows)


def _summary_lines(chosen, results, rows: pd.DataFrame, angles: pd.DataFrame, protocols, cd_limit: float) -> list:
    others = [p for p in protocols if p != REFERENCE]
    lines = [f"Airfoils: {len(chosen)} ("
             + ", ".join(f"{g} {sum(g in e['groups'] for e in chosen.values())}" for g in ("flagged", "top", "control")) + ")",
             f"Reference: {REFERENCE} (fresh XFOIL process per angle) | E agree within {E_TOLERANCE:.0%} | "
             f"CD on a different branch when off by more than {CD_TOLERANCE:.0%}", ""]
    for p in protocols:
        runs = [results.get((k, p), {}) for k in chosen]
        ok = sum(1 for r in runs if any(v is not None for v in r.get("E", {}).values()))
        below = sum(1 for r in runs if r.get("cd_min") is not None and r["cd_min"] < cd_limit)
        lines.append(f"{p:<11} simulated {ok:>4} of {len(runs)} | polar CDmin below the laminar limit ({cd_limit:.5f}): {below}")
    if "E_zero_out" in rows and rows["dataset_E"].notna().any():
        both = rows.dropna(subset=["dataset_E", "E_zero_out"])
        same = (abs(both["E_zero_out"] / both["dataset_E"] - 1) <= E_TOLERANCE)
        lines.append(f"Label check: zero_out reproduces the dataset E in {int(same.sum())} of {len(both)} rows "
                     f"({same.mean():.1%}); the dataset was built with zero_out, so this should be ~100%.")
    verdicts = {}
    for p in others:
        ratio = rows[f"E_{p}_vs_{REFERENCE}"].dropna()
        agree = (abs(ratio - 1) <= E_TOLERANCE)
        sub = angles[angles["protocol"] == p] if len(angles) else angles
        off = (abs(sub["ratio"] - 1) > CD_TOLERANCE) if len(sub) else pd.Series(dtype=bool)
        airfoils_off = sub.loc[off, "airfoil"].nunique() if len(sub) else 0
        lines += ["", f"--- {p} vs {REFERENCE} " + "-" * 60,
                  f"E rows agreeing: {int(agree.sum())} of {len(ratio)} ({agree.mean():.1%})" if len(ratio) else "E rows agreeing: -",
                  f"E ratio {p}/{REFERENCE}: median {ratio.median():.3f}, max {ratio.max():.3f}" if len(ratio) else "",
                  f"Angles on a different CD branch: {int(off.sum())} of {len(off)} ({off.mean():.1%}) in {airfoils_off} airfoil(s)"
                  if len(off) else "Angles compared: 0"]
        for group in ("flagged", "top", "control"):
            g = rows[rows["groups"].str.contains(group)][f"E_{p}_vs_{REFERENCE}"].dropna()
            if len(g):
                lines.append(f"    {group:<8} E agree {(abs(g - 1) <= E_TOLERANCE).mean():>6.1%} of {len(g)} rows, median ratio {g.median():.3f}")
        if len(off) and off.any():
            hit = sub.loc[off, "alpha"]
            lines.append(f"    different-branch angles: {int((hit.abs() <= 5).sum())} within +-5 deg, "
                         f"{int(((hit.abs() > 5) & (hit.abs() <= 10)).sum())} at 5-10 deg, {int((hit.abs() > 10).sum())} beyond 10 deg")
        # E reaches AVL only through the six key points: name the one that moved most where E disagrees
        moved = []
        for key, entry in chosen.items():
            kp_p, kp_ref = (results.get((key, q), {}).get("key_points") for q in (p, REFERENCE))
            rel = rows.loc[rows["airfoil"] == entry["id"], f"E_{p}_vs_{REFERENCE}"].dropna()
            if kp_p and kp_ref and len(rel) and (abs(rel - 1) > E_TOLERANCE).any():
                change = [abs(a - b) / max(abs(b), 1e-9) for a, b in zip(kp_p, kp_ref)]
                moved.append(KEY_POINT_NAMES[int(np.argmax(change))])
        if moved:
            counts = pd.Series(moved).value_counts()
            lines.append(f"    key point that moved most where E disagrees: "
                         + ", ".join(f"{name} ({n})" for name, n in counts.items())
                         + "  [CLneg/CDneg: negative-stall point, CLpos/CDpos: positive-stall point]")
        pre_stall = sub[sub["alpha"].abs() <= PRE_STALL] if len(sub) else sub
        same_branch = float((abs(pre_stall["ratio"] - 1) <= CD_TOLERANCE).mean()) if len(pre_stall) else None
        if same_branch is not None:
            lines.append(f"    pre-stall polar (|alpha| <= {PRE_STALL:g} deg): same CD branch at {same_branch:.1%} of "
                         f"{len(pre_stall)} angles")
        verdicts[p] = (same_branch, agree.mean() if len(ratio) else None)
    if "zero_out" in verdicts:
        polar_ok, e_ok = verdicts["zero_out"]
        lines.append("")
        if polar_ok is not None:
            lines.append(f"RESULT: before stall, zero_out polars match the history-free reference ({polar_ok:.1%} of angles)."
                         if polar_ok >= 0.99 else
                         f"[WARNING] Before stall, zero_out polars leave the history-free branch at {1 - polar_ok:.1%} of "
                         f"angles: the sweep history still decides the labels (see the angles CSV).")
        if e_ok is not None and e_ok < 0.95:
            lines.append(f"[NOTE] E differs by more than {E_TOLERANCE:.0%} in {1 - e_ok:.0%} of rows. When the pre-stall polars "
                         f"agree, the difference enters through the key points, which are fitted over the whole polar, "
                         f"including the post-stall angles where every protocol is history-dependent (key point line above).")
    return lines


# =============================================================================================== #
# === Unit test: the sweep protocol check === #
# =============================================================================================== #

if __name__ == "__main__":
    from file_handling_support.dataset_names import CURRENT_DATASET, dataset_folder, dataset_number_arg

    parser = argparse.ArgumentParser(description="Sweep protocol check: does an XFOIL label depend on the alpha-sweep history?",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="See the module docstring for the protocols and outputs.")
    parser.add_argument("--dataset", type=dataset_number_arg, default=CURRENT_DATASET, help="Dataset whose airfoils are sampled (top and control groups).")
    parser.add_argument("--csv", help="Explicit regressor CSV instead of the dataset's.")
    parser.add_argument("--flagged-from", help="Rectification rows CSV for the 'flagged' group (default: latest dataset 17 rectification).")
    parser.add_argument("--flagged", type=int, default=20, help="Airfoils from the rectification with the largest E drop.")
    parser.add_argument("--top", type=int, default=10, help="Highest-E airfoils of the dataset.")
    parser.add_argument("--controls", type=int, default=20, help="Random airfoils of the dataset.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the control sample.")
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS),
                        help=f"Protocols to run; '{REFERENCE}' is always included as the reference.")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Parallel processes.")
    args = parser.parse_args()
    check_dataset_kp_file(args.dataset)
    protocols = [REFERENCE] + [p for p in args.protocols if p != REFERENCE]
    missing = [p for p in protocols if p not in SWEEP_MODES]
    if missing:
        sys.exit(f"[ERROR] This copy of src/simulation_support/alpha_sweep.py has no {', '.join(missing)} sweep "
                 f"(it offers {', '.join(SWEEP_MODES)}).\n"
                 f"        Update src/simulation_support/alpha_sweep.py and src/simulation_support/polar_creation.py "
                 f"from the commit that added this tool, then run it again.")

    chosen, csv_path, rows_path = _choose_airfoils(args)
    print(f"Dataset CSV: {csv_path.relative_to(PROJECT_ROOT) if csv_path.is_relative_to(PROJECT_ROOT) else csv_path}")
    if rows_path:
        print(f"Flagged shapes from: {rows_path.relative_to(PROJECT_ROOT) if rows_path.is_relative_to(PROJECT_ROOT) else rows_path}")
    print(f"{len(chosen)} airfoils x {len(protocols)} protocols = {len(chosen) * len(protocols)} simulations "
          f"(a cold polar takes about one XFOIL start per angle, so it is the slow one).")

    results = simulate_shapes(chosen, protocols, args.workers, title="Sweep protocol check",
                              header=f"{len(chosen)} airfoils x {len(protocols)} protocols ({', '.join(protocols)})")
    reference_ok = sum(1 for k in chosen if any(v is not None for v in results.get((k, REFERENCE), {}).get("E", {}).values()))
    if reference_ok == 0:
        errors = sorted({str(results.get((k, REFERENCE), {}).get("error")) for k in chosen})
        sys.exit(f"[ERROR] Every {REFERENCE} (reference) simulation failed, so nothing can be compared. No files written.\n"
                 + "\n".join(f"        {e[:160]}" for e in errors[:3]))
    rows, angles = _compare(chosen, results, protocols)
    cd_limit = laminar_cd_limit(2.5e5)
    lines = _summary_lines(chosen, results, rows, angles, protocols, cd_limit)

    out_dir = dataset_folder(args.dataset, PROJECT_ROOT) / "sweep_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {"report": out_dir / f"sweep_check_{stamp}_report.txt", "rows": out_dir / f"sweep_check_{stamp}_rows.csv",
             "angles": out_dir / f"sweep_check_{stamp}_angles.csv", "json": out_dir / f"sweep_check_{stamp}.json"}
    rows.to_csv(paths["rows"], index=False)
    angles.to_csv(paths["angles"], index=False)
    with open(paths["json"], "w") as f:
        json.dump(jsonable({"protocols": protocols, "csv": str(csv_path), "flagged_from": str(rows_path) if rows_path else None,
                            "airfoils": {e["id"]: {"cst": e["cst"], "groups": sorted(e["groups"]), "dataset_E": e["dataset_E"],
                                                   "runs": {p: results.get((k, p)) for p in protocols}}
                                         for k, e in chosen.items()}}), f)
    bar = "=" * 80
    text = [bar, f"SWEEP PROTOCOL CHECK ({datetime.now().strftime('%Y-%m-%d %H:%M')})", bar, *lines, "", "Files:",
            *[f"  {p.relative_to(PROJECT_ROOT)}" for p in paths.values()], bar]
    paths["report"].write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n" + "\n".join(text))
