"""
Checks whether XFOIL labels depend on the alpha-sweep protocol (convergence history).

At Re = 2.5e5 a laminar separation bubble can make XFOIL's viscous solution bistable: for the same shape
and angle there may be a low-drag and a higher-drag solution, and a warm start from the previous angle
decides which one XFOIL returns. The legacy 'sequential' sweep (from -20 deg upwards) lands on a low-drag
branch for many airfoils (dataset 17 artifact). This tool simulates a sample of airfoils with three
protocols and compares polars and E:

  cold        a fresh XFOIL process for every angle: no history at all (the reference)
  zero_out    the current pipeline: two warm-started passes from 0 deg outwards
  sequential  the legacy sweep from the lowest angle upwards

If zero_out matches cold, the dataset labels do not depend on the sweep history.

Sample
------
  flagged   airfoils the dataset 17 rectification corrected (largest E drop first): known bistable shapes
  top       the highest-E airfoils of the dataset being checked
  control   random airfoils of that dataset

Outputs go to data/datasets/dataset_<N>/sweep_check/: a report (.txt), one row per airfoil and
configuration (_rows.csv), the per-angle CD comparison (_angles.csv) and every run with its polar (.json).

Examples
--------
    python src/utilities/sweep_protocol_check.py --dataset 17.5
    python src/utilities/sweep_protocol_check.py --dataset 17.5 --flagged 40 --top 10 --controls 40 --workers 8
    python src/utilities/sweep_protocol_check.py --dataset 17.5 --protocols cold zero_out
"""
import argparse
import hashlib
import json
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import file_handling_support.database_correction as dc  # noqa: E402
from file_handling_support.dataset_names import dataset_number_arg  # noqa: E402
from ux_support.progress import run_with_progress  # noqa: E402

PROTOCOLS = ("cold", "zero_out", "sequential")
REFERENCE = "cold"
# Two CD values of the same angle are on different branches when they differ by more than this
CD_TOLERANCE = 0.05
# Two E values agree when they differ by at most this fraction
E_TOLERANCE = 0.05
NACA_KP_FILE = "naca_0012_M0-9281_RE250k_M0-093_RE250000.txt"
CASE_FILE = "Hossain_sqwing_case.run"
CONFIGS = ["rec", "flat", "60DEG"]


def temp_root() -> Path:
    # Short root: XFOIL reads at most 64 characters of a file name (see simulation_support.file_names)
    return Path(r"C:\sweep_check_temp") if sys.platform == "win32" else Path.home() / ".sweep_check_temp"


def settings_for(protocol: str) -> dc.Simulation_Settings:
    exe = ".exe" if sys.platform == "win32" else ""
    support = PROJECT_ROOT / "data" / "support"
    return dc.Simulation_Settings(
        n_cst=5, aseq=[-20.0, 20.0, 0.5], reynolds=2.5e5, mach=0.09281,
        base_airfoil=str(support / "naca0012_unit.csv"), naca_kp_file=str(support / "naca_kp" / NACA_KP_FILE),
        case_file=str(support / CASE_FILE), xfoil_exe=str(PROJECT_ROOT / "bin" / f"xfoil{exe}"),
        avl_exe=str(PROJECT_ROOT / "bin" / f"avl{exe}"), sweep_mode=protocol, kp_seed=42,
    )


# =============================================================================================== #
# === Sample === #
# =============================================================================================== #

def cst_columns(df: pd.DataFrame) -> list:
    lower = sorted((c for c in df.columns if c.startswith("cst_l_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    upper = sorted((c for c in df.columns if c.startswith("cst_u_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    return lower + upper


def airfoil_key(cst) -> str:
    return ",".join(f"{v:.6f}" for v in cst)


def latest_rectification_rows() -> Path:
    rows = sorted((PROJECT_ROOT / "data" / "datasets" / "dataset_17" / "rectification").glob("rectification_*_rows.csv"))
    return rows[-1] if rows else None


def choose_airfoils(args) -> dict:
    """({key: {"cst", "groups", "dataset_E": {config: E}, "id"}}, dataset CSV, rectification rows CSV)."""
    chosen = {}

    def add(cst, group, dataset_E=None):
        key = airfoil_key(cst)
        entry = chosen.setdefault(key, {"cst": [float(v) for v in cst], "groups": set(), "dataset_E": {},
                                        "id": "a" + hashlib.sha1(key.encode()).hexdigest()[:8]})
        entry["groups"].add(group)
        entry["dataset_E"].update(dataset_E or {})

    csv_path = Path(args.csv) if args.csv else (PROJECT_ROOT / "data" / "datasets" / f"dataset_{args.dataset}"
                                                 / f"regressor_dataset_{args.dataset}.csv")
    df = pd.read_csv(csv_path)
    cols = cst_columns(df)
    configs = [c for c in CONFIGS if c in df.columns]
    df["config"] = df[configs].idxmax(axis=1)
    df["key"] = [airfoil_key(row) for row in df[cols].to_numpy()]
    by_airfoil = df.groupby("key")
    best = by_airfoil["E"].max().sort_values(ascending=False)
    lookup = {key: (group[cols].iloc[0].to_numpy(), dict(zip(group["config"], group["E"]))) for key, group in by_airfoil}

    rows_path = Path(args.flagged_from) if args.flagged_from else latest_rectification_rows()
    if args.flagged and rows_path and rows_path.exists():
        rows = pd.read_csv(rows_path)
        rows = rows[rows["outcome"] == "corrected"].copy()
        rows["drop"] = rows["dataset_E"] - rows["rectified_E"]
        rcols = cst_columns(rows)
        for _, group in sorted(rows.groupby("airfoil"), key=lambda g: -g[1]["drop"].max())[:args.flagged]:
            cst = group[rcols].iloc[0].to_numpy()
            add(cst, "flagged", lookup.get(airfoil_key(cst), (None, {}))[1])
    elif args.flagged:
        print("[NOTE] No rectification rows file found; the 'flagged' group is empty.")
    for key in best.index[:args.top]:
        add(lookup[key][0], "top", lookup[key][1])
    rng = np.random.default_rng(args.seed)
    for key in rng.choice(best.index.to_numpy(), size=min(args.controls, len(best)), replace=False):
        add(lookup[key][0], "control", lookup[key][1])
    return chosen, csv_path, rows_path


# =============================================================================================== #
# === Simulation === #
# =============================================================================================== #

def run_task(task: dict) -> dict:
    """One airfoil, one protocol, in this worker's own temp folder (complete chain: polar -> key points -> AVL)."""
    settings = dc.Simulation_Settings(**task["settings"])
    run_dir = Path(task["temp_root"]) / f"w{dc._WORKER['slot']}"
    try:
        record = dc.simulate_airfoil_once(settings, task["airfoil_id"], task["cst"], CONFIGS, run_dir, store_polar=True)
    except Exception as e:
        record = {"status": "error", "error": f"{type(e).__name__}: {e}", "E": {}}
    dc._worker_status("idle")
    dc._worker_progress()
    return {"key": task["key"], "protocol": task["protocol"], "run": record}


def simulate(chosen: dict, protocols, workers: int) -> dict:
    """{(key, protocol): run record}."""
    from dataclasses import asdict
    root = temp_root()
    tasks = []
    for key, entry in chosen.items():
        for protocol in protocols:
            tasks.append({"key": key, "protocol": protocol, "cst": entry["cst"], "airfoil_id": entry["id"],
                          "settings": asdict(settings_for(protocol)), "temp_root": str(root)})
    # Cold polars take longest; starting them first keeps every worker busy until the end
    tasks.sort(key=lambda t: t["protocol"] != "cold")
    workers = max(1, min(workers, len(tasks)))
    results = {}
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        status_list = manager.list(["idle"] * workers)
        slot_queue = manager.Queue()
        for i in range(workers):
            slot_queue.put(i)
        pool = multiprocessing.Pool(processes=workers, initializer=dc._rectifier_worker_init,
                                    initargs=(slot_queue, progress_queue, status_list))

        def job():
            for result in pool.imap_unordered(run_task, tasks):
                results[(result["key"], result["protocol"])] = result["run"]

        try:
            run_with_progress(job, len(tasks), progress_queue, title="Sweep protocol check",
                              header=f"{len(chosen)} airfoils x {len(protocols)} protocols ({', '.join(protocols)})",
                              status_list=status_list, n_workers=workers, console_desc="Simulations", unit="run")
        finally:
            pool.terminate()
            pool.join()
    return results


# =============================================================================================== #
# === Comparison === #
# =============================================================================================== #

def polar_comparison(run_a: dict, run_b: dict):
    """Per common angle: (alpha, CD_a, CD_b). Empty when either run has no polar."""
    pa, pb = run_a.get("polar"), run_b.get("polar")
    if not pa or not pb:
        return []
    cd_b = dict(zip(pb["alpha"], pb["CD"]))
    return [(a, cd, cd_b[a]) for a, cd in zip(pa["alpha"], pa["CD"]) if a in cd_b]


def compare(chosen: dict, results: dict, protocols) -> tuple:
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


def summary_lines(chosen, results, rows: pd.DataFrame, angles: pd.DataFrame, protocols, cd_limit: float) -> list:
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
            lines.append(f"    different-branch angles span {hit.min():g} to {hit.max():g} deg")
        verdicts[p] = agree.mean() if len(ratio) else None
    zo = verdicts.get("zero_out")
    if zo is not None:
        lines += ["", ("RESULT: zero_out labels match the history-free reference; the sweep protocol does not decide them."
                       if zo >= 0.95 else
                       f"[WARNING] zero_out disagrees with the history-free reference on {1 - zo:.0%} of rows: "
                       "the labels still depend on the sweep history (see the rows and angles CSVs).")]
    return lines


# =============================================================================================== #
# === Main === #
# =============================================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="See the module docstring for the protocols and outputs.")
    parser.add_argument("--dataset", type=dataset_number_arg, default="17.5", help="Dataset whose airfoils are sampled (top and control groups).")
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
    protocols = [REFERENCE] + [p for p in args.protocols if p != REFERENCE]

    chosen, csv_path, rows_path = choose_airfoils(args)
    print(f"Dataset CSV: {csv_path.relative_to(PROJECT_ROOT) if csv_path.is_relative_to(PROJECT_ROOT) else csv_path}")
    if rows_path:
        print(f"Flagged shapes from: {rows_path.relative_to(PROJECT_ROOT) if rows_path.is_relative_to(PROJECT_ROOT) else rows_path}")
    print(f"{len(chosen)} airfoils x {len(protocols)} protocols = {len(chosen) * len(protocols)} simulations "
          f"(a cold polar takes about one XFOIL start per angle, so it is the slow one).")

    results = simulate(chosen, protocols, args.workers)
    rows, angles = compare(chosen, results, protocols)
    cd_limit = dc.laminar_cd_limit(2.5e5)
    lines = summary_lines(chosen, results, rows, angles, protocols, cd_limit)

    out_dir = PROJECT_ROOT / "data" / "datasets" / f"dataset_{args.dataset}" / "sweep_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {"report": out_dir / f"sweep_check_{stamp}_report.txt", "rows": out_dir / f"sweep_check_{stamp}_rows.csv",
             "angles": out_dir / f"sweep_check_{stamp}_angles.csv", "json": out_dir / f"sweep_check_{stamp}.json"}
    rows.to_csv(paths["rows"], index=False)
    angles.to_csv(paths["angles"], index=False)
    with open(paths["json"], "w") as f:
        json.dump(dc._jsonable({"protocols": protocols, "csv": str(csv_path), "flagged_from": str(rows_path) if rows_path else None,
                                "airfoils": {e["id"]: {"cst": e["cst"], "groups": sorted(e["groups"]), "dataset_E": e["dataset_E"],
                                                       "runs": {p: results.get((k, p)) for p in protocols}}
                                             for k, e in chosen.items()}}), f)
    bar = "=" * 80
    text = [bar, f"SWEEP PROTOCOL CHECK ({datetime.now().strftime('%Y-%m-%d %H:%M')})", bar, *lines, "", "Files:",
            *[f"  {p.relative_to(PROJECT_ROOT)}" for p in paths.values()], bar]
    paths["report"].write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n" + "\n".join(text))


if __name__ == "__main__":
    main()
