"""
Neighbour-consistency diagnosis and rectification of a generated regressor dataset.

A dataset row is suspicious when its efficiency E differs strongly from the median E of its grid neighbours
(same wing configuration, one grid step away in a single CST coefficient). Suspicious airfoils are simulated
with fully independent runs and every row is decided by strict majority.

The simulation chain is deterministic (dataset 17 diagnosis, 2026-10-28: identical polars and E in every
run, pool and single process alike), so repeating a run cannot change a value; only a different pipeline
can. The default pipeline is therefore the 'zero_out' alpha sweep with a seeded key-point fit, and rectify
uses one run per airfoil. See simulation_support.alpha_sweep for why the legacy 'sequential' sweep is not
recommended.

Modes
-----
diagnose   Re-simulates a sample (strongest outliers, top-E designs and random unflagged controls) and reports
           how repeatable the simulation chain is and where run-to-run variation comes from (XFOIL polar,
           key-point fit or AVL). Does not modify the dataset.
rectify    Rounds of flag -> simulate -> decide until no new outliers appear, then writes a rectified copy of
           the dataset. Runs already stored by a previous diagnosis or rectification are reused.

Both modes keep every simulation in data/datasets/dataset_<N>/rectification/evaluations.json.
The underlying logic lives in file_handling_support.database_correction (Dataset_Rectifier).

Examples
--------
    python src/utilities/dataset_rectification.py diagnose --dataset 17
    python src/utilities/dataset_rectification.py diagnose --dataset 17 --flagged 0 --top 0 --controls 300 --runs 1
    python src/utilities/dataset_rectification.py diagnose --dataset 17 --sweep sequential --kp-seed none
    python src/utilities/dataset_rectification.py rectify --dataset 17 --yes
"""
import argparse
import ast
import multiprocessing
import platform
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from file_handling_support.database_correction import (  # noqa: E402
    Dataset_Rectifier, Simulation_Settings, display_available, grid_levels_from_ranges,
)
from simulation_support.alpha_sweep import DEFAULT_SWEEP_MODE, SWEEP_MODES  # noqa: E402

# =============================================================================================== #
# === Dataset properties === #
# =============================================================================================== #

# Flow regimes of datasets whose log predates the "Flow regime" block written by the generators.
# Dataset 17: values from Hossain_DB_Gen.py at the time it was generated.
KNOWN_FLOW_REGIMES = {
    17: {"reynolds": 2.5e5, "mach": 0.09281, "aseq": [-20.0, 20.0, 0.5], "base_airfoil": "naca0012_unit.csv",
         "naca_kp_file": "naca_0012_M0-9281_RE250k_M0-093_RE250000.txt"},
}
DEFAULT_CASE_FILE = "Hossain_sqwing_case.run"
DEFAULT_BASE_AIRFOIL = "naca0012_unit.csv"
CONFIG_CLASS_IDENTIFIERS = {"HossainWingRec_Creation": "rec", "HossainWingFlat_Creation": "flat",
                            "HossainWing60DEG_Creation": "60DEG"}


def find_dataset_log(dataset_num: int):
    logs = sorted((PROJECT_ROOT / "data" / "logs").glob(f"Dataset_{dataset_num}_*_data_points.log"))
    return logs[-1] if logs else None


def _literal(text: str):
    """Python literal from a log line; plain fractions such as 1/24 (older logs) are evaluated first."""
    text = re.sub(r'(?<![\w.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', lambda m: repr(float(m.group(1)) / float(m.group(2))), text)
    return ast.literal_eval(text.strip())


def _seed(text: str):
    """Key-point seed from the CLI or a log: an integer, or 'none' for the legacy unseeded fit."""
    text = str(text).strip()
    return None if text.lower() == "none" else int(text)


def parse_dataset_log(path: Path) -> dict:
    """
    Reads n_cst, CST ranges, wing configurations and (when present) the flow regime from a generation log.
    Logs with a single range dictionary apply it to both surfaces.
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    props = {}
    dicts, section = {}, None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("CST Order (n_cst):"):
            props["n_cst"] = int(stripped.split(":")[1])
        elif stripped == "Lower surface:":
            section = "lower"
        elif stripped == "Upper surface:":
            section = "upper"
        elif stripped == "CST Coefficient Ranges:":
            section = "both"
        elif stripped.startswith("{") and section:
            dicts[section] = _literal(stripped)
            section = None
        elif stripped.startswith("- ") and stripped[2:] in CONFIG_CLASS_IDENTIFIERS:
            props.setdefault("configs", []).append(CONFIG_CLASS_IDENTIFIERS[stripped[2:]])
        elif stripped.startswith("Re = "):
            props["reynolds"] = float(stripped.split("=", 1)[1])
        elif stripped.startswith("Mach = "):
            props["mach"] = float(stripped.split("=", 1)[1])
        elif stripped.startswith("ASEQ = "):
            props["aseq"] = [float(v) for v in _literal(stripped.split("=", 1)[1])]
        elif stripped.startswith("Base airfoil = "):
            props["base_airfoil"] = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("NACA key point file = "):
            props["naca_kp_file"] = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("Sweep mode = "):
            props["generation_sweep"] = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("Key point seed = "):
            props["generation_kp_seed"] = _seed(stripped.split("=", 1)[1])
    if "lower" in dicts or "upper" in dicts:
        props["ranges_lower"], props["ranges_upper"] = dicts.get("lower"), dicts.get("upper")
    elif "both" in dicts:
        props["ranges_lower"] = props["ranges_upper"] = dicts["both"]
    return props


def binary_paths():
    suffix = ".exe" if platform.system() == "Windows" else ""
    return PROJECT_ROOT / "bin" / f"xfoil{suffix}", PROJECT_ROOT / "bin" / f"avl{suffix}"


def temp_root() -> Path:
    """Short scratch path: XFOIL and AVL cannot handle long file names."""
    return Path("C:/rectify_temp") if platform.system() == "Windows" else Path.home() / ".rectify_temp"


# =============================================================================================== #
# === Property resolution and confirmation === #
# =============================================================================================== #

EDITABLE = [
    ("csv", "Dataset CSV", str),
    ("n_cst", "CST order (n_cst)", int),
    ("ranges_lower", "CST ranges, lower surface", _literal),
    ("ranges_upper", "CST ranges, upper surface", _literal),
    ("reynolds", "Reynolds number", float),
    ("mach", "Mach number", float),
    ("aseq", "Alpha sequence [start, end, step]", lambda t: [float(v) for v in _literal(t)]),
    ("base_airfoil", "Base airfoil (data/support)", str),
    ("naca_kp_file", "NACA key point file (data/support/naca_kp)", str),
    ("case_file", "AVL case file (data/support)", str),
]


def resolve_properties(args) -> tuple:
    """Values and their origin, by precedence: command line > dataset log > known regimes > defaults."""
    values, sources = {}, {}

    def put(name, value, source):
        if value is not None and name not in values:
            values[name], sources[name] = value, source

    cli = {"n_cst": args.n_cst, "reynolds": args.re, "mach": args.mach, "aseq": args.aseq,
           "base_airfoil": args.base_airfoil, "naca_kp_file": args.naca_kp, "case_file": args.case_file,
           "csv": args.csv}
    for name, value in cli.items():
        put(name, value, "command line")
    log_path = find_dataset_log(args.dataset)
    log_props = parse_dataset_log(log_path) if log_path else {}
    for name, value in log_props.items():
        put(name, value, f"log ({log_path.name})")
    for name, value in KNOWN_FLOW_REGIMES.get(args.dataset, {}).items():
        put(name, value, "known regime table")
    put("csv", str(PROJECT_ROOT / "data" / "datasets" / f"dataset_{args.dataset}" / f"regressor_dataset_{args.dataset}.csv"), "default")
    put("case_file", DEFAULT_CASE_FILE, "default")
    put("base_airfoil", DEFAULT_BASE_AIRFOIL, "default")
    return values, sources, log_path


def print_properties(values: dict, sources: dict, args) -> None:
    print("\n" + "=" * 100)
    print(f"Dataset {args.dataset} | mode: {args.mode}")
    print("=" * 100)
    for name, label, _ in EDITABLE:
        value = values.get(name, "<missing>")
        print(f"  {label:<44} {str(value):<40} [{sources.get(name, '-')}]")
    print("-" * 100)
    print(f"  {'Pipeline variant':<44} sweep = {args.sweep}, key-point seed = {args.kp_seed}")
    print(f"  {'Outlier rule':<44} |E - neighbour median| > {args.threshold}, >= {args.min_neighbours} neighbours")
    print(f"  {'Decision':<44} strict majority of {args.runs} runs, agreement tolerance {args.tolerance:.1%}")
    workers = "1 (single process)" if args.single_process else str(args.workers)
    print(f"  {'Workers':<44} {workers}")
    print("=" * 100)


def confirm_properties(values: dict, sources: dict, args) -> dict:
    """Shows the resolved values and asks for confirmation; a 'no' allows editing each value in turn."""
    while True:
        missing = [label for name, label, _ in EDITABLE if name not in values]
        print_properties(values, sources, args)
        if args.yes and not missing:
            return values
        if missing:
            print(f"Missing values: {', '.join(missing)}. Please provide them.")
            answer = "n"
        else:
            answer = input("Use these values? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return values
        print("Press Enter to keep the current value.")
        for name, label, parse in EDITABLE:
            while True:
                raw = input(f"  {label} [{values.get(name, '')}]: ").strip()
                if not raw:
                    if name in values:
                        break
                    continue
                try:
                    values[name], sources[name] = parse(raw), "entered"
                    break
                except (ValueError, SyntaxError) as e:
                    print(f"    Not a valid value ({e}). Try again.")
        args.yes = False


def support_file(name: str, *subdirs: str) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = PROJECT_ROOT / "data" / "support" / Path(*subdirs) / name
    if not path.exists():
        raise FileNotFoundError(f"Support file not found: {path}")
    return path


# =============================================================================================== #
# === Command line === #
# =============================================================================================== #

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    data = common.add_argument_group("dataset")
    data.add_argument("--dataset", type=int, required=True, help="Dataset number, e.g. 17.")
    data.add_argument("--csv", help="Regressor CSV to process (default: data/datasets/dataset_<N>/regressor_dataset_<N>.csv).")
    data.add_argument("--work-dir", help="Folder for the evaluation store and reports (default: data/datasets/dataset_<N>/rectification).")

    flow = common.add_argument_group("flow regime and support files (default: dataset log, then the known-regime table)")
    flow.add_argument("--n-cst", type=int, help="CST order.")
    flow.add_argument("--re", type=float, help="Reynolds number.")
    flow.add_argument("--mach", type=float, help="Mach number.")
    flow.add_argument("--aseq", type=float, nargs=3, metavar=("START", "END", "STEP"), help="Alpha sequence in degrees.")
    flow.add_argument("--base-airfoil", help="Base airfoil CSV in data/support (or an absolute path).")
    flow.add_argument("--naca-kp", help="NACA 0012 key point file in data/support/naca_kp (or an absolute path).")
    flow.add_argument("--case-file", help="AVL run case file in data/support (or an absolute path).")

    pipe = common.add_argument_group("simulation pipeline")
    pipe.add_argument("--sweep", choices=SWEEP_MODES, default=DEFAULT_SWEEP_MODE,
                      help="XFOIL alpha sweep. 'zero_out' runs two processes that both start at 0 deg; 'sequential' "
                           "walks the whole sequence from the lowest angle in one process (legacy generation behavior, "
                           "not recommended: it produces a spurious low-drag branch, see simulation_support.alpha_sweep).")
    pipe.add_argument("--kp-seed", type=_seed, default=42,
                      help="Seed for the Gaussian-process key-point fit; 'none' reproduces the legacy unseeded fit.")

    rule = common.add_argument_group("outlier rule and decision")
    rule.add_argument("--threshold", type=float, default=15.0, help="Flag rows with |E - neighbour median| above this value.")
    rule.add_argument("--min-neighbours", type=int, default=3, help="Rows with fewer grid neighbours are never flagged.")
    rule.add_argument("--runs", type=int, default=None,
                      help="Independent simulations per airfoil (default: 5 for diagnose, to keep measuring "
                           "repeatability; 1 for rectify, since the simulation chain is deterministic).")
    rule.add_argument("--tolerance", type=float, default=0.05,
                      help="Relative tolerance for two simulated E values to count as the same result.")

    run = common.add_argument_group("execution")
    run.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1),
                     help="Parallel worker processes (CPU count - 1 on this machine).")
    run.add_argument("--single-process", action="store_true",
                     help="Run every simulation in this process, one after another. Runs are labelled so the "
                          "diagnosis can compare repeatability between pool and single-process execution.")
    gui = run.add_mutually_exclusive_group()
    gui.add_argument("--gui", dest="gui", action="store_true", default=argparse.SUPPRESS,
                     help="Force the progress window (default: window when a display is detected).")
    gui.add_argument("--no-gui", dest="gui", action="store_false", default=argparse.SUPPRESS, help="Use the console progress bar.")
    run.add_argument("--yes", action="store_true", help="Accept the resolved values without the confirmation prompt.")

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] + "\n\n" + __doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run '<mode> --help' for the options of each mode.")
    modes = parser.add_subparsers(dest="mode", required=True)

    diag = modes.add_parser("diagnose", parents=[common], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            help="Measure repeatability on a sample; the dataset is not modified.",
                            description="Re-simulates the strongest outliers, the top-E designs and random unflagged "
                                        "controls, then writes diagnosis_<time>_report.txt, _rows.csv and .json to the work folder.")
    diag.add_argument("--flagged", type=int, default=20, help="Number of strongest flagged airfoils to simulate.")
    diag.add_argument("--all-flagged", action="store_true", help="Simulate every flagged airfoil (prepares a full rectification).")
    diag.add_argument("--top", type=int, default=5, help="Number of top-E dataset airfoils to simulate.")
    diag.add_argument("--controls", type=int, default=10, help="Number of random unflagged airfoils to simulate.")
    diag.add_argument("--refits", type=int, default=5, help="Extra key-point fits on the first polar of each airfoil.")
    diag.add_argument("--sample-seed", type=int, default=0, help="Seed for choosing the control airfoils.")

    rect = modes.add_parser("rectify", parents=[common], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            help="Correct the dataset, reusing runs from earlier diagnoses.",
                            description="Rounds of flag -> simulate -> majority decision until no new outliers appear. "
                                        "Writes the rectified CSV and rectification_<time>_report.txt, _rows.csv and .json.")
    rect.add_argument("--max-rounds", type=int, default=5, help="Maximum flag/simulate/decide rounds.")
    rect.add_argument("--output", help="Rectified CSV path (default: next to the dataset, *_rectified.csv).")
    return parser


# =============================================================================================== #
# === Main === #
# =============================================================================================== #

def main() -> None:
    args = build_parser().parse_args()
    if args.runs is None:
        args.runs = 5 if args.mode == "diagnose" else 1
    values, sources, log_path = resolve_properties(args)
    if log_path is None:
        print(f"[INFO] No generation log found for dataset {args.dataset} in data/logs.")
    values = confirm_properties(values, sources, args)

    xfoil, avl = binary_paths()
    for binary in (xfoil, avl):
        if not binary.exists():
            raise FileNotFoundError(f"Binary not found: {binary}")
    settings = Simulation_Settings(
        n_cst=values["n_cst"], aseq=list(values["aseq"]), reynolds=values["reynolds"], mach=values["mach"],
        base_airfoil=str(support_file(values["base_airfoil"])), naca_kp_file=str(support_file(values["naca_kp_file"], "naca_kp")),
        case_file=str(support_file(values["case_file"])), xfoil_exe=str(xfoil), avl_exe=str(avl),
        sweep_mode=args.sweep, kp_seed=args.kp_seed,
    )
    generation = (values.get("generation_sweep"), values.get("generation_kp_seed", "unknown"))
    if values.get("generation_sweep") and generation != (args.sweep, args.kp_seed):
        print(f"[NOTE] The dataset was generated with sweep = {generation[0]}, key-point seed = {generation[1]}; "
              f"this run uses sweep = {args.sweep}, key-point seed = {args.kp_seed}.")
    dataset_dir = PROJECT_ROOT / "data" / "datasets" / f"dataset_{args.dataset}"
    work_dir = Path(args.work_dir) if args.work_dir else dataset_dir / "rectification"
    grid = None
    if values.get("ranges_lower") and values.get("ranges_upper"):
        grid = grid_levels_from_ranges(values["n_cst"], values["ranges_lower"], values["ranges_upper"])

    print("\nLoading dataset and building the neighbour grid...")
    rectifier = Dataset_Rectifier(
        csv_path=values["csv"], settings=settings, work_dir=work_dir, temp_root=temp_root(), grid_levels=grid,
        threshold=args.threshold, min_neighbours=args.min_neighbours, n_runs=args.runs, agreement_tolerance=args.tolerance,
        workers=args.workers, single_process=args.single_process, use_gui=getattr(args, "gui", None),
    )
    stored = len(rectifier.store["evaluations"][settings.signature_id()])
    print(f"Rows: {len(rectifier.E)} | airfoils: {len(rectifier.airfoils)} | CST values snapped to the grid: "
          f"{rectifier.snapped_values} (max shift {rectifier.snap_distance:.2e})")
    print(f"Evaluation store: {rectifier.store_path}")
    print(f"  Pipeline {settings.signature_id()}: {stored} airfoil(s) already simulated.")
    for sid, (sig, n) in rectifier.other_signatures().items():
        print(f"  Other pipeline {sid}: {n} airfoil(s) (sweep = {sig.get('sweep_mode')}, key-point seed = {sig.get('kp_seed')}); "
              f"not reused by this run.")
    print(f"Progress display: {'window' if rectifier.use_gui else 'console'} "
          f"(display {'detected' if display_available() else 'not detected'})")

    if args.mode == "diagnose":
        paths = rectifier.diagnose(n_flagged=args.flagged, n_controls=args.controls, n_top=args.top,
                                   all_flagged=args.all_flagged, kp_refits=args.refits, seed=args.sample_seed)
    else:
        previous = sorted(work_dir.glob("diagnosis_*_report.txt"))
        print(f"Previous diagnoses in the work folder: {len(previous)}" + (f" (latest: {previous[-1].name})" if previous else ""))
        csv_path = Path(values["csv"])
        output = Path(args.output) if args.output else csv_path.with_name(f"{csv_path.stem}_rectified{csv_path.suffix}")
        paths = rectifier.rectify(output_csv=output, max_rounds=args.max_rounds)

    print("\nAll outputs:")
    for name, path in paths.items():
        print(f"  {name:<8} {Dataset_Rectifier._short_path(path)}")


if __name__ == "__main__":
    main()
