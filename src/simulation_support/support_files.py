"""
Names of the support files every simulation uses (data/support), defined once, and the check that a dataset was built
with them.

NACA_KP_FILE holds the key points of the NACA 0012: the CDCL drag polar of the winglet sections of the flat and 60DEG
wings (see simulation_support.geo_file_creation). It was made with utilities/airfoil_kp_exporter.py (zero_out sweep,
key-point seed 42), the chain the datasets use.

LEGACY_NACA_KP_FILE is the file of datasets 17 and 17.5. Its polar was an old sequential sweep with CDmin = 0.00128,
below the laminar limit. Those two datasets were also built with the rec wing at twice its span (see
HossainWingRec_Creation), so their labels cannot be mixed with new simulations: check_dataset_kp_file refuses them.
"""
import sys
from pathlib import Path

NACA_KP_FILE = "naca_0012_M0-093_RE250000_zero_out_s42.txt"
LEGACY_NACA_KP_FILE = "naca_0012_M0-9281_RE250k_M0-093_RE250000.txt"
CASE_FILE = "Hossain_sqwing_case.run"
BASE_AIRFOIL = "naca0012_unit.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def naca_kp_path(project_root=PROJECT_ROOT) -> Path:
    """Path of the NACA 0012 key-point file of the winglet sections."""
    return Path(project_root) / "data" / "support" / "naca_kp" / NACA_KP_FILE


def dataset_kp_file(dataset: str, project_root=PROJECT_ROOT):
    """
    Name of the key-point file a dataset was built with, from its generation log (an expansion uses its parent's).
    A log without the line is older than it, so the dataset used the legacy file. None if there is no log.
    """
    from file_handling_support.dataset_names import parent_dataset
    number = parent_dataset(dataset) or dataset
    logs = sorted((Path(project_root) / "data" / "logs").glob(f"Dataset_{number}_*_data_points.log"))
    if not logs:
        return None
    for line in logs[-1].read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("NACA key point file = "):
            return line.split("=", 1)[1].strip()
    return LEGACY_NACA_KP_FILE


def check_dataset_kp_file(dataset: str, project_root=PROJECT_ROOT) -> None:
    """Stops the program if the dataset was built with another key-point file: its labels come from other wings than
    the ones simulated now (datasets 17 and 17.5, see the module docstring)."""
    used = dataset_kp_file(dataset, project_root)
    if used is None:
        print(f"[WARNING] No generation log of dataset {dataset} in data/logs: cannot check how its wings were simulated.")
    elif used != NACA_KP_FILE:
        sys.exit(f"[ERROR] Dataset {dataset} was built with {used},\n"
                 f"        the simulations now use {NACA_KP_FILE}.\n"
                 f"        Its labels cannot be mixed with new simulations (datasets 17 and 17.5 also have the rec wing at "
                 f"twice its span). Use dataset 17.6 or later.")
