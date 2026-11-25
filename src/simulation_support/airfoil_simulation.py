"""
One complete, independent simulation of an airfoil, the chain every dataset label comes from:

    CST coefficients -> .dat airfoil -> XFOIL polar -> key points (GPR fit) -> one .avl wing geometry and one AVL run
    per wing configuration -> CL, CD, E = CL / CD

simulate_airfoil_once runs it in its own folder and returns a JSON-ready record. It is the building block of the
process pools in the Dataset_Rectifier (file_handling_support.database_correction), the sweep protocol check and
the iterative inference (simulation_support.sweep_protocol_check.simulate_shapes). A pool worker keeps a small
state (WORKER_STATE: its slot number, the progress queue and the status list of the progress window) set by
worker_init.

Moved here from file_handling_support/database_correction.py (2026-11-04) so every simulation driver shares it.
"""
import hashlib
import json
import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from cst_modeling.foil import check_validity

from geometric_support import Airfoil_Generator
from .alpha_sweep import DEFAULT_SWEEP_MODE, SWEEP_REVISIONS
from .avl_simulation import AVL_Run_Case
from .geo_file_creation import HossainWingRec_Creation, HossainWingFlat_Creation, HossainWing60DEG_Creation
from .geometry_checks import GeometryCheckError, reference_check
from .key_point_processing import Key_Point_Extraction, key_points_hash, key_points_result
from .polar_creation import Xfoil_Polar_Creator as XPR

GEO_CREATORS_BY_IDENTIFIER: Dict[str, Type] = {
    "rec": HossainWingRec_Creation,
    "flat": HossainWingFlat_Creation,
    "60DEG": HossainWing60DEG_Creation,
}
LEGACY_KP_HASH = "4a054da99275"  # key points of datasets 17 and 17.5 (LEGACY_NACA_KP_FILE)
KEY_POINT_NAMES: List[str] = ["CLneg", "CDneg", "CLcdmin", "CDmin", "CLpos", "CDpos"]


@dataclass
class Simulation_Settings:
    """
    Everything that defines one simulation of an airfoil: flow regime, support files, binaries and the
    optional pipeline variants (alpha sweep mode and key-point fit seed). The signature (see `signature`)
    excludes machine-specific paths, so stored runs stay reusable on another machine. The winglet key points are part
    of it by file name and by value, so a file rewritten under the same name does not reuse runs of the old one.
    """
    n_cst: int
    aseq: List[float]
    reynolds: float
    mach: float
    base_airfoil: str
    naca_kp_file: str
    case_file: str
    xfoil_exe: str
    avl_exe: str
    sweep_mode: str = DEFAULT_SWEEP_MODE
    kp_seed: Optional[int] = 42
    xfoil_iter: int = 200

    def angles(self) -> List[float]:
        start, stop, step = self.aseq
        return [start + j * step for j in range(int(round((stop - start) / step)) + 1)]

    def signature(self) -> dict:
        signature = {
            "n_cst": self.n_cst, "aseq": list(self.aseq), "reynolds": self.reynolds, "mach": self.mach,
            "base_airfoil": Path(self.base_airfoil).name, "naca_kp_file": Path(self.naca_kp_file).name,
            "case_file": Path(self.case_file).name, "sweep_mode": self.sweep_mode, "kp_seed": self.kp_seed,
            "xfoil_iter": self.xfoil_iter,
        }
        # Revision 1 is left out so signatures stored before revisions existed keep their id
        revision = SWEEP_REVISIONS.get(self.sweep_mode, 1)
        if revision > 1:
            signature["sweep_revision"] = revision
        # Left out for the legacy key points, so signatures stored with them keep their id
        if Path(self.naca_kp_file).exists():
            kp_hash = key_points_hash(self.naca_kp_file)
            if kp_hash != LEGACY_KP_HASH:
                signature["naca_kp_hash"] = kp_hash
        return signature

    def signature_id(self) -> str:
        return hashlib.sha1(json.dumps(self.signature(), sort_keys=True).encode()).hexdigest()[:12]


def laminar_cd_limit(reynolds: float) -> float:
    """
    Lowest physically plausible 2D section drag: the friction drag of a flat plate of zero thickness with
    fully laminar flow on both surfaces. Blasius' laminar boundary layer gives a mean skin-friction
    coefficient Cf = 1.328 / sqrt(Re) per surface (referred to the chord); two wetted surfaces double it:
    CD_limit = 2 x 1.328 / sqrt(Re), e.g. 0.00531 at Re = 2.5e5. A real airfoil adds pressure drag and, once
    the flow turns turbulent, more friction, so a polar CD below this value points to a non-physical XFOIL
    solution (the legacy sequential sweep produced CDmin ~ 0.002, see simulation_support.alpha_sweep).
    """
    return 2 * 1.328 / math.sqrt(reynolds)


def jsonable(value):
    """Recursively converts numpy values, sets and NaN into plain JSON values (run records, reports)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# --- Worker process state (one airfoil task at a time per process) --- #
WORKER_STATE = {"slot": 0, "progress": None, "status": None, "generators": {}, "checked_wings": set()}


def worker_init(slot_queue, progress_queue, status_list) -> None:
    """Pool initializer: takes a free slot number and the shared progress queue / status list."""
    WORKER_STATE.update(slot=slot_queue.get(), progress=progress_queue, status=status_list, generators={})


def worker_status(text: str) -> None:
    """Shows `text` as this worker's line in the progress window (no-op outside a pool)."""
    status = WORKER_STATE["status"]
    if status is not None:
        try:
            status[WORKER_STATE["slot"]] = text
        except Exception:
            pass


def worker_progress() -> None:
    """Reports one finished unit to the progress window (no-op outside a pool)."""
    if WORKER_STATE["progress"] is not None:
        WORKER_STATE["progress"].put(1)


def _worker_generator(settings: Simulation_Settings) -> Airfoil_Generator:
    key = (settings.base_airfoil, settings.n_cst)
    if key not in WORKER_STATE["generators"]:
        WORKER_STATE["generators"][key] = Airfoil_Generator(base_airfoil_csv=settings.base_airfoil, n_cst=settings.n_cst, verbose=False)
    return WORKER_STATE["generators"][key]


def read_polar_arrays(polar_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converged (alpha, CL, CD) rows of an XFOIL polar file, parsed exactly as Key_Point_Extraction does."""
    if not Path(polar_path).exists():
        return np.array([]), np.array([]), np.array([])
    reader = Key_Point_Extraction(polar_file_paths=[str(polar_path)])
    try:
        alpha, cl, cd = reader._polar_cleanup(*reader._polar_extract(str(polar_path)))
    except Exception:
        return np.array([]), np.array([]), np.array([])
    return alpha, cl, cd


def read_force_file(force_path: Path) -> Tuple[float, float]:
    """(CLtot, CDtot) of an AVL force file; NaN for a missing file or value."""
    cl_tot, cd_tot = math.nan, math.nan
    if not force_path.exists():
        return cl_tot, cd_tot
    with open(force_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'CLtot =' in line:
                cl_tot = float(line.split('=')[1].split()[0])
            elif 'CDtot =' in line:
                cd_tot = float(line.split('=')[1].split()[0])
    return cl_tot, cd_tot


def read_span_efficiency(force_path: Path) -> float:
    """Span efficiency e of an AVL force file (Trefftz plane; inviscid Oswald factor); NaN if it is missing."""
    if not Path(force_path).exists():
        return math.nan
    match = re.search(r"CYff\s*=\s*\S+\s+e\s*=\s*([-+0-9.Ee]+)", Path(force_path).read_text(encoding="utf-8", errors="ignore"))
    return float(match.group(1)) if match else math.nan


def polar_key_points(polar_path: Path, seed: Optional[int]) -> List[float]:
    """The six key points (KEY_POINT_NAMES order) of a polar, from the same GPR fit the dataset used."""
    low_cl, high_cl, min_cd = Key_Point_Extraction([str(polar_path)], random_state=seed)._interpolate_and_extract(str(polar_path))
    return [float(low_cl[0]), float(low_cl[1]), float(min_cd[0]), float(min_cd[1]), float(high_cl[0]), float(high_cl[1])]


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None
    except OSError:
        return None


def simulate_airfoil_once(settings: Simulation_Settings, airfoil_id: str, cst: List[float], configs: List[str],
                          run_dir: Path, kp_refits: int = 0, store_polar: bool = False, store_files: bool = False) -> dict:
    """
    One complete, independent simulation of an airfoil: .dat generation -> XFOIL polar -> key points ->
    one AVL geometry and AVL run per wing configuration. Returns a JSON-ready run record.
    kp_refits > 0 repeats the key-point fit on the same polar to measure the fit's own variance.
    store_polar keeps the converged polar (alpha, CL, CD) in the record; store_files keeps the text of every .avl
    geometry and AVL force file (record["files"][config] = {"avl", "force"}), so each stage's output can be read
    later without archives. The .dat is not stored: the CST coefficients regenerate it exactly.
    Statuses: ok, partial (some configurations failed), invalid_geometry (check_validity failed, not simulated),
    key_points_failed, avl_failed, error.
    """
    t0 = time.time()
    record = {"status": "ok", "error": None, "polar_points": 0, "polar_hash": None, "cd_min": None, "key_points": None,
              "kp_refits": [], "CL": {}, "CD": {}, "E": {}, "e": {}, "seconds": None}
    if store_files:
        record["files"] = {}
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    polar_dir, geo_dir, force_dir = run_dir / "polars", run_dir / "geo", run_dir / "forces"
    for d in (polar_dir, geo_dir, force_dir):
        d.mkdir(parents=True, exist_ok=True)
    try:
        worker_status(f"{airfoil_id}: airfoil")
        generator = _worker_generator(settings)
        x, yu, yl = generator.generate(cst_coeffs=np.asarray(cst, dtype=float))
        if not check_validity(x, yu, yl, verbose=False):
            record["status"] = "invalid_geometry"
            return record
        generator._create_dat_file(output_dir=run_dir, n_cst=settings.n_cst, x=x, yl=yl, yu=yu, file_name=airfoil_id)
        dat_path = run_dir / f"{airfoil_id}.dat"

        worker_status(f"{airfoil_id}: XFOIL ({settings.sweep_mode})")
        polar_path = polar_dir / f"polar_{airfoil_id}.txt"
        XPR(xfoil_exe_path=settings.xfoil_exe, main_dir=run_dir, airfoil_dir=run_dir, polar_dir=polar_dir,
            verbose=False).run_simulation(dat_path, str(polar_path), settings.angles(), visc=settings.reynolds,
                                          mach=settings.mach, iter=settings.xfoil_iter, sweep_mode=settings.sweep_mode)
        alpha, cl, cd = read_polar_arrays(polar_path)
        record["polar_points"] = int(len(alpha))
        record["cd_min"] = float(np.min(cd)) if len(cd) else None
        record["polar_hash"] = hashlib.sha1(np.round(np.stack([alpha, cl, cd]), 6).tobytes()).hexdigest()[:16] if len(alpha) else None
        if store_polar:
            record["polar"] = {"alpha": alpha.tolist(), "CL": cl.tolist(), "CD": cd.tolist()}

        worker_status(f"{airfoil_id}: key points")
        try:
            kp = polar_key_points(polar_path, settings.kp_seed)
        except Exception as e:
            record.update(status="key_points_failed", error=str(e))
            return record
        record["key_points"] = kp
        record["kp_refits"] = [polar_key_points(polar_path, settings.kp_seed) for _ in range(kp_refits)]

        placeholder = [{'file_path': f"{airfoil_id}.txt", **key_points_result(kp)}]
        case_copy = run_dir / Path(settings.case_file).name
        shutil.copy(settings.case_file, case_copy)
        for identifier in configs:
            worker_status(f"{airfoil_id}: AVL {identifier}")
            GEO_CREATORS_BY_IDENTIFIER[identifier](
                key_points_data=placeholder, airfoil_abs_paths=[str(dat_path)],
                naca_key_points_data=str(settings.naca_kp_file), output_dir=str(geo_dir), verbose=False, pb=False,
            ).create_geo_files()
            geo_path = geo_dir / f"{airfoil_id}_{identifier}_geo.avl"
            force_path = force_dir / f"force_{airfoil_id}_{identifier}.txt"
            cl_tot = cd_tot = math.nan
            if geo_path.exists():
                if identifier not in WORKER_STATE["checked_wings"]:  # once per process and configuration
                    reference_check(geo_path)
                    WORKER_STATE["checked_wings"].add(identifier)
                AVL_Run_Case(geo_files=[str(geo_path)], case_file=str(case_copy), avl_exe_path=settings.avl_exe,
                             output_dir=force_dir).run()
                cl_tot, cd_tot = read_force_file(force_path)
            e = cl_tot / cd_tot if math.isfinite(cl_tot) and math.isfinite(cd_tot) and cd_tot != 0 else math.nan
            record["CL"][identifier] = None if math.isnan(cl_tot) else cl_tot
            record["CD"][identifier] = None if math.isnan(cd_tot) else cd_tot
            record["E"][identifier] = None if math.isnan(e) else e
            span_efficiency = read_span_efficiency(force_path)
            record["e"][identifier] = None if math.isnan(span_efficiency) else span_efficiency
            if store_files:
                record["files"][identifier] = {"avl": _read_text(geo_path), "force": _read_text(force_path)}
        if any(v is None for v in record["E"].values()):
            record["status"] = "avl_failed" if all(v is None for v in record["E"].values()) else "partial"
        return record
    except GeometryCheckError:  # the code builds a wrong wing: never a result of one airfoil
        raise
    except Exception as e:
        record.update(status="error", error=f"{type(e).__name__}: {e}")
        return record
    finally:
        record["seconds"] = round(time.time() - t0, 2)
        shutil.rmtree(run_dir, ignore_errors=True)
