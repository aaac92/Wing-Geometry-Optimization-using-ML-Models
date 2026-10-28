import hashlib
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from tqdm import tqdm
from cst_modeling.foil import check_validity

from simulation_support import Xfoil_Polar_Creator as XPR
from simulation_support import Key_Point_Extraction, AVL_Run_Case
from simulation_support import HossainWingRec_Creation, HossainWingFlat_Creation, HossainWing60DEG_Creation
from geometric_support import Airfoil_Generator
try:
    from ux_support import Progress_GUI
except ImportError:  # tkinter missing (headless install); only the GUI progress view needs it
    Progress_GUI = None


@dataclass
class AccelerationConfig:
    """
    Tunable parameters for the per-pass acceleration mechanism.

    Acceleration is a per-pass trust ledger (`accel_value`, reset to
    `initial` at the start of every top/bottom/middle pass) that grows
    when a resolved point's original database value is confirmed correct
    ('validated'), and shrinks when a point needed correction or was
    refused/dropped -- by different amounts, since a refusal is stronger
    evidence of an unreliable point than one that just needed correcting.

    As the ledger moves away from `initial`, it shrinks how many of the
    pass's `n_target` points actually get tested -- dropped from the
    least-extreme, not-yet-tested end of the current ranking first, down
    to `floor_fraction * n_target` -- and grows the tested count back
    toward `n_target` if trust erodes again. Never expands past `n_target`.

    See Database_Corrector._accel_n_effective / _accel_update.
    """
    enabled: bool = True
    initial: float = 1.0
    reward_validated: float = 1.0
    penalty_corrected: float = 1.0
    penalty_dropped: float = 2.0
    streak_bonus: float = 0.15
    streak_cap: int = 10
    step_size: float = 3.0
    hysteresis: float = 0.0
    floor_fraction: float = 0.10
    cap: float = 20.0


class Database_Corrector:
    """
    Validates and corrects outlier aerodynamic efficiency (E = CL/CD) values
    in a regressor training CSV by re-simulating the full aerodynamic pipeline
    from CST coefficients, without multiprocessing.

    Correction logic:

    1. Back up the input CSV as <stem>_original<ext>.
    2. Correct the top `top_bottom_fraction` of rows by E (highest to lowest).
    3. Correct the bottom `top_bottom_fraction` of rows by E (lowest to highest).
    4. Optionally correct `middle_fraction` of rows selected by deviation from
       the median E (disabled by default).
    5. Overwrite the original CSV with corrected data.

    Validation criterion
    --------------------
    Two consecutive simulations of the same point must agree within
    `error_threshold` (fractional absolute error) for the new E value and
    refreshed key point values to be written. Points that do not converge
    within `max_attempts` are refused (dropped from the database).

    Recovery
    --------
    Progress is checkpointed under the temp directory after every point
    resolves (and after every simulation attempt within a point), so a
    crashed or interrupted `run_correction()` call resumes automatically
    on the next invocation rather than restarting. Checkpoint state lives
    entirely under the temp root and is cleared on a clean completion, so
    callers need no recovery-specific logic of their own.

    Supported dataset formats
    -------------------------
    - Efficiency mode only: last CSV column must be 'E'.
    - Single-configuration: provide `default_geo_creator_class`.
    - Multi-configuration (one-hot encoded): provide `configs` as a list of
      (class, identifier_string) tuples, matching Hossain_DB_Gen's format.

    Notes
    -----
    Key point columns (CLneg, CDneg, CLcdmin, CDmin, CLpos, CDpos) are 
    recomputed and overwritten for any validated point if `append_kp` is True, 
    since GPR-based key point extraction carries run-to-run variance.
    """

    # Known config identifiers, must match Training_Data_Generator.config_headers
    _KNOWN_CONFIGS: List[str] = ["rec", "flat", "60DEG"]

    # Decimal places used to build the airfoil-lookup index key. Trades a
    # small chance of missing a true match sitting right at a rounding
    # boundary (safely falls back to CST regeneration) for O(1) lookups
    # instead of an exact per-point tolerance scan.
    _CST_MATCH_DECIMALS: int = 4

    def __init__(
        self,
        csv_file_path,
        base_airfoil_csv,
        xfoil_exe_path,
        avl_exe_path,
        case_file_path,
        n_cst: int,
        aseq: List[float],
        Re: float,
        Mach: float,
        naca_key_points_data,
        configs: Optional[List[Tuple[Type, str]]] = None,
        default_geo_creator_class: Optional[Type] = None,
        error_threshold: float = 0.05,
        max_attempts: int = 6,
        top_bottom_fraction: float = 0.01,
        middle_fraction: float = 0.0,
        source_airfoils_dir=None,
        temp_dir=None,
        verbose: bool = False,
        append_kp: bool = False,
        accel_config: Optional[AccelerationConfig] = None,
    ):
        """
        Parameters
        ----------
        csv_file_path : str or Path
            Path to the regressor training CSV to correct.
        base_airfoil_csv : str or Path
            Base airfoil CSV for Airfoil_Generator (e.g. naca0012_unit.csv).
        xfoil_exe_path : str or Path
            Path to the XFOIL executable.
        avl_exe_path : str or Path
            Path to the AVL executable.
        case_file_path : str or Path
            Path to the AVL .run case file.
        n_cst : int
            CST polynomial order. Must match the CSV column count.
        aseq : list of float
            [start_alpha, end_alpha, step] for the XFOIL polar sweep.
        Re : float
            Reynolds number for the XFOIL polar sweep. Must match the flow
            condition the target database was generated under.
        Mach : float
            Mach number for the XFOIL polar sweep. Must match the flow
            condition the target database was generated under.
        naca_key_points_data : str or Path
            Path to the NACA 0012 CDCL reference file required by the geo
            creator classes (per-section CDCL, since this AVL build lacks
            the global CDCL option).
        configs : list of (class, str) tuples, optional
            Wing configuration classes paired with their identifier strings.
            Required if the CSV contains one-hot configuration columns.
            Raises immediately if config columns are present but this is None.
        default_geo_creator_class : type, optional
            Geo creator class for single-configuration CSVs (no config columns).
            Required if `configs` is None. Raises if neither is provided.
        error_threshold : float
            Maximum fractional absolute error for validation (default 0.02 = 2%).
        max_attempts : int
            Maximum simulation attempts per point before deletion (default 10).
        top_bottom_fraction : float
            Fraction of total rows to correct from each E distribution extreme
            (default 0.05 = 5%).
        middle_fraction : float
            Fraction of rows to sample from the middle of the E distribution
            after top/bottom passes. Rows are selected by highest absolute
            deviation from the median E. Set to 0.0 to disable (default).
        source_airfoils_dir : str or Path, optional
            Directory of already-generated airfoil .dat files from the
            original database run (e.g. the driver's
            ``training_temp_root / "airfoils"``), each carrying a
            CST-coefficient header. When provided, points are sourced by
            looking up a matching file here first, falling back to CST
            regeneration only when no unambiguous match exists (zero or
            multiple files share the same rounded CST key). When omitted,
            every point is always regenerated from CST coefficients.
        temp_dir : Path, optional
            Root directory for simulation intermediates. Kept independent
            of the driver's own temp root (not nested under it) to avoid
            excessive path length under the nested per-point subdirectories,
            which XFOIL and AVL do not tolerate well. Defaults to
            ``~/.correction_temp`` on Linux/macOS (dot-prefixed to stay
            hidden) or ``C:\\correction_temp`` on Windows (hidden via
            ``attrib +H`` in `_setup_temp_dirs`).
        verbose : bool
            If True, mirrors log output to stdout.
        append_kp : bool
            If True, requires and writes key point tracking columns.
        accel_config : AccelerationConfig, optional
            Tunable parameters for the per-pass acceleration mechanism
            (see AccelerationConfig). Defaults to AccelerationConfig()
            when omitted, i.e. acceleration is on with its own defaults.
            Pass AccelerationConfig(enabled=False) to disable it and test
            the full n_target/n_middle every pass, matching pre-acceleration
            behavior.
        """
        self.csv_file_path = Path(csv_file_path)
        self.base_airfoil_csv = Path(base_airfoil_csv)
        self.xfoil_exe_path = Path(xfoil_exe_path)
        self.avl_exe_path = Path(avl_exe_path)
        self.case_file_path = Path(case_file_path)
        self.n_cst = n_cst
        self.aseq = aseq
        self.Re = Re
        self.Mach = Mach
        self.naca_key_points_data = Path(naca_key_points_data)
        self.error_threshold = error_threshold
        self.max_attempts = max_attempts
        self.top_bottom_fraction = top_bottom_fraction
        self.middle_fraction = middle_fraction
        self.source_airfoils_dir = Path(source_airfoils_dir) if source_airfoils_dir else None
        self.verbose = verbose
        self.append_kp = append_kp
        self.accel_config = accel_config if accel_config is not None else AccelerationConfig()

        # Populated once per run_correction() call by _build_airfoil_index();
        # empty means every point falls back to CST regeneration.
        self._airfoil_index: Dict[Tuple[float, ...], List[Path]] = {}

        # Build internal config map: identifier string -> geo creator class
        self._config_map: Dict[str, Type] = {}
        if configs is not None:
            self._config_map = {identifier: cls for cls, identifier in configs}

        # Temp directory tree
        if temp_dir:
            self._temp_root = Path(temp_dir)
        elif os.name == 'nt':
            self._temp_root = Path(r"C:\correction_temp")
        else:
            self._temp_root = Path.home() / '.correction_temp'
        self._airfoils_temp = self._temp_root / "airfoils"
        self._polars_temp = self._temp_root / "polars"
        self._geo_temp = self._temp_root / "geometry files"
        self._force_temp = self._temp_root / "force files"
        self._support_temp = self._temp_root / "support_files"
        self._recovery_temp = self._temp_root / "recovery"
        self._checkpoint_path = self._recovery_temp / "checkpoint.json"
        self._checkpoint_df_path = self._recovery_temp / "checkpoint_df.csv"
        self._point_progress_path = self._recovery_temp / "current_point.json"

        # Logger — named per file to avoid conflicts with other module loggers
        log_path = self.csv_file_path.parent / f"{self.csv_file_path.stem}_correction.log"
        self._logger = logging.getLogger(f"Database_Corrector.{self.csv_file_path.stem}")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh = logging.FileHandler(log_path)
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)
            if self.verbose:
                ch = logging.StreamHandler()
                ch.setFormatter(fmt)
                self._logger.addHandler(ch)

        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

        header_df = pd.read_csv(self.csv_file_path, nrows=0)
        columns = list(header_df.columns)

        # Output mode validation: last column must be 'E'
        if columns[-1] != 'E':
            raise ValueError(
                f"Last column is '{columns[-1]}', expected 'E'. "
                "Database_Corrector supports Efficiency mode only "
                "(single 'E' output column). Verify the database was "
                "generated with output_mode='Efficiency'."
            )

        # CST column identification
        cst_l_pat = re.compile(r'^cst_l_(\d+)$')
        cst_u_pat = re.compile(r'^cst_u_(\d+)$')
        self._cst_l_cols: List[str] = sorted(
            [c for c in columns if cst_l_pat.match(c)],
            key=lambda x: int(cst_l_pat.match(x).group(1))
        )
        self._cst_u_cols: List[str] = sorted(
            [c for c in columns if cst_u_pat.match(c)],
            key=lambda x: int(cst_u_pat.match(x).group(1))
        )
        found_cst = len(self._cst_l_cols) + len(self._cst_u_cols)
        expected_cst = 2 * n_cst
        if found_cst != expected_cst:
            raise ValueError(
                f"n_cst={n_cst} implies {expected_cst} CST columns "
                f"({n_cst} lower + {n_cst} upper), but CSV contains "
                f"{len(self._cst_l_cols)} lower and {len(self._cst_u_cols)} upper."
            )

        # Key point column presence check conditionally executed
        self._kp_cols: List[str] = [
            "CLneg", "CDneg", "CLcdmin", "CDmin", "CLpos", "CDpos"
        ]
        if self.append_kp:
            missing_kp = [k for k in self._kp_cols if k not in columns]
            if missing_kp:
                raise ValueError(
                    f"Expected key point columns not found in CSV: {missing_kp}."
                    "Make sure database was created with 'append_kp = True'."
                )

        # Configuration column detection and validation
        self._config_cols: List[str] = [c for c in columns if c in self._KNOWN_CONFIGS]
        if self._config_cols:
            if not self._config_map:
                raise ValueError(
                    f"CSV contains configuration columns {self._config_cols} but "
                    "no 'configs' argument was provided. This indicates either a "
                    "bug in training_csv_creator (config columns should not exist "
                    "for a single-configuration dataset) or a missing constructor "
                    "argument."
                )
            missing_in_map = [c for c in self._config_cols if c not in self._config_map]
            if missing_in_map:
                raise ValueError(
                    f"Configuration columns {missing_in_map} are present in the CSV "
                    "but have no corresponding entry in the provided 'configs'. "
                    "Every config column must map to a geo creator class."
                )
            self._default_geo_class: Optional[Type] = None
        else:
            if default_geo_creator_class is None:
                raise ValueError(
                    "No configuration columns found in CSV and "
                    "'default_geo_creator_class' was not provided. "
                    "A geo creator class is required for single-configuration databases."
                )
            self._default_geo_class = default_geo_creator_class

    # ── Internal utilities ───────────────────────────────────────────────── #

    def _log(self, message: str, level: str = 'info') -> None:
        """Dispatches to the named logger at the given level."""
        getattr(self._logger, level)(message)

    def _setup_temp_dirs(self) -> None:
        """Creates the temp directory tree and hides the root on Windows."""
        for d in [
            self._temp_root, self._airfoils_temp, self._polars_temp,
            self._geo_temp, self._force_temp, self._support_temp,
            self._recovery_temp,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        if os.name == 'nt':
            subprocess.run(
                ['attrib', '+H', str(self._temp_root)],
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        shutil.copy(self.case_file_path, self._support_temp)

    def _teardown_temp_dirs(self) -> None:
        """Removes the entire temp directory tree."""
        if self._temp_root.exists():
            shutil.rmtree(self._temp_root)

    def _read_dat_cst_header(self, file_path: Path, expected_count: int) -> List[float]:
        """
        Reads a CST-coefficient header from a .dat file's first 1KB.
        Adapted from db_airfoil_cst_finder.py's read_dat_header: tries a
        bracketed, comma- or space-delimited text header first, falls back
        to raw float32 binary. Returns [] if neither yields `expected_count`
        values.

        Values are expected in the same order `_source_airfoil` builds them
        for comparison: CST lower coefficients followed by CST upper
        coefficients.
        """
        try:
            with open(file_path, "rb") as f:
                raw = f.read(1024)
            text = raw.decode(errors="ignore")

            match = re.search(r"\[([^\]]+)\]", text)
            if match:
                # Header writers have used both comma-delimited
                # ("[0.1, 0.2]") and space-delimited ("[0.1 0.2]") bracketed
                # lists -- split on either so both are parsed instead of the
                # space-delimited case silently failing to yield any values.
                numbers = re.split(r"[,\s]+", match.group(1).strip())
                values = [float(n) for n in numbers if n != ""]
                return values[:expected_count]

            with open(file_path, "rb") as f:
                data = f.read(expected_count * 4)
            if len(data) == expected_count * 4:
                return list(struct.unpack(f"{expected_count}f", data))
        except (OSError, ValueError, struct.error) as e:
            self._log(f"Could not read CST header from {file_path.name}: {e}", 'warning')
        return []

    def _build_airfoil_index(self) -> Dict[Tuple[float, ...], List[Path]]:
        """
        Walks `source_airfoils_dir` once, mapping a rounded CST-coefficient
        tuple (see `_CST_MATCH_DECIMALS`) to the list of files sharing it.
        Built fresh on every `run_correction()` call, fresh or resumed --
        the source directory is immutable during correction, so this is
        cheaper and safer than persisting/invalidating a cached index.
        Returns {} (every point falls back to regeneration) if
        `source_airfoils_dir` wasn't provided or doesn't exist.
        """
        if self.source_airfoils_dir is None or not self.source_airfoils_dir.exists():
            if self.source_airfoils_dir is not None:
                self._log(
                    f"source_airfoils_dir not found: {self.source_airfoils_dir}. "
                    "All points will be regenerated from CST.", 'warning'
                )
            return {}

        expected_count = len(self._cst_l_cols) + len(self._cst_u_cols)
        dat_files = list(self.source_airfoils_dir.rglob("*.dat"))

        index: Dict[Tuple[float, ...], List[Path]] = {}
        for file_path in tqdm(dat_files, desc="Indexing available airfoils"):
            header_values = self._read_dat_cst_header(file_path, expected_count)
            if len(header_values) != expected_count:
                continue
            key = tuple(round(v, self._CST_MATCH_DECIMALS) for v in header_values)
            index.setdefault(key, []).append(file_path)

        self._log(
            f"Airfoil index built: {len(dat_files)} files scanned, "
            f"{len(index)} unique CST keys."
        )
        return index

    def _source_airfoil(self, row: pd.Series, label: str) -> Tuple[Path, str, bool]:
        """
        Resolves the airfoil .dat file for one point. Called once per point
        (not per attempt -- CST-defined geometry is invariant across
        retries of the same point). Reuses an already-sourced file on disk
        first (covers resuming mid-point after a crash) -- but only if its
        embedded CST header still matches this row's own coefficients; a
        mismatch means the file is a stale leftover from an earlier
        run/crash (e.g. a different row previously resolved to the same
        label), so it is discarded and re-sourced rather than trusted.
        Otherwise tries the prebuilt index, falling back to CST regeneration
        whenever no unambiguous match exists.

        Returns
        -------
        (airfoil_path, actual_airfoil_name, is_reconstruction)
            `actual_airfoil_name` is the original source file's name for an
            index-matched airfoil, or the reconstructed file's own name when
            no source match exists (or a resumed cache-hit had none) --
            `is_reconstruction` distinguishes the two for logging.
        """
        def _header_derived_name(path: Path) -> Optional[str]:
            """
            Recovers the original name embedded in a .dat header (the
            `file_name` an earlier `_create_dat_file` call was given, or an
            index-matched file's own original name -- both survive a
            straight `shutil.copy`). Used only for a resumed cache-hit,
            where neither branch below runs to tell us directly. Returns
            None if the header doesn't match the expected format.
            """
            try:
                with open(path, "rb") as f:
                    text = f.read(1024).decode(errors="ignore")
            except OSError:
                return None
            match = re.match(r"^([^\r\n\[]+?)\s+based on", text)
            return match.group(1).strip() if match else None

        airfoil_path = self._airfoils_temp / f"{label}.dat"
        cst_coeffs = (
            list(row[self._cst_l_cols].values) +
            list(row[self._cst_u_cols].values)
        )

        if airfoil_path.exists():
            expected_count = len(self._cst_l_cols) + len(self._cst_u_cols)
            cached_values = self._read_dat_cst_header(airfoil_path, expected_count)
            cached_key = tuple(round(v, self._CST_MATCH_DECIMALS) for v in cached_values)
            current_key = tuple(round(v, self._CST_MATCH_DECIMALS) for v in cst_coeffs)
            if len(cached_values) == expected_count and cached_key == current_key:
                header_name = _header_derived_name(airfoil_path)
                if header_name is not None:
                    return airfoil_path, header_name, (header_name == label)
                return airfoil_path, airfoil_path.name, False
            self._log(
                f"  Cached airfoil temp file for '{label}' does not match "
                f"this row's CST coefficients (stale leftover from a "
                f"previous run/crash); discarding and re-sourcing.",
                'warning',
            )
            try:
                airfoil_path.unlink()
            except OSError:
                pass

        if self._airfoil_index:
            key = tuple(round(v, self._CST_MATCH_DECIMALS) for v in cst_coeffs)
            matches = self._airfoil_index.get(key, [])
            if len(matches) == 1:
                shutil.copy(matches[0], airfoil_path)
                self._log(
                    f"  Sourced airfoil for '{label}' from index match: "
                    f"{matches[0]}"
                )
                return airfoil_path, matches[0].name, False
            elif len(matches) > 1:
                self._log(
                    f"  Ambiguous airfoil match for '{label}' "
                    f"({len(matches)} files share this CST key); regenerating.",
                    'warning',
                )
            # 0 matches falls through silently -- expected occasionally
            # (e.g. a point whose source airfoil was itself refused in an
            # earlier correction pass), not worth a warning every time.

        airfoil_creator = Airfoil_Generator(
            base_airfoil_csv=self.base_airfoil_csv,
            n_cst=self.n_cst,
            verbose=False,
        )
        x_mod, yu_mod, yl_mod = airfoil_creator.generate(cst_coeffs=cst_coeffs)
        airfoil_creator._create_dat_file(
            output_dir=self._airfoils_temp,
            n_cst=self.n_cst,
            x=x_mod,
            yl=yl_mod,
            yu=yu_mod,
            file_name=label,
        )
        self._log(
            f"  Sourced airfoil for '{label}' via CST regeneration "
            f"(no unambiguous match in source_airfoils_dir)."
        )
        return airfoil_path, airfoil_path.name, True

    def _atomic_write_json(self, path: Path, data: dict) -> None:
        """Writes JSON via a temp file + os.replace so a crash mid-write can't corrupt it."""
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp_path, path)

    def _save_checkpoint(
        self,
        df: pd.DataFrame,
        total_rows: int,
        n_target: int,
        passes_complete: set,
        current_pass: Optional[str],
        processed_this_pass: int,
        point_counter: int,
        corrected_indices: set,
        refused_indices: set,
        stats: dict,
        accel_value: float,
        accel_streak: int,
        accel_prev_d: int,
    ) -> None:
        """
        Persists pass-level progress: the working dataframe (with the pandas
        index preserved as stable row identifiers, since refused rows are
        dropped without renumbering) plus enough state to resume the correct
        pass at the correct position. `point_counter` is tracked separately
        from `processed_this_pass` (rather than derived from it) so label
        numbering stays monotonic across a resume -- resetting it would let
        a freshly-numbered label collide with a stale temp file left over
        from a differently-numbered point before the crash. Written after
        every point resolves.

        `accel_value`/`accel_streak`/`accel_prev_d` are the per-pass
        acceleration ledger (see AccelerationConfig / _accel_n_effective).
        Persisted so a resumed run reconstructs the exact effective target
        the interrupted run was using, rather than recomputing one from
        scratch -- see `_accel_n_effective`.
        """
        tmp_df_path = self._checkpoint_df_path.with_suffix('.csv.tmp')
        df.to_csv(tmp_df_path, index=True)
        os.replace(tmp_df_path, self._checkpoint_df_path)

        self._atomic_write_json(self._checkpoint_path, {
            'csv_file_path': str(self.csv_file_path),
            'total_rows': total_rows,
            'n_target': n_target,
            'passes_complete': sorted(passes_complete),
            'current_pass': current_pass,
            'processed_this_pass': processed_this_pass,
            'point_counter': point_counter,
            'corrected_indices': sorted(corrected_indices),
            'refused_indices': sorted(refused_indices),
            'stats': stats,
            'accel_value': accel_value,
            'accel_streak': accel_streak,
            'accel_prev_d': accel_prev_d,
        })

    def _load_checkpoint(self) -> Optional[dict]:
        """
        Returns the checkpoint state dict if a valid, matching checkpoint
        exists, else None. Mismatched `csv_file_path` (a leftover checkpoint
        from a different target CSV sharing this temp dir) is treated as
        no checkpoint, with a warning, rather than resumed against.
        """
        if not self._checkpoint_path.exists() or not self._checkpoint_df_path.exists():
            return None
        try:
            with open(self._checkpoint_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self._log(f"Checkpoint unreadable ({e}), starting fresh.", 'warning')
            return None

        if state.get('csv_file_path') != str(self.csv_file_path):
            self._log(
                "Checkpoint found but targets a different CSV "
                f"({state.get('csv_file_path')}); starting fresh.", 'warning'
            )
            return None
        return state

    def _save_point_progress(
        self,
        idx: int,
        label: str,
        stored_e: float,
        last_sim_e: Optional[float],
        consecutive_pass: int,
        next_attempt: int,
        attempt_log: List[dict],
    ) -> None:
        """
        Persists in-flight attempt-loop state for the single point currently
        being validated, so a crash mid-point resumes at `next_attempt`
        instead of re-running already-completed attempts. `next_attempt` is
        tracked explicitly (not inferred from len(attempt_log)) since a
        failed simulation attempt consumes an attempt without adding a
        log entry.
        """
        self._atomic_write_json(self._point_progress_path, {
            'csv_file_path': str(self.csv_file_path),
            'idx': idx,
            'label': label,
            'stored_e': stored_e,
            'last_sim_e': last_sim_e,
            'consecutive_pass': consecutive_pass,
            'next_attempt': next_attempt,
            'attempt_log': attempt_log,
        })

    def _load_point_progress(self, idx: int) -> Optional[dict]:
        """
        Returns saved attempt-loop state for `idx` if present and matching,
        else None. A mismatched idx means the point in progress at crash
        time differs from the one about to be processed (shouldn't happen
        under normal resume, since selection is deterministic given the
        checkpointed df and index sets) -- treated as no saved state.
        """
        if not self._point_progress_path.exists():
            return None
        try:
            with open(self._point_progress_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self._log(f"Point-progress file unreadable ({e}), starting attempt 1.", 'warning')
            return None

        if state.get('csv_file_path') != str(self.csv_file_path) or state.get('idx') != idx:
            self._log(
                f"Point-progress file found but doesn't match idx={idx}; "
                "starting attempt 1.", 'warning'
            )
            return None
        return state

    def _clear_point_progress(self) -> None:
        """Removes the in-flight point-progress file once a point resolves."""
        try:
            self._point_progress_path.unlink()
        except FileNotFoundError:
            pass

    def _clean_temp_for_label(self, label: str, dirs: Optional[List[Path]] = None) -> None:
        """
        Deletes all temp files whose filename contains `label`.
        `dirs` defaults to all four temp dirs; pass a subset to scope the
        sweep (e.g. geo/force dirs only, so as not to touch the point-scoped
        airfoil .dat file that must survive across attempts).
        """
        if dirs is None:
            dirs = [self._airfoils_temp, self._polars_temp,
                    self._geo_temp, self._force_temp]
        for d in dirs:
            for f in d.glob(f"*{label}*"):
                try:
                    f.unlink()
                except OSError:
                    pass

    def _get_geo_creator_class(self, row: pd.Series) -> Optional[Type]:
        """
        Returns the geo creator class for a given row.

        For single-configuration CSVs, returns `_default_geo_class`.
        For multi-configuration CSVs, reads the one-hot encoding and maps
        the active config column to its corresponding class via `_config_map`.
        """
        if not self._config_cols:
            return self._default_geo_class
        for col in self._config_cols:
            if row[col] == 1:
                return self._config_map[col]
        self._log(
            f"No configuration column is set to 1. "
            f"Config values: {row[self._config_cols].to_dict()}",
            'warning',
        )
        return None

    def _extract_force_data(
        self, file_path: Path
    ) -> Optional[Tuple[str, float, float]]:
        """Extracts airfoil name, CLtot, and CDtot from an AVL force file."""
        airfoil, cl_tot, cd_tot = None, None, None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if "Configuration:" in line and " for " in line:
                        airfoil = line.split(" for ", 1)[1].strip()
                    elif 'CLtot =' in line:
                        cl_tot = float(line.split('=')[1].strip())
                    elif 'CDtot =' in line:
                        cd_tot = float(line.split('=')[1].strip())
            if all((airfoil, cl_tot is not None, cd_tot is not None)):
                return airfoil, cl_tot, cd_tot
            self._log(f"Incomplete force data in {file_path.name}.", 'warning')
            return None
        except FileNotFoundError:
            self._log(f"Force file not found: {file_path}", 'error')
            return None
        except (ValueError, IndexError) as e:
            self._log(f"Error parsing force file {file_path.name}: {e}", 'error')
            return None

    def _format_row_for_log(self, row: pd.Series) -> str:
        """Returns a structured string of all row features for log output."""
        cst_l = [round(float(v), 8) for v in row[self._cst_l_cols].values]
        cst_u = [round(float(v), 8) for v in row[self._cst_u_cols].values]
        parts = [f"CST_lower={cst_l}", f"CST_upper={cst_u}"]
        
        if self.append_kp:
            kp = {k: round(float(row[k]), 6) for k in self._kp_cols}
            parts.append(f"KP={kp}")
            
        if self._config_cols:
            parts.append(f"Config={row[self._config_cols].to_dict()}")
            
        return " | ".join(parts)

    def _simulate_point(
        self, row: pd.Series, label: str, airfoil_path: Path
    ) -> Optional[Tuple[float, List[float]]]:
        """
        Executes the full aerodynamic simulation pipeline for one data point.

        Pipeline
        --------
        1. Compute drag polar (XFOIL) from the already-sourced airfoil.
        2. Extract key points via GPR interpolation (Key_Point_Extraction).
        3. Determine geo creator class from one-hot config encoding.
        4. Build AVL wing geometry file (geo creator class).
        5. Compute aerodynamic force coefficients (AVL_Run_Case).
        6. Return E = CLtot / CDtot and the six key point values.

        Parameters
        ----------
        row : pd.Series
            Data row containing CST coefficients and configuration encoding.
        label : str
            Unique label for temp file naming.
        airfoil_path : Path
            Path to the airfoil .dat file for this point, sourced once per
            point (by lookup or CST regeneration) rather than per attempt --
            see `_source_airfoil`.

        Returns
        -------
        (sim_E, kp_values) on success, None on any failure.
        kp_values order: [CLneg, CDneg, CLcdmin, CDmin, CLpos, CDpos]
        """
        try:
            # XFOIL polar curve creation
            num_steps = int((self.aseq[1] - self.aseq[0]) / self.aseq[2]) + 1
            angles = [self.aseq[0] + j * self.aseq[2] for j in range(num_steps)]
            polar_path = self._polars_temp / f"polar_{label}.txt"

            xfoil_case = XPR(
                xfoil_exe_path=str(self.xfoil_exe_path),
                main_dir=self._temp_root,
                airfoil_dir=self._airfoils_temp,
                polar_dir=self._polars_temp,
                verbose=False,
                process_logging=False
            )
            xfoil_case.run_simulation(
                airfoil_file_path=airfoil_path,
                polar_file_path=str(polar_path),
                alfa_sequence=angles,
                visc=self.Re,
                mach=self.Mach,
            )

            # Key point extraction
            kp_extract = Key_Point_Extraction(polar_file_paths=[str(polar_path)])
            low_cl_point, high_cl_point, min_cd_point = (
                kp_extract._interpolate_and_extract(str(polar_path))
            )
            kp_values: List[float] = [
                low_cl_point[0],   # CLneg
                low_cl_point[1],   # CDneg
                min_cd_point[0],   # CLcdmin
                min_cd_point[1],   # CDmin
                high_cl_point[0],  # CLpos
                high_cl_point[1],  # CDpos
            ]
            placeholder_list = [{
                # Must strip (via the geo creator's own
                # "polar_"/"key_polar_"/".txt" stripping) to exactly
                # airfoil_path.stem, since that's the key the geo creator
                # looks itself up by (airfoil_path_map keyed on the airfoil's
                # own filename stem) -- NOT polar_path.name, which is
                # attempt-scoped and no longer matches the airfoil's
                # point-scoped stem now that they're sourced independently.
                'file_path': f"{airfoil_path.stem}.txt",
                'min_cl_point': {'CL': low_cl_point[0],  'CD': low_cl_point[1]},
                'max_cl_point': {'CL': high_cl_point[0], 'CD': high_cl_point[1]},
                'min_cd_point': {'CL': min_cd_point[0],  'CD': min_cd_point[1]},
            }]

            # Geo creator class selection
            geo_creator_class = self._get_geo_creator_class(row)
            if geo_creator_class is None:
                return None

            # AVL geometry file creation
            geo_creator = geo_creator_class(
                key_points_data=placeholder_list,
                airfoil_abs_paths=[airfoil_path],
                naca_key_points_data=str(self.naca_key_points_data),
                output_dir=str(self._geo_temp),
                verbose=False,
                pb=False,
            )
            geo_creator.create_geo_files()

            if geo_creator.errors:
                for err in geo_creator.errors:
                    self._log(f"  Geo creator error: {err}", 'warning')

            # Output is named {airfoil_stem}_{config}_geo.avl by every geo
            # creator class -- no attempt-specific component, since the
            # creator names files after the airfoil it was given, not after
            # our per-attempt label.
            avl_files = list(self._geo_temp.glob(f"{airfoil_path.stem}_*_geo.avl"))
            if not avl_files:
                self._log(
                    f"No .avl geo file found matching "
                    f"'{airfoil_path.stem}_*_geo.avl' after geo creator ran.",
                    'error',
                )
                return None
            geo_path = avl_files[0]

            # AVL force coefficient calculation
            case_file_temp = self._support_temp / self.case_file_path.name
            avl_case = AVL_Run_Case(
                geo_files=[geo_path],
                case_file=str(case_file_temp),
                avl_exe_path=str(self.avl_exe_path),
                output_dir=self._force_temp,
            )
            avl_case.run()

            # Parse force file; name derived from geo file stem
            force_stem = geo_path.stem.replace("_geo", "")
            force_path = self._force_temp / f"force_{force_stem}.txt"
            result = self._extract_force_data(force_path)
            if result is None:
                return None

            _, cl, cd = result
            if cd == 0:
                self._log(f"CDtot=0 for label '{label}', E is undefined.", 'warning')
                return None

            return cl / cd, kp_values

        except Exception as e:
            self._log(f"Simulation pipeline failed for '{label}': {e}", 'error')
            return None

    def _correct_point(
        self, df: pd.DataFrame, idx: int, label: str,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Validates and corrects a single data point.

        Validation criterion
        --------------------
        Two consecutive simulations must produce E values within
        `error_threshold` of each other. The reference for each comparison
        is the result of the immediately preceding simulation (or the stored
        CSV value for the first attempt). Values are written to the dataframe
        only upon confirmed validation. KP values are optionally updated.

        Resumes mid-attempt-loop from a saved point-progress file when one
        matches `idx` (see `_save_point_progress`/`_load_point_progress`),
        so a crash between attempts doesn't discard already-completed ones.

        Parameters
        ----------
        df : pd.DataFrame
            Working copy of the training database.
        idx : int
            Row index of the point to validate.
        label : str
            Unique label for temp file naming and log identification.
        on_status : callable, optional
            Called with a short human-readable string at the same points
            the attempt loop already logs (result of each attempt,
            validation, failure) -- lets a caller mirror live status
            somewhere other than the log file, e.g. a GUI detail view.
            No-op when omitted.

        Returns
        -------
        (df, outcome) where outcome is 'validated', 'corrected', or 'refused'.
        """
        def _status(text: str) -> None:
            if on_status is not None:
                on_status(text)

        row = df.loc[idx]
        stored_e: float = row['E']
        airfoil_path, actual_airfoil_name, is_reconstruction = self._source_airfoil(row, label)
        _status(f"{label}: sourcing airfoil / starting attempt 1")

        # 1-based CSV line number: header occupies line 1, so DataFrame
        # index 0 is line 2.
        database_line_id = idx + 2
        airfoil_field_label = (
            "Airfoil for validation (reconstruction)" if is_reconstruction
            else "Airfoil for validation"
        )

        resumed = self._load_point_progress(idx)
        if resumed is not None:
            last_sim_e: Optional[float] = resumed['last_sim_e']
            consecutive_pass: int = resumed['consecutive_pass']
            attempt_log: List[dict] = resumed['attempt_log']
            start_attempt = resumed['next_attempt']
            self._log(
                f"\n  Point \"{label}\" idx: {database_line_id} | "
                f"Stored E={stored_e:.8f} | "
                f"{airfoil_field_label}: {actual_airfoil_name} | "
                f"Resuming from attempt {start_attempt} "
                f"(consecutive_pass={consecutive_pass})"
            )
        else:
            last_sim_e = None
            consecutive_pass = 0
            attempt_log = []
            start_attempt = 1
            self._log(
                f"\n  Point \"{label}\" idx: {database_line_id} | "
                f"Stored E={stored_e:.8f} | "
                f"{airfoil_field_label}: {actual_airfoil_name}"
            )
        self._log(f"  Features: {self._format_row_for_log(row)}")

        for attempt in range(start_attempt, self.max_attempts + 1):
            attempt_label = f"{label}_a{attempt}"

            result = self._simulate_point(df.loc[idx], attempt_label, airfoil_path)

            # Clean temp files produced by this attempt before the next iteration
            self._clean_temp_for_label(attempt_label)
            # AVL geo files ("{label}_{config}_geo.avl") and force files
            # ("force_{label}_{config}.txt") are named after the point-scoped
            # label, not attempt_label (see _simulate_point), so the sweep
            # above never touches them. Clean them by `label` here, scoped to
            # just these two dirs, so a failed/stale attempt's output can't
            # be silently read by a later attempt's _extract_force_data call.
            self._clean_temp_for_label(label, dirs=[self._geo_temp, self._force_temp])

            if result is None:
                self._log(
                    f"  Attempt {attempt}/{self.max_attempts}: simulation failed.",
                    'warning',
                )
                _status(f"{label}: attempt {attempt}/{self.max_attempts} failed")
                consecutive_pass = 0
                self._save_point_progress(
                    idx, label, stored_e, last_sim_e, consecutive_pass,
                    attempt + 1, attempt_log,
                )
                continue

            sim_e, kp_values = result
            reference = last_sim_e if last_sim_e is not None else stored_e
            error = (
                abs(sim_e - reference) / abs(reference)
                if reference != 0 else float('inf')
            )

            attempt_log.append({
                'attempt': attempt,
                'sim_E': sim_e,
                'reference': reference,
                'error': error,
                'kp_values': kp_values,
            })
            self._log(
                f"  Attempt {attempt}/{self.max_attempts}: "
                f"sim_E={sim_e:.8f} | ref={reference:.8f} | "
                f"error={error:.4%} | consecutive_pass={consecutive_pass}"
            )
            _status(
                f"{label}: attempt {attempt}/{self.max_attempts} | "
                f"sim_E={sim_e:.4f} | error={error:.2%}"
            )

            if error <= self.error_threshold:
                consecutive_pass += 1
                if consecutive_pass == 2:
                    # Two consecutive agreements: write validated E and conditionally write corresponding KP values
                    df.loc[idx, 'E'] = sim_e
                    if self.append_kp:
                        for col, val in zip(self._kp_cols, kp_values):
                            df.loc[idx, col] = val
                            
                    outcome = 'validated' if len(attempt_log) == 2 else 'corrected'
                    
                    log_msg = f"  {outcome.upper()} | New E={sim_e:.8f} | "
                    if self.append_kp:
                        log_msg += "KP updated | "
                    log_msg += f"Total attempts used: {attempt}"
                    
                    self._log(log_msg)
                    _status(f"{label}: {outcome} (E={sim_e:.4f})")
                    self._clear_point_progress()
                    self._clean_temp_for_label(label)
                    return df, outcome
            else:
                consecutive_pass = 0

            last_sim_e = sim_e
            self._save_point_progress(
                idx, label, stored_e, last_sim_e, consecutive_pass,
                attempt + 1, attempt_log,
            )

        # Max attempts exhausted without two consecutive passes
        self._log(
            f"  FAILED: max_attempts={self.max_attempts} reached without "
            f"validation. Row at idx={idx} will be refused (removed from DB).",
            'error',
        )
        self._log(f"  Full attempt history for idx={idx}:")
        for entry in attempt_log:
            kp_log_str = f"KP={[round(v, 6) for v in entry['kp_values']]}" if self.append_kp else "KP_mode= Off"
            self._log(
                f"    Attempt {entry['attempt']}: "
                f"sim_E={entry['sim_E']:.8f} | "
                f"ref={entry['reference']:.8f} | "
                f"error={entry['error']:.4%} | "
                f"{kp_log_str}"
            )

        self._clear_point_progress()
        self._clean_temp_for_label(label)
        _status(f"{label}: refused (max attempts reached)")
        df = df.drop(index=idx)
        return df, 'refused'

    # ── Acceleration ──────────────────────────────────────────────────── #

    def _accel_n_effective(
        self, accel_value: float, prev_d: int, n_target: int,
    ) -> Tuple[int, int]:
        """
        Maps the current per-pass acceleration ledger to how many of
        `n_target` points should actually be tested, and the drop count
        `d` used to get there (`n_target - d`).

        `d` is derived from net movement off `accel_config.initial`, in
        units of `accel_config.step_size`, clamped so testing never drops
        below `floor_fraction * n_target` and never exceeds `n_target`
        (acceleration only ever reduces the per-pass workload).

        When `accel_config.hysteresis > 0`, `d` only moves once `net`
        clears a margin past the next threshold in either direction
        (Schmitt-trigger style dead-band) rather than at the bare
        division boundary, to avoid the count flapping by +/-1 when `net`
        sits close to a boundary. `prev_d` carries that state across
        calls -- pass back the `d` this function last returned, and
        persist it in the checkpoint so a resumed run reconstructs the
        exact target the interrupted run was using rather than a value
        recomputed from a possibly-changed config.

        With acceleration disabled, always returns `(n_target, 0)`.
        """
        cfg = self.accel_config
        if not cfg.enabled or cfg.step_size <= 0:
            return n_target, 0

        n_floor = max(1, math.ceil(cfg.floor_fraction * n_target))
        max_d = max(0, n_target - n_floor)
        net = accel_value - cfg.initial

        if cfg.hysteresis > 0:
            d = prev_d
            while d < max_d and net >= (d + 1) * cfg.step_size + cfg.hysteresis:
                d += 1
            while d > 0 and net < d * cfg.step_size - cfg.hysteresis:
                d -= 1
        else:
            d = math.floor(net / cfg.step_size)

        d = max(0, min(d, max_d))
        return n_target - d, d

    def _accel_update(
        self, accel_value: float, accel_streak: int, outcome: str,
    ) -> Tuple[float, int]:
        """
        Updates the per-pass acceleration ledger from one resolved point's
        classification, which is exactly `outcome` from `_correct_point`:

        - 'validated' (original database value confirmed correct): streak
          increments and the reward is amplified by
          `1 + streak_bonus * min(streak - 1, streak_cap)`, giving
          consecutive validations a growing bonus without letting one long
          streak grow unbounded.
        - 'corrected' (value needed correcting but converged): streak
          resets, ledger drops by `penalty_corrected`.
        - 'refused' (dropped, never converged): streak resets, ledger
          drops by `penalty_dropped` (default larger -- a stronger signal
          of an unreliable point than one that just needed correcting).

        Clamped to `accel_config.initial +/- accel_config.cap` so the
        ledger itself can't run away over a long pass; the operationally
        meaningful, further-clamped quantity is `_accel_n_effective`'s
        `d`, not this raw value.

        No-op (returns inputs unchanged) when acceleration is disabled.
        """
        cfg = self.accel_config
        if not cfg.enabled:
            return accel_value, accel_streak

        if outcome == 'validated':
            accel_streak += 1
            multiplier = 1 + cfg.streak_bonus * min(accel_streak - 1, cfg.streak_cap)
            accel_value += cfg.reward_validated * multiplier
        else:
            accel_streak = 0
            accel_value -= cfg.penalty_corrected if outcome == 'corrected' else cfg.penalty_dropped

        lo, hi = cfg.initial - cfg.cap, cfg.initial + cfg.cap
        accel_value = min(max(accel_value, lo), hi)
        return accel_value, accel_streak

    def _run_top_bottom_pass(
        self, on_progress: Callable[[int], None], on_status: Callable[[str], None],
        on_accel: Callable[[int, float], None],
        df: pd.DataFrame, direction: str, ascending: bool, n_target: int,
        processed_this_pass: int, point_counter: int,
        corrected_indices: set, refused_indices: set, stats: dict,
        passes_complete: set, total_rows: int,
        accel_value: float, accel_streak: int, accel_prev_d: int,
    ) -> Tuple[pd.DataFrame, set, set, dict]:
        """
        Body of one top or bottom pass -- see run_correction for the dynamic
        re-sort rationale. `n_effective` (how many of `n_target` are
        actually tested this pass) is recomputed from the acceleration
        ledger at the top of every iteration, so it can shrink OR grow
        back across the loop's lifetime -- selection itself (dynamic
        re-sort, first not-yet-resolved index) is unaffected; acceleration
        only moves the stopping line.
        """
        n_effective, accel_prev_d = self._accel_n_effective(accel_value, accel_prev_d, n_target)
        while processed_this_pass < n_effective:
            sorted_idx = df['E'].sort_values(ascending=ascending).index.tolist()
            next_idx = next(
                (i for i in sorted_idx if i not in corrected_indices and i not in refused_indices),
                None,
            )
            if next_idx is None:
                self._log(f"No more unprocessed rows for {direction} pass.")
                break

            point_counter += 1
            label = f"{direction}_{point_counter}"
            self._log(f"\n--- [{direction.upper()}] Point {processed_this_pass + 1}/{n_effective} ---")
            on_status(f"Starting {label} ({processed_this_pass + 1}/{n_effective})")

            df, outcome = self._correct_point(df, next_idx, label, on_status=on_status)

            if outcome in ('validated', 'corrected'):
                corrected_indices.add(next_idx)
                stats[outcome] += 1
            else:
                refused_indices.add(next_idx)
                stats['refused'] += 1

            accel_value, accel_streak = self._accel_update(accel_value, accel_streak, outcome)
            n_effective, accel_prev_d = self._accel_n_effective(accel_value, accel_prev_d, n_target)

            processed_this_pass += 1
            on_progress(processed_this_pass)
            on_accel(n_target - n_effective, accel_value)
            self._save_checkpoint(
                df, total_rows, n_target, passes_complete, direction,
                processed_this_pass, point_counter,
                corrected_indices, refused_indices, stats,
                accel_value, accel_streak, accel_prev_d,
            )

        passes_complete.add(direction)
        self._save_checkpoint(
            df, total_rows, n_target, passes_complete, None, 0, 0,
            corrected_indices, refused_indices, stats,
            self.accel_config.initial, 0, 0,
        )
        return df, corrected_indices, refused_indices, stats

    def _run_middle_pass(
        self, on_progress: Callable[[int], None], on_status: Callable[[str], None],
        on_accel: Callable[[int, float], None],
        df: pd.DataFrame, n_middle: int, already_done: int,
        corrected_indices: set, refused_indices: set, stats: dict,
        passes_complete: set, total_rows: int, n_target: int,
        accel_value: float, accel_streak: int, accel_prev_d: int,
    ) -> Tuple[pd.DataFrame, set, set, dict]:
        """
        Body of the middle pass -- see run_correction for why the candidate
        list is recomputed fresh (unaffected by acceleration: the ranking
        itself doesn't change, only how much of it gets consumed).
        `n_effective` bounds `processed_this_pass` exactly as in
        `_run_top_bottom_pass`; because `middle_order` is a fixed
        precomputed list rather than a live re-sort, an explicit cursor
        walks it in rank order instead of a `for` loop, so the stopping
        condition can be re-checked (and can grow back) on every step.
        """
        median_e = df['E'].median()
        remaining_target = max(0, n_middle - already_done)
        eligible_idx = df.index.difference(
            pd.Index(list(corrected_indices | refused_indices))
        )
        middle_order = (
            df.loc[eligible_idx, 'E']
            .sub(median_e)
            .abs()
            .sort_values(ascending=False)
            .head(remaining_target)
            .index.tolist()
        )

        point_counter = already_done
        processed_this_pass = already_done
        n_effective, accel_prev_d = self._accel_n_effective(accel_value, accel_prev_d, n_middle)
        cursor = 0
        while cursor < len(middle_order) and processed_this_pass < n_effective:
            next_idx = middle_order[cursor]
            cursor += 1
            if next_idx in corrected_indices or next_idx in refused_indices:
                continue
            point_counter += 1
            label = f"middle_{point_counter}"
            self._log(f"\n--- [MIDDLE] Point {point_counter}/{n_effective} ---")
            on_status(f"Starting {label} ({point_counter}/{n_effective})")

            df, outcome = self._correct_point(df, next_idx, label, on_status=on_status)
            if outcome in ('validated', 'corrected'):
                corrected_indices.add(next_idx)
                stats[outcome] += 1
            else:
                refused_indices.add(next_idx)
                stats['refused'] += 1

            accel_value, accel_streak = self._accel_update(accel_value, accel_streak, outcome)
            n_effective, accel_prev_d = self._accel_n_effective(accel_value, accel_prev_d, n_middle)

            processed_this_pass += 1
            on_progress(processed_this_pass)
            on_accel(n_middle - n_effective, accel_value)
            self._save_checkpoint(
                df, total_rows, n_target, passes_complete, 'middle',
                processed_this_pass, point_counter,
                corrected_indices, refused_indices, stats,
                accel_value, accel_streak, accel_prev_d,
            )

        passes_complete.add('middle')
        self._save_checkpoint(
            df, total_rows, n_target, passes_complete, None, 0, 0,
            corrected_indices, refused_indices, stats,
            self.accel_config.initial, 0, 0,
        )
        return df, corrected_indices, refused_indices, stats

    def _run_pass_with_gui(
        self,
        pass_fn: Callable[..., Tuple[pd.DataFrame, set, set, dict]],
        pass_kwargs: dict,
        n_target: int,
        initial_completed: int,
        window_title: str,
        header_text: str,
    ) -> Tuple[pd.DataFrame, set, set, dict]:
        """
        Runs `pass_fn(on_progress=..., on_status=..., **pass_kwargs)` in a
        background thread while a Progress_GUI window owns the main thread
        -- mirrors the threading.Thread + gui.run() + thread.join() pattern
        Hossain_DB_Gen_linux.py already uses for its own multiprocessing
        stages, adapted here to a single sequential worker (num_workers=1)
        instead of a process pool. Replaces the tqdm bars, which were
        getting corrupted by other tools' stdout output interleaving with
        their carriage-return redraws during validation.

        The window is guaranteed to close regardless of how pass_fn exits
        -- full completion, an early break when no unprocessable rows
        remain, or an exception -- by padding the progress queue up to
        n_target in a finally block; otherwise Progress_GUI's own
        completion check (files_completed >= total_files) would never
        fire and the window would hang open indefinitely. Any exception
        from pass_fn is captured and re-raised here, on the main thread,
        once the window has closed -- run_correction's checkpoint/recovery
        flow depends on failures actually propagating, which the driver's
        existing multiprocessing usage of this same GUI does not need to
        guarantee (its worker pool isn't relied on to fail loudly).

        Also seeds and threads through the acceleration display: an
        `on_accel(dropped, value)` callback alongside `on_progress`/
        `on_status`, mutating a shared `acceleration_state` dict the GUI
        polls the same way it already polls `worker_status_list`. Requests
        the GUI's detailed_only/show_acceleration mode (see GUI.py) --
        default-window-is-detail-view with the acceleration readout,
        gated behind `accel_config.enabled` so a disabled config falls
        back to the plain detail view with no readout.
        """
        progress_queue: "queue.Queue" = queue.Queue()
        worker_status_list = ["Starting..."]

        accel_value0 = pass_kwargs.get('accel_value', self.accel_config.initial)
        accel_prev_d0 = pass_kwargs.get('accel_prev_d', 0)
        _, initial_d = self._accel_n_effective(accel_value0, accel_prev_d0, n_target)
        acceleration_state = {'dropped': initial_d, 'value': accel_value0}

        state = {'processed': initial_completed, 'error': None, 'result': None}

        def on_progress(processed_count: int) -> None:
            state['processed'] = processed_count
            progress_queue.put(1)

        def on_status(text: str) -> None:
            worker_status_list[0] = text

        def on_accel(dropped: int, value: float) -> None:
            acceleration_state['dropped'] = dropped
            acceleration_state['value'] = value

        def _target():
            try:
                state['result'] = pass_fn(
                    on_progress=on_progress, on_status=on_status,
                    on_accel=on_accel, **pass_kwargs,
                )
            except Exception as e:
                state['error'] = e
            finally:
                remaining = max(0, n_target - state['processed'])
                for _ in range(remaining):
                    progress_queue.put(1)

        thread = threading.Thread(target=_target)
        thread.start()

        gui = Progress_GUI(
            total_files=n_target,
            progress_queue=progress_queue,
            worker_status_list=worker_status_list,
            num_workers=1,
            window_title=window_title,
            header_text=header_text,
            initial_completed=initial_completed,
            detailed_only=True,
            show_acceleration=self.accel_config.enabled,
            acceleration_state=acceleration_state,
        )
        gui.run()
        thread.join()

        if state['error'] is not None:
            raise state['error']
        return state['result']

    def run_correction(self) -> None:
        """
        Executes the full correction pipeline.

        Pass order
        ----------
        1. Top pass    — highest E values descending (dynamic re-sort each step).
        2. Bottom pass — lowest E values ascending (dynamic re-sort each step).
        3. Middle pass — highest absolute deviation from median E (optional,
                         controlled by `middle_fraction`; disabled by default).

        Dynamic re-sorting ensures that a corrected value which shifts rank
        (after E is updated) is correctly skipped in favor of the next
        unprocessed extreme value. The middle pass recomputes its candidate
        list once per entry (including on resume) rather than re-sorting
        every iteration, matching the original single-run behavior; this
        recomputation is deterministic given the checkpointed df and index
        sets, so it's safe to repeat on resume without re-processing points.

        Acceleration
        ------------
        Each pass tracks a trust ledger (see `AccelerationConfig`) that
        shrinks how many of its target points actually get tested as
        validated points accumulate, and grows the tested count back if
        points start needing correction or refusal. Resets per pass;
        persisted in the checkpoint so a resume uses the exact effective
        target the interrupted run was using.

        Recovery
        --------
        If a checkpoint from an interrupted run exists and targets this same
        CSV, resumes from it: skips whichever passes are already marked
        complete, and continues the in-progress pass (if any) from its
        recorded position rather than restarting it. See the class docstring.

        Output
        ------
        Overwrites `csv_file_path` with the corrected dataframe.
        The untouched original is preserved as <stem>_original<ext>.
        A correction log is written to <stem>_correction.log alongside the CSV.
        """
        start_time = time.time()
        self._log(f"=== Database_Corrector starting: {self.csv_file_path.name} ===")

        # Backup original file
        backup_path = self.csv_file_path.with_stem(
            self.csv_file_path.stem + "_original"
        )
        if not backup_path.exists():
            shutil.copy(self.csv_file_path, backup_path)
            self._log(f"Original backed up as: {backup_path.name}")
        else:
            self._log(f"Backup already exists ({backup_path.name}), skipping copy.")

        self._setup_temp_dirs()
        self._airfoil_index = self._build_airfoil_index()

        try:
            checkpoint = self._load_checkpoint()
            if checkpoint is not None:
                df = pd.read_csv(self._checkpoint_df_path, index_col=0)
                total_rows = checkpoint['total_rows']
                n_target = checkpoint['n_target']
                passes_complete: set = set(checkpoint['passes_complete'])
                resumed_pass: Optional[str] = checkpoint['current_pass']
                resumed_processed = checkpoint['processed_this_pass']
                resumed_point_counter = checkpoint['point_counter']
                corrected_indices: set = set(checkpoint['corrected_indices'])
                refused_indices: set = set(checkpoint['refused_indices'])
                stats = checkpoint['stats']
                stats.setdefault('corrected', 0)
                # Per-pass acceleration ledger. .get(...) defaults cover a
                # checkpoint written before this field existed -- resumes
                # as if this pass had just started at accel_config.initial.
                resumed_accel_value = checkpoint.get('accel_value', self.accel_config.initial)
                resumed_accel_streak = checkpoint.get('accel_streak', 0)
                resumed_accel_prev_d = checkpoint.get('accel_prev_d', 0)
                self._log(
                    f"Resuming from checkpoint: passes_complete={sorted(passes_complete)} | "
                    f"in-progress={resumed_pass} | "
                    f"validated={stats['validated']} | corrected={stats['corrected']} | refused={stats['refused']}"
                )
            else:
                df = pd.read_csv(self.csv_file_path)
                total_rows = len(df)
                n_target = max(1, int(total_rows * self.top_bottom_fraction))
                passes_complete = set()
                resumed_pass = None
                resumed_processed = 0
                resumed_point_counter = 0
                corrected_indices = set()
                refused_indices = set()
                stats = {'validated': 0, 'corrected': 0, 'refused': 0}
                resumed_accel_value = self.accel_config.initial
                resumed_accel_streak = 0
                resumed_accel_prev_d = 0
                self._log(
                    f"Total rows: {total_rows} | "
                    f"Target per top/bottom pass: {n_target} "
                    f"({self.top_bottom_fraction:.1%})"
                )

            # ── Top and bottom passes (dynamic re-sort per iteration) ── #
            for direction, ascending in [('top', False), ('bottom', True)]:
                if direction in passes_complete:
                    self._log(f"\n=== {direction.upper()} pass already complete, skipping ===")
                    continue

                self._log(
                    f"\n=== {direction.upper()} pass "
                    f"({'descending' if not ascending else 'ascending'} E) ==="
                )
                if resumed_pass == direction:
                    processed_this_pass = resumed_processed
                    point_counter = resumed_point_counter
                    accel_value = resumed_accel_value
                    accel_streak = resumed_accel_streak
                    accel_prev_d = resumed_accel_prev_d
                    self._log(f"  Resuming at point {processed_this_pass + 1}/{n_target}")
                else:
                    processed_this_pass = 0
                    point_counter = 0
                    accel_value = self.accel_config.initial
                    accel_streak = 0
                    accel_prev_d = 0

                df, corrected_indices, refused_indices, stats = self._run_pass_with_gui(
                    pass_fn=self._run_top_bottom_pass,
                    pass_kwargs=dict(
                        df=df, direction=direction, ascending=ascending, n_target=n_target,
                        processed_this_pass=processed_this_pass, point_counter=point_counter,
                        corrected_indices=corrected_indices, refused_indices=refused_indices,
                        stats=stats, passes_complete=passes_complete, total_rows=total_rows,
                        accel_value=accel_value, accel_streak=accel_streak, accel_prev_d=accel_prev_d,
                    ),
                    n_target=n_target,
                    initial_completed=processed_this_pass,
                    window_title="Database Correction Progress",
                    header_text=f"{direction.capitalize()} pass ({'descending' if not ascending else 'ascending'} E)",
                )

            if self.middle_fraction > 0 and 'middle' not in passes_complete:
                n_middle = max(1, int(total_rows * self.middle_fraction))
                already_done = resumed_processed if resumed_pass == 'middle' else 0
                if resumed_pass == 'middle':
                    already_accel_value = resumed_accel_value
                    already_accel_streak = resumed_accel_streak
                    already_accel_prev_d = resumed_accel_prev_d
                else:
                    already_accel_value = self.accel_config.initial
                    already_accel_streak = 0
                    already_accel_prev_d = 0
                self._log(
                    f"\n=== MIDDLE pass: {n_middle} rows "
                    f"({self.middle_fraction:.1%}), selected by absolute "
                    f"deviation from median E ==="
                )
                if already_done:
                    self._log(f"  Resuming at point {already_done + 1}/{n_middle}")

                df, corrected_indices, refused_indices, stats = self._run_pass_with_gui(
                    pass_fn=self._run_middle_pass,
                    pass_kwargs=dict(
                        df=df, n_middle=n_middle, already_done=already_done,
                        corrected_indices=corrected_indices, refused_indices=refused_indices,
                        stats=stats, passes_complete=passes_complete,
                        total_rows=total_rows, n_target=n_target,
                        accel_value=already_accel_value, accel_streak=already_accel_streak,
                        accel_prev_d=already_accel_prev_d,
                    ),
                    n_target=n_middle,
                    initial_completed=already_done,
                    window_title="Database Correction Progress",
                    header_text="Middle pass (deviation from median E)",
                )

            # Overwrite the original CSV with corrected data
            df.to_csv(self.csv_file_path, index=False)
            self._log(f"\n=== Correction complete ===")
            self._log(f"Output saved to: {self.csv_file_path.name}")
            self._log(
                f"Validated: {stats['validated']} | "
                f"Corrected: {stats['corrected']} | "
                f"Refused: {stats['refused']} | "
                f"Remaining rows: {len(df)}/{total_rows}"
            )

            # Temp dir (including all checkpoint state) is only torn down on a
            # clean, full completion so it remains available for debugging
            # and resuming after a crash or interrupt.
            self._teardown_temp_dirs()

        except Exception as e:
            self._log(f"Critical error during correction: {e}", 'error')
            raise

        finally:
            elapsed = time.time() - start_time
            self._log(f"=== Finished in {elapsed:.2f}s ===")
            for handler in self._logger.handlers[:]:
                handler.close()
                self._logger.removeHandler(handler)


# =============================================================================================== #
# === Neighbour-consistency diagnosis and rectification === #
# =============================================================================================== #

GEO_CREATORS_BY_IDENTIFIER: Dict[str, Type] = {
    "rec": HossainWingRec_Creation,
    "flat": HossainWingFlat_Creation,
    "60DEG": HossainWing60DEG_Creation,
}
KEY_POINT_NAMES: List[str] = ["CLneg", "CDneg", "CLcdmin", "CDmin", "CLpos", "CDpos"]
STORE_FORMAT_VERSION = 1


def display_available() -> bool:
    """
    True when a graphical display can be used. On Linux this requires DISPLAY or WAYLAND_DISPLAY;
    Windows and macOS sessions are assumed to have one.
    """
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def grid_levels_from_ranges(n_cst: int, ranges_lower: dict, ranges_upper: dict) -> List[np.ndarray]:
    """
    Grid values of every CST coefficient (lower surface first, then upper), built exactly like
    geometric_support.create_modification_combinations: np.linspace(start, stop, round((stop - start) / step) + 1).
    Accepts the 'LE'/'inter'/'TE' and the older 'first'/'middle'/'last' key names.
    """
    def levels(rng):
        start, stop, step = rng
        if step == 0 or start == stop:
            return np.array([float(start)])
        return np.linspace(start, stop, int(round((stop - start) / step)) + 1)

    def surface(ranges):
        first = ranges.get('LE', ranges.get('first'))
        middle = ranges.get('inter', ranges.get('middle'))
        last = ranges.get('TE', ranges.get('last'))
        return [levels(first)] + [levels(middle)] * (n_cst - 2) + [levels(last)]

    return surface(ranges_lower) + surface(ranges_upper)


@dataclass
class Simulation_Settings:
    """
    Everything that defines one simulation of an airfoil: flow regime, support files, binaries and the
    optional pipeline variants (alpha sweep mode and key-point fit seed). The signature (see `signature`)
    excludes machine-specific paths, so stored runs stay reusable on another machine.
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
    sweep_mode: str = "sequential"
    kp_seed: Optional[int] = None
    xfoil_iter: int = 200

    def angles(self) -> List[float]:
        start, stop, step = self.aseq
        return [start + j * step for j in range(int(round((stop - start) / step)) + 1)]

    def signature(self) -> dict:
        return {
            "n_cst": self.n_cst, "aseq": list(self.aseq), "reynolds": self.reynolds, "mach": self.mach,
            "base_airfoil": Path(self.base_airfoil).name, "naca_kp_file": Path(self.naca_kp_file).name,
            "case_file": Path(self.case_file).name, "sweep_mode": self.sweep_mode, "kp_seed": self.kp_seed,
            "xfoil_iter": self.xfoil_iter,
        }

    def signature_id(self) -> str:
        return hashlib.sha1(json.dumps(self.signature(), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class Neighbour_Flags:
    """Result of one neighbour-consistency scan over the dataset rows."""
    neighbour_median: np.ndarray
    neighbour_count: np.ndarray
    deviation: np.ndarray
    flagged_rows: np.ndarray
    airfoil_keys: List[str] = field(default_factory=list)


# --- Worker process state (one airfoil task at a time per process) --- #
_WORKER = {"slot": 0, "progress": None, "status": None, "generators": {}}


def _rectifier_worker_init(slot_queue, progress_queue, status_list) -> None:
    _WORKER.update(slot=slot_queue.get(), progress=progress_queue, status=status_list, generators={})


def _worker_status(text: str) -> None:
    status = _WORKER["status"]
    if status is not None:
        try:
            status[_WORKER["slot"]] = text
        except Exception:
            pass


def _worker_progress() -> None:
    if _WORKER["progress"] is not None:
        _WORKER["progress"].put(1)


def _worker_generator(settings: Simulation_Settings) -> Airfoil_Generator:
    key = (settings.base_airfoil, settings.n_cst)
    if key not in _WORKER["generators"]:
        _WORKER["generators"][key] = Airfoil_Generator(base_airfoil_csv=settings.base_airfoil, n_cst=settings.n_cst, verbose=False)
    return _WORKER["generators"][key]


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


def _read_force_file(force_path: Path) -> Tuple[float, float]:
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


def _key_points(polar_path: Path, seed: Optional[int]) -> List[float]:
    low_cl, high_cl, min_cd = Key_Point_Extraction([str(polar_path)], random_state=seed)._interpolate_and_extract(str(polar_path))
    return [float(low_cl[0]), float(low_cl[1]), float(min_cd[0]), float(min_cd[1]), float(high_cl[0]), float(high_cl[1])]


def simulate_airfoil_once(settings: Simulation_Settings, airfoil_id: str, cst: List[float], configs: List[str],
                          run_dir: Path, kp_refits: int = 0, store_polar: bool = False) -> dict:
    """
    One complete, independent simulation of an airfoil: .dat generation -> XFOIL polar -> key points ->
    one AVL geometry and AVL run per wing configuration. Returns a JSON-ready run record.
    kp_refits > 0 repeats the key-point fit on the same polar to measure the fit's own variance.
    """
    t0 = time.time()
    record = {"status": "ok", "error": None, "polar_points": 0, "polar_hash": None, "key_points": None,
              "kp_refits": [], "CL": {}, "CD": {}, "E": {}, "seconds": None}
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    polar_dir, geo_dir, force_dir = run_dir / "polars", run_dir / "geo", run_dir / "forces"
    for d in (polar_dir, geo_dir, force_dir):
        d.mkdir(parents=True, exist_ok=True)
    try:
        _worker_status(f"{airfoil_id}: airfoil")
        generator = _worker_generator(settings)
        x, yu, yl = generator.generate(cst_coeffs=np.asarray(cst, dtype=float))
        if not check_validity(x, yu, yl, verbose=False):
            record["status"] = "invalid_geometry"
            return record
        generator._create_dat_file(output_dir=run_dir, n_cst=settings.n_cst, x=x, yl=yl, yu=yu, file_name=airfoil_id)
        dat_path = run_dir / f"{airfoil_id}.dat"

        _worker_status(f"{airfoil_id}: XFOIL ({settings.sweep_mode})")
        polar_path = polar_dir / f"polar_{airfoil_id}.txt"
        XPR(xfoil_exe_path=settings.xfoil_exe, main_dir=run_dir, airfoil_dir=run_dir, polar_dir=polar_dir,
            verbose=False).run_simulation(dat_path, str(polar_path), settings.angles(), visc=settings.reynolds,
                                          mach=settings.mach, iter=settings.xfoil_iter, sweep_mode=settings.sweep_mode)
        alpha, cl, cd = read_polar_arrays(polar_path)
        record["polar_points"] = int(len(alpha))
        record["polar_hash"] = hashlib.sha1(np.round(np.stack([alpha, cl, cd]), 6).tobytes()).hexdigest()[:16] if len(alpha) else None
        if store_polar:
            record["polar"] = {"alpha": alpha.tolist(), "CL": cl.tolist(), "CD": cd.tolist()}

        _worker_status(f"{airfoil_id}: key points")
        try:
            kp = _key_points(polar_path, settings.kp_seed)
        except Exception as e:
            record.update(status="key_points_failed", error=str(e))
            return record
        record["key_points"] = kp
        record["kp_refits"] = [_key_points(polar_path, settings.kp_seed) for _ in range(kp_refits)]

        placeholder = [{
            'file_path': f"{airfoil_id}.txt",
            'min_cl_point': {'CL': kp[0], 'CD': kp[1]},
            'max_cl_point': {'CL': kp[4], 'CD': kp[5]},
            'min_cd_point': {'CL': kp[2], 'CD': kp[3]},
        }]
        case_copy = run_dir / Path(settings.case_file).name
        shutil.copy(settings.case_file, case_copy)
        for identifier in configs:
            _worker_status(f"{airfoil_id}: AVL {identifier}")
            GEO_CREATORS_BY_IDENTIFIER[identifier](
                key_points_data=placeholder, airfoil_abs_paths=[str(dat_path)],
                naca_key_points_data=str(settings.naca_kp_file), output_dir=str(geo_dir), verbose=False, pb=False,
            ).create_geo_files()
            geo_path = geo_dir / f"{airfoil_id}_{identifier}_geo.avl"
            cl_tot = cd_tot = math.nan
            if geo_path.exists():
                AVL_Run_Case(geo_files=[str(geo_path)], case_file=str(case_copy), avl_exe_path=settings.avl_exe,
                             output_dir=force_dir).run()
                cl_tot, cd_tot = _read_force_file(force_dir / f"force_{airfoil_id}_{identifier}.txt")
            e = cl_tot / cd_tot if math.isfinite(cl_tot) and math.isfinite(cd_tot) and cd_tot != 0 else math.nan
            record["CL"][identifier] = None if math.isnan(cl_tot) else cl_tot
            record["CD"][identifier] = None if math.isnan(cd_tot) else cd_tot
            record["E"][identifier] = None if math.isnan(e) else e
        if any(v is None for v in record["E"].values()):
            record["status"] = "avl_failed" if all(v is None for v in record["E"].values()) else "partial"
        return record
    except Exception as e:
        record.update(status="error", error=f"{type(e).__name__}: {e}")
        return record
    finally:
        record["seconds"] = round(time.time() - t0, 2)
        shutil.rmtree(run_dir, ignore_errors=True)


def simulate_airfoil_task(task: dict) -> dict:
    """
    Pool entry point. Runs task['n_runs'] independent simulations of one airfoil inside this worker's own
    temp folder and reports one progress step per run. Only the first run performs the key-point refits.
    """
    settings = Simulation_Settings(**task["settings"])
    slot_dir = Path(task["temp_root"]) / f"w{_WORKER['slot']}"
    runs = []
    for k in range(task["n_runs"]):
        try:
            run = simulate_airfoil_once(settings, task["airfoil_id"], task["cst"], task["configs"], slot_dir / f"r{k}",
                                        kp_refits=task["kp_refits"] if k == 0 else 0, store_polar=task["store_polar"])
        except Exception as e:
            run = {"status": "error", "error": f"{type(e).__name__}: {e}", "E": {}}
        run.update(mode=task["mode"], workers=task["workers"], timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        runs.append(run)
        _worker_progress()
    _worker_status("idle")
    return {"key": task["key"], "runs": runs}


def cluster_values(values: List[float], rel_tol: float) -> List[List[float]]:
    """Groups sorted values whose consecutive gap is within rel_tol of the larger value; largest group first."""
    groups: List[List[float]] = []
    for v in sorted(values):
        if groups and abs(v - groups[-1][-1]) <= rel_tol * max(abs(v), abs(groups[-1][-1]), 1e-12):
            groups[-1].append(v)
        else:
            groups.append([v])
    return sorted(groups, key=len, reverse=True)


def majority_decision(values: List[float], n_runs: int, rel_tol: float, original: float) -> dict:
    """
    Decision for one dataset row from its repeated simulations.
      confirmed  -- a strict majority of the n_runs agrees and matches the dataset value
      corrected  -- a strict majority agrees on a different value (written to the dataset)
      unresolved -- no strict majority (dataset value kept, row reported)
      failed     -- no successful simulation (dataset value kept, row reported)
    """
    valid = [v for v in values if v is not None and math.isfinite(v)]
    if not valid:
        return {"outcome": "failed", "value": original, "agreement": 0.0, "n_valid": 0, "clusters": 0}
    groups = cluster_values(valid, rel_tol)
    top = groups[0]
    agreement = len(top) / n_runs
    if len(top) * 2 <= n_runs:
        return {"outcome": "unresolved", "value": original, "agreement": agreement, "n_valid": len(valid), "clusters": len(groups)}
    value = float(np.median(top))
    same = abs(value - original) <= rel_tol * max(abs(original), 1e-12)
    return {"outcome": "confirmed" if same else "corrected", "value": value, "agreement": agreement,
            "n_valid": len(valid), "clusters": len(groups)}


class Dataset_Rectifier:
    """
    Neighbour-consistency diagnosis and rectification of a gridded regressor dataset (CST coefficients +
    one-hot wing configuration + E).

    Outlier criterion
    -----------------
    Two dataset rows are neighbours when they share the wing configuration and their CST vectors differ by
    exactly one grid step in exactly one coefficient. A row is flagged when it has at least `min_neighbours`
    neighbours and |E - median(neighbour E)| > `threshold`. Work is done per airfoil (unique CST vector): all
    of its configuration rows are re-simulated together, since they share the polar and key points.

    Repeated simulation and decision
    --------------------------------
    Every selected airfoil is simulated `n_runs` times, each run fully independent (new XFOIL and AVL
    processes, own temp folder). Each row is decided by strict majority (see `majority_decision`).

    Evaluation store
    ----------------
    Every run is kept in `<work_dir>/evaluations.json`, grouped by the pipeline signature of
    Simulation_Settings. Diagnosis and rectification both read and extend it, so rectification only
    simulates what a previous diagnosis (or an interrupted rectification) has not already simulated.
    Runs record whether they came from the process pool or from single-process mode.
    """

    def __init__(self, csv_path, settings: Simulation_Settings, work_dir, temp_root,
                 grid_levels: Optional[List[np.ndarray]] = None, threshold: float = 15.0, min_neighbours: int = 3,
                 n_runs: int = 5, agreement_tolerance: float = 0.05, workers: Optional[int] = None,
                 single_process: bool = False, use_gui: Optional[bool] = None, log: Callable[[str], None] = print):
        self.csv_path = Path(csv_path)
        self.settings = settings
        self.work_dir = Path(work_dir)
        self.temp_root = Path(temp_root)
        self.threshold = threshold
        self.min_neighbours = min_neighbours
        self.n_runs = n_runs
        self.tol = agreement_tolerance
        self.single_process = single_process
        self.workers = 1 if single_process else max(1, workers or (multiprocessing.cpu_count() - 1))
        self.use_gui = display_available() if use_gui is None else use_gui
        self.log = log
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.df = pd.read_csv(self.csv_path, float_precision="round_trip")
        self.target = self.df.columns[-1]
        features = list(self.df.columns[:-1])
        self.cst_cols = (sorted([c for c in features if re.fullmatch(r'cst_l_\d+', c)], key=lambda c: int(c.rsplit('_', 1)[1])) +
                         sorted([c for c in features if re.fullmatch(r'cst_u_\d+', c)], key=lambda c: int(c.rsplit('_', 1)[1])))
        self.config_cols = [c for c in features if c not in self.cst_cols]
        if not self.config_cols:
            raise NotImplementedError("Single-configuration datasets are not supported yet: their wing geometry creator "
                                      "(RecWingGeo_Creation) has not been updated to the current interface.")
        unknown = [c for c in self.config_cols if c not in GEO_CREATORS_BY_IDENTIFIER]
        if unknown:
            raise ValueError(f"No geometry creator registered for configuration column(s) {unknown}. "
                             f"Known: {list(GEO_CREATORS_BY_IDENTIFIER)}")
        if len(self.cst_cols) != 2 * settings.n_cst:
            raise ValueError(f"Dataset has {len(self.cst_cols)} CST columns but n_cst = {settings.n_cst}.")

        self.E = self.df[self.target].to_numpy(dtype=float)
        self.row_config = np.array(self.config_cols)[self.df[self.config_cols].to_numpy().argmax(axis=1)]
        self._build_grid(grid_levels)
        self._build_neighbours()
        self.store_path = self.work_dir / "evaluations.json"
        self.store = self._load_store()

    # --- Dataset structure --- #

    def _build_grid(self, grid_levels: Optional[List[np.ndarray]]) -> None:
        """Snaps CST values to the exact generation grid and indexes every row by its grid position."""
        raw = self.df[self.cst_cols].to_numpy(dtype=float)
        if grid_levels is None:
            grid_levels = [np.unique(np.round(raw[:, j], 6)) for j in range(raw.shape[1])]
        self.grid_levels = [np.asarray(g, dtype=float) for g in grid_levels]
        level_idx = np.zeros(raw.shape, dtype=np.int64)
        exact = np.zeros(raw.shape)
        self.snap_distance = 0.0
        for j, levels in enumerate(self.grid_levels):
            idx = np.abs(raw[:, j, None] - levels[None, :]).argmin(axis=1)
            level_idx[:, j] = idx
            exact[:, j] = levels[idx]
            self.snap_distance = max(self.snap_distance, float(np.abs(raw[:, j] - levels[idx]).max()))
        steps = [float(np.min(np.diff(g))) if len(g) > 1 else 1.0 for g in self.grid_levels]
        if self.snap_distance > 0.25 * min(steps):
            raise ValueError(f"Dataset CST values are up to {self.snap_distance:.4g} away from the grid levels. "
                             f"The CST ranges do not describe this dataset.")
        self.snapped_values = int((np.abs(raw - exact) > 1e-12).sum())
        self.cst_exact = exact
        self.level_idx = level_idx

        keys = [",".join(f"{v:.10f}" for v in row) for row in exact]
        self.row_key = np.array(keys)
        self.airfoils: Dict[str, dict] = {}
        for i, key in enumerate(keys):
            entry = self.airfoils.setdefault(key, {"cst": exact[i].tolist(), "rows": [],
                                                   "id": "a" + hashlib.sha1(key.encode()).hexdigest()[:8]})
            entry["rows"].append(i)

    def _build_neighbours(self) -> None:
        """(rows, 2 x coefficients) matrix of neighbour row indices, -1 where the neighbour is missing."""
        n_levels = np.array([len(g) for g in self.grid_levels], dtype=np.int64)
        radix = np.cumprod(np.concatenate([[1], n_levels[:-1]]))
        config_code = pd.Index(self.config_cols).get_indexer(self.row_config).astype(np.int64)
        code = (self.level_idx * radix).sum(axis=1) + config_code * int(np.prod(n_levels))
        order = np.argsort(code)
        sorted_code = code[order]
        neighbours = np.full((len(code), 2 * len(n_levels)), -1, dtype=np.int64)
        for j in range(len(n_levels)):
            for s, d in enumerate((-1, 1)):
                target_level = self.level_idx[:, j] + d
                valid = (target_level >= 0) & (target_level < n_levels[j])
                target = code + d * radix[j]
                pos = np.clip(np.searchsorted(sorted_code, target), 0, len(code) - 1)
                hit = valid & (sorted_code[pos] == target)
                neighbours[hit, 2 * j + s] = order[pos[hit]]
        self.neighbours = neighbours

    def flag(self, E: np.ndarray) -> Neighbour_Flags:
        """Neighbour-consistency scan of the E values given (dataset or partially rectified)."""
        nb_values = np.where(self.neighbours >= 0, E[np.clip(self.neighbours, 0, None)], np.nan)
        count = np.isfinite(nb_values).sum(axis=1)
        median = np.full(len(E), np.nan)
        has = count > 0
        median[has] = np.nanmedian(nb_values[has], axis=1)
        deviation = E - median
        flagged = (count >= self.min_neighbours) & (np.abs(deviation) > self.threshold)
        severity: Dict[str, float] = {}
        for i in np.flatnonzero(flagged):
            key = self.row_key[i]
            severity[key] = max(severity.get(key, 0.0), abs(float(deviation[i])))
        keys = sorted(severity, key=severity.get, reverse=True)
        return Neighbour_Flags(median, count, deviation, flagged, keys)

    # --- Evaluation store --- #

    def _load_store(self) -> dict:
        if self.store_path.exists():
            with open(self.store_path, 'r') as f:
                store = json.load(f)
        else:
            store = {"format": STORE_FORMAT_VERSION, "dataset_csv": self.csv_path.name, "signatures": {}, "evaluations": {}}
        sid = self.settings.signature_id()
        store["signatures"][sid] = self.settings.signature()
        store["evaluations"].setdefault(sid, {})
        return store

    def _save_store(self) -> None:
        tmp = self.store_path.with_suffix(".json.tmp")
        with open(tmp, 'w') as f:
            json.dump(self.store, f)
        os.replace(tmp, self.store_path)

    def runs_for(self, key: str) -> List[dict]:
        return self.store["evaluations"][self.settings.signature_id()].get(key, {}).get("runs", [])

    def other_signatures(self) -> Dict[str, Tuple[dict, int]]:
        """Stored pipeline signatures other than the current one, with their number of evaluated airfoils."""
        sid = self.settings.signature_id()
        return {s: (sig, len(self.store["evaluations"].get(s, {}))) for s, sig in self.store["signatures"].items() if s != sid}

    # --- Execution --- #

    def evaluate(self, requests: Dict[str, int], kp_refits: int = 0, store_polar: bool = False,
                 title: str = "Dataset rectification") -> None:
        """
        Simulates every requested airfoil the requested number of additional times (key -> runs) and appends
        the runs to the evaluation store. Uses the process pool or single-process mode, with the progress GUI
        when a display is available and a console progress bar otherwise.
        """
        requests = {k: n for k, n in requests.items() if n > 0}
        if not requests:
            return
        sid = self.settings.signature_id()
        mode = "single" if self.single_process else "pool"
        workers = 1 if self.single_process else min(self.workers, len(requests))
        settings = asdict(self.settings)
        tasks = []
        for key, n in requests.items():
            airfoil = self.airfoils[key]
            has_refits = any(r.get("kp_refits") for r in self.runs_for(key))
            tasks.append({"key": key, "airfoil_id": airfoil["id"], "cst": airfoil["cst"], "configs": list(self.config_cols),
                          "n_runs": n, "kp_refits": 0 if has_refits else kp_refits, "store_polar": store_polar,
                          "settings": settings, "temp_root": str(self.temp_root), "mode": mode, "workers": workers})
        total = sum(t["n_runs"] for t in tasks)
        header = (f"{len(tasks)} airfoil(s), {total} simulation(s) | {mode} mode, {workers} worker(s) | "
                  f"sweep: {self.settings.sweep_mode}, key-point seed: {self.settings.kp_seed}")
        self.log(f"[RUN] {header}")
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

        state = {"error": None, "done": 0, "last_save": time.time()}

        def store_result(result: dict) -> None:
            entry = self.store["evaluations"][sid].setdefault(result["key"], {"cst": self.airfoils[result["key"]]["cst"], "runs": []})
            entry["runs"].extend(result["runs"])
            state["done"] += 1
            if state["done"] % 5 == 0 or time.time() - state["last_save"] > 60:
                self._save_store()
                state["last_save"] = time.time()

        try:
            if self.single_process:
                progress_queue = queue.Queue()
                status_list = ["idle"]
                _WORKER.update(slot=0, progress=progress_queue, status=status_list, generators={})

                def job():
                    try:
                        for task in tasks:
                            store_result(simulate_airfoil_task(task))
                    except BaseException as e:
                        state["error"] = e
                    finally:
                        for _ in range(total):
                            progress_queue.put(1)

                self._run_with_progress(job, total, progress_queue, status_list, 1, title, header)
            else:
                with multiprocessing.Manager() as manager:
                    progress_queue = manager.Queue()
                    status_list = manager.list(["idle"] * workers)
                    slot_queue = manager.Queue()
                    for i in range(workers):
                        slot_queue.put(i)
                    pool = multiprocessing.Pool(processes=workers, initializer=_rectifier_worker_init,
                                                initargs=(slot_queue, progress_queue, status_list))

                    def job():
                        try:
                            for result in pool.imap_unordered(simulate_airfoil_task, tasks):
                                store_result(result)
                        except BaseException as e:
                            state["error"] = e
                        finally:
                            for _ in range(total):
                                progress_queue.put(1)

                    try:
                        self._run_with_progress(job, total, progress_queue, status_list, workers, title, header)
                    finally:
                        pool.terminate()
                        pool.join()
        finally:
            self._save_store()
            shutil.rmtree(self.temp_root, ignore_errors=True)
        if state["error"] is not None:
            raise state["error"]

    def _run_with_progress(self, job: Callable[[], None], total: int, progress_queue, status_list, n_workers: int,
                           title: str, header: str) -> None:
        """Runs `job` in a background thread while the main thread shows its progress."""
        thread = threading.Thread(target=job, daemon=True)
        gui = None
        if self.use_gui and Progress_GUI is not None:
            try:
                gui = Progress_GUI(total_files=total, progress_queue=progress_queue, worker_status_list=status_list,
                                   num_workers=n_workers, window_title=title, header_text=header, detailed_only=True)
            except Exception as e:
                self.log(f"[INFO] Progress window unavailable ({e}); using the console progress bar.")
                gui = None
        thread.start()
        if gui is not None:
            gui.run()
        else:
            done = 0
            with tqdm(total=total, desc="Simulations", unit="run") as bar:
                while thread.is_alive() or done < total:
                    try:
                        progress_queue.get(timeout=0.5)
                        step = min(1, total - done)
                        done += step
                        bar.update(step)
                    except queue.Empty:
                        if not thread.is_alive():
                            break
                    if done >= total:
                        break
        thread.join()

    # --- Analysis --- #

    def _row_decisions(self, key: str, runs: List[dict]) -> Dict[int, dict]:
        """Majority decision for every dataset row of one airfoil, from its first n_runs runs."""
        runs = runs[:self.n_runs]
        decisions = {}
        for row in self.airfoils[key]["rows"]:
            identifier = self.row_config[row]
            values = [r.get("E", {}).get(identifier) for r in runs]
            decisions[row] = majority_decision(values, self.n_runs, self.tol, float(self.E[row]))
            decisions[row]["runs_E"] = values
        return decisions

    def _airfoil_diagnostics(self, key: str, runs: List[dict]) -> dict:
        """Where the run-to-run variation of one airfoil comes from: XFOIL polar, key-point fit, or neither."""
        runs = runs[:self.n_runs]
        hashes = [r.get("polar_hash") for r in runs if r.get("polar_hash")]
        polar_identical = len(hashes) >= 2 and len(set(hashes)) == 1
        polar_delta = None
        polars = [r["polar"] for r in runs if r.get("polar")]
        if len(polars) >= 2:
            common = set(polars[0]["alpha"])
            for p in polars[1:]:
                common &= set(p["alpha"])
            if common:
                def at(p, name):
                    lookup = dict(zip(p["alpha"], p[name]))
                    return np.array([lookup[a] for a in sorted(common)])
                polar_delta = {name: float(np.max(np.ptp(np.stack([at(p, name) for p in polars]), axis=0))) for name in ("CL", "CD")}
        kps = np.array([r["key_points"] for r in runs if r.get("key_points")])
        kp_run_spread = dict(zip(KEY_POINT_NAMES, np.ptp(kps, axis=0).tolist())) if len(kps) >= 2 else None
        refits = next((r["kp_refits"] + [r["key_points"]] for r in runs if r.get("kp_refits")), None)
        kp_refit_spread = dict(zip(KEY_POINT_NAMES, np.ptp(np.array(refits), axis=0).tolist())) if refits else None
        decisions = self._row_decisions(key, runs)
        repeatable = all(d["clusters"] <= 1 and d["n_valid"] == len(runs) for d in decisions.values())
        if repeatable:
            source = "repeatable"
        elif not polar_identical and len(hashes) >= 2:
            source = "xfoil"
        elif any(r.get("status") != "ok" for r in runs):
            source = "simulation_failures"
        elif kp_refit_spread and max(kp_refit_spread.values()) > 0:
            source = "key_point_fit"
        else:
            source = "avl_or_geometry"
        return {"polar_identical": polar_identical, "polar_runs": len(hashes), "polar_points": [r.get("polar_points") for r in runs],
                "polar_max_delta": polar_delta, "kp_run_spread": kp_run_spread, "kp_refit_spread": kp_refit_spread,
                "repeatable": repeatable, "variation_source": source, "decisions": decisions,
                "statuses": [r.get("status") for r in runs], "modes": sorted({r.get("mode", "pool") for r in runs})}

    def _mode_study(self, keys: List[str]) -> List[dict]:
        """Repeatability split by execution mode (pool vs single process), for airfoils with >= 2 runs in a mode."""
        table = []
        for mode in ("pool", "single"):
            stable = total = 0
            for key in keys:
                runs = [r for r in self.runs_for(key) if r.get("mode", "pool") == mode]
                if len(runs) < 2:
                    continue
                total += 1
                ok = True
                for row in self.airfoils[key]["rows"]:
                    vals = [r.get("E", {}).get(self.row_config[row]) for r in runs]
                    vals = [v for v in vals if v is not None]
                    if len(vals) != len(runs) or len(cluster_values(vals, self.tol)) > 1:
                        ok = False
                stable += ok
            if total:
                table.append({"mode": mode, "airfoils": total, "repeatable": stable, "share": stable / total})
        return table

    def _flag_summary(self, flags: Neighbour_Flags) -> dict:
        dev = flags.deviation[flags.flagged_rows]
        return {"rows": int(len(self.E)), "rows_flagged": int(flags.flagged_rows.sum()),
                "flagged_above_neighbours": int((dev > 0).sum()), "flagged_below_neighbours": int((dev < 0).sum()),
                "airfoils": len(self.airfoils), "airfoils_flagged": len(flags.airfoil_keys),
                "mean_neighbours": float(flags.neighbour_count.mean())}

    # --- Modes --- #

    def diagnose(self, n_flagged: int = 20, n_controls: int = 10, n_top: int = 5, all_flagged: bool = False,
                 kp_refits: int = 5, seed: int = 0) -> dict:
        """
        Re-simulates a sample of airfoils n_runs times each: the most strongly flagged ones (or all flagged
        ones), the top-E dataset airfoils and a random sample of unflagged controls. Writes a report, a
        per-row CSV and a JSON analysis to work_dir, and extends the evaluation store. Returns the paths.
        """
        flags = self.flag(self.E)
        groups: Dict[str, set] = {}
        for key in (flags.airfoil_keys if all_flagged else flags.airfoil_keys[:n_flagged]):
            groups.setdefault(key, set()).add("flagged")
        best_per_airfoil = {}
        for i, key in enumerate(self.row_key):
            best_per_airfoil[key] = max(best_per_airfoil.get(key, -np.inf), self.E[i])
        for key in sorted(best_per_airfoil, key=best_per_airfoil.get, reverse=True)[:n_top]:
            groups.setdefault(key, set()).add("top")
        flagged_set = set(flags.airfoil_keys)
        unflagged = [k for k in self.airfoils if k not in flagged_set]
        rng = np.random.default_rng(seed)
        for key in rng.choice(unflagged, size=min(n_controls, len(unflagged)), replace=False).tolist() if unflagged else []:
            groups.setdefault(key, set()).add("control")

        requests = {k: self.n_runs - len(self.runs_for(k)) for k in groups}
        reused = sum(1 for k in groups if requests[k] <= 0)
        self.log(f"[DIAGNOSE] {len(groups)} airfoil(s) selected; {reused} already have {self.n_runs} stored run(s).")
        self.evaluate(requests, kp_refits=kp_refits, store_polar=True, title="Dataset diagnosis")

        analysis = {key: self._airfoil_diagnostics(key, self.runs_for(key)) for key in groups}
        summary = self._diagnosis_summary(flags, groups, analysis)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        paths = {
            "report": self.work_dir / f"diagnosis_{stamp}_report.txt",
            "rows": self.work_dir / f"diagnosis_{stamp}_rows.csv",
            "json": self.work_dir / f"diagnosis_{stamp}.json",
            "store": self.store_path,
        }
        self._rows_table(analysis, groups).to_csv(paths["rows"], index=False)
        with open(paths["json"], 'w') as f:
            json.dump(_jsonable({"settings": self.settings.signature(), "signature_id": self.settings.signature_id(),
                                 "summary": summary, "airfoils": {k: {"id": self.airfoils[k]["id"], "groups": sorted(groups[k]), **v}
                                                                  for k, v in analysis.items()}}), f, indent=2)
        paths["report"].write_text(self._diagnosis_report(summary, analysis, groups), encoding='utf-8')
        return paths

    def rectify(self, output_csv, max_rounds: int = 5) -> dict:
        """
        Rounds of: flag on the current E -> simulate flagged airfoils that have fewer than n_runs stored runs ->
        majority decision per row -> update E. Airfoils decided in an earlier round are not re-flagged.
        Stops when a round has no new flagged airfoils or after max_rounds. Writes the rectified CSV (only the
        target column changes) plus a report, a per-row CSV and a JSON of decisions. Returns the paths.
        """
        E_current = self.E.copy()
        decided: Dict[str, Dict[int, dict]] = {}
        rounds = []
        for rnd in range(1, max_rounds + 1):
            flags = self.flag(E_current)
            keys = [k for k in flags.airfoil_keys if k not in decided]
            if not keys:
                self.log(f"[RECTIFY] Round {rnd}: no new flagged airfoils. Done.")
                break
            requests = {k: self.n_runs - len(self.runs_for(k)) for k in keys}
            reused = sum(1 for n in requests.values() if n <= 0)
            self.log(f"[RECTIFY] Round {rnd}: {int(flags.flagged_rows.sum())} flagged row(s) in {len(keys)} new airfoil(s); "
                     f"{reused} reuse stored runs, {len(keys) - reused} need simulation.")
            self.evaluate(requests, title=f"Dataset rectification - round {rnd}")
            outcomes = {"confirmed": 0, "corrected": 0, "unresolved": 0, "failed": 0}
            for key in keys:
                decided[key] = self._row_decisions(key, self.runs_for(key))
                for row, d in decided[key].items():
                    outcomes[d["outcome"]] += 1
                    if d["outcome"] in ("confirmed", "corrected"):
                        E_current[row] = d["value"]
            rounds.append({"round": rnd, "flagged_rows": int(flags.flagged_rows.sum()), "new_airfoils": len(keys),
                           "reused_airfoils": reused, **outcomes})
        final_flags = self.flag(E_current)

        rectified = self.df.copy()
        rectified[self.target] = E_current
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        rectified.to_csv(output_csv, index=False)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        paths = {
            "dataset": output_csv,
            "report": self.work_dir / f"rectification_{stamp}_report.txt",
            "rows": self.work_dir / f"rectification_{stamp}_rows.csv",
            "json": self.work_dir / f"rectification_{stamp}.json",
            "store": self.store_path,
        }
        rows = []
        for key, decisions in decided.items():
            for row, d in decisions.items():
                rows.append({"airfoil": self.airfoils[key]["id"], "row": row, "config": self.row_config[row],
                             **{c: self.df.iat[row, self.df.columns.get_loc(c)] for c in self.cst_cols},
                             "dataset_E": self.E[row], "rectified_E": E_current[row], "outcome": d["outcome"],
                             "agreement": d["agreement"], "valid_runs": d["n_valid"], "clusters": d["clusters"],
                             "runs_E": ";".join("" if v is None else f"{v:.4f}" for v in d["runs_E"])})
        pd.DataFrame(rows).to_csv(paths["rows"], index=False)
        summary = {"settings": self.settings.signature(), "signature_id": self.settings.signature_id(),
                   "n_runs": self.n_runs, "threshold": self.threshold, "min_neighbours": self.min_neighbours,
                   "agreement_tolerance": self.tol, "rounds": rounds,
                   "initial_flags": self._flag_summary(self.flag(self.E)), "final_flags": self._flag_summary(final_flags),
                   "dataset_best_before": float(np.max(self.E)), "dataset_best_after": float(np.max(E_current)),
                   "rows_changed": int((np.abs(E_current - self.E) > 1e-12).sum())}
        with open(paths["json"], 'w') as f:
            json.dump(_jsonable({**summary, "decisions": {self.airfoils[k]["id"]: {"cst": self.airfoils[k]["cst"],
                                                                                    "rows": {str(r): d for r, d in v.items()}}
                                                           for k, v in decided.items()}}), f, indent=2)
        paths["report"].write_text(self._rectification_report(summary, E_current, decided), encoding='utf-8')
        return paths

    # --- Reports --- #

    def _rows_table(self, analysis: Dict[str, dict], groups: Dict[str, set]) -> pd.DataFrame:
        rows = []
        for key, a in analysis.items():
            for row, d in a["decisions"].items():
                rows.append({"airfoil": self.airfoils[key]["id"], "groups": "+".join(sorted(groups[key])),
                             "config": self.row_config[row], **{c: self.cst_exact[row, j] for j, c in enumerate(self.cst_cols)},
                             "dataset_E": self.E[row], "majority_E": d["value"] if d["outcome"] in ("confirmed", "corrected") else None,
                             "outcome": d["outcome"], "agreement": d["agreement"], "clusters": d["clusters"],
                             "variation_source": a["variation_source"], "polar_identical": a["polar_identical"],
                             "runs_E": ";".join("" if v is None else f"{v:.4f}" for v in d["runs_E"])})
        return pd.DataFrame(rows)

    def _diagnosis_summary(self, flags: Neighbour_Flags, groups: Dict[str, set], analysis: Dict[str, dict]) -> dict:
        def share(keys, predicate):
            keys = list(keys)
            return (sum(1 for k in keys if predicate(analysis[k])) / len(keys)) if keys else None

        by_group = {}
        for group in ("flagged", "top", "control"):
            keys = [k for k in analysis if group in groups[k]]
            rows = [d for k in keys for d in analysis[k]["decisions"].values()]
            by_group[group] = {
                "airfoils": len(keys),
                "repeatable": share(keys, lambda a: a["repeatable"]),
                "rows": len(rows),
                "rows_reproduced": (sum(d["outcome"] == "confirmed" for d in rows) / len(rows)) if rows else None,
                "rows_corrected": sum(d["outcome"] == "corrected" for d in rows),
                "rows_unresolved": sum(d["outcome"] == "unresolved" for d in rows),
                "rows_failed": sum(d["outcome"] == "failed" for d in rows),
            }
        sources: Dict[str, int] = {}
        for a in analysis.values():
            sources[a["variation_source"]] = sources.get(a["variation_source"], 0) + 1
        refit_keys = [k for k in analysis if analysis[k]["kp_refit_spread"] is not None]
        return {
            "settings": self.settings.signature(), "signature_id": self.settings.signature_id(),
            "n_runs": self.n_runs, "threshold": self.threshold, "min_neighbours": self.min_neighbours,
            "agreement_tolerance": self.tol, "snapped_values": self.snapped_values, "max_snap_distance": self.snap_distance,
            "flags": self._flag_summary(flags), "groups": by_group, "variation_sources": sources,
            "polar_identical_share": share([k for k in analysis if analysis[k]["polar_runs"] >= 2], lambda a: a["polar_identical"]),
            "kp_refit_variable_share": share(refit_keys, lambda a: max(a["kp_refit_spread"].values()) > 0),
            "mode_study": self._mode_study(list(analysis)),
            "other_signatures": {s: {"signature": sig, "airfoils": n} for s, (sig, n) in self.other_signatures().items()},
        }

    def _diagnosis_report(self, summary: dict, analysis: Dict[str, dict], groups: Dict[str, set]) -> str:
        pct = lambda v: "-" if v is None else f"{100 * v:.1f}%"
        f = summary["flags"]
        lines = ["=" * 110, f"Dataset diagnosis: {self.csv_path.name}", "=" * 110,
                 f"Pipeline: {json.dumps(summary['settings'])}",
                 f"Signature: {summary['signature_id']} | runs per airfoil: {self.n_runs} | agreement tolerance: {pct(self.tol)}",
                 f"CST values snapped to the generation grid: {summary['snapped_values']} (max shift {summary['max_snap_distance']:.2e})",
                 "",
                 "--- Neighbour-consistency scan " + "-" * 79,
                 f"Rows: {f['rows']} | mean neighbours per row: {f['mean_neighbours']:.1f} | threshold: |E - neighbour median| > {self.threshold}",
                 f"Flagged rows: {f['rows_flagged']} ({f['flagged_above_neighbours']} above, {f['flagged_below_neighbours']} below their neighbours)",
                 f"Flagged airfoils: {f['airfoils_flagged']} of {f['airfoils']}",
                 "",
                 "--- Repeatability (same airfoil, independent runs) " + "-" * 59,
                 f"Identical XFOIL polars across runs: {pct(summary['polar_identical_share'])} of airfoils with >= 2 successful polars",
                 f"Key-point fit changes when refitted on the same polar: {pct(summary['kp_refit_variable_share'])} of airfoils",
                 "Source of run-to-run variation: " + ", ".join(f"{k}: {v}" for k, v in sorted(summary["variation_sources"].items())),
                 ""]
        lines.append(f"{'Group':<10} {'Airfoils':>8} {'Repeatable':>10} {'Rows':>5} {'Reproduced':>10} {'Corrected':>9} {'Unresolved':>10} {'Failed':>6}")
        for g, s in summary["groups"].items():
            lines.append(f"{g:<10} {s['airfoils']:>8} {pct(s['repeatable']):>10} {s['rows']:>5} {pct(s['rows_reproduced']):>10} "
                         f"{s['rows_corrected']:>9} {s['rows_unresolved']:>10} {s['rows_failed']:>6}")
        if summary["mode_study"]:
            lines += ["", "--- Execution mode study " + "-" * 85]
            for m in summary["mode_study"]:
                lines.append(f"{m['mode']:<7} airfoils with >= 2 runs: {m['airfoils']:>4} | repeatable: {m['repeatable']:>4} ({pct(m['share'])})")
        if summary["other_signatures"]:
            lines += ["", "--- Other pipelines in the evaluation store " + "-" * 66]
            for s, o in summary["other_signatures"].items():
                lines.append(f"{s}: {o['airfoils']} airfoil(s) | {json.dumps(o['signature'])}")
        lines += ["", "--- Airfoils " + "-" * 97,
                  f"{'Airfoil':<10} {'Groups':<16} {'Cfg':<6} {'Dataset E':>9} {'Majority':>9} {'Outcome':<11} {'Agree':>6} {'Source':<20} Runs E"]
        order = sorted(analysis, key=lambda k: -max(self.E[r] for r in self.airfoils[k]["rows"]))
        for key in order:
            a = analysis[key]
            for row, d in sorted(a["decisions"].items(), key=lambda x: self.row_config[x[0]]):
                majority = f"{d['value']:.2f}" if d["outcome"] in ("confirmed", "corrected") else "-"
                runs = " ".join("fail" if v is None else f"{v:.2f}" for v in d["runs_E"])
                lines.append(f"{self.airfoils[key]['id']:<10} {'+'.join(sorted(groups[key])):<16} {self.row_config[row]:<6} "
                             f"{self.E[row]:>9.2f} {majority:>9} {d['outcome']:<11} {pct(d['agreement']):>6} {a['variation_source']:<20} {runs}")
        lines.append("=" * 110)
        return "\n".join(lines) + "\n"

    def _rectification_report(self, summary: dict, E_current: np.ndarray, decided: Dict[str, Dict[int, dict]]) -> str:
        fi, ff = summary["initial_flags"], summary["final_flags"]
        lines = ["=" * 110, f"Dataset rectification: {self.csv_path.name}", "=" * 110,
                 f"Pipeline: {json.dumps(summary['settings'])}",
                 f"Signature: {summary['signature_id']} | runs per airfoil: {self.n_runs} | threshold: {self.threshold} | "
                 f"agreement tolerance: {100 * self.tol:.1f}%", "",
                 f"{'Round':>5} {'Flagged rows':>12} {'New airfoils':>12} {'Reused':>6} {'Confirmed':>9} {'Corrected':>9} {'Unresolved':>10} {'Failed':>6}"]
        for r in summary["rounds"]:
            lines.append(f"{r['round']:>5} {r['flagged_rows']:>12} {r['new_airfoils']:>12} {r['reused_airfoils']:>6} "
                         f"{r['confirmed']:>9} {r['corrected']:>9} {r['unresolved']:>10} {r['failed']:>6}")
        lines += ["",
                  f"Flagged rows before: {fi['rows_flagged']} | after: {ff['rows_flagged']} (the remaining ones were confirmed, unresolved or failed)",
                  f"Rows changed: {summary['rows_changed']}",
                  f"Dataset best E before: {summary['dataset_best_before']:.4f} | after: {summary['dataset_best_after']:.4f}",
                  "", "E distribution (rows)    before     after"]
        for lo, hi in ((50, np.inf), (45, 50), (40, 45), (30, 40), (-np.inf, 30)):
            label = f"E >= {lo}" if hi == np.inf else (f"E < {hi}" if lo == -np.inf else f"{lo} <= E < {hi}")
            before = int(((self.E >= lo) & (self.E < hi)).sum())
            after = int(((E_current >= lo) & (E_current < hi)).sum())
            lines.append(f"{label:<22} {before:>8} {after:>9}")
        lines += ["", "--- Top 15 rows after rectification " + "-" * 73,
                  f"{'Row':>7} {'Cfg':<6} {'Dataset E':>9} {'Rectified':>9}  Status"]
        decided_rows = {row: d for decisions in decided.values() for row, d in decisions.items()}
        for row in np.argsort(-E_current)[:15]:
            status = decided_rows[row]["outcome"] if row in decided_rows else "not flagged"
            lines.append(f"{row:>7} {self.row_config[row]:<6} {self.E[row]:>9.4f} {E_current[row]:>9.4f}  {status}")
        lines.append("=" * 110)
        return "\n".join(lines) + "\n"


def _jsonable(value):
    """Recursively converts numpy values, sets and NaN into plain JSON values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
