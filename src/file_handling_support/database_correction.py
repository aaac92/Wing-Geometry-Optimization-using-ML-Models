import json
import logging
import os
import queue
import re
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type

import pandas as pd
from tqdm import tqdm

from simulation_support import Xfoil_Polar_Creator as XPR
from simulation_support import Key_Point_Extraction, AVL_Run_Case
from geometric_support import Airfoil_Generator
from ux_support import Progress_GUI


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
        error_threshold: float = 0.02,
        max_attempts: int = 6,
        top_bottom_fraction: float = 0.01,
        middle_fraction: float = 0.0,
        source_airfoils_dir=None,
        temp_dir=None,
        verbose: bool = False,
        append_kp: bool = False,
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
        bracketed, comma-separated text header first, falls back to raw
        float32 binary. Returns [] if neither yields `expected_count` values.
        """
        try:
            with open(file_path, "rb") as f:
                raw = f.read(1024)
            text = raw.decode(errors="ignore")

            match = re.search(r"\[([^\]]+)\]", text)
            if match:
                numbers = match.group(1).split(",")
                values = [float(n.strip()) for n in numbers if n.strip() != ""]
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

    def _source_airfoil(self, row: pd.Series, label: str) -> Path:
        """
        Resolves the airfoil .dat file for one point. Called once per point
        (not per attempt -- CST-defined geometry is invariant across
        retries of the same point). Reuses an already-sourced file on disk
        first (covers resuming mid-point after a crash), then tries the
        prebuilt index, falling back to CST regeneration whenever no
        unambiguous match exists -- mirrors current behavior exactly, just
        without repeating the work on every attempt.
        """
        airfoil_path = self._airfoils_temp / f"{label}.dat"
        if airfoil_path.exists():
            return airfoil_path

        cst_coeffs = (
            list(row[self._cst_l_cols].values) +
            list(row[self._cst_u_cols].values)
        )

        if self._airfoil_index:
            key = tuple(round(v, self._CST_MATCH_DECIMALS) for v in cst_coeffs)
            matches = self._airfoil_index.get(key, [])
            if len(matches) == 1:
                shutil.copy(matches[0], airfoil_path)
                return airfoil_path
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
        return airfoil_path

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

    def _clean_temp_for_label(self, label: str) -> None:
        """Deletes all temp files whose filename contains `label`."""
        for d in [self._airfoils_temp, self._polars_temp,
                  self._geo_temp, self._force_temp]:
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
        cst_l = list(row[self._cst_l_cols].values.round(8))
        cst_u = list(row[self._cst_u_cols].values.round(8))
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
        (df, outcome) where outcome is 'validated' or 'refused'.
        """
        def _status(text: str) -> None:
            if on_status is not None:
                on_status(text)

        row = df.loc[idx]
        stored_e: float = row['E']
        airfoil_path = self._source_airfoil(row, label)
        _status(f"{label}: sourcing airfoil / starting attempt 1")

        resumed = self._load_point_progress(idx)
        if resumed is not None:
            last_sim_e: Optional[float] = resumed['last_sim_e']
            consecutive_pass: int = resumed['consecutive_pass']
            attempt_log: List[dict] = resumed['attempt_log']
            start_attempt = resumed['next_attempt']
            self._log(
                f"\n  Point '{label}' (idx={idx}) | Stored E={stored_e:.8f} | "
                f"Resuming from attempt {start_attempt} "
                f"(consecutive_pass={consecutive_pass})"
            )
        else:
            last_sim_e = None
            consecutive_pass = 0
            attempt_log = []
            start_attempt = 1
            self._log(f"\n  Point '{label}' (idx={idx}) | Stored E={stored_e:.8f}")
        self._log(f"  Features: {self._format_row_for_log(row)}")

        for attempt in range(start_attempt, self.max_attempts + 1):
            attempt_label = f"{label}_a{attempt}"

            result = self._simulate_point(df.loc[idx], attempt_label, airfoil_path)

            # Clean temp files produced by this attempt before the next iteration
            self._clean_temp_for_label(attempt_label)

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
                            
                    log_msg = f"  VALIDATED | New E={sim_e:.8f} | "
                    if self.append_kp:
                        log_msg += "KP updated | "
                    log_msg += f"Total attempts used: {attempt}"
                    
                    self._log(log_msg)
                    _status(f"{label}: validated (E={sim_e:.4f})")
                    self._clear_point_progress()
                    self._clean_temp_for_label(label)
                    return df, 'validated'
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

    def _run_top_bottom_pass(
        self, on_progress: Callable[[int], None], on_status: Callable[[str], None],
        df: pd.DataFrame, direction: str, ascending: bool, n_target: int,
        processed_this_pass: int, point_counter: int,
        corrected_indices: set, refused_indices: set, stats: dict,
        passes_complete: set, total_rows: int,
    ) -> Tuple[pd.DataFrame, set, set, dict]:
        """Body of one top or bottom pass -- see run_correction for the dynamic re-sort rationale."""
        while processed_this_pass < n_target:
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
            self._log(f"\n--- [{direction.upper()}] Point {processed_this_pass + 1}/{n_target} ---")
            on_status(f"Starting {label} ({processed_this_pass + 1}/{n_target})")

            df, outcome = self._correct_point(df, next_idx, label, on_status=on_status)

            if outcome == 'validated':
                corrected_indices.add(next_idx)
                stats['validated'] += 1
            else:
                refused_indices.add(next_idx)
                stats['refused'] += 1

            processed_this_pass += 1
            on_progress(processed_this_pass)
            self._save_checkpoint(
                df, total_rows, n_target, passes_complete, direction,
                processed_this_pass, point_counter,
                corrected_indices, refused_indices, stats,
            )

        passes_complete.add(direction)
        self._save_checkpoint(
            df, total_rows, n_target, passes_complete, None, 0, 0,
            corrected_indices, refused_indices, stats,
        )
        return df, corrected_indices, refused_indices, stats

    def _run_middle_pass(
        self, on_progress: Callable[[int], None], on_status: Callable[[str], None],
        df: pd.DataFrame, n_middle: int, already_done: int,
        corrected_indices: set, refused_indices: set, stats: dict,
        passes_complete: set, total_rows: int, n_target: int,
    ) -> Tuple[pd.DataFrame, set, set, dict]:
        """Body of the middle pass -- see run_correction for why the candidate list is recomputed fresh."""
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
        for next_idx in middle_order:
            if next_idx in corrected_indices or next_idx in refused_indices:
                continue
            point_counter += 1
            label = f"middle_{point_counter}"
            self._log(f"\n--- [MIDDLE] Point {point_counter}/{n_middle} ---")
            on_status(f"Starting {label} ({point_counter}/{n_middle})")

            df, outcome = self._correct_point(df, next_idx, label, on_status=on_status)
            if outcome == 'validated':
                corrected_indices.add(next_idx)
                stats['validated'] += 1
            else:
                refused_indices.add(next_idx)
                stats['refused'] += 1

            processed_this_pass += 1
            on_progress(processed_this_pass)
            self._save_checkpoint(
                df, total_rows, n_target, passes_complete, 'middle',
                processed_this_pass, point_counter,
                corrected_indices, refused_indices, stats,
            )

        passes_complete.add('middle')
        self._save_checkpoint(
            df, total_rows, n_target, passes_complete, None, 0, 0,
            corrected_indices, refused_indices, stats,
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
        """
        progress_queue: "queue.Queue" = queue.Queue()
        worker_status_list = ["Starting..."]
        state = {'processed': initial_completed, 'error': None, 'result': None}

        def on_progress(processed_count: int) -> None:
            state['processed'] = processed_count
            progress_queue.put(1)

        def on_status(text: str) -> None:
            worker_status_list[0] = text

        def _target():
            try:
                state['result'] = pass_fn(on_progress=on_progress, on_status=on_status, **pass_kwargs)
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
                self._log(
                    f"Resuming from checkpoint: passes_complete={sorted(passes_complete)} | "
                    f"in-progress={resumed_pass} | "
                    f"validated={stats['validated']} | refused={stats['refused']}"
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
                stats = {'validated': 0, 'refused': 0}
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
                    self._log(f"  Resuming at point {processed_this_pass + 1}/{n_target}")
                else:
                    processed_this_pass = 0
                    point_counter = 0

                df, corrected_indices, refused_indices, stats = self._run_pass_with_gui(
                    pass_fn=self._run_top_bottom_pass,
                    pass_kwargs=dict(
                        df=df, direction=direction, ascending=ascending, n_target=n_target,
                        processed_this_pass=processed_this_pass, point_counter=point_counter,
                        corrected_indices=corrected_indices, refused_indices=refused_indices,
                        stats=stats, passes_complete=passes_complete, total_rows=total_rows,
                    ),
                    n_target=n_target,
                    initial_completed=processed_this_pass,
                    window_title="Database Correction Progress",
                    header_text=f"{direction.capitalize()} pass ({'descending' if not ascending else 'ascending'} E)",
                )

            if self.middle_fraction > 0 and 'middle' not in passes_complete:
                n_middle = max(1, int(total_rows * self.middle_fraction))
                already_done = resumed_processed if resumed_pass == 'middle' else 0
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