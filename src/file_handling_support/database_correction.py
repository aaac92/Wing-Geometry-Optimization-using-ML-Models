import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd

from simulation_support import Xfoil_Polar_Creator as XPR
from simulation_support import Key_Point_Extraction, AVL_Run_Case
from geometric_support import Airfoil_Generator


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
    within `max_attempts` are deleted from the database.

    Supported dataset formats
    -------------------------
    - Efficiency mode only: last CSV column must be 'E'.
    - Single-configuration: provide `default_geo_creator_class`.
    - Multi-configuration (one-hot encoded): provide `configs` as a list of
      (class, identifier_string) tuples, matching Hossain_DB_Gen's format.

    Notes
    -----
    Key point columns (CLneg, CDneg, CLcdmin, CDmin, CLpos, CDpos) are always
    recomputed and overwritten for any validated point, since GPR-based key
    point extraction carries run-to-run variance.
    """

    # Known config identifiers, must match Training_Data_Generator.config_headers
    _KNOWN_CONFIGS: List[str] = ["rec", "flat", "60DEG"]

    def __init__(
        self,
        csv_file_path,
        base_airfoil_csv,
        xfoil_exe_path,
        avl_exe_path,
        case_file_path,
        n_cst: int,
        aseq: List[float],
        configs: Optional[List[Tuple[Type, str]]] = None,
        default_geo_creator_class: Optional[Type] = None,
        error_threshold: float = 0.02,
        max_attempts: int = 10,
        top_bottom_fraction: float = 0.05,
        middle_fraction: float = 0.0,
        temp_dir=None,
        verbose: bool = False,
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
        temp_dir : Path, optional
            Root directory for simulation intermediates.
            Defaults to C:\\db_correction_temp.
        verbose : bool
            If True, mirrors log output to stdout.
        """
        self.csv_file_path = Path(csv_file_path)
        self.base_airfoil_csv = Path(base_airfoil_csv)
        self.xfoil_exe_path = Path(xfoil_exe_path)
        self.avl_exe_path = Path(avl_exe_path)
        self.case_file_path = Path(case_file_path)
        self.n_cst = n_cst
        self.aseq = aseq
        self.error_threshold = error_threshold
        self.max_attempts = max_attempts
        self.top_bottom_fraction = top_bottom_fraction
        self.middle_fraction = middle_fraction
        self.verbose = verbose

        # Build internal config map: identifier string -> geo creator class
        self._config_map: Dict[str, Type] = {}
        if configs is not None:
            self._config_map = {identifier: cls for cls, identifier in configs}

        # Temp directory tree
        self._temp_root = Path(temp_dir) if temp_dir else Path(r"C:\db_correction_temp")
        self._airfoils_temp = self._temp_root / "airfoils"
        self._polars_temp = self._temp_root / "polars"
        self._geo_temp = self._temp_root / "geometry files"
        self._force_temp = self._temp_root / "force files"
        self._support_temp = self._temp_root / "support_files"

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

        # Key point column presence check
        self._kp_cols: List[str] = [
            "CLneg", "CDneg", "CLcdmin", "CDmin", "CLpos", "CDpos"
        ]
        missing_kp = [k for k in self._kp_cols if k not in columns]
        if missing_kp:
            raise ValueError(
                f"Expected key point columns not found in CSV: {missing_kp}."
                "Make sure database was created with the updated training_csv_creator."
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
        ]:
            d.mkdir(parents=True, exist_ok=True)
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
        kp = {k: round(float(row[k]), 6) for k in self._kp_cols}
        parts = [f"CST_lower={cst_l}", f"CST_upper={cst_u}", f"KP={kp}"]
        if self._config_cols:
            parts.append(f"Config={row[self._config_cols].to_dict()}")
        return " | ".join(parts)

    def _simulate_point(
        self, row: pd.Series, label: str
    ) -> Optional[Tuple[float, List[float]]]:
        """
        Executes the full aerodynamic simulation pipeline for one data point.

        Pipeline
        --------
        1. Generate airfoil geometry from CST coefficients (Airfoil_Generator).
        2. Compute drag polar (XFOIL).
        3. Extract key points via GPR interpolation (Key_Point_Extraction).
        4. Determine geo creator class from one-hot config encoding.
        5. Build AVL wing geometry file (geo creator class).
        6. Compute aerodynamic force coefficients (AVL_Run_Case).
        7. Return E = CLtot / CDtot and the six key point values.

        Parameters
        ----------
        row : pd.Series
            Data row containing CST coefficients and configuration encoding.
        label : str
            Unique label for temp file naming.

        Returns
        -------
        (sim_E, kp_values) on success, None on any failure.
        kp_values order: [CLneg, CDneg, CLcdmin, CDmin, CLpos, CDpos]
        """
        try:
            # Extract CST coefficients: lower surface first, then upper
            cst_coeffs = (
                list(row[self._cst_l_cols].values) +
                list(row[self._cst_u_cols].values)
            )

            # Airfoil geometry
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
            airfoil_path = self._airfoils_temp / f"{label}.dat"

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
            )
            xfoil_case.run_simulation(
                airfoil_file_path=airfoil_path,
                polar_file_path=str(polar_path),
                alfa_sequence=angles,
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
                'file_path': polar_path.name,
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
                output_dir=str(self._geo_temp),
                verbose=False,
                pb=False,
            )
            geo_creator.create_geo_files()

            avl_files = list(self._geo_temp.glob(f"*{label}*_geo.avl"))
            if not avl_files:
                self._log(
                    f"No .avl geo file found matching '*{label}*_geo.avl' "
                    "after geo creator ran.", 'error'
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
        self, df: pd.DataFrame, idx: int, label: str
    ) -> Tuple[pd.DataFrame, str]:
        """
        Validates and corrects a single data point.

        Validation criterion
        --------------------
        Two consecutive simulations must produce E values within
        `error_threshold` of each other. The reference for each comparison
        is the result of the immediately preceding simulation (or the stored
        CSV value for the first attempt). Values are written to the dataframe
        only upon confirmed validation. KP values are always updated on
        validation.

        Parameters
        ----------
        df : pd.DataFrame
            Working copy of the training database.
        idx : int
            Row index of the point to validate.
        label : str
            Unique label for temp file naming and log identification.

        Returns
        -------
        (df, outcome) where outcome is 'validated' or 'deleted'.
        """
        row = df.loc[idx]
        stored_e: float = row['E']
        self._log(f"\n  Point '{label}' (idx={idx}) | Stored E={stored_e:.8f}")
        self._log(f"  Features: {self._format_row_for_log(row)}")

        last_sim_e: Optional[float] = None
        consecutive_pass: int = 0
        attempt_log: List[dict] = []

        for attempt in range(1, self.max_attempts + 1):
            attempt_label = f"{label}_a{attempt}"

            result = self._simulate_point(df.loc[idx], attempt_label)

            # Clean temp files produced by this attempt before the next iteration
            self._clean_temp_for_label(attempt_label)

            if result is None:
                self._log(
                    f"  Attempt {attempt}/{self.max_attempts}: simulation failed.",
                    'warning',
                )
                consecutive_pass = 0
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

            if error <= self.error_threshold:
                consecutive_pass += 1
                if consecutive_pass == 2:
                    # Two consecutive agreements: write validated E and corresponding KP values
                    df.loc[idx, 'E'] = sim_e
                    for col, val in zip(self._kp_cols, kp_values):
                        df.loc[idx, col] = val
                    self._log(
                        f"  VALIDATED | New E={sim_e:.8f} | "
                        f"KP updated | Total attempts used: {attempt}"
                    )
                    return df, 'validated'
            else:
                consecutive_pass = 0

            last_sim_e = sim_e

        # Max attempts exhausted without two consecutive passes
        self._log(
            f"  FAILED: max_attempts={self.max_attempts} reached without "
            f"validation. Row at idx={idx} will be removed from DB.",
            'error',
        )
        self._log(f"  Full attempt history for idx={idx}:")
        for entry in attempt_log:
            self._log(
                f"    Attempt {entry['attempt']}: "
                f"sim_E={entry['sim_E']:.8f} | "
                f"ref={entry['reference']:.8f} | "
                f"error={entry['error']:.4%} | "
                f"KP={[round(v, 6) for v in entry['kp_values']]}"
            )

        df = df.drop(index=idx)
        return df, 'deleted'

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
        unprocessed extreme value.

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

        try:
            df = pd.read_csv(self.csv_file_path)
            total_rows = len(df)
            n_target = max(1, int(total_rows * self.top_bottom_fraction))
            self._log(
                f"Total rows: {total_rows} | "
                f"Target per top/bottom pass: {n_target} "
                f"({self.top_bottom_fraction:.1%})"
            )

            corrected_indices: set = set()
            deleted_indices: set = set()
            stats = {'validated': 0, 'deleted': 0}

            # ── Top and bottom passes (dynamic re-sort per iteration) ── #
            for direction, ascending in [('top', False), ('bottom', True)]:
                self._log(
                    f"\n=== {direction.upper()} pass "
                    f"({'descending' if not ascending else 'ascending'} E) ==="
                )
                processed_this_pass = 0
                point_counter = 0

                while processed_this_pass < n_target:
                    # Re-sort on current df values after each correction so that a corrected E that shifted rank is not re-selected.
                    sorted_idx = (
                        df['E'].sort_values(ascending=ascending).index.tolist()
                    )
                    next_idx = next(
                        (
                            i for i in sorted_idx
                            if i not in corrected_indices
                            and i not in deleted_indices
                        ),
                        None,
                    )
                    if next_idx is None:
                        self._log(f"No more unprocessed rows for {direction} pass.")
                        break

                    point_counter += 1
                    label = f"{direction}_{point_counter}"
                    self._log(
                        f"\n--- [{direction.upper()}] "
                        f"Point {processed_this_pass + 1}/{n_target} ---"
                    )

                    df, outcome = self._correct_point(df, next_idx, label)

                    if outcome == 'validated':
                        corrected_indices.add(next_idx)
                        stats['validated'] += 1
                    else:
                        deleted_indices.add(next_idx)
                        stats['deleted'] += 1

                    processed_this_pass += 1

            if self.middle_fraction > 0:
                n_middle = max(1, int(total_rows * self.middle_fraction))
                self._log(
                    f"\n=== MIDDLE pass: {n_middle} rows "
                    f"({self.middle_fraction:.1%}), selected by absolute "
                    f"deviation from median E ==="
                )
                median_e = df['E'].median()
                # Eligible: rows not already processed in top/bottom passes
                eligible_idx = df.index.difference(
                    pd.Index(list(corrected_indices | deleted_indices))
                )
                middle_order = (
                    df.loc[eligible_idx, 'E']
                    .sub(median_e)
                    .abs()
                    .sort_values(ascending=False)
                    .head(n_middle)
                    .index.tolist()
                )

                for point_counter, next_idx in enumerate(middle_order, start=1):
                    if next_idx in corrected_indices or next_idx in deleted_indices:
                        continue
                    label = f"middle_{point_counter}"
                    self._log(
                        f"\n--- [MIDDLE] Point {point_counter}/{n_middle} ---"
                    )
                    df, outcome = self._correct_point(df, next_idx, label)
                    if outcome == 'validated':
                        corrected_indices.add(next_idx)
                        stats['validated'] += 1
                    else:
                        deleted_indices.add(next_idx)
                        stats['deleted'] += 1

            # Overwrite the original CSV with corrected data
            df.to_csv(self.csv_file_path, index=False)
            self._log(f"\n=== Correction complete ===")
            self._log(f"Output saved to: {self.csv_file_path.name}")
            self._log(
                f"Validated: {stats['validated']} | "
                f"Deleted: {stats['deleted']} | "
                f"Remaining rows: {len(df)}/{total_rows}"
            )

        except Exception as e:
            self._log(f"Critical error during correction: {e}", 'error')
            raise

        finally:
            self._teardown_temp_dirs()
            elapsed = time.time() - start_time
            self._log(f"=== Finished in {elapsed:.2f}s ===")
            for handler in self._logger.handlers[:]:
                handler.close()
                self._logger.removeHandler(handler)