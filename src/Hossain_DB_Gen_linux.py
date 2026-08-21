import numpy as np
import os
import shutil
import threading
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from cst_modeling.foil import check_validity
from multiprocessing import Pool, Manager, cpu_count
# Import custom package class and methods
from simulation_support import Xfoil_Polar_Creator as XPR
from simulation_support import Key_Point_Extraction, HossainWingRec_Creation, HossainWing60DEG_Creation, HossainWingFlat_Creation, AVL_Run_Case
from geometric_support import Airfoil_Generator, Validate_Airfoil, create_modification_combinations
from file_handling_support import compression_Alma, Training_Data_Generator, Database_Corrector, get_all_files_from
from ux_support import Progress_GUI, recovery_json, recovery_Fabius

# Get the path to the src directory.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Define final directories
dataset_build_data = project_root / "data" / "dataset_build_data"
dataset_build_data.mkdir(exist_ok=True)
airfoils_dir = project_root / "data" / "dataset_build_data" / "airfoils"
force_files_dir = project_root / "data" / "dataset_build_data" / "force files"
polars_dir = project_root / "data" / "dataset_build_data" / "polars"
geo_files_dir = project_root / "data" / "dataset_build_data" / "geometry files"
support_files_dir = project_root / "data" / "support"
naca_kp_dir = project_root / "data" / "support" / "naca_kp"
logs_dir = project_root / "data" / "logs"
datasets_dir = project_root / "data" / "datasets"
binaries_dir = project_root / "bin"
# Define temp directories
# Temp directories with a short path are needed for XFOIL and AVL
training_temp_root = Path.home() / '.training_temp' # The dot prefix naturally hides the folder in Linux; user-writable, avoids needing root on '/'
os.makedirs(training_temp_root, exist_ok=True)
airfoils_temp_dir = training_temp_root / "airfoils"
force_files_temp_dir = training_temp_root / "force files"
polars_temp_dir = training_temp_root / "polars"
AI_results_temp_dir = training_temp_root / "AI results"
support_files_temp_dir = training_temp_root / "support_files"
naca_kp_temp_dir = training_temp_root / "support_files" / "naca_kp"
geo_files_temp_dir = training_temp_root / "geometry files"
training_files_temp_dir = training_temp_root / "training files"
# Define .rar paths and names for recovery logic
dataset_num = 17
unmended_dataset_path = project_root / "data" / "datasets" / f"dataset_{dataset_num}"
dataset_dir = datasets_dir / f"dataset_{dataset_num}"
airfoil_dataset_name = f"airfoils_dataset_{dataset_num}"
airfoil_rar_path = airfoils_dir / f"{airfoil_dataset_name}.rar"
polar_dataset_name = f"polars_dataset_{dataset_num}"
polar_rar_path = polars_dir / f"{polar_dataset_name}.rar"
geo_dataset_name = f"geo_files_dataset_{dataset_num}"
geo_rar_path = geo_files_dir / f"{geo_dataset_name}.rar"
force_dataset_name = f"force_files_dataset_{dataset_num}"
force_rar_path = force_files_dir / f"{force_dataset_name}.rar"
training_files_dataset_name = f"training_files_dataset_{dataset_num}"
training_files_rar_path = unmended_dataset_path / f"{training_files_dataset_name}.rar"
mended_files_dataset_name = f"final_training_files_dataset_{dataset_num}"
mended_files_rar_path = dataset_dir / f"{mended_files_dataset_name}.rar"

# === Recovery Handling === #
rec_json = recovery_json()
rec_Fabius = recovery_Fabius()

# === Aux objects === #
rar_Alma = compression_Alma()

# === Database Logic === #
def Airfoil_Database_Generation(base_airfoil_path, n_cst, cmb_datalist, generate_invalid=False, **kwargs):
    '''
    Fills a folder with generated airfoils matching the combination list and geometrically filtered for real plausible airfoils
    
    Parameters
    ----------
    base_airfoil_path : str, rstr
        Takes the absolute path to the base airfoil for modification

    n_cst : int
        Order of CST coefficients

    cmb_datalist : np.ndarray
        A list containing rows of cst coeeffiecients, format in Airfoil_Generator
    '''
    # Unpack filter parameters
    threshold_max_t = kwargs.get('threshold_max_t')
    threshold_min_t = kwargs.get('threshold_min_t')
    threshold_t_x_max = kwargs.get('threshold_t_x_max')
    threshold_t_x_min = kwargs.get('threshold_t_x_min')
    activate_curvature_cont = kwargs.get('activate_curvature_cont')
    threshold_rLE = kwargs.get('threshold_rLE')
    monotonic_activation = kwargs.get('monotonic_activation')
    threshold_max_camber = kwargs.get('threshold_max_camber')
    threshold_min_camber = kwargs.get('threshold_min_camber')

    # Recovery check
    if rec_Fabius.check_stage_status(stage_name="Airfoil Genesis", current_rar=airfoil_rar_path, required_deps=None):
        return

    os.makedirs(airfoils_temp_dir, exist_ok=True)
    invalid_output_dir = airfoils_temp_dir / "invalid airfoils"
    log_file_path = airfoils_temp_dir / "Invalid_airfoils_log.txt"

    # --- Recovery Logic --- #
    existing_airfoils = {os.path.splitext(f.name)[0] for f in airfoils_temp_dir.glob('*.dat')}
    start_index = 0
    if existing_airfoils:
        # Find the highest numbered existing airfoil to resume
        max_num = 0
        for name in existing_airfoils:
            try:
                num = int(name.split('_')[-1])
                if num > max_num:
                    max_num = num
            except (ValueError, IndexError):
                continue
        start_index = max_num

        # Delete the last airfoil file to prevent corruption issues
        last_airfoil_path = airfoils_temp_dir / f"Test_{start_index}.dat"
        if last_airfoil_path.exists():
            last_airfoil_path.unlink()
            start_index -= 1

        print(f"Resuming geometric database generation from airfoil Test_{start_index + 1}")

    # --- Generation Logic --- #
    airfoil_creator = Airfoil_Generator(
            base_airfoil_csv=base_airfoil_path,
            n_cst=n_cst,
            verbose=False    
        )
    for i, cmb in enumerate(tqdm(cmb_datalist, desc="Generating Airfoils", disable=False)):
        # Run the generation process
        airfoil_name = f"Test_{i+1}"
        x_mod, yu_mod, yl_mod = airfoil_creator.generate(
            cst_coeffs=cmb,
            show_plot=False,
            debug=False
            )
        
        # Silent check_validity before creating mod airfoil
        log_reason = None
        validity = check_validity(x_mod, yu_mod, yl_mod, verbose=False)

        if validity == True:
            valid = 'valid'
            try: # Wrapped in try block in case FoilGeoFeatures raises an unexpected error
                va = Validate_Airfoil(
                    x=x_mod, 
                    yu=yu_mod, 
                    yl=yl_mod,
                    cst_coeffs=cmb_datalist[i],
                    n_cst=n_cst,
                    name=airfoil_name,
                    verbose=False
                    )
                
                if valid == 'valid':
                    valid, log_reason = va.thickness_filters(th_t_min=threshold_min_t, th_t_max=threshold_max_t, th_t_x_max=threshold_t_x_max, th_t_x_min=threshold_t_x_min)
                if valid == 'valid':
                    valid, log_reason = va.sharpness_filter(curv_cont=activate_curvature_cont, th_rLE=threshold_rLE)
                if valid == 'valid':
                    if monotonic_activation:
                        valid, log_reason = va.monotonic_thickness_filter()
                if valid == 'valid':
                    valid, log_reason = va.camber_filters(th_c_max=threshold_max_camber, th_c_min=threshold_min_camber)

                if valid == 'invalid':
                    validity = False

                if validity:
                    airfoil_creator._create_dat_file(
                        output_dir=airfoils_temp_dir, 
                        n_cst = n_cst,
                        x=x_mod,
                        yl=yl_mod,
                        yu=yu_mod,
                        file_name=airfoil_name,
                        debug=False
                        )
            except ValueError as e:
                print(f"Could not create Geo Features Object due to unexpected error: {e}")
        else:
            cst_coeffs = cmb_datalist[i]
            cmb_l = np.array(cst_coeffs[:n_cst])
            cmb_u = np.array(cst_coeffs[n_cst:])
            reason = "Check_validity failed"
            log_line = f"{airfoil_name}.dat: cmb_l = {cmb_l}, cmb_u = {cmb_u} | Reason: {reason}\n"
            valid, log_reason = 'invalid', log_line
        
        if not validity: # If validity == False then we write to log, optionally save airfoil
            if generate_invalid:
                airfoil_creator._create_dat_file(
                    output_dir=invalid_output_dir,
                    n_cst = n_cst,
                    x=x_mod,
                    yl=yl_mod,
                    yu=yu_mod,
                    file_name=airfoil_name,
                    debug=False
                )
            with open(log_file_path, "a") as log:
                log.write(f"{log_reason}\n")

    # Save airfoils to main path in compressed form
    rar_Alma.compress_folder(airfoils_temp_dir, airfoils_dir, airfoil_dataset_name)

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Geometric database generation completed at {current_time}.")

    return

def _resolve_polar_status(polar_abs_path, ASEQ, completion_flag):
    '''
    Determines an existing polar file's status against ASEQ, and repairs the
    completion flag if the file already has every required angle but the
    flag write itself was interrupted (e.g. killed between the last ALFA and
    QUIT/close). Without this repair, such a file recomputes an empty angle
    list on every future resume attempt and is silently treated as "not
    complete" forever, since it never re-enters run_simulation() to reach
    the completion_flag() call.

    Returns:
        status (str): 'complete', 'partial', or 'corrupt'
        angles_remaining (list[float] or None): angles still needed;
            None for 'corrupt' (caller decides the full-range reprocessing).
    '''
    with open(polar_abs_path, 'r') as f:
        content = f.read()

    if completion_flag in content:
        return 'complete', []

    lines = content.splitlines()

    # XFOIL always emits "alpha CL CD CDp CM Top_Xtr Bot_Xtr" as the column
    # header, regardless of version or flow regime. Data rows only ever
    # appear below it (past its dashed underline), so scanning must be
    # scoped to that region -- otherwise numeric tokens in the metadata
    # above (e.g. "1 1 Reynolds number fixed ... Mach number fixed") can be
    # misread as a valid alpha/CL data row.
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().split()[:2] == ['alpha', 'CL']:
            header_idx = i
            break

    if header_idx is None:
        return 'corrupt', None  # Column header missing -- malformed file

    last_alfa = -999
    for line in reversed(lines[header_idx + 2:]):
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                last_alfa = float(parts[0])
                float(parts[1])
                break
            except (ValueError, IndexError):
                continue

    if last_alfa <= -999:
        return 'corrupt', None  # Header present but no data rows beneath it

    start_angle = last_alfa + ASEQ[2]
    num_steps = int((ASEQ[1] - start_angle) / ASEQ[2]) + 1
    angles_remaining = [start_angle + j * ASEQ[2] for j in range(num_steps)]

    if not angles_remaining:
        # Every required angle is already in the file -- repair the flag.
        try:
            with open(polar_abs_path, "a") as f:
                f.write(f"\n{completion_flag}")
        except IOError as e:
            print(f"[ERROR] Could not repair completion flag on {polar_abs_path}: {e}")
        return 'complete', []

    return 'partial', angles_remaining

def polar_worker(args):
    '''
    Worker function to run XFOIL simulation for a single airfoil.
    '''
    airfoil_abs_path, polar_abs_path, ASEQ, Re, Mach, binaries_dir, training_temp_root, airfoils_temp_dir, polars_temp_dir, progress_queue, index, worker_status_list, verbose, corrupted_files_list = args

    # Obtain airfoil name for GUI
    airfoil_name = os.path.basename(airfoil_abs_path)
    worker_status_list[index] = airfoil_name
    
    # --- Recovery Logic for each polar file --- #
    completion_flag = "XFOIL process completed successfully"
    if os.path.exists(polar_abs_path):
        status, angles_remaining = _resolve_polar_status(polar_abs_path, ASEQ, completion_flag)

        if status == 'complete':
            if progress_queue:
                progress_queue.put(1)
            return  # Skip if already completed (or just repaired)

        if status == 'partial':
            ANGLES = angles_remaining
        else: # 'corrupt' -- file exists but is empty or corrupt
            if verbose:
                print(f"[INFO] Corrupted file detected and deleted: {polar_abs_path}. Reverting to fresh file.")
            corrupted_files_list.append(polar_abs_path)
            try:
                os.remove(polar_abs_path)
            except OSError:
                pass
            num_steps = int((ASEQ[1] - ASEQ[0]) / ASEQ[2]) + 1
            ANGLES = [ASEQ[0] + j * ASEQ[2] for j in range(num_steps)]

    else: # File doesn't exist yet
         num_steps = int((ASEQ[1] - ASEQ[0]) / ASEQ[2]) + 1
         ANGLES = [ASEQ[0] + j * ASEQ[2] for j in range(num_steps)]

    if not ANGLES: # Nothing to compute
        if progress_queue:
            progress_queue.put(1)
            worker_status_list[index] = "Idle"
        return

    # --- Calculate Polar --- #
    try:
        xfoil_exe = os.path.join(binaries_dir, 'xfoil')
        # Create an instance of the runner
        xfoil_case = XPR(
            xfoil_exe_path=xfoil_exe,
            main_dir=training_temp_root,
            airfoil_dir=airfoils_temp_dir,
            polar_dir=polars_temp_dir,
            verbose=False
            )

        # Run the simulation
        xfoil_case.run_simulation(
            airfoil_file_path=airfoil_abs_path,
            polar_file_path=polar_abs_path,
            alfa_sequence=ANGLES,
            visc = Re,
            mach=Mach
        )

    except Exception as e:
        print(f"\n[ERROR] A critical error occurred while processing {os.path.basename(airfoil_abs_path)}: {e}")
    finally:
        if progress_queue:
            progress_queue.put(1)

def Polar_Database_Generation(ASEQ, Re, Mach, verbose=False):
    '''
    Runs XFOIL to create drag polars for each airfoil using multiprocessing and a GUI.

    Parameters:
    ==============
    ASEQ (list): List containing: 
                    ASEQ[0]: Smallest angle of attack to test.
                    ASEQ[1]: Largest angle of attack to test.
                    ASEQ[2]: Resolution for intermediate angles of attack.
    verbose (bool): Toggles prints and tqdm if any are present.
    '''
    # Recovery check
    rec_dependencies = [(airfoil_rar_path, airfoils_temp_dir)]
    if rec_Fabius.check_stage_status(stage_name="Polar Formulation", current_rar=polar_rar_path, required_deps=rec_dependencies):
        return

    airfoil_abs_paths = get_all_files_from(airfoils_temp_dir, extension='dat')
    if not airfoil_abs_paths:
        print("No airfoils found to process for polar generation.")
        return

    # --- Pre-allocation recovery filtering --- #
    # polar_worker() also checks this flag, but only after a task has already
    # been dispatched to a pool worker. On a resumed run that check has to be
    # re-run for every already-completed airfoil (potentially tens of
    # thousands), which is what was driving the rapid XFOIL spawn/kill cycling.
    # Filtering here removes completed airfoils from the queue before the Pool
    # is ever created, matching the intended recovery design.
    completion_flag = "XFOIL process completed successfully"
    complete_paths = []
    partial_paths = []   # polar file exists but no flag yet -> polar_worker resumes mid-sequence
    fresh_paths = []     # no polar file yet -> starts from ASEQ[0]

    corrupted_count = 0
    for file in airfoil_abs_paths:
        airfoil_name = os.path.splitext(os.path.basename(file))[0]
        polar_abs_path = os.path.join(polars_temp_dir, f"polar_{airfoil_name}.txt")
        if os.path.exists(polar_abs_path):
            status, _ = _resolve_polar_status(polar_abs_path, ASEQ, completion_flag)
            if status == 'complete':  # includes flag repaired just now
                complete_paths.append(file)
            elif status == 'partial':
                partial_paths.append(file)
            else: # 'corrupt'
                if verbose:
                    print(f"[INFO] Corrupted file detected and deleted: {polar_abs_path}. Reverting to fresh start.")
                corrupted_count += 1
                try:
                    os.remove(polar_abs_path)
                except OSError:
                    pass
                fresh_paths.append(file)
        else:
            fresh_paths.append(file)

    if corrupted_count and not verbose:
        print(f"[INFO] {corrupted_count} corrupted polar file(s) detected and deleted. Reverting to fresh start.")
            
    os.makedirs(polars_temp_dir, exist_ok=True)  # first-ever run: XPR only creates this once a task spawns
    polar_files_on_disk = get_all_files_from(polars_temp_dir, extension='txt')
    matched_polar_count = len(complete_paths) + len(partial_paths)
    print(
        f"[INFO] Recovery scan: {len(airfoil_abs_paths)} airfoil(s) found | "
        f"{len(polar_files_on_disk)} polar file(s) on disk | "
        f"{len(complete_paths)} confirmed complete (flagged) | "
        f"{len(partial_paths)} partial (will resume) | "
        f"{len(fresh_paths)} not started."
    )
    if len(polar_files_on_disk) != matched_polar_count:
        print(
            f"[WARNING] {len(polar_files_on_disk)} polar file(s) on disk vs "
            f"{matched_polar_count} matched to a current airfoil by name "
            f"({len(polar_files_on_disk) - matched_polar_count} unmatched) -- check for a naming mismatch."
        )

    pending_airfoil_paths = partial_paths + fresh_paths

    if not pending_airfoil_paths:
        print("[INFO] All airfoils already have completed polars. Nothing to dispatch.")
        rar_Alma.compress_folder(polars_temp_dir, polars_dir, polar_dataset_name)
        return

    num_workers = min(cpu_count() - 1, len(pending_airfoil_paths))

    with Manager() as manager:
        progress_queue = manager.Queue()
        worker_status_list = manager.list(["Initializing..."] * num_workers)
        corrupted_files_list = manager.list()

        tasks = []
        for i, file in enumerate(pending_airfoil_paths):
            airfoil_name = os.path.splitext(os.path.basename(file))[0]
            polar_abs_path = os.path.join(polars_temp_dir, f"polar_{airfoil_name}.txt")
            tasks.append((file, polar_abs_path, ASEQ, Re, Mach, binaries_dir, training_temp_root, airfoils_temp_dir, polars_temp_dir, progress_queue, i % num_workers, worker_status_list, verbose, corrupted_files_list))

        # Define the processing job to run in a separate thread
        def processing_job():
            with Pool(processes=num_workers, maxtasksperchild=100) as pool:
                pool.map(polar_worker, tasks)

        # --- Setup and run GUI ---
        header = f"Creating polars from {ASEQ[0]}° to {ASEQ[1]}° (Resolution: {ASEQ[2]}°)"
        gui = Progress_GUI(
            total_files=len(tasks),
            progress_queue=progress_queue,
            worker_status_list=worker_status_list,
            num_workers=num_workers,
            window_title="Polar Calculation Progress",
            header_text=header
        )

        # Start the processing in a background thread
        proc_thread = threading.Thread(target=processing_job)
        proc_thread.start()

        # Run the GUI in the main thread
        gui.run()

        proc_thread.join() # Wait for the background process to finish

        if corrupted_files_list and not verbose:
            print(f"[INFO] {len(corrupted_files_list)} corrupted polar file(s) detected and deleted. Reverting to fresh file.")

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[INFO] Polar database generation completed at {current_time}.")
    rar_Alma.compress_folder(polars_temp_dir, polars_dir, polar_dataset_name)
    
    return

def Geometric_File_Creation(configs):
    '''
    Generates AVL's .geo files that build the wing geometry, this script focuses on replicating Hossain's experiment so there are 3 wing configurations.
    '''
    # Recovery check
    rec_dependencies = [
        (airfoil_rar_path, airfoils_temp_dir),
        (polar_rar_path, polars_temp_dir)
    ]
    if rec_Fabius.check_stage_status(stage_name="AVL Geometric File Formation", current_rar=geo_rar_path, required_deps=rec_dependencies):
        return

    polars_abs_paths = get_all_files_from(polars_temp_dir)
    print("\n--- Extracting key aerodynamic points from polar data ---")
    key_points = Key_Point_Extraction(polar_file_paths=polars_abs_paths, verbose=True)
    key_points_data = key_points.process_files()
    KP_error_log = os.path.join(geo_files_temp_dir, 'KP_error_log.txt')
    key_points.save_error_log(KP_error_log)
    
    airfoil_abs_paths = get_all_files_from(airfoils_temp_dir, extension='dat')

    # NACA 0012 CDCL support file (AVL 3.32 requires section-local CDCL)
    naca_0012_kp_path = os.path.join(naca_kp_dir, "naca_0012_M0-9281_RE250k_M0-093_RE250000.txt")

    for creator_class, identifier in configs:
        geo_creator = creator_class(
            key_points_data=key_points_data,
            airfoil_abs_paths=airfoil_abs_paths,
            naca_key_points_data=naca_0012_kp_path,
            output_dir=str(geo_files_temp_dir),
            verbose=False, 
            pb=True
        )

        pattern = f"*_{identifier}_geo.avl"
        existing_type_files = list(geo_files_temp_dir.glob(pattern))

        if existing_type_files:
            existing_type_files.sort(key=lambda x: x.stat().st_mtime)
            last_file = existing_type_files[-1]
            
            try:
                last_file.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {last_file.name} for recovery: {e}")

        # --- File Creation & Logging ---
        print(f"\n--- Creating {identifier} geometry files ---")
        with tqdm(total=len(geo_creator.airfoil_path_map), desc=f"{identifier} geo files", disable=False) as bar:
            geo_creator.create_geo_files(progress_callback=bar.update)
        
        error_log_name = f'geo_{identifier}_error_log.txt'
        geo_creator.save_error_log(os.path.join(geo_files_temp_dir, error_log_name))

    rar_Alma.compress_folder(geo_files_temp_dir, geo_files_dir, geo_dataset_name)

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Geometric file for AVL database generated at {current_time}.")

    return

def avl_worker(args):
    '''
    Worker function to run AVL simulation for a single geometry file.
    '''
    geo_path, case_file, avl_exe, output_dir, progress_queue, index, worker_status_list = args

    # Obtain geometric file for GUI
    geo_name = os.path.basename(geo_path)
    worker_status_list[index] = geo_name
    
    try:
        avl_case = AVL_Run_Case(
            geo_files=[geo_path],
            case_file=case_file,
            avl_exe_path=avl_exe,
            output_dir=output_dir,
            debug=False
        )
        avl_case.run()
    except Exception as e:
        print(f"\n[ERROR] A critical error occurred while processing {os.path.basename(geo_path)}: {e}")
        error_log_path = os.path.join(output_dir, "avl_error.log")
        with open(error_log_path, "a") as log_file:
            log_file.write(f"Failed to process {geo_name}: {e}\n")
    finally:
        worker_status_list[index] = "Idle"
        if progress_queue:
            progress_queue.put(1)

def Final_Coefficient_Generation(verbose=False):
    '''
    Uses AVL to calculate the force coefficients from each geometric file using the global case file.
    '''
    # Recovery check
    rec_dependencies = [(geo_rar_path, geo_files_temp_dir)]
    if rec_Fabius.check_stage_status(stage_name="Force Computation", current_rar=force_rar_path, required_deps=rec_dependencies):
        return

    avl_exe = os.path.join(binaries_dir, 'avl')
    case_file = os.path.join(support_files_dir, "Hossain_sqwing_case.run")
    support_files_temp_dir.mkdir(exist_ok=True)
    case_file_temp = os.path.join(support_files_temp_dir, "Hossain_sqwing_case.run")
    shutil.copyfile(case_file, case_file_temp)

    geo_abs_paths = get_all_files_from(geo_files_temp_dir, extension='avl')
    existing_force_files = {
        f.stem.replace("force_", "") for f in force_files_temp_dir.glob('force_*.txt')
    }
    geo_files_to_process = [
        g for g in geo_abs_paths if Path(g).stem.replace("_geo", "") not in existing_force_files
    ]

    if not geo_files_to_process:
        print("All force files appear to be generated. Compressing now.")
        rar_Alma.compress_folder(force_files_temp_dir, force_files_dir, force_dataset_name)
        return

    num_workers = min(cpu_count() - 2, len(geo_files_to_process))

    # chunksize=1: AVL_Run_Case's retry loop (up to 3 attempts, ~30s timeout per
    # command) makes per-task duration highly variable. Any chunksize > 1 locks a
    # worker into finishing its whole assigned batch before it can pull more work,
    # so a worker stuck with several slow/retrying tasks stalls while workers with
    # "easy" chunks run out of work and idle. Per-task IPC overhead is negligible
    # next to multi-second AVL subprocess calls, so there's no real cost to this.
    chunksize = 1
    if existing_force_files:
        print(f"Resuming with {len(geo_files_to_process)} remaining simulations.")
    print(f"Distributing {len(geo_files_to_process)} tasks with a chunksize of {chunksize}.")
    with Manager() as manager:
        progress_queue = manager.Queue()
        worker_status_list = manager.list(["Initializing..."] * num_workers)

        tasks = [(path, case_file_temp, avl_exe, force_files_temp_dir, progress_queue, i % num_workers, worker_status_list) for i, path in enumerate(geo_files_to_process)]

        def processing_job():
            with Pool(processes=num_workers, maxtasksperchild=100) as pool:
                for _ in pool.imap_unordered(avl_worker, tasks, chunksize=chunksize):
                    pass

        # --- Setup and run GUI ---
        with open(case_file_temp, 'r') as f:
            header = f.readline().strip() # Read first line for header

        gui = Progress_GUI(
            total_files=len(tasks),
            progress_queue=progress_queue,
            worker_status_list=worker_status_list,
            num_workers=num_workers,
            window_title="AVL Coefficient Calculation",
            header_text=header
        )

        proc_thread = threading.Thread(target=processing_job)
        proc_thread.start()

        gui.run()

        proc_thread.join()

    rar_Alma.compress_folder(force_files_temp_dir, force_files_dir, force_dataset_name)

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Final simulation completed at {current_time}.")
    error_log_path = os.path.join(project_root, "avl_error.log")
    if os.path.exists(error_log_path):
        print(f"[WARNING] Some files failed to process. Check '{error_log_path}' for details.")

    return

def Final_Database_Generation(dataset_dir):
    '''
    Creates the final training files for the classifier and regressor models.
    '''
    # Recovery check
    rec_dependencies = [
        (airfoil_rar_path, airfoils_temp_dir),
        (geo_rar_path, geo_files_temp_dir),
        (force_rar_path, force_files_temp_dir)
    ]
    if rec_Fabius.check_stage_status(stage_name="Training Files Integration", current_rar=training_files_rar_path, required_deps=rec_dependencies):
        return

    invalid_airfoils_dir = airfoils_temp_dir / "Invalid_airfoils_log.txt"

    regression_file = f"regressor_dataset_{dataset_num}.csv"
    classification_file = f"classifier_dataset_{dataset_num}.csv"

    # Check if files need to be generated
    generate_regression = not (training_files_temp_dir / regression_file).exists() or (training_files_temp_dir / regression_file).stat().st_size == 0
    generate_classification = not (training_files_temp_dir / classification_file).exists() or (training_files_temp_dir / classification_file).stat().st_size == 0

    if generate_regression or generate_classification:
        generator_eff = Training_Data_Generator(
            force_files_dir = force_files_temp_dir,
            airfoils_dir = airfoils_temp_dir,
            invalid_airfoils_path = invalid_airfoils_dir,
            geo_files_dir = geo_files_temp_dir,
            output_dir = training_files_temp_dir,
            n_cst = n_cst,
            verbose = False,
            output_mode = 'Efficiency',
            append_kp=False
        )

        generator_eff.generate_training_files(
            regression_filename = regression_file,
            classification_filename = classification_file
            )

    rar_Alma.compress_folder(training_files_temp_dir, dataset_dir, training_files_dataset_name)

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Training files completed at {current_time}.")

    return

def Database_Correction(dataset_dir, n_cst, ASEQ, Re, Mach, configs):
    '''
    Executes validation and correction of outlier efficiency values (mostly due to buggy multiprocessing implementation) in the generated regression training database.
    '''
    # Recovery check
    rec_dependencies = [(training_files_rar_path, training_files_temp_dir)]
    if rec_Fabius.check_stage_status(stage_name="Database Mending", current_rar=mended_files_rar_path, required_deps=rec_dependencies):
        return

    regression_file = f"regressor_dataset_{dataset_num}.csv"
    csv_path = training_files_temp_dir / regression_file

    if not csv_path.exists():
        print(f"[ERROR] Target CSV {csv_path} not found. Verify recovery logic. Ensure Final_Database_Generation completed successfully.")
        return

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Database correction stage started at {current_time}.")

    base_airfoil_path = os.path.join(support_files_dir, 'naca0012_unit.csv')
    xfoil_exe = os.path.join(binaries_dir, 'xfoil')
    avl_exe = os.path.join(binaries_dir, 'avl')
    case_file = os.path.join(support_files_dir, "Hossain_sqwing_case.run")
    # Kept independent of training_temp_root (not nested under it) -- XFOIL/AVL
    # don't tolerate the resulting path length well.
    correction_temp_dir = Path.home() / '.correction_temp'
    naca_0012_kp_path = os.path.join(naca_kp_dir, "naca_0012_M0-9281_RE250k_M0-093_RE250000.txt")

    copernicus = Database_Corrector(
        csv_file_path=csv_path,
        base_airfoil_csv=base_airfoil_path,
        xfoil_exe_path=xfoil_exe,
        avl_exe_path=avl_exe,
        case_file_path=case_file,
        n_cst=n_cst,
        aseq=ASEQ,
        Re=Re,
        Mach=Mach,
        naca_key_points_data=naca_0012_kp_path,
        configs=configs,
        middle_fraction=0.01,
        source_airfoils_dir=airfoils_temp_dir,
        temp_dir=correction_temp_dir,
        verbose=False
    )
    
    copernicus.run_correction()

    # Compress the corrected dataset to a new archive to preserve the original
    rar_Alma.compress_folder(training_files_temp_dir, dataset_dir, mended_files_dataset_name)

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Database correction stage completed at {current_time}.")

    return

if __name__ == "__main__":
    # === Timer for Fun  === #
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start = datetime.now()
    print(f"Database generation started at: {start_time}.")
    # === Geometric Generation Setup === #
    base_airfoil = os.path.join(support_files_dir,'naca0012_unit.csv')
    n_cst = 5

    # Ranges for values that modify CST Curves (starting_value, end_value, step)
    ranges_upper = {
        'LE': (0, 0.125, 0.0625),       # range for value closest to LE
        'inter': (0, 0.125, 1/24),      # range for intermediate points
        'TE': (0, 0.125, 0.125)         # range for value closest to TE
    }
    ranges_lower = {
        'LE': (-0.125, 0, 0.0625),      # range for value closest to LE
        'inter': (0, 0.125, 1/24),      # range for intermediate points
        'TE': (0, 0.125, 0.125)         # range for value closest to TE
    }
    cmb_datalist = create_modification_combinations(n_cst, ranges_upper, ranges_lower, debug=True)

    # Wing configurations in case they are used
    configs = [
        (HossainWingRec_Creation, "rec"),
        (HossainWingFlat_Creation, "flat"),
        (HossainWing60DEG_Creation, "60DEG")
    ]
    num_configs = len(configs)

    # Geometric Filters thresholds and activations
    geo_filt_parm = {
        'threshold_max_t': 0.25,
        'threshold_min_t': 0.03,
        'threshold_t_x_max': 0.45,
        'threshold_t_x_min': 0.18,
        'activate_curvature_cont': True,
        'threshold_rLE': 0.01,
        'monotonic_activation': True,
        'threshold_max_camber': 0.12,
        'threshold_min_camber': -0.02
    }

    threshold_max_t = geo_filt_parm['threshold_max_t']
    threshold_min_t = geo_filt_parm['threshold_min_t']
    threshold_t_x_max = geo_filt_parm['threshold_t_x_max']
    threshold_t_x_min = geo_filt_parm['threshold_t_x_min']
    activate_curvature_cont = geo_filt_parm['activate_curvature_cont']
    threshold_rLE = geo_filt_parm['threshold_rLE']
    monotonic_activation = geo_filt_parm['monotonic_activation']
    threshold_max_camber = geo_filt_parm['threshold_max_camber']
    threshold_min_camber = geo_filt_parm['threshold_min_camber']

    # === Polar Alpha Range and Conditions === #
    ASEQ = [-20, 20, 0.5]
    Re = 2.5e5
    Mach = 0.09281

    # === Dataset Parameter Logging === # 
    # Dataset Number is a Global Variable, select at the top of the file.
    num_combinations = cmb_datalist.shape[0]*num_configs
    log_filename = logs_dir / f"Dataset_{dataset_num}_{num_combinations}_data_points.log"

    if not log_filename.exists():
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Dataset {dataset_num} Generation Parameters\n")
            log_file.write("="*30 + "\n")
            log_file.write(f"CST Order (n_cst): {n_cst}\n")
            log_file.write("\nGeometric Filters:\n")
            log_file.write(f"  thickness_filters: threshold_max_t = {threshold_max_t}, threshold_min_t = {threshold_min_t}, threshold_t_x_max = {threshold_t_x_max}, threshold_t_x_min = {threshold_t_x_min}\n")
            log_file.write(f"  sharpness_filters: activate_curvature_cont = {activate_curvature_cont}, threshold_rLE = {threshold_rLE}\n")
            log_file.write(f"  monotonic_thickness_filter: {monotonic_activation}\n")
            log_file.write(f"  camber_filters: threshold_max_camber = {threshold_max_camber}, threshold_min_camber = {threshold_min_camber}\n")
            log_file.write("\nCST Coefficient Ranges:\n")
            log_file.write("\nLower surface:\n")
            log_file.write(f"  {ranges_lower}\n")
            log_file.write("\nUpper surface:\n")
            log_file.write(f"  {ranges_upper}\n")
            if configs:
                log_file.write("\nUsing the following wing configurations:\n")
                # Loop through the list of tuples
                for config_obj, _ in configs:
                    # Use __name__ to get the string representation of the class
                    log_file.write(f" - {config_obj.__name__}\n")

    # === Execution with Recovery === #
    status = rec_json.get_recovery_status()

    if not status.get('airfoil_database_generation_complete'):
        Airfoil_Database_Generation(base_airfoil, n_cst, cmb_datalist, **geo_filt_parm)
        status['airfoil_database_generation_complete'] = True
        rec_json.save_recovery_status(status)

    if not status.get('polar_database_generation_complete'):
        Polar_Database_Generation(ASEQ, Re, Mach)
        status['polar_database_generation_complete'] = True
        rec_json.save_recovery_status(status)

    if not status.get('geometric_file_creation_complete'):
        Geometric_File_Creation(configs)
        status['geometric_file_creation_complete'] = True
        rec_json.save_recovery_status(status)

    if not status.get('final_coefficient_generation_complete'):
        Final_Coefficient_Generation()
        status['final_coefficient_generation_complete'] = True
        rec_json.save_recovery_status(status)

    if not status.get('final_database_generation_complete'):
        Final_Database_Generation(dataset_dir)
        status['final_database_generation_complete'] = True
        rec_json.save_recovery_status(status)

    if not status.get('database_correction_complete'):
        Database_Correction(dataset_dir, n_cst, ASEQ, Re, Mach, configs)
        status['database_correction_complete'] = True
        rec_json.save_recovery_status(status)

    # Execution completed successfully
    shutil.rmtree(training_temp_root)
    rec_json.recover_cleanup()

    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    end = datetime.now()
    total = end - start
    with open(log_filename, 'a') as log_file:
        log_file.write(f"\nExecution start: {start_time} | Execution end: {end_time}\n")
        log_file.write(f"Total execution time: {total}\n")
    print(f"Cleanup complete, database completed at {end_time}.")
    print(f"Total execution time: {total}")