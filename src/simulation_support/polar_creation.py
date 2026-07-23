import functools
import logging
import multiprocessing.pool
import os
import traceback
from pathlib import Path
from winpty import PtyProcess
import psutil

# Logging helpers to find memory leak
class FsyncFileHandler(logging.FileHandler):
    """FileHandler that forces each record to physical disk immediately,
    so log lines survive a hard reset instead of sitting in OS page cache."""
    def emit(self, record):
        super().emit(record)
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except (OSError, ValueError):
            pass
def _resource_snapshot():
    try:
        p = psutil.Process()
        return p.num_handles(), p.num_threads()
    except Exception:
        return -1, -1

# Useful class
class Xfoil_Polar_Creator:
    """
    Manages XFOIL simulations in a clean, temporary environment to avoid pathing issues.
    """
    def __init__(self, xfoil_exe_path, main_dir, airfoil_dir, polar_dir, verbose:bool=False, process_logging:bool=False, mdes_seq:bool=False):
        """
        Initializes the runner and sets up the temporary development environment.

        Args:
            xfoil_exe_path (str): The original, absolute path to the xfoil.exe executable.
            main_dir (str): The working directory, ideally use the root of a system drive to avoid long paths with conflicting directory names.
            airfoil_dir (str): The airfoil directory inside the working directory.
            polar_dir (str): The polar directory inside the working directory.
            verbose (bool): Trigger for console messages.
            process_logging (bool): Used to log individual XFOIL process to debug a "memory leak", do not activate for normal use, only activate if the process crashes 
            mid session. DEACTIVATE IN DATABASE CORRECTION PHASE, logging will interfere with _rmtree and crash.
            mdes_seq (bool): Trigger for the MDES geometry smoothing sequence.
        """
        self.verbose = verbose
        self.process_logging = process_logging
        self.mdes_seq = mdes_seq

        if self.verbose:
            print("="*50)
            print("Initializing XFoilRunner and Temporary Environment")
            print("="*50)

        # Validate the original XFOIL path
        if not os.path.exists(xfoil_exe_path):
            raise FileNotFoundError(f"XFOIL executable not found at: {xfoil_exe_path}")

        # Set up directory paths
        self.main_dir = main_dir
        self.xfoil_path = xfoil_exe_path
        self.airfoil_dir = airfoil_dir
        self.polar_dir = Path(polar_dir)

        # Maybe not necessary
        #self.temp_xfoil_path = temp_dir / "support_files"

        # Copy xfoil.exe to the clean, temporary path
        #print(f"[SETUP] Copying '{xfoil_exe_path}' to '{self.temp_xfoil_path}'")
        #shutil.copy(xfoil_exe_path, self.temp_xfoil_path)

        # Create the sub-directory
        self.polar_dir.mkdir(exist_ok=True)

        if self.verbose:
            print("\n[INFO] Temporary environment is ready.")
            print(f"  - Main Directory: {self.main_dir}")
            print(f"  - XFOIL Path: {self.xfoil_path}")
            print(f"  - Airfoil Temp Dir: {self.airfoil_dir}")
            print(f"  - Polar Temp Dir: {self.polar_dir}")
            print("="*50)

        # Logging setup
        if self.process_logging:
            self._debug_logger = logging.getLogger(f"XPR.{os.getpid()}")
            self._debug_logger.setLevel(logging.DEBUG)
            if not self._debug_logger.handlers:
                log_dir = Path(main_dir) / "debug_logs"
                log_dir.mkdir(exist_ok=True)
                fh = FsyncFileHandler(log_dir / f"xfoil_debug_{os.getpid()}.log")
                fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
                self._debug_logger.addHandler(fh)
            self._spawn_count = 0
            self._close_count = 0

    def initialize_xfoil(self, airfoil_file_path, polar_file_path, iter, visc, mach):
        """
        Runs the standard initialization of XFOIL and tries to process rough airfoils, or skips that for special cases.
        
        :param airfoil_file_path: The relative path to the source airfoil .dat file.
        :param polar_file_path: The final relative path to save the output polar file to.
        :param iter: Number of XFOIL iterations before flagging non-convergence
        :param visc: Reynolds Number for the experiment using viscosity
        :param mach: Fluid Mach speed
        """
        for attempt in range(4):
            # --- LAUNCH XFOIL FROM THE CLEAN TEMP DIRECTORY --- #
            # BEGIN LOGGING BLOCK
            if self.process_logging:
                h, t = _resource_snapshot()
                self._spawn_count += 1
                self._debug_logger.debug(
                    f"SPAWN #{self._spawn_count} attempt={attempt} "
                    f"airfoil={Path(airfoil_file_path).name} handles={h} threads={t}"
                )
            proc = PtyProcess.spawn([str(self.xfoil_path)])
            # BEGIN LOGGING BLOCK
            if self.process_logging:
                self._debug_logger.debug(f"  spawned pid={getattr(proc, 'pid', 'n/a')}")

            try:
                # --- SCRIPTED XFOIL COMMANDS --- #
                if not self._send_command(proc, f"LOAD {airfoil_file_path}"): raise RuntimeError(f"LOAD for {airfoil_file_path} failed")
                if attempt < 3 and self.mdes_seq:
                    # Try the MDES sequence (cleans airfoil geometry for easier convergence in simulations)
                    mdes_success = True
                    if not self._send_command(proc, "MDES"): mdes_success = False
                    elif not self._send_command(proc, "FILT"): mdes_success = False
                    elif not self._send_command(proc, "EXEC"): mdes_success = False
                    elif not self._send_command(proc, ""): mdes_success = False

                    if not mdes_success:
                        raise RuntimeError("MDES sequence failed")
                if not self._send_command(proc, "PANE"): raise RuntimeError(f"PANE for {airfoil_file_path} failed")
                if not self._send_command(proc, "OPER", expect_prompt=".OPERi   c>"): raise RuntimeError(f"Failed to initiate OPER command for {airfoil_file_path}")
                if not self._send_command(proc, f"ITER {iter}", expect_prompt=".OPERi   c>"): raise RuntimeError(f"Failed to set ITER {iter} for {airfoil_file_path}")
                if not self._send_command(proc, f"VISC {visc}", expect_prompt=".OPERv   c>"): raise RuntimeError(f"Failed to set VISC {visc} for {airfoil_file_path}")
                if not self._send_command(proc, f"MACH {mach}", expect_prompt=".OPERv   c>"): raise RuntimeError(f"Failed to set MACH {mach} for {airfoil_file_path}")

                # Set up polar accumulation using the temporary polar path
                if not self._send_command(proc, "PACC", expect_prompt='Enter  polar save filename  OR  <return> for no file   s>'): raise RuntimeError(f"Failed to setup PACC for {airfoil_file_path}")
                if not self._send_command(proc, str(polar_file_path), expect_prompt='Enter  polar dump filename  OR  <return> for no file   s>'): raise RuntimeError(f"Setting the polar filename for {airfoil_file_path} failed")
                if not self._send_command(proc, "", expect_prompt=".OPERva   c>"): raise RuntimeError(f"Final confirmation for {airfoil_file_path}'s polar simulation failed")

                return proc

            except RuntimeError as e:
                # BEGIN LOGGING BLOCK
                if self.process_logging:
                    self._debug_logger.debug(f"  FAIL attempt={attempt} reason={e}")
                if proc:
                    try:
                        proc.close(force=True)
                        # BEGIN LOGGING BLOCK
                        if self.process_logging:
                            self._close_count += 1
                            self._debug_logger.debug(
                                f"  CLOSE #{self._close_count} ok "
                                f"(spawn_close_delta={self._spawn_count - self._close_count})"
                            )
                    except Exception as close_err:
                        # BEGIN LOGGING BLOCK
                        if self.process_logging:
                            self._debug_logger.debug(f"  CLOSE FAILED: {type(close_err).__name__}: {close_err}")
                    proc = None

                if attempt == 3:
                    return False
            
            continue # Restart the loop for the next attempt

        return False

    
    def run_simulation(self, airfoil_file_path, polar_file_path, alfa_sequence, visc=2.88e6, mach=0.15, iter=200):
        """
        Executes an XFOIL simulation for a single airfoil file.

        Args:
            airfoil_path (str): The relative path to the source airfoil .dat file.
            polar_file_path (str): The final relative path to save the output polar file to.
            alfa_sequence (list): A list of angles of attack.
            visc (float): Reynolds number for the simulation.
            mach (float): Mach number for the simulation.
            iter (int): Maximum number of iterations for the solver.
        """
        # Ensure the final output directory for the polar exists
        os.makedirs(os.path.dirname(polar_file_path), exist_ok=True)

        # Define paths for temporary files for this specific run
        airfoil_file_path = Path(airfoil_file_path)

        if self.verbose:
            print(f"\n--- Starting XFOIL for {airfoil_file_path.name} ---")

        proc = None
        try:
            # Initialize xfoil with given values
            proc = self.initialize_xfoil(airfoil_file_path, polar_file_path, iter, visc, mach)
            if proc is False:
                return False  # All internal retries in initialize_xfoil exhausted; abort this airfoil.

            # Run the angle of attack sequence
            for alfa in alfa_sequence:
                # Check for a valid process variable
                if proc != None:
                    if not self._send_command(proc, f"ALFA {alfa}", expect_prompt=".OPERva   c>", timeout_sec=10):
                        # Close process and make proc variable invalid, in case of failure.
                        print(f"[WARNING] Could not execute ALFA {alfa} for {airfoil_file_path.name}. Re-initializing and continuing.")
                        proc.close()
                        proc = None
                        continue
                else:
                    # Reinitilaize XFOIL and run current ALFA angle for invalid proc variable
                    proc = self.initialize_xfoil(airfoil_file_path, polar_file_path, iter, visc, mach)
                    if proc is False:
                        return False  # Re-init exhausted its retries; abort this airfoil.
                    if not self._send_command(proc, f"ALFA {alfa}", expect_prompt=".OPERva   c>", timeout_sec=10):
                        # Repeat failure logic
                        print(f"[WARNING] Could not execute ALFA {alfa}. Re-initializing and continuing.")
                        proc.close()
                        proc = None
                        continue    


            # --- FINISH AND CLEAN UP PROCESS ---
            if self.verbose:
                print("[INFO] Finalizing XFOIL process.")
            if not self._send_command(proc, "PACC", expect_prompt=".OPERv   c>"): return False
            if not self._send_command(proc, ""): return False # Exit OPER mode
            if not self._send_command(proc, "QUIT", quit=True): return False
            proc.close()

            self.completion_flag(polar_file_path)

            return True

        except Exception as e:
            # BEGIN LOGGING BLOCK
            if self.process_logging:
                self._debug_logger.debug(
                    f"[FATAL] airfoil={airfoil_file_path.name} "
                    f"error={type(e).__name__}: {e}\n{traceback.format_exc()}"
                )
            print(f"[FATAL] An unexpected error occurred: {e}")
            return False
        finally:
            # Ensure the process is terminated if it's still alive
            if proc and proc.isalive():
                try:
                    proc.close(force=True)
                    # BEGIN LOGGING BLOCK
                    if self.process_logging:
                        self._close_count += 1
                except Exception as close_err:
                    # BEGIN LOGGING BLOCK
                    if self.process_logging:
                        self._debug_logger.debug(f"  finally-CLOSE FAILED: {close_err}")
            # BEGIN LOGGING BLOCK
            if self.process_logging:
                h, t = _resource_snapshot()
                self._debug_logger.debug(
                    f"END airfoil={airfoil_file_path.name} handles={h} threads={t} "
                    f"spawn_close_delta={self._spawn_count - self._close_count}"
                )
            if self.verbose:
                print(f"# --- Finished XFOIL for {airfoil_file_path.name} --- #")

    def completion_flag(self, polar_path):
        try:
            with open(polar_path, "a") as f:
                f.write("\nXFOIL process completed successfully")
            if self.verbose:
                print(f"[INFO] Completion flag added to {polar_path}")
        except IOError as e:
            print(f"[ERROR] Could not write completion flag to {polar_path}: {e}")

    def _timeout(self, max_timeout):
        """Timeout decorator, parameter in seconds."""
        def timeout_decorator(item):
            @functools.wraps(item)
            def func_wrapper(*args, **kwargs):
                pool = multiprocessing.pool.ThreadPool(processes=1)
                async_result = pool.apply_async(item, args, kwargs)
                try:
                    return async_result.get(max_timeout)
                except multiprocessing.TimeoutError:
                    if self.verbose:
                        print(f"[ERROR] Function {item.__name__} timed out after {max_timeout} seconds.")
                    return None
                finally:
                    pool.terminate()
            return func_wrapper
        return timeout_decorator

    def _send_command(self, proc, command, expect_prompt="XFOIL   c>", timeout_sec=8, quit=False):
        """
        Sends a command to the XFOIL process and waits for the expected prompt.
        """
        if self.verbose:
            print(f"[CMD] Sending: {command if command else '-> <enter>'}")
        @self._timeout(max_timeout=timeout_sec)
        def timed_write(process, cmd, prompt):
            output_buffer = ""
            try:
                process.write(cmd + "\r\n")
                while True:
                    line = process.read(128)
                    if not line: return (False, output_buffer)
                    output_buffer += line
                    if prompt in output_buffer: return (True, output_buffer)
            except EOFError:
                return (False, output_buffer)

        result_tuple = timed_write(proc, command, expect_prompt)
        if result_tuple is None:
            if self.verbose:
                print(f"[ERROR] Timeout waiting for prompt '{expect_prompt}' after sending '{command}'.")
            return False
        success, _ = result_tuple
        if not success and not quit:
            if self.verbose:
                print(f"[ERROR] Failed to get prompt '{expect_prompt}' after sending '{command}'.")
            return False
        return True