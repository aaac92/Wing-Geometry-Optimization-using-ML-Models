import logging
import os
import sys
import traceback
from pathlib import Path
import psutil
from .alpha_sweep import run_sweep

IS_WINDOWS = sys.platform == "win32"

# Windows drives XFOIL through winpty, Linux through pexpect
if IS_WINDOWS:
    import multiprocessing.pool
    from winpty import PtyProcess
else:
    import pexpect

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
        return (p.num_handles() if IS_WINDOWS else p.num_fds()), p.num_threads()
    except Exception:
        return -1, -1

# Useful class
class Xfoil_Polar_Creator:
    """
    Manages XFOIL simulations in a clean, temporary environment to avoid pathing issues.
    Works on Windows (winpty) and Linux (pexpect); the platform is detected automatically.
    """
    def __init__(self, xfoil_exe_path, main_dir, airfoil_dir, polar_dir, verbose:bool=False, process_logging:bool=False, mdes_seq:bool=False):
        """
        Initializes the runner and sets up the temporary development environment.

        Args:
            xfoil_exe_path (str): The original, absolute path to the xfoil executable (xfoil.exe on Windows).
            main_dir (str): The working directory. On Windows, ideally use the root of a system drive to avoid long paths.
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

    def _spawn(self):
        if IS_WINDOWS:
            return PtyProcess.spawn([str(self.xfoil_path)])
        return pexpect.spawn(str(self.xfoil_path), encoding='utf-8')

    def initialize_xfoil(self, airfoil_file_path, polar_file_path, iter, visc, mach):
        """
        Runs the standard initialization of XFOIL and tries to process rough airfoils.
        Returns a tuple containing the active process (or False) and the most recent error message.
        """
        last_error_msg = "Unknown initialization failure."
        for attempt in range(4):
            # --- LAUNCH XFOIL FROM THE CLEAN TEMP DIRECTORY --- #
            if self.process_logging:
                h, t = _resource_snapshot()
                self._spawn_count += 1
                self._debug_logger.debug(
                    f"SPAWN #{self._spawn_count} attempt={attempt} "
                    f"airfoil={Path(airfoil_file_path).name} handles/fds={h} threads={t}"
                )

            proc = self._spawn()

            if self.process_logging:
                self._debug_logger.debug(f"  spawned pid={getattr(proc, 'pid', 'n/a')}")

            try:
                # --- SCRIPTED XFOIL COMMANDS --- #
                if not self._send_command(proc, f"LOAD {airfoil_file_path}", timeout_sec=15): raise RuntimeError(f"LOAD for {airfoil_file_path} failed")
                if attempt < 3 and self.mdes_seq:
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

                # Handling floating point truncation overwrite prompt safely
                match_idx = self._send_command(proc, str(polar_file_path), expect_prompt=['Enter  polar dump filename  OR  <return> for no file   s>', "Set current parameters to old save file values ? y/n>"])
                if match_idx == -1:
                    raise RuntimeError(f"Setting the polar filename for {airfoil_file_path} as {polar_file_path} failed")
                elif match_idx == 1:
                    if not self._send_command(proc, "y", expect_prompt='Enter  polar dump filename  OR  <return> for no file   s>'): raise RuntimeError(f"Failed to confirm regime parameter overwrite for {airfoil_file_path}")
                    if not self._send_command(proc, "", expect_prompt=".OPERva   c>"): raise RuntimeError(f"Failed to omit polar dump file for {airfoil_file_path}.")
                elif match_idx == 0:
                    if not self._send_command(proc, "", expect_prompt=".OPERva   c>"): raise RuntimeError(f"Failed to omit polar dump file for {airfoil_file_path}.")

                return proc, "Success"

            except RuntimeError as e:
                last_error_msg = str(e)
                if self.process_logging:
                    self._debug_logger.debug(f"  FAIL attempt={attempt} reason={last_error_msg}")
                if proc:
                    try:
                        proc.close(force=True)
                        if self.process_logging:
                            self._close_count += 1
                            self._debug_logger.debug(
                                f"  CLOSE #{self._close_count} ok "
                                f"(spawn_close_delta={self._spawn_count - self._close_count})"
                            )
                    except Exception as close_err:
                        if self.process_logging:
                            self._debug_logger.debug(f"  CLOSE FAILED: {type(close_err).__name__}: {close_err}")
                    proc = None

                if attempt == 3:
                    return False, last_error_msg

            continue

        return False, last_error_msg

    def run_simulation(self, airfoil_file_path, polar_file_path, alfa_sequence, visc=2.88e6, mach=0.15, iter=200, sweep_mode=None):
        """
        Executes an XFOIL simulation for a single airfoil file and appends the completion flag on success.

        Args:
            airfoil_file_path (str): The path to the source airfoil .dat file.
            polar_file_path (str): The path to save the output polar file to.
            alfa_sequence (list): A list of angles of attack.
            visc (float): Reynolds number for the simulation.
            mach (float): Mach number for the simulation.
            iter (int): Maximum number of iterations for the solver.
            sweep_mode (str): None for the default ('zero_out'), or 'sequential' for the legacy sweep.
                See simulation_support.alpha_sweep before using 'sequential'.
        """
        os.makedirs(os.path.dirname(polar_file_path), exist_ok=True)
        return run_sweep(self, airfoil_file_path, polar_file_path, alfa_sequence, visc=visc, mach=mach, iter=iter, sweep_mode=sweep_mode)

    def alpha_pass(self, airfoil_file_path, polar_file_path, alfa_sequence, visc, mach, iter):
        """
        Walks one XFOIL process through alfa_sequence in the given order, writing every converged angle to
        polar_file_path. Does not write the completion flag. Returns True if the pass reached the end.
        """
        airfoil_file_path = Path(airfoil_file_path)

        if self.verbose:
            print(f"\n--- Starting XFOIL for {airfoil_file_path.name} ---")

        proc = None
        try:
            proc, init_msg = self.initialize_xfoil(airfoil_file_path, polar_file_path, iter, visc, mach)
            if proc is False:
                print(f"[INFO] {airfoil_file_path.name}: exhausted all initialization attempts. Reason: {init_msg}. Aborting run.")
                return False

            for alfa in alfa_sequence:
                attempts = 0
                success = False

                while attempts < 2 and not success:
                    if proc is None:
                        proc, init_msg = self.initialize_xfoil(airfoil_file_path, polar_file_path, iter, visc, mach)
                        if proc is False:
                            print(f"[INFO] {airfoil_file_path.name}: exhausted initialization during alpha sweep. Reason: {init_msg}.")
                            return False

                    if not self._send_command(proc, f"ALFA {alfa}", expect_prompt=".OPERva   c>", timeout_sec=10):
                        attempts += 1
                        print(f"[WARNING] Could not execute ALFA {alfa} for {airfoil_file_path.name} (Attempt {attempts}/2). Re-initializing.")
                        proc.close(force=True)
                        proc = None
                    else:
                        success = True

                if not success:
                    print(f"[WARNING] ALFA {alfa} completely failed for {airfoil_file_path.name}. Skipping this angle.")

            # --- FINISH AND CLEAN UP PROCESS ---
            if self.verbose:
                print("[INFO] Finalizing XFOIL process.")
            if proc is not None:
                if not self._send_command(proc, "PACC", expect_prompt=".OPERv   c>"): return False
                if not self._send_command(proc, ""): return False
                if not self._send_command(proc, "QUIT", quit=True): return False
                proc.close()

            return True

        except Exception as e:
            if self.process_logging:
                self._debug_logger.debug(
                    f"[FATAL] airfoil={airfoil_file_path.name} "
                    f"error={type(e).__name__}: {e}\n{traceback.format_exc()}"
                )
            print(f"[FATAL] An unexpected error occurred: {e}")
            return False
        finally:
            if proc and proc.isalive():
                try:
                    proc.close(force=True)
                    if self.process_logging:
                        self._close_count += 1
                except Exception as close_err:
                    if self.process_logging:
                        self._debug_logger.debug(f"  finally-CLOSE FAILED: {close_err}")

            if self.process_logging:
                h, t = _resource_snapshot()
                self._debug_logger.debug(
                    f"END airfoil={airfoil_file_path.name} handles/fds={h} threads={t} "
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

    def _send_command(self, proc, command, expect_prompt="XFOIL   c>", timeout_sec=8, quit=False):
        """
        Sends a command to the XFOIL process and waits for the expected prompt.
        expect_prompt may be a list: the index of the first prompt seen is returned, or -1 on failure.
        """
        is_list = isinstance(expect_prompt, list)
        failed = -1 if is_list else False
        if proc is None:
            if self.verbose:
                print(f"[ERROR] _send_command called with no active XFOIL process (command='{command}').")
            return failed

        if self.verbose:
            print(f"[CMD] Sending: {command if command else '-> <enter>'}")

        prompts = expect_prompt if is_list else [expect_prompt]
        if IS_WINDOWS:
            index, closed = self._winpty_exchange(proc, command, prompts, timeout_sec)
        else:
            index, closed = self._pexpect_exchange(proc, command, prompts, timeout_sec)

        if index is not None:
            return index if is_list else True
        if closed and quit:
            return 0 if is_list else True
        if self.verbose:
            reason = "XFOIL closed unexpectedly" if closed else "Timeout"
            print(f"[ERROR] {reason} while waiting for '{expect_prompt}' after sending '{command}'.")
        return failed

    def _pexpect_exchange(self, proc, command, prompts, timeout_sec):
        """Returns (index of the matched prompt or None, process closed)."""
        try:
            proc.sendline(command)
            return proc.expect_exact(prompts, timeout=timeout_sec), False
        except pexpect.TIMEOUT:
            return None, False
        except pexpect.EOF:
            return None, True

    def _winpty_exchange(self, proc, command, prompts, timeout_sec):
        """Returns (index of the matched prompt or None, process closed). winpty reads block, hence the thread timeout."""
        def exchange():
            output_buffer = ""
            try:
                proc.write(command + "\r\n")
                while True:
                    chunk = proc.read(128)
                    if not chunk:
                        return None, True
                    output_buffer += chunk
                    hits = [(output_buffer.find(p), i) for i, p in enumerate(prompts) if p in output_buffer]
                    if hits:
                        return min(hits)[1], False
            except EOFError:
                return None, True

        pool = multiprocessing.pool.ThreadPool(processes=1)
        try:
            return pool.apply_async(exchange).get(timeout_sec)
        except multiprocessing.TimeoutError:
            return None, False
        finally:
            pool.terminate()
