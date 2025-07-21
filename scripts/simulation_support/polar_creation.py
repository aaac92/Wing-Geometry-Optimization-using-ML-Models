import functools
import multiprocessing.pool
import os
import shutil
import ctypes
from pathlib import Path
from winpty import PtyProcess

class XFoilRunner:
    """
    Manages XFOIL simulations in a clean, temporary environment to avoid pathing issues.
    """

    def _create_hidden_directory(self, dir_path: Path):
        """Creates a directory if it doesn't exist and makes it hidden on Windows."""
        if not dir_path.exists():
            print(f"[INFO] Creating hidden directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
            try:
                # Set the hidden attribute on the directory
                FILE_ATTRIBUTE_HIDDEN = 0x02
                ctypes.windll.kernel32.SetFileAttributesW(str(dir_path), FILE_ATTRIBUTE_HIDDEN)
            except Exception as e:
                print(f"[WARNING] Could not set hidden attribute on {dir_path}. Error: {e}")

    def __init__(self, xfoil_exe_path, disk="P", dev_dirname="dev", airfoil_subdir="airfoils", polar_subdir="polars"):
        """
        Initializes the runner and sets up the temporary development environment.

        Args:
            xfoil_exe_path (str): The original, absolute path to the xfoil.exe executable.
            disk (str): The disk to create the temporary dev folder on (e.g., 'P', 'C').
            dev_dirname (str): The name of the main temporary directory.
            airfoil_subdir (str): The name of the subdirectory for temporary airfoils.
            polar_subdir (str): The name of the subdirectory for temporary polars.
        """
        print("="*50)
        print("Initializing XFoilRunner and Temporary Environment")
        print("="*50)

        # 1. Validate the original XFOIL path
        if not os.path.exists(xfoil_exe_path):
            raise FileNotFoundError(f"Original XFOIL executable not found at: {xfoil_exe_path}")

        # 2. Set up the temporary directory paths
        self.dev_dir = Path(f"{disk}:/{dev_dirname}")
        self.temp_xfoil_path = self.dev_dir / "xfoil.exe"
        self.airfoil_dir = self.dev_dir / airfoil_subdir
        self.polar_dir = self.dev_dir / polar_subdir

        # 3. Create the hidden parent directory
        self._create_hidden_directory(self.dev_dir)

        # 4. Copy xfoil.exe to the clean, temporary path
        print(f"[SETUP] Copying '{xfoil_exe_path}' to '{self.temp_xfoil_path}'")
        shutil.copy(xfoil_exe_path, self.temp_xfoil_path)

        # 5. Create the sub-directories
        self.airfoil_dir.mkdir(exist_ok=True)
        self.polar_dir.mkdir(exist_ok=True)

        print("\n[INFO] Temporary environment is ready.")
        print(f"  - Main Directory: {self.dev_dir}")
        print(f"  - XFOIL Path: {self.temp_xfoil_path}")
        print(f"  - Airfoil Temp Dir: {self.airfoil_dir}")
        print(f"  - Polar Temp Dir: {self.polar_dir}")
        print("="*50)


    def run_simulation(self, airfoil_path, polar_path, alfa_sequence, visc=2.88e6, mach=0.15, iter=200):
        """
        Executes an XFOIL simulation for a single airfoil file.

        Args:
            airfoil_path (str): The relative path to the source airfoil .dat file.
            polar_path (str): The final relative path to save the output polar file to.
            alfa_sequence (list): A list of angles of attack.
            visc (float): Reynolds number for the simulation.
            mach (float): Mach number for the simulation.
            iter (int): Maximum number of iterations for the solver.
        """
        # Ensure the final output directory for the polar exists
        os.makedirs(os.path.dirname(polar_path), exist_ok=True)

        # Define paths for temporary files for this specific run
        source_airfoil = Path(airfoil_path)
        temp_airfoil = self.airfoil_dir / source_airfoil.name
        temp_polar = self.polar_dir / Path(polar_path).name

        if not source_airfoil.exists():
            print(f"[ERROR] Source airfoil file not found: {source_airfoil}")
            return False

        # Copy airfoil to the temp directory for this run
        shutil.copy(source_airfoil, temp_airfoil)
        print(f"\n--- Starting XFOIL for {source_airfoil.name} ---")
        print(f"[INFO] Copied '{source_airfoil.name}' to '{temp_airfoil}' for simulation.")

        proc = None
        try:
            # --- LAUNCH XFOIL FROM THE CLEAN TEMP DIRECTORY ---
            proc = PtyProcess.spawn([str(self.temp_xfoil_path)])

            # --- SCRIPTED XFOIL COMMANDS ---
            # Load the temporary airfoil file
            if not self._send_command(proc, f"LOAD {temp_airfoil}"): return False
            if not self._send_command(proc, "PANE"): return False
            if not self._send_command(proc, "OPER", expect_prompt=".OPERi   c>"): return False
            if not self._send_command(proc, f"ITER {iter}", expect_prompt=".OPERi   c>"): return False
            if not self._send_command(proc, f"VISC {visc}", expect_prompt=".OPERv   c>"): return False
            if not self._send_command(proc, f"MACH {mach}", expect_prompt=".OPERv   c>"): return False

            # Set up polar accumulation using the temporary polar path
            if not self._send_command(proc, "PACC", expect_prompt='Enter  polar save filename  OR  <return> for no file   s>'): return False
            if not self._send_command(proc, str(temp_polar), expect_prompt='Enter  polar dump filename  OR  <return> for no file   s>'): return False
            if not self._send_command(proc, "", expect_prompt=".OPERva   c>"): return False

            # Run the angle of attack sequence
            for alfa in alfa_sequence:
                if not self._send_command(proc, f"ALFA {alfa}", expect_prompt=".OPERva   c>", timeout_sec=10):
                    print(f"[WARNING] Could not execute ALFA {alfa}. Re-initializing and continuing.")
                    self._send_command(proc, "INIT", expect_prompt=".OPERva   c>")
                    continue

            # --- FINISH AND CLEAN UP PROCESS ---
            print("[INFO] Finalizing XFOIL process.")
            if not self._send_command(proc, "PACC", expect_prompt=".OPERv   c>"): return False
            if not self._send_command(proc, ""): return False # Exit OPER mode
            if not self._send_command(proc, "QUIT", quit=True): return False
            proc.close()

            # --- COPY POLAR TO FINAL DESTINATION ---
            print(f"[SUCCESS] Simulation complete for {source_airfoil.name}.")
            if temp_polar.exists():
                print(f"[INFO] Copying result from '{temp_polar}' to '{polar_path}'.")
                shutil.copy(temp_polar, polar_path)
            else:
                print(f"[WARNING] Polar file was not created at '{temp_polar}'. Cannot copy result.")

            return True

        except Exception as e:
            print(f"[FATAL] An unexpected error occurred: {e}")
            return False
        finally:
            # Clean up the single temporary airfoil file used in this run
            if temp_airfoil.exists():
                os.remove(temp_airfoil)
            # Ensure the process is terminated if it's still alive
            if proc and proc.isalive():
                proc.close(force=True)
            print(f"--- Finished XFOIL for {source_airfoil.name} ---")

    def completion_flag(self, polar_path):
        try:
            with open(polar_path, "a") as f:
                f.write("\nXFOIL process completed successfully")
            print(f"[INFO] Completion flag added to {polar_path}")
        except IOError as e:
            print(f"[ERROR] Could not write completion flag to {polar_path}: {e}")
    
    def cleanup(self):
        """
        Deletes the entire temporary development directory.
        Call this once after all simulations are complete.
        """
        print("\n" + "="*50)
        print(f"Cleaning up temporary environment at: {self.dev_dir}")
        print("="*50)
        if self.dev_dir.exists():
            try:
                shutil.rmtree(self.dev_dir)
                print(f"[INFO] Successfully deleted directory: {self.dev_dir}")
            except OSError as e:
                print(f"[ERROR] Failed to delete directory {self.dev_dir}: {e}")
        else:
            print(f"[INFO] Directory {self.dev_dir} not found, skipping cleanup.")

    # _send_command and _timeout methods remain the same as your original script
    # (Including them here for completeness)

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
        # This function can remain as it was in your original script
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
            print(f"[ERROR] Timeout waiting for prompt '{expect_prompt}' after sending '{command}'.")
            return False
        success, _ = result_tuple
        if not success and not quit:
            print(f"[ERROR] Failed to get prompt '{expect_prompt}' after sending '{command}'.")
            return False
        return True