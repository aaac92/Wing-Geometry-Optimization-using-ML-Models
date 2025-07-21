import os
import shutil
import multiprocessing.pool
import functools
from pathlib import Path
from winpty import PtyProcess
import ctypes

# === Timeout Decorator === #
def timeout(max_timeout, default=None):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            try:
                return async_result.get(max_timeout)
            except multiprocessing.TimeoutError:
                # The original print is kept but will only show if the function itself doesn't handle the debug flag.
                # The class method _send is designed to be quiet if debug is False.
                print(f"Function {item.__name__} timed out after {max_timeout} seconds.")
                return default
            finally:
                pool.terminate()
        return func_wrapper
    return timeout_decorator

class AVL_Run_Case:
    """
    A class to automate AVL simulations for a batch of geometry files.

    This class handles the creation of a temporary hidden directory,
    copies necessary files, runs AVL for each geometry, copies the
    results to a specified output folder, and cleans up all temporary files.
    """
    def __init__(self, geo_files: list, case_file: str, avl_exe: str, output_dir: str, debug: bool = False):
        """
        Initializes the AVLBatchProcessor.

        Args:
            geo_files (list): A list of absolute string paths to the geometry (.avl) files.
            case_file (str): An absolute string path to the case file.
            avl_exe (str): An absolute string path to the avl.exe executable.
            output_dir (str): An absolute string path to the folder where output force files will be saved.
            debug (bool, optional): If True, enables detailed console messages. Defaults to False.
        """
        self.geo_files = [Path(p) for p in geo_files]
        self.case_file = Path(case_file)
        self.avl_exe = Path(avl_exe)
        self.output_dir = Path(output_dir)
        self.debug = debug

        # --- Define temporary directory structure ---
        self.temp_dir = Path("P:/dev_avl")
        self.temp_geo_dir = self.temp_dir / "geo_files"
        self.temp_force_dir = self.temp_dir / "force_files"
        self.temp_avl_exe_path = self.temp_dir / self.avl_exe.name
        self.temp_case_file_path = self.temp_dir / self.case_file.name

    def _send(self, proc, command: str, expect_prompt: str = "AVL   c>"):
        """Sends a command to the AVL process and waits for the expected prompt."""
        if self.debug:
            print(f"[DEBUG] Sending command: {command}")
        try:
            proc.write(command + "\r\n")
            output = ""
            while True:
                line = proc.read(64)
                if not line:
                    if self.debug:
                        print("[DEBUG] No more output from process. EOF.")
                    return False
                output += line
                if expect_prompt in output:
                    if self.debug:
                        print(f"[DEBUG] Prompt '{expect_prompt}' received.")
                    return True
        except EOFError:
            if self.debug:
                print("[DEBUG] EOFError encountered.")
            return False

    def _setup_temp_dir(self):
        """Creates the temporary directory structure and copies necessary files."""
        if self.debug:
            print(f"[DEBUG] Setting up temporary directory at: {self.temp_dir}")

        # Create directories
        self.temp_dir.mkdir(exist_ok=True)
        self.temp_geo_dir.mkdir(exist_ok=True)
        self.temp_force_dir.mkdir(exist_ok=True)
        
        # --- Make the temp directory hidden (Windows specific) ---
        try:
            FILE_ATTRIBUTE_HIDDEN = 0x02
            ctypes.windll.kernel32.SetFileAttributesW(str(self.temp_dir), FILE_ATTRIBUTE_HIDDEN)
            if self.debug:
                print(f"[DEBUG] Set hidden attribute on {self.temp_dir}")
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Could not set hidden attribute on {self.temp_dir}: {e}")


        # --- Copy files to temp directory ---
        shutil.copy(self.avl_exe, self.temp_avl_exe_path)
        shutil.copy(self.case_file, self.temp_case_file_path)
        for geo_file in self.geo_files:
            shutil.copy(geo_file, self.temp_geo_dir / geo_file.name)
        
        if self.debug:
            print("[DEBUG] All necessary files copied to temporary directory.")

    def _copy_results(self):
        """Copies the generated force files from the temp folder to the final output directory."""
        if self.debug:
            print(f"[DEBUG] Copying results to: {self.output_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for force_file in self.temp_force_dir.glob('*.txt'):
            shutil.copy(force_file, self.output_dir / force_file.name)
            if self.debug:
                print(f"[DEBUG] Copied {force_file.name}")

    def _cleanup_temp_dir(self):
        """Removes the entire temporary directory."""
        if self.temp_dir.exists():
            # Remove hidden attribute before deleting to avoid permissions issues
            try:
                ctypes.windll.kernel32.SetFileAttributesW(str(self.temp_dir), 0x80) # FILE_ATTRIBUTE_NORMAL
            except Exception as e:
                 if self.debug:
                    print(f"[DEBUG] Could not remove hidden attribute: {e}")
            shutil.rmtree(self.temp_dir)
            if self.debug:
                print(f"[DEBUG] Temporary directory {self.temp_dir} has been removed.")

    def run(self):
        """
        Orchestrates the entire batch simulation process.
        Sets up the environment, runs AVL for each geometry file,
        copies results, and cleans up. This version does NOT change
        the script's working directory.
        """
        self._setup_temp_dir()
        
        # A try...finally block ensures cleanup happens even if errors occur
        try:
            timed_send = timeout(8)(self._send)

            # --- Loop through each geometry file ---
            for geo_file in self.temp_geo_dir.glob('*.avl'):
                if self.debug:
                    print(f"\n--- Processing {geo_file.name} ---")

                airfoil_name = geo_file.stem 
                force_file_path = self.temp_force_dir / f"force_{airfoil_name}.txt"

                # Launch avl.exe using its absolute path from the temp directory
                proc = PtyProcess.spawn(str(self.temp_avl_exe_path))
                
                # --- AVL Command Sequence (using absolute paths) ---
                if not timed_send(proc, f"LOAD {geo_file.resolve()}"): continue
                if not timed_send(proc, f"CASE {self.temp_case_file_path.resolve()}"): continue
                if not timed_send(proc, "OPER", expect_prompt='.OPER (case 1/1)   c>'): continue
                if not timed_send(proc, "X", expect_prompt='.OPER (case 1/1)   c>'): continue
                if not timed_send(proc, "FT", expect_prompt='Enter filename, or <return> for screen output   s>'): continue
                if not timed_send(proc, str(force_file_path.resolve()), expect_prompt='.OPER (case 1/1)   c>'): continue
                if not timed_send(proc, ""): continue 
                timed_send(proc, "QUIT")
                
                proc.close()
                if self.debug:
                    print(f"--- Finished processing {geo_file.name} ---")

            # --- Copy results back after all simulations are done ---
            self._copy_results()

        finally:
            # --- Cleanup ---
            self._cleanup_temp_dir()