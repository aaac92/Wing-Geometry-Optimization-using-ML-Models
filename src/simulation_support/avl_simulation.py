import os
import multiprocessing.pool
import functools
from pathlib import Path
from winpty import PtyProcess

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
    def __init__(self, geo_files: list, case_file: str, avl_exe_path: str, output_dir: str, debug: bool = False):
        """
        Initializes the AVLBatchProcessor.

        Args:
            geo_files (list): A list of absolute string paths to the geometry (.avl) files.
            case_file (str): An absolute string path to the case file.
            avl_exe (str): An absolute string path to the avl.exe executable.
            output_dir (str): An absolute string path to the folder where output force files will be saved.
            debug (bool, optional): If True, enables detailed console messages. Defaults to False.
        """
        if not os.path.exists(avl_exe_path):
            raise FileNotFoundError(f"AVL executable not found at: {avl_exe_path}")

        self.geo_files = [Path(p) for p in geo_files]
        self.case_file = case_file
        self.avl_exe = avl_exe_path
        self.output_dir = Path(output_dir)
        self.debug = debug

        self.output_dir.mkdir(exist_ok=True)

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

    def _execute_avl_commands(self, proc, timed_send, geo_file, force_file_path):
        """
        Executes the sequence of AVL commands.
        Returns True on success, False on timeout/failure.
        """
        try:
            # --- AVL Command Sequence ---
            if not timed_send(proc, f"LOAD {geo_file}"): return False
            if not timed_send(proc, f"CASE {self.case_file}"): return False
            if not timed_send(proc, "OPER", expect_prompt='.OPER (case 1/1)   c>'): return False
            if not timed_send(proc, "X", expect_prompt='.OPER (case 1/1)   c>'): return False
            if not timed_send(proc, "FT", expect_prompt='Enter filename, or <return> for screen output   s>'): return False
            if not timed_send(proc, str(force_file_path), expect_prompt='.OPER (case 1/1)   c>'): return False
            if not timed_send(proc, ""): return False # Advance to the AVL prompt
            timed_send(proc, "QUIT")
            return True
        except TimeoutError:
            if self.debug:
                print("A command timed out.")
            return False

    def run(self):
        """
        Orchestrates the entire batch simulation process.
        Runs AVL for each geometry file
        """
        
        # A try...finally block ensures cleanup happens even if errors occur
        try:
            retry = 3
            timed_send = timeout(30)(self._send)

            # --- Loop through each geometry file ---
            for geo_file in self.geo_files:
                success = False
                for attempt in range(1, retry + 1):
                    proc = None
                    try:
                        if self.debug:
                            print(f"\n--- Processing {geo_file.name} (Attempt {attempt}/{retry}) ---")

                        geometry_name = geo_file.stem.replace("_geo", "")
                        force_file_path = self.output_dir / f"force_{geometry_name}.txt"

                        # Launch avl.exe
                        proc = PtyProcess.spawn([str(self.avl_exe)])

                        # --- AVL Command Sequence (using absolute paths) ---
                        if self._execute_avl_commands(proc, timed_send, geo_file, force_file_path):
                            success = True
                            if self.debug:
                                print(f"--- Successfully processed {geo_file.name} ---")
                            break
                    except Exception as e:
                        # Catch unexpected errors during an attempt
                        if self.debug:
                            print(f"An unexpected error occurred on attempt {attempt}: {e}")
                    finally:
                        # Cleanup: ensure the process is closed after each attempt
                        if proc:
                            if proc.isalive():
                                proc.close(force=True)
                    if not success and self.debug:
                        print(f"Attempt {attempt} failed. Retrying...")
                
                # After all retries, if still not successful, raise an error
                if not success:
                    # This exception will be caught and logged in the avl runner.
                    raise RuntimeError(f"Failed to process {geo_file.name} after {retry} attempts.")

        except Exception as e:
            if self.debug:
                print(f"An error ocurred: {e}")