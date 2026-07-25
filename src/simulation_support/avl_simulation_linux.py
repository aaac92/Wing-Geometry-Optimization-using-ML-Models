import os
from pathlib import Path
import pexpect  # Replaces winpty and multiprocessing.pool

class AVL_Run_Case:
    """
    A class to automate AVL simulations for a batch of geometry files on Linux.

    This class handles the creation of a temporary directory,
    copies necessary files, runs AVL for each geometry, copies the
    results to a specified output folder, and cleans up all temporary files.
    """
    def __init__(self, geo_files: list, case_file: str, avl_exe_path: str, output_dir: str, debug: bool = False):
        """
        Initializes the AVLBatchProcessor.

        Args:
            geo_files (list): A list of absolute string paths to the geometry (.avl) files.
            case_file (str): An absolute string path to the case file.
            avl_exe_path (str): An absolute string path to the AVL executable.
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

    def _send(self, proc, command: str, expect_prompt: str = "AVL   c>", timeout_sec: int = 30):
        """Sends a command to the AVL process and waits for the expected prompt using pexpect."""
        if self.debug:
            print(f"[DEBUG] Sending command: {command}")
        try:
            # sendline automatically adds the necessary return carriage for Linux
            proc.sendline(command)
            # wait specifically for the exact string to appear in the terminal
            proc.expect_exact(expect_prompt, timeout=timeout_sec)
            
            if self.debug:
                print(f"[DEBUG] Prompt '{expect_prompt}' received.")
            return True
            
        except pexpect.TIMEOUT:
            if self.debug:
                print(f"[DEBUG] Timeout waiting for prompt '{expect_prompt}' after sending '{command}'.")
            return False
        except pexpect.EOF:
            # If the command was to quit, an EOF is a successful exit
            if command == "QUIT":
                return True
            if self.debug:
                print("[DEBUG] EOFError encountered (Process closed unexpectedly).")
            return False

    def _execute_avl_commands(self, proc, geo_file, force_file_path):
        """
        Executes the sequence of AVL commands.
        Returns True on success, False on timeout/failure.
        """
        # The timed_send wrapper is no longer needed since timeout is natively handled in _send
        if not self._send(proc, f"LOAD {geo_file}"): return False
        if not self._send(proc, f"CASE {self.case_file}"): return False
        if not self._send(proc, "OPER", expect_prompt='.OPER (case 1/1)   c>'): return False
        if not self._send(proc, "X", expect_prompt='.OPER (case 1/1)   c>'): return False
        if not self._send(proc, "FT", expect_prompt='Enter filename, or <return> for screen output   s>'): return False
        if not self._send(proc, str(force_file_path), expect_prompt='.OPER (case 1/1)   c>'): return False
        if not self._send(proc, ""): return False # Advance to the AVL prompt
        self._send(proc, "QUIT")
        return True

    def run(self):
        """
        Orchestrates the entire batch simulation process.
        Runs AVL for each geometry file
        """
        # A try...finally block ensures cleanup happens even if errors occur
        try:
            retry = 3

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

                        # Launch native Linux avl process
                        proc = pexpect.spawn(str(self.avl_exe), encoding='utf-8')

                        # --- AVL Command Sequence (using absolute paths) ---
                        if self._execute_avl_commands(proc, geo_file, force_file_path):
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
                        if proc and proc.isalive():
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